/**
 * @file convergence_profile.cpp
 * @brief Records marginal error at every convergence checkpoint.
 *
 * For each (N, epsilon) configuration, runs the Sinkhorn solver in
 * incremental batches of iterations.  After each batch the solver reports
 * its marginal error, giving a full convergence trace.
 *
 * Approach:
 *   Because the SinkhornSolver reinitialises dual potentials on every
 *   call to solve(), we cannot simply call it repeatedly in short bursts
 *   to accumulate an error trace.  Instead we use the kernel launch
 *   functions directly, managing GPU memory and the Sinkhorn loop on the
 *   host side, which mirrors the logic in sinkhorn_solver.cpp but records
 *   the error at every convergence check interval.
 *
 * Configurations:
 *   N       in {256, 1024, 4096}
 *   epsilon in {0.1, 0.01, 0.001}
 *
 * Output:  experiments/data/convergence_profile.csv
 * Columns: n, epsilon, iteration, marginal_error
 */

#include "sinkhorn/config.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <vector>

// ============================================================================
// Forward declarations of kernel launchers (from sinkhorn_kernel.cu)
// ============================================================================

namespace fastsinkhorn {

void launchUpdateAlpha(
    const float* d_C, const float* d_log_nu, const float* d_beta,
    float* d_alpha, int n, int m, float eps);

void launchUpdateBeta(
    const float* d_C, const float* d_log_mu, const float* d_alpha,
    float* d_beta, int n, int m, float eps);

float launchComputeMarginalError(
    const float* d_C, const float* d_alpha, const float* d_beta,
    const float* d_log_mu, const float* d_log_nu, const float* d_mu,
    float* d_errors, float* d_scalar,
    int n, int m, float eps);

} // namespace fastsinkhorn

using namespace fastsinkhorn;

// ============================================================================
// CUDA Error Checking
// ============================================================================

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                 \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

// ============================================================================
// Helpers
// ============================================================================

/**
 * @brief Generate a normalized Gaussian distribution on a uniform [0,1] grid.
 */
static std::vector<float> makeGaussian(int n, float center, float sigma) {
    std::vector<float> dist(n);
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        float x = static_cast<float>(i) / (n - 1);
        float val = expf(-0.5f * (x - center) * (x - center) / (sigma * sigma));
        dist[i] = val;
        sum += val;
    }
    for (int i = 0; i < n; ++i) dist[i] /= sum;
    return dist;
}

/**
 * @brief Build the squared-Euclidean cost matrix on a uniform [0,1] grid.
 */
static std::vector<float> buildCostMatrix(int N) {
    std::vector<float> C(static_cast<size_t>(N) * N);
    for (int i = 0; i < N; ++i) {
        float xi = static_cast<float>(i) / (N - 1);
        for (int j = 0; j < N; ++j) {
            float yj = static_cast<float>(j) / (N - 1);
            C[static_cast<size_t>(i) * N + j] = (xi - yj) * (xi - yj);
        }
    }
    return C;
}

// ============================================================================
// Core: run Sinkhorn loop with error recording
// ============================================================================

/**
 * @brief Run the log-domain Sinkhorn iteration loop, recording marginal
 *        error at every @p check_interval iterations.
 *
 * @param mu           Host source distribution (size N)
 * @param nu           Host target distribution (size N)
 * @param C            Host cost matrix (N x N, row-major)
 * @param N            Problem size (square: n = m = N)
 * @param eps          Regularization parameter
 * @param max_iter     Maximum number of Sinkhorn iterations
 * @param check_interval  Record error every this many iterations
 * @param csv          File handle to write CSV rows
 */
static void runConvergenceTrace(
    const std::vector<float>& mu,
    const std::vector<float>& nu,
    const std::vector<float>& C,
    int N, float eps, int max_iter, int check_interval,
    FILE* csv)
{
    // ------------------------------------------------------------------
    // Allocate GPU memory
    // ------------------------------------------------------------------
    float* d_C       = nullptr;
    float* d_mu      = nullptr;
    float* d_log_mu  = nullptr;
    float* d_log_nu  = nullptr;
    float* d_alpha   = nullptr;
    float* d_beta    = nullptr;
    float* d_errors  = nullptr;
    float* d_scalar  = nullptr;

    size_t mat_bytes = static_cast<size_t>(N) * N * sizeof(float);
    size_t vec_bytes = N * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_C,      mat_bytes));
    CUDA_CHECK(cudaMalloc(&d_mu,     vec_bytes));
    CUDA_CHECK(cudaMalloc(&d_log_mu, vec_bytes));
    CUDA_CHECK(cudaMalloc(&d_log_nu, vec_bytes));
    CUDA_CHECK(cudaMalloc(&d_alpha,  vec_bytes));
    CUDA_CHECK(cudaMalloc(&d_beta,   vec_bytes));
    CUDA_CHECK(cudaMalloc(&d_errors, vec_bytes));
    CUDA_CHECK(cudaMalloc(&d_scalar, sizeof(float)));

    // ------------------------------------------------------------------
    // Upload data
    // ------------------------------------------------------------------
    CUDA_CHECK(cudaMemcpy(d_C,  C.data(),  mat_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mu, mu.data(), vec_bytes, cudaMemcpyHostToDevice));

    // Compute log distributions on host, upload
    std::vector<float> log_mu(N), log_nu(N);
    for (int i = 0; i < N; ++i) log_mu[i] = logf(fmaxf(mu[i], 1e-30f));
    for (int i = 0; i < N; ++i) log_nu[i] = logf(fmaxf(nu[i], 1e-30f));

    CUDA_CHECK(cudaMemcpy(d_log_mu, log_mu.data(), vec_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_log_nu, log_nu.data(), vec_bytes, cudaMemcpyHostToDevice));

    // Initialise potentials to zero
    CUDA_CHECK(cudaMemset(d_alpha, 0, vec_bytes));
    CUDA_CHECK(cudaMemset(d_beta,  0, vec_bytes));

    // ------------------------------------------------------------------
    // Record initial error (iteration 0, before any updates)
    // ------------------------------------------------------------------
    {
        float err0 = launchComputeMarginalError(
            d_C, d_alpha, d_beta, d_log_mu, d_log_nu, d_mu,
            d_errors, d_scalar, N, N, eps);
        fprintf(csv, "%d,%.4g,%d,%.2e\n", N, eps, 0, err0);
        printf("    iter %5d  error = %.2e\n", 0, err0);
    }

    // ------------------------------------------------------------------
    // Sinkhorn loop
    // ------------------------------------------------------------------
    for (int iter = 0; iter < max_iter; ++iter) {
        launchUpdateAlpha(d_C, d_log_nu, d_beta, d_alpha, N, N, eps);
        launchUpdateBeta(d_C, d_log_mu, d_alpha, d_beta, N, N, eps);

        if ((iter + 1) % check_interval == 0) {
            float error = launchComputeMarginalError(
                d_C, d_alpha, d_beta, d_log_mu, d_log_nu, d_mu,
                d_errors, d_scalar, N, N, eps);

            fprintf(csv, "%d,%.4g,%d,%.2e\n", N, eps, iter + 1, error);

            if ((iter + 1) % (check_interval * 10) == 0 || iter + 1 <= check_interval) {
                printf("    iter %5d  error = %.2e\n", iter + 1, error);
            }

            // Stop early if converged to machine-level accuracy
            if (error < 1e-12f) {
                printf("    converged at iter %d (error < 1e-12)\n", iter + 1);
                break;
            }
        }
    }

    // ------------------------------------------------------------------
    // Free GPU memory
    // ------------------------------------------------------------------
    cudaFree(d_C);
    cudaFree(d_mu);
    cudaFree(d_log_mu);
    cudaFree(d_log_nu);
    cudaFree(d_alpha);
    cudaFree(d_beta);
    cudaFree(d_errors);
    cudaFree(d_scalar);
}

// ============================================================================
// Main
// ============================================================================

int main() {
    // ------------------------------------------------------------------
    // GPU information
    // ------------------------------------------------------------------
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    printf("=== Fast-Sinkhorn-CUDA: Convergence Profile ===\n");
    printf("GPU: %s\n", prop.name);
    printf("  Global memory: %.0f MB\n\n",
           prop.totalGlobalMem / (1024.0 * 1024.0));

    // ------------------------------------------------------------------
    // Experiment grid
    // ------------------------------------------------------------------
    const std::vector<int>   sizes    = {256, 1024, 4096};
    const std::vector<float> epsilons = {0.1f, 0.01f, 0.001f};
    const int kMaxIter       = 5000;
    const int kCheckInterval = 10;

    // ------------------------------------------------------------------
    // Open output CSV
    // ------------------------------------------------------------------
    FILE* csv = fopen("experiments/data/convergence_profile.csv", "w");
    if (!csv) {
        fprintf(stderr, "ERROR: Cannot open "
                        "experiments/data/convergence_profile.csv for writing.\n");
        return 1;
    }
    fprintf(csv, "n,epsilon,iteration,marginal_error\n");

    // ------------------------------------------------------------------
    // Sweep
    // ------------------------------------------------------------------
    for (int N : sizes) {
        // Check GPU memory
        size_t free_mem = 0, total_mem = 0;
        cudaMemGetInfo(&free_mem, &total_mem);
        size_t required = static_cast<size_t>(N) * N * sizeof(float) * 3;
        if (free_mem < required) {
            printf("Skipping N=%d: insufficient GPU memory\n", N);
            continue;
        }

        auto mu = makeGaussian(N, 0.3f, 0.08f);
        auto nu = makeGaussian(N, 0.7f, 0.08f);
        auto C  = buildCostMatrix(N);

        for (float eps : epsilons) {
            printf("  N=%d  eps=%.3f\n", N, eps);
            runConvergenceTrace(mu, nu, C, N, eps,
                                kMaxIter, kCheckInterval, csv);
            printf("\n");
        }
    }

    fclose(csv);
    printf("Results written to experiments/data/convergence_profile.csv\n");
    return 0;
}
