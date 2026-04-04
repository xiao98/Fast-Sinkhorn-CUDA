/**
 * @file sinkhorn_solver.cpp
 * @brief Host-side Sinkhorn solver orchestration.
 *
 * Manages GPU memory allocation, data transfers, kernel launches,
 * and convergence control for the log-domain Sinkhorn algorithm.
 */

#include "sinkhorn/sinkhorn_solver.h"
#include "sinkhorn/cost_matrix.h"
#include "sinkhorn/memory_profiler.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <numeric>

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
// Forward declarations of kernel launchers (defined in sinkhorn_kernel.cu)
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

float launchComputeTransportCost(
    const float* d_C, const float* d_alpha, const float* d_beta,
    const float* d_log_mu, const float* d_log_nu,
    float* d_row_costs, float* d_scalar,
    int n, int m, float eps);

void launchComputeTransportPlan(
    const float* d_C, const float* d_alpha, const float* d_beta,
    const float* d_log_mu, const float* d_log_nu, float* d_pi,
    int n, int m, float eps);

// ============================================================================
// SinkhornSolver::Impl — Private Implementation (PIMPL)
// ============================================================================

struct SinkhornSolver::Impl {
    SinkhornConfig config;
    MemoryProfiler profiler;

    // GPU device pointers
    float* d_C       = nullptr;  // Cost matrix [n × m]
    float* d_mu      = nullptr;  // Source distribution [n]
    float* d_nu      = nullptr;  // Target distribution [m]
    float* d_log_mu  = nullptr;  // log(μ) [n]
    float* d_log_nu  = nullptr;  // log(ν) [m]
    float* d_alpha   = nullptr;  // Dual potential α [n]
    float* d_beta    = nullptr;  // Dual potential β [m]
    float* d_errors  = nullptr;  // Per-row marginal errors [n]
    float* d_scalar  = nullptr;  // Scalar workspace [1]

    // Problem dimensions
    int n_ = 0;
    int m_ = 0;

    // CUDA events for timing
    cudaEvent_t start_event, stop_event;

    Impl(const SinkhornConfig& cfg) : config(cfg) {
        CUDA_CHECK(cudaEventCreate(&start_event));
        CUDA_CHECK(cudaEventCreate(&stop_event));
    }

    ~Impl() {
        freeDevice();
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
    }

    void allocateDevice(int n, int m) {
        freeDevice();
        n_ = n;
        m_ = m;

        CUDA_CHECK(cudaMalloc(&d_C,      n * m * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_mu,     n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_nu,     m * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_log_mu, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_log_nu, m * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_alpha,  n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_beta,   m * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_errors, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_scalar, sizeof(float)));

        // Initialize potentials to zero
        CUDA_CHECK(cudaMemset(d_alpha, 0, n * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_beta,  0, m * sizeof(float)));

        if (config.enable_profiling) {
            profiler.snapshot("After allocation");
        }
    }

    void freeDevice() {
        if (d_C)      { cudaFree(d_C);      d_C = nullptr; }
        if (d_mu)     { cudaFree(d_mu);     d_mu = nullptr; }
        if (d_nu)     { cudaFree(d_nu);     d_nu = nullptr; }
        if (d_log_mu) { cudaFree(d_log_mu); d_log_mu = nullptr; }
        if (d_log_nu) { cudaFree(d_log_nu); d_log_nu = nullptr; }
        if (d_alpha)  { cudaFree(d_alpha);  d_alpha = nullptr; }
        if (d_beta)   { cudaFree(d_beta);   d_beta = nullptr; }
        if (d_errors) { cudaFree(d_errors); d_errors = nullptr; }
        if (d_scalar) { cudaFree(d_scalar); d_scalar = nullptr; }
    }

    void uploadDistributions(const std::vector<float>& mu, const std::vector<float>& nu) {
        // Upload raw distributions
        CUDA_CHECK(cudaMemcpy(d_mu, mu.data(), n_ * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_nu, nu.data(), m_ * sizeof(float), cudaMemcpyHostToDevice));

        // Compute and upload log distributions
        std::vector<float> log_mu(n_), log_nu(m_);
        for (int i = 0; i < n_; ++i) log_mu[i] = logf(fmaxf(mu[i], 1e-30f));
        for (int j = 0; j < m_; ++j) log_nu[j] = logf(fmaxf(nu[j], 1e-30f));

        CUDA_CHECK(cudaMemcpy(d_log_mu, log_mu.data(), n_ * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_log_nu, log_nu.data(), m_ * sizeof(float), cudaMemcpyHostToDevice));
    }

    SinkhornResult runSinkhornIterations() {
        SinkhornResult result;
        float eps = config.epsilon;

        // Start timing
        CUDA_CHECK(cudaEventRecord(start_event));

        if (config.enable_profiling) {
            profiler.snapshot("Before iterations");
        }

        // ================================================================
        // Main Sinkhorn Loop
        // ================================================================
        int iter = 0;
        for (; iter < config.max_iterations; ++iter) {

            // Step 1: Update α given β
            //   α_i = -ε · LSE_j( (β_j - C_{ij}) / ε + log(ν_j) )
            launchUpdateAlpha(d_C, d_log_nu, d_beta, d_alpha, n_, m_, eps);

            // Step 2: Update β given α
            //   β_j = -ε · LSE_i( (α_i - C_{ij}) / ε + log(μ_i) )
            launchUpdateBeta(d_C, d_log_mu, d_alpha, d_beta, n_, m_, eps);

            // Step 3: Check convergence periodically
            if ((iter + 1) % config.convergence_check_interval == 0) {
                float error = launchComputeMarginalError(
                    d_C, d_alpha, d_beta, d_log_mu, d_log_nu, d_mu,
                    d_errors, d_scalar, n_, m_, eps);

                if (config.verbose) {
                    printf("  Iteration %d: marginal error = %.6e\n", iter + 1, error);
                }

                result.marginal_error = error;

                if (error < config.convergence_threshold) {
                    result.converged = true;
                    ++iter;
                    break;
                }
            }
        }
        result.iterations = iter;

        // ================================================================
        // Compute transport cost
        // ================================================================
        result.transport_cost = launchComputeTransportCost(
            d_C, d_alpha, d_beta, d_log_mu, d_log_nu,
            d_errors, d_scalar, n_, m_, eps);

        // Sinkhorn distance = transport_cost (the regularized objective)
        result.sinkhorn_distance = result.transport_cost;

        // Stop timing
        CUDA_CHECK(cudaEventRecord(stop_event));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        CUDA_CHECK(cudaEventElapsedTime(&result.elapsed_ms, start_event, stop_event));

        if (config.enable_profiling) {
            profiler.snapshot("After iterations");
            result.peak_memory_bytes = profiler.getPeakMemoryUsage();
        }

        return result;
    }
};

// ============================================================================
// SinkhornSolver — Public Interface Implementation
// ============================================================================

SinkhornSolver::SinkhornSolver(const SinkhornConfig& config)
    : pImpl_(new Impl(config)) {}

SinkhornSolver::~SinkhornSolver() { delete pImpl_; }

SinkhornSolver::SinkhornSolver(SinkhornSolver&& other) noexcept
    : pImpl_(other.pImpl_) { other.pImpl_ = nullptr; }

SinkhornSolver& SinkhornSolver::operator=(SinkhornSolver&& other) noexcept {
    if (this != &other) {
        delete pImpl_;
        pImpl_ = other.pImpl_;
        other.pImpl_ = nullptr;
    }
    return *this;
}

SinkhornResult SinkhornSolver::solve(
    const std::vector<float>& mu,
    const std::vector<float>& nu,
    const std::vector<float>& C,
    int n, int m)
{
    // Allocate GPU memory
    pImpl_->allocateDevice(n, m);

    // Upload cost matrix
    CUDA_CHECK(cudaMemcpy(pImpl_->d_C, C.data(), n * m * sizeof(float),
                           cudaMemcpyHostToDevice));

    // Upload distributions
    pImpl_->uploadDistributions(mu, nu);

    // Run solver
    return pImpl_->runSinkhornIterations();
}

SinkhornResult SinkhornSolver::solveFromPoints(
    const std::vector<float>& mu,
    const std::vector<float>& nu,
    const std::vector<float>& X,
    const std::vector<float>& Y,
    int n, int m, int dim)
{
    // Allocate GPU memory
    pImpl_->allocateDevice(n, m);

    // Upload point clouds and compute cost matrix on GPU
    float* d_X = nullptr;
    float* d_Y = nullptr;
    CUDA_CHECK(cudaMalloc(&d_X, n * dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Y, m * dim * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_X, X.data(), n * dim * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Y, Y.data(), m * dim * sizeof(float), cudaMemcpyHostToDevice));

    // Compute cost matrix on GPU
    computeSquaredEuclideanCost(d_X, d_Y, pImpl_->d_C, n, m, dim);

    // Free temporary point cloud buffers
    cudaFree(d_X);
    cudaFree(d_Y);

    // Upload distributions
    pImpl_->uploadDistributions(mu, nu);

    // Run solver
    return pImpl_->runSinkhornIterations();
}

void SinkhornSolver::getTransportPlan(std::vector<float>& pi, int n, int m) {
    pi.resize(n * m);
    float* d_pi = nullptr;
    CUDA_CHECK(cudaMalloc(&d_pi, n * m * sizeof(float)));

    launchComputeTransportPlan(
        pImpl_->d_C, pImpl_->d_alpha, pImpl_->d_beta,
        pImpl_->d_log_mu, pImpl_->d_log_nu, d_pi, n, m,
        pImpl_->config.epsilon);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(pi.data(), d_pi, n * m * sizeof(float),
                           cudaMemcpyDeviceToHost));
    cudaFree(d_pi);
}

void SinkhornSolver::getDualPotentials(std::vector<float>& alpha, std::vector<float>& beta) {
    int n = pImpl_->n_;
    int m = pImpl_->m_;
    alpha.resize(n);
    beta.resize(m);

    CUDA_CHECK(cudaMemcpy(alpha.data(), pImpl_->d_alpha, n * sizeof(float),
                           cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(beta.data(), pImpl_->d_beta, m * sizeof(float),
                           cudaMemcpyDeviceToHost));
}

} // namespace fastsinkhorn
