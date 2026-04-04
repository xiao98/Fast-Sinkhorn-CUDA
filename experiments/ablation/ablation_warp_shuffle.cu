/**
 * @file ablation_warp_shuffle.cu
 * @brief Ablation: warp-shuffle reduction vs shared-memory-only reduction.
 *
 * Compiles both kernel variants into one binary and times each.
 * Output: experiments/data/ablation_warp.csv
 */

#include "sinkhorn/config.h"
#include "sinkhorn/sinkhorn_solver.h"
#include <cuda_runtime.h>
#include <cfloat>
#include <cstdio>
#include <cmath>
#include <vector>
#include <numeric>
#include <string>

using namespace fastsinkhorn;

// ============================================================================
// Shared-memory-ONLY reduction (no warp shuffle) — the ablation variant
// ============================================================================

/**
 * @brief Max reduction using ONLY shared memory (no __shfl_down_sync).
 */
__device__ float sharedMemReduceMax(float val) {
    __shared__ float sdata[256];  // Assumes kBlockSize <= 256
    int tid = threadIdx.x;
    sdata[tid] = val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    return sdata[0];
}

/**
 * @brief Sum reduction using ONLY shared memory (no __shfl_down_sync).
 */
__device__ float sharedMemReduceSum(float val) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    sdata[tid] = val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    return sdata[0];
}

/**
 * @brief Alpha update using shared-memory-only reduction.
 */
__global__ void updateAlphaSharedMemOnly(
    const float* __restrict__ C,
    const float* __restrict__ log_nu,
    const float* __restrict__ beta,
    float*       __restrict__ alpha,
    int n, int m, float eps)
{
    const int i = blockIdx.x;
    if (i >= n) return;

    const float inv_eps = 1.0f / eps;
    const float* Ci = C + i * m;

    float thread_max = -FLT_MAX;
    for (int j = threadIdx.x; j < m; j += blockDim.x) {
        float val = (beta[j] - Ci[j]) * inv_eps + log_nu[j];
        thread_max = fmaxf(thread_max, val);
    }
    float row_max = sharedMemReduceMax(thread_max);

    float thread_sum = 0.0f;
    for (int j = threadIdx.x; j < m; j += blockDim.x) {
        float val = (beta[j] - Ci[j]) * inv_eps + log_nu[j];
        thread_sum += expf(val - row_max);
    }
    float row_sum = sharedMemReduceSum(thread_sum);

    if (threadIdx.x == 0) {
        alpha[i] = -eps * (row_max + logf(fmaxf(row_sum, 1e-30f)));
    }
}

/**
 * @brief Beta update using shared-memory-only reduction.
 */
__global__ void updateBetaSharedMemOnly(
    const float* __restrict__ C,
    const float* __restrict__ log_mu,
    const float* __restrict__ alpha,
    float*       __restrict__ beta,
    int n, int m, float eps)
{
    const int j = blockIdx.x;
    if (j >= m) return;

    const float inv_eps = 1.0f / eps;

    float thread_max = -FLT_MAX;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float val = (alpha[i] - C[i * m + j]) * inv_eps + log_mu[i];
        thread_max = fmaxf(thread_max, val);
    }
    float col_max = sharedMemReduceMax(thread_max);

    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float val = (alpha[i] - C[i * m + j]) * inv_eps + log_mu[i];
        thread_sum += expf(val - col_max);
    }
    float col_sum = sharedMemReduceSum(thread_sum);

    if (threadIdx.x == 0) {
        beta[j] = -eps * (col_max + logf(fmaxf(col_sum, 1e-30f)));
    }
}

// ============================================================================
// Marginal error kernel (uses shared-mem reduction for the ablation path)
// ============================================================================

__global__ void computeMarginalErrorSharedMem(
    const float* __restrict__ C,
    const float* __restrict__ alpha,
    const float* __restrict__ beta,
    const float* __restrict__ log_mu,
    const float* __restrict__ log_nu,
    const float* __restrict__ mu,
    float*       __restrict__ errors,
    int n, int m, float eps)
{
    const int i = blockIdx.x;
    if (i >= n) return;

    const float inv_eps = 1.0f / eps;
    const float* Ci = C + i * m;

    float thread_max = -FLT_MAX;
    for (int j = threadIdx.x; j < m; j += blockDim.x) {
        float val = (alpha[i] + beta[j] - Ci[j]) * inv_eps + log_nu[j];
        thread_max = fmaxf(thread_max, val);
    }
    float row_max = sharedMemReduceMax(thread_max);

    float thread_sum = 0.0f;
    for (int j = threadIdx.x; j < m; j += blockDim.x) {
        float val = (alpha[i] + beta[j] - Ci[j]) * inv_eps + log_nu[j];
        thread_sum += expf(val - row_max);
    }
    float row_sum = sharedMemReduceSum(thread_sum);

    if (threadIdx.x == 0) {
        float log_ri = log_mu[i] + row_max + logf(fmaxf(row_sum, 1e-30f));
        float ri = expf(log_ri);
        errors[i] = fabsf(ri - mu[i]);
    }
}

__global__ void sumReduceShared(const float* input, float* output, int n) {
    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        thread_sum += input[i];
    }
    float total = sharedMemReduceSum(thread_sum);
    if (threadIdx.x == 0) output[0] = total;
}

// ============================================================================
// Standalone solver loop for shared-memory-only variant
// ============================================================================

struct AblationResult {
    float time_ms;
    int iterations;
    float transport_cost;
    bool converged;
};

AblationResult runSharedMemOnlySolver(
    const float* d_C, const float* d_mu, const float* d_nu,
    const float* d_log_mu, const float* d_log_nu,
    int n, int m, float eps, int max_iter, float threshold, int check_interval)
{
    float *d_alpha, *d_beta, *d_errors, *d_scalar;
    cudaMalloc(&d_alpha, n * sizeof(float));
    cudaMalloc(&d_beta, m * sizeof(float));
    cudaMalloc(&d_errors, n * sizeof(float));
    cudaMalloc(&d_scalar, sizeof(float));
    cudaMemset(d_alpha, 0, n * sizeof(float));
    cudaMemset(d_beta, 0, m * sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    AblationResult result = {};
    int iter = 0;
    for (; iter < max_iter; ++iter) {
        updateAlphaSharedMemOnly<<<n, kBlockSize>>>(d_C, d_log_nu, d_beta, d_alpha, n, m, eps);
        updateBetaSharedMemOnly<<<m, kBlockSize>>>(d_C, d_log_mu, d_alpha, d_beta, n, m, eps);

        if ((iter + 1) % check_interval == 0) {
            computeMarginalErrorSharedMem<<<n, kBlockSize>>>(
                d_C, d_alpha, d_beta, d_log_mu, d_log_nu, d_mu, d_errors, n, m, eps);
            sumReduceShared<<<1, kBlockSize>>>(d_errors, d_scalar, n);

            float error;
            cudaMemcpy(&error, d_scalar, sizeof(float), cudaMemcpyDeviceToHost);
            if (error < threshold) {
                result.converged = true;
                ++iter;
                break;
            }
        }
    }
    result.iterations = iter;

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&result.time_ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_alpha);
    cudaFree(d_beta);
    cudaFree(d_errors);
    cudaFree(d_scalar);

    return result;
}

// ============================================================================
// Helpers
// ============================================================================

std::vector<float> makeGaussian(int n, float center, float sigma) {
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

// ============================================================================
// Main
// ============================================================================

int main() {
    const float epsilon = 0.01f;
    const int max_iter = 2000;
    const float threshold = 1e-6f;
    const int check_interval = 10;
    const int warmup_runs = 3;
    const int timed_runs = 10;

    std::vector<int> sizes = {512, 1024, 2048, 4096};

    // Print GPU info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n", prop.name);
    printf("Ablation: Warp Shuffle vs Shared Memory Only\n\n");

    // Open output file
    FILE* fp = fopen("experiments/data/ablation_warp.csv", "w");
    if (!fp) { fprintf(stderr, "Cannot open output file\n"); return 1; }
    fprintf(fp, "config,n,epsilon,time_ms_mean,time_ms_std,iterations,converged,notes\n");

    for (int N : sizes) {
        printf("N = %d:\n", N);

        auto mu = makeGaussian(N, 0.3f, 0.08f);
        auto nu = makeGaussian(N, 0.7f, 0.08f);

        // Build cost matrix
        std::vector<float> C(N * N);
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j) {
                float xi = float(i) / (N - 1), yj = float(j) / (N - 1);
                C[i * N + j] = (xi - yj) * (xi - yj);
            }

        // Upload to GPU
        float *d_C, *d_mu_dev, *d_nu_dev, *d_log_mu, *d_log_nu;
        cudaMalloc(&d_C, N * N * sizeof(float));
        cudaMalloc(&d_mu_dev, N * sizeof(float));
        cudaMalloc(&d_nu_dev, N * sizeof(float));
        cudaMalloc(&d_log_mu, N * sizeof(float));
        cudaMalloc(&d_log_nu, N * sizeof(float));

        cudaMemcpy(d_C, C.data(), N * N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_mu_dev, mu.data(), N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_nu_dev, nu.data(), N * sizeof(float), cudaMemcpyHostToDevice);

        std::vector<float> log_mu(N), log_nu(N);
        for (int i = 0; i < N; ++i) log_mu[i] = logf(fmaxf(mu[i], 1e-30f));
        for (int i = 0; i < N; ++i) log_nu[i] = logf(fmaxf(nu[i], 1e-30f));
        cudaMemcpy(d_log_mu, log_mu.data(), N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_log_nu, log_nu.data(), N * sizeof(float), cudaMemcpyHostToDevice);

        // --- Benchmark: Original (warp shuffle) via library ---
        {
            SinkhornConfig config;
            config.epsilon = epsilon;
            config.max_iterations = max_iter;
            config.convergence_threshold = threshold;
            config.convergence_check_interval = check_interval;

            std::vector<float> times;
            int last_iters = 0;
            bool last_conv = false;

            for (int r = 0; r < warmup_runs + timed_runs; ++r) {
                SinkhornSolver solver(config);
                auto result = solver.solve(mu, nu, C, N, N);
                if (r >= warmup_runs) {
                    times.push_back(result.elapsed_ms);
                    last_iters = result.iterations;
                    last_conv = result.converged;
                }
            }

            float mean = std::accumulate(times.begin(), times.end(), 0.0f) / times.size();
            float sq_sum = 0;
            for (float t : times) sq_sum += (t - mean) * (t - mean);
            float std_dev = sqrtf(sq_sum / times.size());

            fprintf(fp, "warp_shuffle,%d,%.4f,%.3f,%.3f,%d,%d,\n",
                    N, epsilon, mean, std_dev, last_iters, last_conv ? 1 : 0);
            printf("  Warp shuffle: %.2f ± %.2f ms\n", mean, std_dev);
        }

        // --- Benchmark: Shared-memory-only ---
        {
            std::vector<float> times;
            int last_iters = 0;
            bool last_conv = false;

            for (int r = 0; r < warmup_runs + timed_runs; ++r) {
                auto result = runSharedMemOnlySolver(
                    d_C, d_mu_dev, d_nu_dev, d_log_mu, d_log_nu,
                    N, N, epsilon, max_iter, threshold, check_interval);
                if (r >= warmup_runs) {
                    times.push_back(result.time_ms);
                    last_iters = result.iterations;
                    last_conv = result.converged;
                }
            }

            float mean = std::accumulate(times.begin(), times.end(), 0.0f) / times.size();
            float sq_sum = 0;
            for (float t : times) sq_sum += (t - mean) * (t - mean);
            float std_dev = sqrtf(sq_sum / times.size());

            fprintf(fp, "shared_mem_only,%d,%.4f,%.3f,%.3f,%d,%d,\n",
                    N, epsilon, mean, std_dev, last_iters, last_conv ? 1 : 0);
            printf("  Shared mem:   %.2f ± %.2f ms\n", mean, std_dev);
        }

        cudaFree(d_C);
        cudaFree(d_mu_dev);
        cudaFree(d_nu_dev);
        cudaFree(d_log_mu);
        cudaFree(d_log_nu);
    }

    fclose(fp);
    printf("\nResults written to experiments/data/ablation_warp.csv\n");
    return 0;
}
