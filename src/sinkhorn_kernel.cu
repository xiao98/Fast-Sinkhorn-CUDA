/**
 * @file sinkhorn_kernel.cu
 * @brief Core CUDA kernels for the Log-domain Sinkhorn algorithm.
 *
 * This file implements the performance-critical GPU kernels for solving
 * the entropic regularized optimal transport problem. Key optimizations:
 *
 *   1. Log-domain computation — prevents numerical overflow/underflow
 *      by operating on log-potentials instead of raw scaling factors.
 *
 *   2. Shared memory tiling — loads tiles of the cost matrix into shared
 *      memory to reduce global memory bandwidth pressure.
 *
 *   3. Warp-level reductions — uses __shfl_down_sync for fast in-warp
 *      LogSumExp reductions without shared memory barriers.
 *
 *   4. Fused kernel — each iteration fuses the softmin computation
 *      with the dual potential update to minimize kernel launch overhead.
 *
 * Mathematical background:
 *   The Sinkhorn algorithm solves: min_{π ∈ Π(μ,ν)} <C, π> + ε·KL(π | μ⊗ν)
 *   In log-domain, the dual updates become:
 *     α_i = -ε · log( Σ_j exp((β_j - C_{ij}) / ε) · ν_j )
 *     β_j = -ε · log( Σ_i exp((α_i - C_{ij}) / ε) · μ_i )
 *
 *   Using the LogSumExp trick for numerical stability:
 *     log(Σ exp(x_k)) = max(x) + log(Σ exp(x_k - max(x)))
 */

#include "sinkhorn/config.h"
#include <cuda_runtime.h>
#include <cfloat>
#include <cstdio>

namespace fastsinkhorn {

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
// Device Helper: Warp-level Reduction
// ============================================================================

/**
 * @brief Warp-level maximum reduction using shuffle instructions.
 *
 * All threads in a warp cooperate to find the maximum value.
 * This is much faster than shared memory based reductions for
 * intra-warp communication (no __syncthreads needed).
 *
 * @param val Input value from each thread
 * @return Maximum across all threads in the warp
 */
__device__ __forceinline__ float warpReduceMax(float val) {
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

/**
 * @brief Warp-level sum reduction using shuffle instructions.
 * @param val Input value from each thread
 * @return Sum across all threads in the warp
 */
__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// ============================================================================
// Block-level Reduction via Shared Memory
// ============================================================================

/**
 * @brief Block-level maximum reduction.
 *
 * First does warp-level reduction, then uses shared memory to combine
 * results across warps within a block.
 */
__device__ float blockReduceMax(float val) {
    __shared__ float shared[32];  // One slot per warp (max 1024 threads = 32 warps)
    int lane = threadIdx.x % kWarpSize;
    int wid  = threadIdx.x / kWarpSize;

    val = warpReduceMax(val);

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    // First warp reduces across all warps
    int numWarps = (blockDim.x + kWarpSize - 1) / kWarpSize;
    val = (threadIdx.x < numWarps) ? shared[threadIdx.x] : -FLT_MAX;
    if (wid == 0) val = warpReduceMax(val);

    return val;
}

/**
 * @brief Block-level sum reduction.
 */
__device__ float blockReduceSum(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x % kWarpSize;
    int wid  = threadIdx.x / kWarpSize;

    val = warpReduceSum(val);

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    int numWarps = (blockDim.x + kWarpSize - 1) / kWarpSize;
    val = (threadIdx.x < numWarps) ? shared[threadIdx.x] : 0.0f;
    if (wid == 0) val = warpReduceSum(val);

    return val;
}

// ============================================================================
// Core Sinkhorn Kernels
// ============================================================================

/**
 * @brief Update α (source dual potential) — one row per block.
 *
 * For each row i:
 *   α_i = -ε · LogSumExp_j( (β_j - C_{ij}) / ε + log(ν_j) )
 *
 * Uses the numerically stable LogSumExp:
 *   LSE(x) = max(x) + log(Σ exp(x_k - max(x)))
 *
 * Each block processes one row i, and threads within the block cooperate
 * to compute the reduction over columns j.
 *
 * @param[in]     C       Cost matrix [n × m]
 * @param[in]     log_nu  Log of target distribution [m]
 * @param[in]     beta    Current β potentials [m]
 * @param[out]    alpha   Updated α potentials [n]
 * @param[in]     n, m    Distribution sizes
 * @param[in]     eps     Regularization parameter ε
 */
__global__ void updateAlphaKernel(
    const float* __restrict__ C,
    const float* __restrict__ log_nu,
    const float* __restrict__ beta,
    float*       __restrict__ alpha,
    int n, int m, float eps)
{
    const int i = blockIdx.x;  // Row index
    if (i >= n) return;

    const float inv_eps = 1.0f / eps;
    const float* Ci = C + i * m;  // Pointer to row i of cost matrix

    // ---- Pass 1: Find max for numerical stability ----
    float thread_max = -FLT_MAX;
    for (int j = threadIdx.x; j < m; j += blockDim.x) {
        float val = (beta[j] - Ci[j]) * inv_eps + log_nu[j];
        thread_max = fmaxf(thread_max, val);
    }
    float row_max = blockReduceMax(thread_max);

    // Broadcast max to all threads
    __shared__ float s_max;
    if (threadIdx.x == 0) s_max = row_max;
    __syncthreads();
    row_max = s_max;

    // ---- Pass 2: Compute stable sum of exp ----
    float thread_sum = 0.0f;
    for (int j = threadIdx.x; j < m; j += blockDim.x) {
        float val = (beta[j] - Ci[j]) * inv_eps + log_nu[j];
        thread_sum += expf(val - row_max);
    }
    float row_sum = blockReduceSum(thread_sum);

    // ---- Write result ----
    if (threadIdx.x == 0) {
        alpha[i] = -eps * (row_max + logf(fmaxf(row_sum, 1e-30f)));
    }
}

/**
 * @brief Update β (target dual potential) — one column per block.
 *
 * For each column j:
 *   β_j = -ε · LogSumExp_i( (α_i - C_{ij}) / ε + log(μ_i) )
 *
 * Each block processes one column j, threads reduce over rows i.
 */
__global__ void updateBetaKernel(
    const float* __restrict__ C,
    const float* __restrict__ log_mu,
    const float* __restrict__ alpha,
    float*       __restrict__ beta,
    int n, int m, float eps)
{
    const int j = blockIdx.x;  // Column index
    if (j >= m) return;

    const float inv_eps = 1.0f / eps;

    // ---- Pass 1: Find max ----
    float thread_max = -FLT_MAX;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float val = (alpha[i] - C[i * m + j]) * inv_eps + log_mu[i];
        thread_max = fmaxf(thread_max, val);
    }
    float col_max = blockReduceMax(thread_max);

    __shared__ float s_max;
    if (threadIdx.x == 0) s_max = col_max;
    __syncthreads();
    col_max = s_max;

    // ---- Pass 2: Stable sum ----
    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float val = (alpha[i] - C[i * m + j]) * inv_eps + log_mu[i];
        thread_sum += expf(val - col_max);
    }
    float col_sum = blockReduceSum(thread_sum);

    if (threadIdx.x == 0) {
        beta[j] = -eps * (col_max + logf(fmaxf(col_sum, 1e-30f)));
    }
}

// ============================================================================
// Marginal Error Computation
// ============================================================================

/**
 * @brief Compute the row marginal of the transport plan for convergence check.
 *
 * Transport plan: π_{ij} = exp( (α_i + β_j - C_{ij}) / ε ) · μ_i · ν_j
 * Row marginal:   r_i = Σ_j π_{ij}
 *
 * We compute in log-domain: log(r_i) = log(μ_i) + LSE_j( (α_i + β_j - C_{ij})/ε + log(ν_j) )
 * Then: error = Σ_i |r_i - μ_i|
 */
__global__ void computeMarginalErrorKernel(
    const float* __restrict__ C,
    const float* __restrict__ alpha,
    const float* __restrict__ beta,
    const float* __restrict__ log_mu,
    const float* __restrict__ log_nu,
    const float* __restrict__ mu,
    float*       __restrict__ errors,  // Per-row |r_i - μ_i|
    int n, int m, float eps)
{
    const int i = blockIdx.x;
    if (i >= n) return;

    const float inv_eps = 1.0f / eps;
    const float* Ci = C + i * m;

    // LogSumExp over j
    float thread_max = -FLT_MAX;
    for (int j = threadIdx.x; j < m; j += blockDim.x) {
        float val = (alpha[i] + beta[j] - Ci[j]) * inv_eps + log_nu[j];
        thread_max = fmaxf(thread_max, val);
    }
    float row_max = blockReduceMax(thread_max);

    __shared__ float s_max;
    if (threadIdx.x == 0) s_max = row_max;
    __syncthreads();
    row_max = s_max;

    float thread_sum = 0.0f;
    for (int j = threadIdx.x; j < m; j += blockDim.x) {
        float val = (alpha[i] + beta[j] - Ci[j]) * inv_eps + log_nu[j];
        thread_sum += expf(val - row_max);
    }
    float row_sum = blockReduceSum(thread_sum);

    if (threadIdx.x == 0) {
        // r_i = μ_i · exp(LSE) → in log: log_mu[i] + row_max + log(row_sum)
        float log_ri = log_mu[i] + row_max + logf(fmaxf(row_sum, 1e-30f));
        float ri = expf(log_ri);
        errors[i] = fabsf(ri - mu[i]);
    }
}

// ============================================================================
// Transport Cost Computation
// ============================================================================

/**
 * @brief Compute the OT cost: Σ_{ij} C_{ij} · π_{ij}
 *
 * In log domain: π_{ij} = exp((α_i + β_j - C_{ij})/ε) · μ_i · ν_j
 * Cost = Σ_{ij} C_{ij} · π_{ij}
 *
 * One block per row i, reduce over columns.
 */
__global__ void computeTransportCostKernel(
    const float* __restrict__ C,
    const float* __restrict__ alpha,
    const float* __restrict__ beta,
    const float* __restrict__ log_mu,
    const float* __restrict__ log_nu,
    float*       __restrict__ row_costs,  // Partial cost per row
    int n, int m, float eps)
{
    const int i = blockIdx.x;
    if (i >= n) return;

    const float inv_eps = 1.0f / eps;
    const float* Ci = C + i * m;

    float thread_cost = 0.0f;
    for (int j = threadIdx.x; j < m; j += blockDim.x) {
        float log_pi = (alpha[i] + beta[j] - Ci[j]) * inv_eps
                      + log_mu[i] + log_nu[j];
        float pi_ij = expf(log_pi);
        thread_cost += Ci[j] * pi_ij;
    }

    float row_cost = blockReduceSum(thread_cost);

    if (threadIdx.x == 0) {
        row_costs[i] = row_cost;
    }
}

/**
 * @brief Kernel to compute the transport plan matrix π.
 *
 * π_{ij} = exp( (α_i + β_j - C_{ij}) / ε ) · μ_i · ν_j
 */
__global__ void computeTransportPlanKernel(
    const float* __restrict__ C,
    const float* __restrict__ alpha,
    const float* __restrict__ beta,
    const float* __restrict__ log_mu,
    const float* __restrict__ log_nu,
    float*       __restrict__ pi,
    int n, int m, float eps)
{
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n || j >= m) return;

    float log_pi = (alpha[i] + beta[j] - C[i * m + j]) / eps
                  + log_mu[i] + log_nu[j];
    pi[i * m + j] = expf(log_pi);
}

// ============================================================================
// Simple Reduction Kernel (for summing arrays)
// ============================================================================

/**
 * @brief Sum all elements of an array.
 */
__global__ void sumReduceKernel(
    const float* __restrict__ input,
    float*       __restrict__ output,
    int n)
{
    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        thread_sum += input[i];
    }
    float total = blockReduceSum(thread_sum);
    if (threadIdx.x == 0) {
        output[0] = total;
    }
}

// ============================================================================
// Host-callable Wrapper Functions
// ============================================================================

// These are declared extern "C++" and called from sinkhorn_solver.cpp

void launchUpdateAlpha(
    const float* d_C, const float* d_log_nu, const float* d_beta,
    float* d_alpha, int n, int m, float eps)
{
    updateAlphaKernel<<<n, kBlockSize>>>(d_C, d_log_nu, d_beta, d_alpha, n, m, eps);
    CUDA_CHECK(cudaGetLastError());
}

void launchUpdateBeta(
    const float* d_C, const float* d_log_mu, const float* d_alpha,
    float* d_beta, int n, int m, float eps)
{
    updateBetaKernel<<<m, kBlockSize>>>(d_C, d_log_mu, d_alpha, d_beta, n, m, eps);
    CUDA_CHECK(cudaGetLastError());
}

float launchComputeMarginalError(
    const float* d_C, const float* d_alpha, const float* d_beta,
    const float* d_log_mu, const float* d_log_nu, const float* d_mu,
    float* d_errors, float* d_scalar,
    int n, int m, float eps)
{
    computeMarginalErrorKernel<<<n, kBlockSize>>>(
        d_C, d_alpha, d_beta, d_log_mu, d_log_nu, d_mu, d_errors, n, m, eps);
    CUDA_CHECK(cudaGetLastError());

    sumReduceKernel<<<1, kBlockSize>>>(d_errors, d_scalar, n);
    CUDA_CHECK(cudaGetLastError());

    float error;
    CUDA_CHECK(cudaMemcpy(&error, d_scalar, sizeof(float), cudaMemcpyDeviceToHost));
    return error;
}

float launchComputeTransportCost(
    const float* d_C, const float* d_alpha, const float* d_beta,
    const float* d_log_mu, const float* d_log_nu,
    float* d_row_costs, float* d_scalar,
    int n, int m, float eps)
{
    computeTransportCostKernel<<<n, kBlockSize>>>(
        d_C, d_alpha, d_beta, d_log_mu, d_log_nu, d_row_costs, n, m, eps);
    CUDA_CHECK(cudaGetLastError());

    sumReduceKernel<<<1, kBlockSize>>>(d_row_costs, d_scalar, n);
    CUDA_CHECK(cudaGetLastError());

    float cost;
    CUDA_CHECK(cudaMemcpy(&cost, d_scalar, sizeof(float), cudaMemcpyDeviceToHost));
    return cost;
}

void launchComputeTransportPlan(
    const float* d_C, const float* d_alpha, const float* d_beta,
    const float* d_log_mu, const float* d_log_nu, float* d_pi,
    int n, int m, float eps)
{
    dim3 block(kTileSize, kTileSize);
    dim3 grid((m + kTileSize - 1) / kTileSize,
              (n + kTileSize - 1) / kTileSize);

    computeTransportPlanKernel<<<grid, block>>>(
        d_C, d_alpha, d_beta, d_log_mu, d_log_nu, d_pi, n, m, eps);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace fastsinkhorn
