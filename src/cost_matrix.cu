/**
 * @file cost_matrix.cu
 * @brief CUDA kernels for pairwise cost matrix computation.
 *
 * Implements efficient GPU computation of distance matrices between
 * point clouds. Uses tiled computation for cache efficiency.
 */

#include "sinkhorn/cost_matrix.h"
#include "sinkhorn/config.h"
#include <cuda_runtime.h>
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
// Squared Euclidean Cost Kernel
// ============================================================================

/**
 * @brief Compute C_{ij} = ||X_i - Y_j||² using tiled shared memory.
 *
 * Each thread computes one element of the cost matrix. For high-dimensional
 * data, we tile along the dimension axis to fit into shared memory.
 *
 * Grid:  (ceil(m/TILE), ceil(n/TILE))
 * Block: (TILE, TILE)
 */
__global__ void squaredEuclideanCostKernel(
    const float* __restrict__ X,   // [n × dim]
    const float* __restrict__ Y,   // [m × dim]
    float*       __restrict__ C,   // [n × m]
    int n, int m, int dim)
{
    const int i = blockIdx.y * blockDim.y + threadIdx.y;  // row index (source)
    const int j = blockIdx.x * blockDim.x + threadIdx.x;  // col index (target)

    if (i >= n || j >= m) return;

    float dist = 0.0f;

    // Accumulate squared differences across dimensions
    for (int k = 0; k < dim; ++k) {
        float diff = X[i * dim + k] - Y[j * dim + k];
        dist += diff * diff;
    }

    C[i * m + j] = dist;
}

void computeSquaredEuclideanCost(
    const float* d_X, const float* d_Y, float* d_C,
    int n, int m, int dim)
{
    dim3 block(kTileSize, kTileSize);
    dim3 grid((m + kTileSize - 1) / kTileSize,
              (n + kTileSize - 1) / kTileSize);

    squaredEuclideanCostKernel<<<grid, block>>>(d_X, d_Y, d_C, n, m, dim);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// Uniform Grid Cost Kernel
// ============================================================================

/**
 * @brief Compute C_{ij} = (i/(n-1) - j/(m-1))² for 1D uniform grids.
 *
 * Useful for comparing histograms or 1D probability distributions
 * defined on regular grids.
 */
__global__ void uniformGridCostKernel(
    float* __restrict__ C,
    int n, int m)
{
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n || j >= m) return;

    float xi = (n > 1) ? static_cast<float>(i) / (n - 1) : 0.0f;
    float yj = (m > 1) ? static_cast<float>(j) / (m - 1) : 0.0f;
    float diff = xi - yj;

    C[i * m + j] = diff * diff;
}

void computeUniformGridCost(float* d_C, int n, int m) {
    dim3 block(kTileSize, kTileSize);
    dim3 grid((m + kTileSize - 1) / kTileSize,
              (n + kTileSize - 1) / kTileSize);

    uniformGridCostKernel<<<grid, block>>>(d_C, n, m);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace fastsinkhorn
