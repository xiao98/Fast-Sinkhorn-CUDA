#pragma once
/**
 * @file cost_matrix.h
 * @brief CUDA kernels for computing the pairwise cost matrix.
 *
 * Given two point clouds X ∈ ℝ^{n×d} and Y ∈ ℝ^{m×d}, computes the cost
 * matrix C ∈ ℝ^{n×m} where C_{ij} = ||X_i - Y_j||² (squared Euclidean).
 */

#include <cstddef>

namespace fastsinkhorn {

/**
 * @brief Compute the squared Euclidean cost matrix on the GPU.
 *
 * C_{ij} = Σ_k (X_{ik} - Y_{jk})²
 *
 * @param[in]  d_X    Device pointer to source points, shape [n × dim], row-major
 * @param[in]  d_Y    Device pointer to target points, shape [m × dim], row-major
 * @param[out] d_C    Device pointer to cost matrix, shape [n × m], row-major
 * @param[in]  n      Number of source points
 * @param[in]  m      Number of target points
 * @param[in]  dim    Dimensionality of points
 */
void computeSquaredEuclideanCost(
    const float* d_X, const float* d_Y, float* d_C,
    int n, int m, int dim);

/**
 * @brief Compute a 1D cost matrix for histogram-style inputs.
 *
 * For 1D distributions on uniform grids, C_{ij} = (i/n - j/m)²
 *
 * @param[out] d_C    Device pointer to cost matrix, shape [n × m]
 * @param[in]  n      Size of source distribution
 * @param[in]  m      Size of target distribution
 */
void computeUniformGridCost(float* d_C, int n, int m);

} // namespace fastsinkhorn
