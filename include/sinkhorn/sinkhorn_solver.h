#pragma once
/**
 * @file sinkhorn_solver.h
 * @brief High-level interface for the Sinkhorn optimal transport solver.
 *
 * This is the main user-facing API. It orchestrates GPU memory allocation,
 * kernel launches, and convergence control for solving the entropic
 * regularized optimal transport problem:
 *
 *   min_{π ∈ Π(μ,ν)} <C, π> + ε·KL(π | μ⊗ν)
 *
 * The solver uses the log-domain Sinkhorn algorithm for numerical stability,
 * with shared-memory tiling and warp-level reductions for performance.
 */

#include "config.h"
#include <vector>

namespace fastsinkhorn {

/**
 * @brief Sinkhorn solver for entropic regularized optimal transport.
 *
 * Solves the regularized OT problem between two discrete probability
 * distributions. The solver operates entirely on the GPU and only
 * transfers the final result back to the host.
 *
 * Example usage:
 * @code
 *   SinkhornConfig config;
 *   config.epsilon = 0.01f;
 *   config.max_iterations = 500;
 *
 *   SinkhornSolver solver(config);
 *
 *   // Solve with pre-computed cost matrix
 *   std::vector<float> mu = {0.25, 0.25, 0.25, 0.25};
 *   std::vector<float> nu = {0.5, 0.5};
 *   std::vector<float> C = { ... }; // 4×2 cost matrix
 *   auto result = solver.solve(mu, nu, C, 4, 2);
 *
 *   // Or solve from point clouds (auto-computes squared Euclidean cost)
 *   std::vector<float> X = { ... }; // source points
 *   std::vector<float> Y = { ... }; // target points
 *   auto result2 = solver.solveFromPoints(mu, nu, X, Y, n, m, dim);
 * @endcode
 */
class SinkhornSolver {
public:
    /**
     * @brief Construct a solver with the given configuration.
     * @param config Solver parameters (epsilon, max_iter, etc.)
     */
    explicit SinkhornSolver(const SinkhornConfig& config = SinkhornConfig{});

    ~SinkhornSolver();

    // Non-copyable, movable
    SinkhornSolver(const SinkhornSolver&) = delete;
    SinkhornSolver& operator=(const SinkhornSolver&) = delete;
    SinkhornSolver(SinkhornSolver&&) noexcept;
    SinkhornSolver& operator=(SinkhornSolver&&) noexcept;

    /**
     * @brief Solve the OT problem with a pre-computed cost matrix.
     *
     * @param mu Source distribution (n elements, must sum to 1)
     * @param nu Target distribution (m elements, must sum to 1)
     * @param C  Cost matrix (n × m, row-major)
     * @param n  Size of source distribution
     * @param m  Size of target distribution
     * @return   SinkhornResult with transport cost, distance, etc.
     */
    SinkhornResult solve(
        const std::vector<float>& mu,
        const std::vector<float>& nu,
        const std::vector<float>& C,
        int n, int m);

    /**
     * @brief Solve the OT problem from point cloud data.
     *
     * Automatically computes the squared Euclidean cost matrix.
     *
     * @param mu  Source distribution weights
     * @param nu  Target distribution weights
     * @param X   Source points (n × dim, row-major)
     * @param Y   Target points (m × dim, row-major)
     * @param n   Number of source points
     * @param m   Number of target points
     * @param dim Dimensionality
     * @return    SinkhornResult
     */
    SinkhornResult solveFromPoints(
        const std::vector<float>& mu,
        const std::vector<float>& nu,
        const std::vector<float>& X,
        const std::vector<float>& Y,
        int n, int m, int dim);

    /**
     * @brief Extract the transport plan matrix π* after solving.
     *
     * Must be called after solve() or solveFromPoints().
     * Note: This involves computing exp((α_i + β_j - C_{ij}) / ε) on GPU,
     * which can be memory-intensive for large n, m.
     *
     * @param[out] pi  Output transport plan, resized to n×m
     * @param n        Source size
     * @param m        Target size
     */
    void getTransportPlan(std::vector<float>& pi, int n, int m);

    /**
     * @brief Get the dual potentials (α, β) after solving.
     * @param[out] alpha  Dual variable α (size n)
     * @param[out] beta   Dual variable β (size m)
     */
    void getDualPotentials(std::vector<float>& alpha, std::vector<float>& beta);

private:
    struct Impl;
    Impl* pImpl_;
};

} // namespace fastsinkhorn
