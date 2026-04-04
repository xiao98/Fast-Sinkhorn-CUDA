/**
 * @file basic_ot.cpp
 * @brief Example: Optimal transport between two 1D distributions.
 *
 * Demonstrates the basic usage of FastSinkhornCUDA:
 *   1. Define two probability distributions (Gaussians on a grid)
 *   2. Solve the regularized OT problem
 *   3. Display the transport plan and cost
 *
 * This example computes the Wasserstein-like distance between:
 *   μ = Gaussian centered at 0.3
 *   ν = Gaussian centered at 0.7
 */

#include "sinkhorn/sinkhorn_solver.h"
#include "sinkhorn/config.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <numeric>

using namespace fastsinkhorn;

/**
 * @brief Generate a normalized Gaussian distribution on [0,1].
 *
 * Returns a discrete probability vector of size n, representing
 * a Gaussian centered at `center` with standard deviation `sigma`.
 */
std::vector<float> makeGaussian(int n, float center, float sigma) {
    std::vector<float> dist(n);
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        float x = static_cast<float>(i) / (n - 1);
        float val = expf(-0.5f * (x - center) * (x - center) / (sigma * sigma));
        dist[i] = val;
        sum += val;
    }
    // Normalize to make it a probability distribution
    for (int i = 0; i < n; ++i) dist[i] /= sum;
    return dist;
}

/**
 * @brief Print a small matrix to stdout.
 */
void printMatrix(const std::vector<float>& M, int rows, int cols,
                 const char* name, int max_display = 8) {
    printf("\n%s (%d × %d):\n", name, rows, cols);
    int r_show = std::min(rows, max_display);
    int c_show = std::min(cols, max_display);

    for (int i = 0; i < r_show; ++i) {
        printf("  ");
        for (int j = 0; j < c_show; ++j) {
            printf("%.4f ", M[i * cols + j]);
        }
        if (cols > max_display) printf("...");
        printf("\n");
    }
    if (rows > max_display) printf("  ...\n");
}

int main() {
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║          Fast-Sinkhorn-CUDA: Basic OT Example              ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    // ================================================================
    // Problem Setup
    // ================================================================
    const int N = 128;           // Distribution size
    const float epsilon = 0.01f; // Regularization parameter

    printf("Problem Configuration:\n");
    printf("  Distribution size:      N = %d\n", N);
    printf("  Regularization:         ε = %.4f\n", epsilon);

    // Source: Gaussian at 0.3
    auto mu = makeGaussian(N, 0.3f, 0.05f);

    // Target: Gaussian at 0.7
    auto nu = makeGaussian(N, 0.7f, 0.05f);

    printf("  Source distribution:    𝒩(0.3, 0.05²)\n");
    printf("  Target distribution:   𝒩(0.7, 0.05²)\n");

    // Cost matrix: C_{ij} = (i/(N-1) - j/(N-1))²
    std::vector<float> C(N * N);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float xi = static_cast<float>(i) / (N - 1);
            float yj = static_cast<float>(j) / (N - 1);
            C[i * N + j] = (xi - yj) * (xi - yj);
        }
    }

    // ================================================================
    // Solve
    // ================================================================
    printf("\nSolving regularized optimal transport...\n\n");

    SinkhornConfig config;
    config.epsilon = epsilon;
    config.max_iterations = 1000;
    config.convergence_threshold = 1e-6f;
    config.convergence_check_interval = 10;
    config.verbose = true;
    config.enable_profiling = true;

    SinkhornSolver solver(config);
    SinkhornResult result = solver.solve(mu, nu, C, N, N);

    // ================================================================
    // Display Results
    // ================================================================
    printf("\n╔══════════════════════════════════════════════════════════════╗\n");
    printf("║                        Results                             ║\n");
    printf("╠══════════════════════════════════════════════════════════════╣\n");
    printf("║  Transport cost:     %.6f                              ║\n", result.transport_cost);
    printf("║  Sinkhorn distance:  %.6f                              ║\n", result.sinkhorn_distance);
    printf("║  Iterations:         %-6d                                ║\n", result.iterations);
    printf("║  Converged:          %-6s                                ║\n",
           result.converged ? "Yes" : "No");
    printf("║  Marginal error:     %.2e                            ║\n", result.marginal_error);
    printf("║  GPU time:           %.2f ms                             ║\n", result.elapsed_ms);
    printf("╚══════════════════════════════════════════════════════════════╝\n");

    // Extract and display transport plan (small portion)
    std::vector<float> pi;
    solver.getTransportPlan(pi, N, N);
    printMatrix(pi, N, N, "Transport Plan π*");

    // Extract dual potentials
    std::vector<float> alpha, beta;
    solver.getDualPotentials(alpha, beta);

    printf("\nDual Potentials (first 8 values):\n");
    printf("  α: ");
    for (int i = 0; i < std::min(N, 8); ++i) printf("%.4f ", alpha[i]);
    printf("\n  β: ");
    for (int j = 0; j < std::min(N, 8); ++j) printf("%.4f ", beta[j]);
    printf("\n");

    // ================================================================
    // Theoretical Check
    // ================================================================
    printf("\n--- Sanity Check ---\n");
    printf("  For two Gaussians separated by Δ = 0.4:\n");
    printf("  Expected W₂² ≈ Δ² = 0.16 (unregularized)\n");
    printf("  Computed regularized cost = %.6f\n", result.transport_cost);
    printf("  (Regularized cost should be close but slightly larger due to ε)\n");

    return 0;
}
