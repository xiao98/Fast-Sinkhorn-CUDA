/**
 * @file scaling_experiment.cpp
 * @brief GPU scaling experiment for Fast-Sinkhorn-CUDA with CSV output.
 *
 * Measures how execution time and memory scale with problem size N for a
 * fixed regularization epsilon = 0.01. Problem sizes range from 64 to
 * 16384; sizes that exceed available GPU memory are automatically skipped.
 *
 * For each size: 3 warmup runs + 10 timed runs.
 *
 * Output:  experiments/data/scaling_ours.csv
 * Columns: n, m, epsilon, time_ms_mean, time_ms_std, iterations,
 *          peak_memory_bytes, cost_matrix_mb, transport_cost, converged
 */

#include "sinkhorn/sinkhorn_solver.h"
#include "sinkhorn/config.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>

using namespace fastsinkhorn;

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

/**
 * @brief Compute mean and standard deviation from a vector of samples.
 */
static void computeStats(const std::vector<float>& v,
                          float& mean, float& stddev) {
    float s = 0.0f;
    for (float x : v) s += x;
    mean = s / static_cast<float>(v.size());

    float sq = 0.0f;
    for (float x : v) sq += (x - mean) * (x - mean);
    stddev = sqrtf(sq / static_cast<float>(v.size()));
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

    printf("=== Fast-Sinkhorn-CUDA: Scaling Experiment ===\n");
    printf("GPU: %s\n", prop.name);
    printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("  SM count:           %d\n", prop.multiProcessorCount);
    printf("  Global memory:      %.0f MB\n",
           prop.totalGlobalMem / (1024.0 * 1024.0));
    printf("\n");

    // ------------------------------------------------------------------
    // Experiment parameters
    // ------------------------------------------------------------------
    const std::vector<int> sizes = {
        64, 128, 256, 512, 1024, 2048, 4096, 8192, 12288, 16384
    };
    const float epsilon    = 0.01f;
    const int   kWarmupRuns = 3;
    const int   kTimedRuns  = 10;

    // ------------------------------------------------------------------
    // Open output CSV
    // ------------------------------------------------------------------
    FILE* csv = fopen("experiments/data/scaling_ours.csv", "w");
    if (!csv) {
        fprintf(stderr, "ERROR: Cannot open experiments/data/scaling_ours.csv "
                        "for writing.\n");
        return 1;
    }
    fprintf(csv, "n,m,epsilon,time_ms_mean,time_ms_std,iterations,"
                 "peak_memory_bytes,cost_matrix_mb,transport_cost,converged\n");

    printf("Fixed epsilon = %.4f\n", epsilon);
    printf("Running %d sizes (%d warmup + %d timed runs each)\n\n",
           static_cast<int>(sizes.size()), kWarmupRuns, kTimedRuns);

    // ------------------------------------------------------------------
    // Sweep over problem sizes
    // ------------------------------------------------------------------
    for (int N : sizes) {
        // Estimate memory: cost matrix (N*N*4) + auxiliary buffers (~3*N*4)
        // Use a conservative factor of 3x the cost matrix size.
        size_t cost_matrix_bytes = static_cast<size_t>(N) * N * sizeof(float);
        float  cost_matrix_mb   = cost_matrix_bytes / (1024.0f * 1024.0f);
        size_t required         = cost_matrix_bytes * 3;

        size_t free_mem = 0, total_mem = 0;
        cudaMemGetInfo(&free_mem, &total_mem);

        if (free_mem < required) {
            printf("  Skipping N=%5d: insufficient GPU memory "
                   "(need %.0f MB, free %.0f MB)\n",
                   N, required / (1024.0 * 1024.0),
                   free_mem / (1024.0 * 1024.0));
            continue;
        }

        printf("  N=%5d  (cost matrix %.1f MB)  ", N, cost_matrix_mb);
        fflush(stdout);

        auto mu = makeGaussian(N, 0.3f, 0.08f);
        auto nu = makeGaussian(N, 0.7f, 0.08f);
        auto C  = buildCostMatrix(N);

        SinkhornConfig config;
        config.epsilon                   = epsilon;
        config.max_iterations            = 5000;
        config.convergence_threshold     = 1e-6f;
        config.convergence_check_interval = 20;
        config.verbose                   = false;
        config.enable_profiling          = true;

        SinkhornSolver solver(config);

        // Warmup runs
        for (int w = 0; w < kWarmupRuns; ++w) {
            solver.solve(mu, nu, C, N, N);
        }

        // Timed runs
        std::vector<float> times(kTimedRuns);
        SinkhornResult last;
        for (int t = 0; t < kTimedRuns; ++t) {
            last = solver.solve(mu, nu, C, N, N);
            times[t] = last.elapsed_ms;
        }

        float mean_ms = 0.0f, std_ms = 0.0f;
        computeStats(times, mean_ms, std_ms);

        // Write CSV row
        fprintf(csv, "%d,%d,%.4g,%.3f,%.3f,%d,%zu,%.2f,%.6f,%d\n",
                N, N, epsilon, mean_ms, std_ms,
                last.iterations, last.peak_memory_bytes,
                cost_matrix_mb, last.transport_cost,
                last.converged ? 1 : 0);

        printf("mean=%.3f ms  std=%.3f ms  iters=%d  peak_mem=%zu B  "
               "cost=%.6f  %s\n",
               mean_ms, std_ms, last.iterations,
               last.peak_memory_bytes, last.transport_cost,
               last.converged ? "converged" : "NOT converged");
    }

    fclose(csv);

    // ------------------------------------------------------------------
    // Print scaling analysis
    // ------------------------------------------------------------------
    printf("\nResults written to experiments/data/scaling_ours.csv\n");
    return 0;
}
