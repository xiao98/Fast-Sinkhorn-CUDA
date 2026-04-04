/**
 * @file ablation_check_interval.cpp
 * @brief Ablation study: effect of convergence check interval on runtime.
 *
 * Checks how frequently inspecting the marginal error (which triggers a
 * GPU-to-CPU synchronisation) impacts total solve time. Uses a fixed
 * problem size N = 2048 and epsilon = 0.01, varying the convergence check
 * interval across {1, 5, 10, 20, 50, 100}.
 *
 * For each interval: 3 warmup runs + 10 timed runs.
 *
 * Output:  experiments/data/ablation_checkinterval.csv
 * Columns: config, n, epsilon, time_ms_mean, time_ms_std, iterations,
 *          converged, notes
 */

#include "sinkhorn/sinkhorn_solver.h"
#include "sinkhorn/config.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <string>
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

    printf("=== Fast-Sinkhorn-CUDA: Ablation — Check Interval ===\n");
    printf("GPU: %s\n", prop.name);
    printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("  SM count:           %d\n", prop.multiProcessorCount);
    printf("  Global memory:      %.0f MB\n\n",
           prop.totalGlobalMem / (1024.0 * 1024.0));

    // ------------------------------------------------------------------
    // Experiment parameters
    // ------------------------------------------------------------------
    const int   N       = 2048;
    const float epsilon = 0.01f;
    const std::vector<int> intervals = {1, 5, 10, 20, 50, 100};
    const int kWarmupRuns = 3;
    const int kTimedRuns  = 10;

    printf("Fixed: N = %d, epsilon = %.4f\n", N, epsilon);
    printf("Testing check intervals: ");
    for (int iv : intervals) printf("%d ", iv);
    printf("\n(%d warmup + %d timed runs each)\n\n", kWarmupRuns, kTimedRuns);

    // ------------------------------------------------------------------
    // Prepare distributions and cost matrix once
    // ------------------------------------------------------------------
    auto mu = makeGaussian(N, 0.3f, 0.08f);
    auto nu = makeGaussian(N, 0.7f, 0.08f);
    auto C  = buildCostMatrix(N);

    // ------------------------------------------------------------------
    // Open output CSV
    // ------------------------------------------------------------------
    FILE* csv = fopen("experiments/data/ablation_checkinterval.csv", "w");
    if (!csv) {
        fprintf(stderr, "ERROR: Cannot open "
                        "experiments/data/ablation_checkinterval.csv "
                        "for writing.\n");
        return 1;
    }
    fprintf(csv, "config,n,epsilon,time_ms_mean,time_ms_std,"
                 "iterations,converged,notes\n");

    // ------------------------------------------------------------------
    // Sweep over check intervals
    // ------------------------------------------------------------------
    printf("%-24s %8s %8s %8s %9s\n",
           "Config", "Mean(ms)", "Std(ms)", "Iters", "Converged");
    printf("--------------------------------------------------------------\n");

    for (int interval : intervals) {
        char configName[64];
        snprintf(configName, sizeof(configName),
                 "check_interval_%d", interval);

        SinkhornConfig config;
        config.epsilon                   = epsilon;
        config.max_iterations            = 5000;
        config.convergence_threshold     = 1e-6f;
        config.convergence_check_interval = interval;
        config.verbose                   = false;
        config.enable_profiling          = false;

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

        // Build notes string
        // A higher check interval reduces sync overhead but may overshoot
        // the true convergence point, resulting in extra iterations.
        char notes[128] = "";
        if (interval == 1) {
            snprintf(notes, sizeof(notes),
                     "sync every iter (max overhead)");
        } else if (interval >= 50) {
            snprintf(notes, sizeof(notes),
                     "rare sync (may overshoot convergence)");
        }

        // Write CSV row
        fprintf(csv, "%s,%d,%.4g,%.3f,%.3f,%d,%d,%s\n",
                configName, N, epsilon, mean_ms, std_ms,
                last.iterations, last.converged ? 1 : 0, notes);

        printf("%-24s %8.3f %8.3f %8d %9s\n",
               configName, mean_ms, std_ms, last.iterations,
               last.converged ? "Yes" : "No");
    }

    fclose(csv);

    printf("\nResults written to "
           "experiments/data/ablation_checkinterval.csv\n");
    return 0;
}
