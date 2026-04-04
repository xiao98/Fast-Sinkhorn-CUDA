/**
 * @file bench_ours.cpp
 * @brief Extended baseline benchmark for Fast-Sinkhorn-CUDA with CSV output.
 *
 * Sweeps over problem sizes N in {256, 512, 1024, 2048, 4096, 8192} and
 * regularization parameters epsilon in {0.1, 0.01, 0.001}. For each (N, eps)
 * pair, performs 3 warmup runs followed by 10 timed runs, recording mean
 * and standard deviation of GPU execution time.
 *
 * Output:  experiments/data/baseline_ours.csv
 * Columns: n, m, epsilon, time_ms_mean, time_ms_std, iterations,
 *          transport_cost, converged, gpu
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
    std::string gpuName(prop.name);

    printf("=== Fast-Sinkhorn-CUDA: Baseline Benchmark (Ours) ===\n");
    printf("GPU: %s\n", prop.name);
    printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("  SM count:           %d\n", prop.multiProcessorCount);
    printf("  Global memory:      %.0f MB\n",
           prop.totalGlobalMem / (1024.0 * 1024.0));
    printf("  Memory bandwidth:   %.0f GB/s\n",
           2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
    printf("\n");

    // ------------------------------------------------------------------
    // Experiment grid
    // ------------------------------------------------------------------
    const std::vector<int>   sizes    = {256, 512, 1024, 2048, 4096, 8192};
    const std::vector<float> epsilons = {0.1f, 0.01f, 0.001f};
    const int kWarmupRuns = 3;
    const int kTimedRuns  = 10;

    // ------------------------------------------------------------------
    // Open output CSV
    // ------------------------------------------------------------------
    FILE* csv = fopen("experiments/data/baseline_ours.csv", "w");
    if (!csv) {
        fprintf(stderr, "ERROR: Cannot open experiments/data/baseline_ours.csv "
                        "for writing.\n");
        return 1;
    }
    fprintf(csv, "n,m,epsilon,time_ms_mean,time_ms_std,iterations,"
                 "transport_cost,converged,gpu\n");

    printf("Running benchmark: %d sizes x %d epsilons "
           "(%d warmup + %d timed runs each)\n\n",
           static_cast<int>(sizes.size()),
           static_cast<int>(epsilons.size()),
           kWarmupRuns, kTimedRuns);

    // ------------------------------------------------------------------
    // Sweep
    // ------------------------------------------------------------------
    for (int N : sizes) {
        // Check GPU memory before allocating for this size
        size_t free_mem = 0, total_mem = 0;
        cudaMemGetInfo(&free_mem, &total_mem);
        size_t required = static_cast<size_t>(N) * N * sizeof(float) * 3;
        if (free_mem < required) {
            printf("  Skipping N=%d: insufficient GPU memory "
                   "(need %.0f MB, free %.0f MB)\n",
                   N, required / (1024.0 * 1024.0),
                   free_mem / (1024.0 * 1024.0));
            continue;
        }

        auto mu = makeGaussian(N, 0.3f, 0.08f);
        auto nu = makeGaussian(N, 0.7f, 0.08f);
        auto C  = buildCostMatrix(N);

        for (float eps : epsilons) {
            printf("  N=%5d  eps=%.3f  ", N, eps);
            fflush(stdout);

            SinkhornConfig config;
            config.epsilon                   = eps;
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
            fprintf(csv, "%d,%d,%.4g,%.3f,%.3f,%d,%.6f,%d,%s\n",
                    N, N, eps, mean_ms, std_ms,
                    last.iterations, last.transport_cost,
                    last.converged ? 1 : 0, gpuName.c_str());

            printf("mean=%.3f ms  std=%.3f ms  iters=%d  cost=%.6f  %s\n",
                   mean_ms, std_ms, last.iterations, last.transport_cost,
                   last.converged ? "converged" : "NOT converged");
        }
    }

    fclose(csv);
    printf("\nResults written to experiments/data/baseline_ours.csv\n");
    return 0;
}
