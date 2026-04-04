/**
 * @file benchmark.cpp
 * @brief Performance benchmark for FastSinkhornCUDA.
 *
 * Measures execution time and memory usage across different problem sizes
 * to demonstrate GPU scaling properties. Tests N = 128, 256, 512, 1024,
 * 2048, 4096, 8192.
 *
 * Outputs a formatted table comparing:
 *   - Problem size N
 *   - Memory allocated (N² cost matrix)
 *   - Number of iterations
 *   - GPU computation time
 *   - Throughput (OT pairs / second)
 */

#include "sinkhorn/sinkhorn_solver.h"
#include "sinkhorn/config.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <chrono>

using namespace fastsinkhorn;

/**
 * @brief Generate a normalized Gaussian on a 1D grid.
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
    for (int i = 0; i < n; ++i) dist[i] /= sum;
    return dist;
}

/**
 * @brief Run benchmark for a single problem size.
 */
struct BenchmarkEntry {
    int n;
    float memory_mb;
    int iterations;
    float time_ms;
    float cost;
    bool converged;
};

BenchmarkEntry runBenchmark(int N, float epsilon) {
    BenchmarkEntry entry;
    entry.n = N;
    entry.memory_mb = static_cast<float>(N) * N * sizeof(float) / (1024.0f * 1024.0f);

    auto mu = makeGaussian(N, 0.3f, 0.08f);
    auto nu = makeGaussian(N, 0.7f, 0.08f);

    // Build cost matrix
    std::vector<float> C(N * N);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float xi = static_cast<float>(i) / (N - 1);
            float yj = static_cast<float>(j) / (N - 1);
            C[i * N + j] = (xi - yj) * (xi - yj);
        }
    }

    SinkhornConfig config;
    config.epsilon = epsilon;
    config.max_iterations = 2000;
    config.convergence_threshold = 1e-5f;
    config.convergence_check_interval = 20;
    config.verbose = false;
    config.enable_profiling = true;

    SinkhornSolver solver(config);

    // Warm-up run
    solver.solve(mu, nu, C, N, N);

    // Timed run
    auto result = solver.solve(mu, nu, C, N, N);

    entry.iterations = result.iterations;
    entry.time_ms = result.elapsed_ms;
    entry.cost = result.transport_cost;
    entry.converged = result.converged;

    return entry;
}

int main() {
    printf("╔══════════════════════════════════════════════════════════════════════╗\n");
    printf("║            Fast-Sinkhorn-CUDA: Performance Benchmark               ║\n");
    printf("╚══════════════════════════════════════════════════════════════════════╝\n\n");

    // Print GPU info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n", prop.name);
    printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("  SM count:           %d\n", prop.multiProcessorCount);
    printf("  Global memory:      %.0f MB\n", prop.totalGlobalMem / (1024.0 * 1024.0));
    printf("  Memory bandwidth:   %.0f GB/s\n",
           2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
    printf("\n");

    const float epsilon = 0.01f;
    printf("Configuration: ε = %.4f, convergence threshold = 1e-5\n\n", epsilon);

    // Problem sizes to benchmark
    std::vector<int> sizes = {128, 256, 512, 1024, 2048, 4096};

    // Check if we have enough memory for 8192
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    if (free_mem > static_cast<size_t>(8192) * 8192 * sizeof(float) * 3) {
        sizes.push_back(8192);
    }

    // Run benchmarks
    std::vector<BenchmarkEntry> results;
    for (int N : sizes) {
        printf("  Running N = %5d ...", N);
        fflush(stdout);
        auto entry = runBenchmark(N, epsilon);
        results.push_back(entry);
        printf(" done (%.1f ms)\n", entry.time_ms);
    }

    // Print results table
    printf("\n");
    printf("╔═════════╤══════════╤════════════╤══════════╤═══════════╤═══════════╗\n");
    printf("║    N    │ Mem (MB) │ Iterations │ Time(ms) │  Cost     │ Converged ║\n");
    printf("╠═════════╪══════════╪════════════╪══════════╪═══════════╪═══════════╣\n");

    for (const auto& e : results) {
        printf("║ %7d │ %8.1f │ %10d │ %8.2f │ %9.6f │ %-9s ║\n",
               e.n, e.memory_mb, e.iterations, e.time_ms, e.cost,
               e.converged ? "Yes" : "No");
    }

    printf("╚═════════╧══════════╧════════════╧══════════╧═══════════╧═══════════╝\n");

    // Scaling analysis
    printf("\nScaling Analysis:\n");
    if (results.size() >= 2) {
        for (size_t i = 1; i < results.size(); ++i) {
            float ratio = results[i].time_ms / results[i-1].time_ms;
            float size_ratio = static_cast<float>(results[i].n) / results[i-1].n;
            printf("  N: %d → %d  (%.1fx size)  Time ratio: %.2fx  "
                   "(ideal O(N²): %.1fx)\n",
                   results[i-1].n, results[i].n, size_ratio,
                   ratio, size_ratio * size_ratio);
        }
    }

    return 0;
}
