/**
 * @file stability_epsilon.cpp
 * @brief Numerical stability experiment: sweep epsilon from large to small.
 *
 * Tests log-domain Sinkhorn across a range of epsilon values.
 * Records whether the solver converges, produces NaN, or diverges.
 *
 * Output: experiments/data/stability_epsilon.csv
 */

#include "sinkhorn/sinkhorn_solver.h"
#include "sinkhorn/config.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <numeric>

using namespace fastsinkhorn;

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

int main() {
    const int N = 512;
    const int max_iter = 5000;
    const float threshold = 1e-9f;

    std::vector<float> epsilons = {
        1.0f, 0.5f, 0.1f, 0.05f, 0.01f, 0.005f, 0.001f, 0.0005f, 0.0001f
    };

    auto mu = makeGaussian(N, 0.3f, 0.08f);
    auto nu = makeGaussian(N, 0.7f, 0.08f);

    std::vector<float> C(N * N);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            float xi = float(i) / (N - 1), yj = float(j) / (N - 1);
            C[i * N + j] = (xi - yj) * (xi - yj);
        }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n", prop.name);
    printf("Stability Experiment: epsilon sweep (N=%d)\n\n", N);

    FILE* fp = fopen("experiments/data/stability_epsilon.csv", "w");
    if (!fp) { fprintf(stderr, "Cannot open output file\n"); return 1; }
    fprintf(fp, "method,n,epsilon,time_ms,iterations,transport_cost,marginal_error,converged,has_nan\n");

    for (float eps : epsilons) {
        printf("epsilon = %.5f: ", eps);

        SinkhornConfig config;
        config.epsilon = eps;
        config.max_iterations = max_iter;
        config.convergence_threshold = threshold;
        config.convergence_check_interval = 10;
        config.verbose = false;

        SinkhornSolver solver(config);
        auto result = solver.solve(mu, nu, C, N, N);

        // Check for NaN in result
        bool has_nan = std::isnan(result.transport_cost) || std::isinf(result.transport_cost);

        fprintf(fp, "log_domain,%d,%.6f,%.3f,%d,%.6f,%.2e,%d,%d\n",
                N, eps, result.elapsed_ms, result.iterations,
                has_nan ? 0.0f : result.transport_cost,
                has_nan ? 0.0f : result.marginal_error,
                result.converged ? 1 : 0,
                has_nan ? 1 : 0);

        if (has_nan)
            printf("NaN! (failed)\n");
        else if (result.converged)
            printf("converged in %d iters, cost=%.6f, time=%.2fms\n",
                   result.iterations, result.transport_cost, result.elapsed_ms);
        else
            printf("did not converge (%d iters), error=%.2e\n",
                   result.iterations, result.marginal_error);
    }

    fclose(fp);
    printf("\nResults written to experiments/data/stability_epsilon.csv\n");
    return 0;
}
