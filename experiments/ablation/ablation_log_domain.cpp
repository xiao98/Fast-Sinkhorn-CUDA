/**
 * @file ablation_log_domain.cpp
 * @brief Ablation: log-domain Sinkhorn vs standard-domain Sinkhorn.
 *
 * The standard-domain solver uses the Gibbs kernel K = exp(-C/eps) and
 * scaling vectors u, v. It is expected to fail (NaN) for small epsilon.
 *
 * Output: experiments/data/ablation_logdomain.csv
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
// Standard-domain Sinkhorn (CPU reference for correctness check)
// ============================================================================

struct StandardResult {
    float transport_cost;
    int iterations;
    float marginal_error;
    bool converged;
    bool has_nan;
    float time_ms;
};

StandardResult runStandardDomainCPU(
    const std::vector<float>& mu,
    const std::vector<float>& nu,
    const std::vector<float>& C,
    int n, int m, float eps, int max_iter, float threshold)
{
    // Compute Gibbs kernel K_ij = exp(-C_ij / eps)
    std::vector<float> K(n * m);
    bool has_nan = false;

    for (int i = 0; i < n * m; ++i) {
        K[i] = expf(-C[i] / eps);
        if (std::isnan(K[i]) || std::isinf(K[i])) {
            has_nan = true;
        }
    }

    if (has_nan) {
        return {0.0f, 0, 0.0f, false, true, 0.0f};
    }

    // Scaling vectors u, v
    std::vector<float> u(n, 1.0f);
    std::vector<float> v(m, 1.0f);

    auto start = std::chrono::high_resolution_clock::now();

    int iter = 0;
    for (; iter < max_iter; ++iter) {
        // u_i = mu_i / (K * v)_i
        for (int i = 0; i < n; ++i) {
            float Kv_i = 0.0f;
            for (int j = 0; j < m; ++j) {
                Kv_i += K[i * m + j] * v[j];
            }
            if (Kv_i < 1e-30f) { has_nan = true; break; }
            u[i] = mu[i] / Kv_i;
            if (std::isnan(u[i]) || std::isinf(u[i])) { has_nan = true; break; }
        }
        if (has_nan) break;

        // v_j = nu_j / (K^T * u)_j
        for (int j = 0; j < m; ++j) {
            float Ku_j = 0.0f;
            for (int i = 0; i < n; ++i) {
                Ku_j += K[i * m + j] * u[i];
            }
            if (Ku_j < 1e-30f) { has_nan = true; break; }
            v[j] = nu[j] / Ku_j;
            if (std::isnan(v[j]) || std::isinf(v[j])) { has_nan = true; break; }
        }
        if (has_nan) break;

        // Check convergence every 10 iterations
        if ((iter + 1) % 10 == 0) {
            float error = 0.0f;
            for (int i = 0; i < n; ++i) {
                float row_sum = 0.0f;
                for (int j = 0; j < m; ++j) {
                    row_sum += u[i] * K[i * m + j] * v[j];
                }
                error += fabsf(row_sum - mu[i]);
            }
            if (error < threshold) {
                iter++;
                auto end = std::chrono::high_resolution_clock::now();
                float ms = std::chrono::duration<float, std::milli>(end - start).count();
                // Compute transport cost
                float cost = 0.0f;
                for (int i = 0; i < n; ++i)
                    for (int j = 0; j < m; ++j)
                        cost += C[i * m + j] * u[i] * K[i * m + j] * v[j];
                return {cost, iter, error, true, false, ms};
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    float ms = std::chrono::duration<float, std::milli>(end - start).count();
    return {0.0f, iter, 0.0f, false, has_nan, ms};
}

// ============================================================================
// Helpers
// ============================================================================

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

// ============================================================================
// Main
// ============================================================================

int main() {
    const int N = 1024;
    const int max_iter = 2000;
    const float threshold = 1e-6f;
    const int warmup = 3;
    const int runs = 10;

    std::vector<float> epsilons = {1.0f, 0.1f, 0.05f, 0.01f, 0.005f, 0.001f, 0.0005f, 0.0001f};

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
    printf("Ablation: Log-domain vs Standard-domain Sinkhorn (N=%d)\n\n", N);

    FILE* fp = fopen("experiments/data/ablation_logdomain.csv", "w");
    if (!fp) { fprintf(stderr, "Cannot open output file\n"); return 1; }
    fprintf(fp, "method,n,epsilon,time_ms_mean,time_ms_std,iterations,transport_cost,marginal_error,converged,has_nan\n");

    for (float eps : epsilons) {
        printf("epsilon = %.4f:\n", eps);

        // --- Log-domain (our GPU solver) ---
        {
            SinkhornConfig config;
            config.epsilon = eps;
            config.max_iterations = max_iter;
            config.convergence_threshold = threshold;
            config.convergence_check_interval = 10;

            std::vector<float> times;
            SinkhornResult last_result;

            for (int r = 0; r < warmup + runs; ++r) {
                SinkhornSolver solver(config);
                auto result = solver.solve(mu, nu, C, N, N);
                if (r >= warmup) {
                    times.push_back(result.elapsed_ms);
                    last_result = result;
                }
            }

            float mean = std::accumulate(times.begin(), times.end(), 0.0f) / times.size();
            float sq = 0; for (float t : times) sq += (t - mean) * (t - mean);
            float std_dev = sqrtf(sq / times.size());

            fprintf(fp, "log_domain,%d,%.6f,%.3f,%.3f,%d,%.6f,%.2e,%d,0\n",
                    N, eps, mean, std_dev, last_result.iterations,
                    last_result.transport_cost, last_result.marginal_error,
                    last_result.converged ? 1 : 0);
            printf("  Log-domain:     %.2f ms, iters=%d, conv=%s\n",
                   mean, last_result.iterations, last_result.converged ? "yes" : "no");
        }

        // --- Standard-domain (CPU) ---
        {
            std::vector<float> times;
            StandardResult last_result = {};

            for (int r = 0; r < warmup + runs; ++r) {
                auto result = runStandardDomainCPU(mu, nu, C, N, N, eps, max_iter, threshold);
                if (r >= warmup) {
                    times.push_back(result.time_ms);
                    last_result = result;
                }
            }

            float mean = 0, std_dev = 0;
            if (!last_result.has_nan) {
                mean = std::accumulate(times.begin(), times.end(), 0.0f) / times.size();
                float sq = 0; for (float t : times) sq += (t - mean) * (t - mean);
                std_dev = sqrtf(sq / times.size());
            }

            fprintf(fp, "standard_domain,%d,%.6f,%.3f,%.3f,%d,%.6f,%.2e,%d,%d\n",
                    N, eps,
                    last_result.has_nan ? 0.0f : mean,
                    last_result.has_nan ? 0.0f : std_dev,
                    last_result.iterations,
                    last_result.transport_cost,
                    last_result.marginal_error,
                    last_result.converged ? 1 : 0,
                    last_result.has_nan ? 1 : 0);
            printf("  Standard-domain: %s\n",
                   last_result.has_nan ? "FAILED (NaN)" :
                   (last_result.converged ?
                    (std::string("") + std::to_string(mean) + " ms").c_str() : "did not converge"));
        }

        printf("\n");
    }

    fclose(fp);
    printf("Results written to experiments/data/ablation_logdomain.csv\n");
    return 0;
}
