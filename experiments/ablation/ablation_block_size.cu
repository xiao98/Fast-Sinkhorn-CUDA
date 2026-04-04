/**
 * @file ablation_block_size.cu
 * @brief Ablation: effect of CUDA block size on Sinkhorn performance.
 *
 * Tests block sizes: 64, 128, 256, 512.
 * Uses template parameter to compile kernels for each block size.
 *
 * Output: experiments/data/ablation_blocksize.csv
 */

#include "sinkhorn/config.h"
#include <cuda_runtime.h>
#include <cfloat>
#include <cstdio>
#include <cmath>
#include <vector>
#include <numeric>

using namespace fastsinkhorn;

// ============================================================================
// Templated warp/block reductions (same logic as main kernels)
// ============================================================================

__device__ __forceinline__ float warpMax(float val) {
    for (int o = 16; o > 0; o >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, o));
    return val;
}

__device__ __forceinline__ float warpSum(float val) {
    for (int o = 16; o > 0; o >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, o);
    return val;
}

__device__ float blkMax(float val) {
    __shared__ float s[32];
    int lane = threadIdx.x % 32, wid = threadIdx.x / 32;
    val = warpMax(val);
    if (lane == 0) s[wid] = val;
    __syncthreads();
    int nw = (blockDim.x + 31) / 32;
    val = (threadIdx.x < nw) ? s[threadIdx.x] : -FLT_MAX;
    if (wid == 0) val = warpMax(val);
    return val;
}

__device__ float blkSum(float val) {
    __shared__ float s[32];
    int lane = threadIdx.x % 32, wid = threadIdx.x / 32;
    val = warpSum(val);
    if (lane == 0) s[wid] = val;
    __syncthreads();
    int nw = (blockDim.x + 31) / 32;
    val = (threadIdx.x < nw) ? s[threadIdx.x] : 0.0f;
    if (wid == 0) val = warpSum(val);
    return val;
}

// ============================================================================
// Kernels (same as main but we'll vary block size at launch time)
// ============================================================================

__global__ void alphaKernel(const float* C, const float* log_nu, const float* beta,
                            float* alpha, int n, int m, float eps) {
    int i = blockIdx.x;
    if (i >= n) return;
    float inv_eps = 1.0f / eps;
    const float* Ci = C + i * m;

    float tmax = -FLT_MAX;
    for (int j = threadIdx.x; j < m; j += blockDim.x)
        tmax = fmaxf(tmax, (beta[j] - Ci[j]) * inv_eps + log_nu[j]);
    float rmax = blkMax(tmax);

    __shared__ float sm;
    if (threadIdx.x == 0) sm = rmax;
    __syncthreads();
    rmax = sm;

    float tsum = 0.0f;
    for (int j = threadIdx.x; j < m; j += blockDim.x)
        tsum += expf((beta[j] - Ci[j]) * inv_eps + log_nu[j] - rmax);
    float rsum = blkSum(tsum);

    if (threadIdx.x == 0) alpha[i] = -eps * (rmax + logf(fmaxf(rsum, 1e-30f)));
}

__global__ void betaKernel(const float* C, const float* log_mu, const float* alpha,
                           float* beta, int n, int m, float eps) {
    int j = blockIdx.x;
    if (j >= m) return;
    float inv_eps = 1.0f / eps;

    float tmax = -FLT_MAX;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        tmax = fmaxf(tmax, (alpha[i] - C[i * m + j]) * inv_eps + log_mu[i]);
    float cmax = blkMax(tmax);

    __shared__ float sm;
    if (threadIdx.x == 0) sm = cmax;
    __syncthreads();
    cmax = sm;

    float tsum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        tsum += expf((alpha[i] - C[i * m + j]) * inv_eps + log_mu[i] - cmax);
    float csum = blkSum(tsum);

    if (threadIdx.x == 0) beta[j] = -eps * (cmax + logf(fmaxf(csum, 1e-30f)));
}

__global__ void errorKernel(const float* C, const float* alpha, const float* beta,
                            const float* log_mu, const float* log_nu, const float* mu,
                            float* errors, int n, int m, float eps) {
    int i = blockIdx.x;
    if (i >= n) return;
    float inv_eps = 1.0f / eps;
    const float* Ci = C + i * m;

    float tmax = -FLT_MAX;
    for (int j = threadIdx.x; j < m; j += blockDim.x)
        tmax = fmaxf(tmax, (alpha[i] + beta[j] - Ci[j]) * inv_eps + log_nu[j]);
    float rmax = blkMax(tmax);

    __shared__ float sm;
    if (threadIdx.x == 0) sm = rmax;
    __syncthreads();
    rmax = sm;

    float tsum = 0.0f;
    for (int j = threadIdx.x; j < m; j += blockDim.x)
        tsum += expf((alpha[i] + beta[j] - Ci[j]) * inv_eps + log_nu[j] - rmax);
    float rsum = blkSum(tsum);

    if (threadIdx.x == 0) {
        float ri = expf(log_mu[i] + rmax + logf(fmaxf(rsum, 1e-30f)));
        errors[i] = fabsf(ri - mu[i]);
    }
}

__global__ void sumKernel(const float* in, float* out, int n) {
    float s = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += in[i];
    s = blkSum(s);
    if (threadIdx.x == 0) out[0] = s;
}

// ============================================================================
// Run solver with specific block size
// ============================================================================

struct Result {
    float time_ms;
    int iterations;
    bool converged;
};

Result runWithBlockSize(const float* d_C, const float* d_mu, const float* d_nu,
                        const float* d_log_mu, const float* d_log_nu,
                        int n, int m, float eps, int block_size,
                        int max_iter, float threshold, int check_interval)
{
    float *d_alpha, *d_beta, *d_errors, *d_scalar;
    cudaMalloc(&d_alpha, n * sizeof(float));
    cudaMalloc(&d_beta, m * sizeof(float));
    cudaMalloc(&d_errors, n * sizeof(float));
    cudaMalloc(&d_scalar, sizeof(float));
    cudaMemset(d_alpha, 0, n * sizeof(float));
    cudaMemset(d_beta, 0, m * sizeof(float));

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);

    Result result = {};
    int iter = 0;
    for (; iter < max_iter; ++iter) {
        alphaKernel<<<n, block_size>>>(d_C, d_log_nu, d_beta, d_alpha, n, m, eps);
        betaKernel<<<m, block_size>>>(d_C, d_log_mu, d_alpha, d_beta, n, m, eps);

        if ((iter + 1) % check_interval == 0) {
            errorKernel<<<n, block_size>>>(d_C, d_alpha, d_beta, d_log_mu, d_log_nu, d_mu, d_errors, n, m, eps);
            sumKernel<<<1, block_size>>>(d_errors, d_scalar, n);
            float error;
            cudaMemcpy(&error, d_scalar, sizeof(float), cudaMemcpyDeviceToHost);
            if (error < threshold) { result.converged = true; ++iter; break; }
        }
    }
    result.iterations = iter;

    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    cudaEventElapsedTime(&result.time_ms, t0, t1);

    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    cudaFree(d_alpha); cudaFree(d_beta); cudaFree(d_errors); cudaFree(d_scalar);
    return result;
}

// ============================================================================
// Main
// ============================================================================

std::vector<float> makeGaussian(int n, float center, float sigma) {
    std::vector<float> d(n);
    float sum = 0;
    for (int i = 0; i < n; ++i) {
        float x = float(i) / (n - 1);
        d[i] = expf(-0.5f * (x - center) * (x - center) / (sigma * sigma));
        sum += d[i];
    }
    for (int i = 0; i < n; ++i) d[i] /= sum;
    return d;
}

int main() {
    const int N = 2048;
    const float eps = 0.01f;
    const int max_iter = 2000;
    const float threshold = 1e-6f;
    const int check_interval = 10;
    const int warmup = 3, runs = 10;

    std::vector<int> block_sizes = {64, 128, 256, 512};

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n", prop.name);
    printf("Ablation: Block Size (N=%d, eps=%.4f)\n\n", N, eps);

    auto mu = makeGaussian(N, 0.3f, 0.08f);
    auto nu = makeGaussian(N, 0.7f, 0.08f);

    std::vector<float> C(N * N);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            float xi = float(i) / (N - 1), yj = float(j) / (N - 1);
            C[i * N + j] = (xi - yj) * (xi - yj);
        }

    float *d_C, *d_mu_dev, *d_nu_dev, *d_log_mu, *d_log_nu;
    cudaMalloc(&d_C, N * N * sizeof(float));
    cudaMalloc(&d_mu_dev, N * sizeof(float));
    cudaMalloc(&d_nu_dev, N * sizeof(float));
    cudaMalloc(&d_log_mu, N * sizeof(float));
    cudaMalloc(&d_log_nu, N * sizeof(float));

    cudaMemcpy(d_C, C.data(), N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mu_dev, mu.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nu_dev, nu.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    std::vector<float> lm(N), ln(N);
    for (int i = 0; i < N; ++i) lm[i] = logf(fmaxf(mu[i], 1e-30f));
    for (int i = 0; i < N; ++i) ln[i] = logf(fmaxf(nu[i], 1e-30f));
    cudaMemcpy(d_log_mu, lm.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_log_nu, ln.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    FILE* fp = fopen("experiments/data/ablation_blocksize.csv", "w");
    fprintf(fp, "config,n,epsilon,time_ms_mean,time_ms_std,iterations,converged,notes\n");

    for (int bs : block_sizes) {
        printf("Block size = %d:\n", bs);

        std::vector<float> times;
        int last_iters = 0;
        bool last_conv = false;

        for (int r = 0; r < warmup + runs; ++r) {
            auto result = runWithBlockSize(d_C, d_mu_dev, d_nu_dev, d_log_mu, d_log_nu,
                                           N, N, eps, bs, max_iter, threshold, check_interval);
            if (r >= warmup) {
                times.push_back(result.time_ms);
                last_iters = result.iterations;
                last_conv = result.converged;
            }
        }

        float mean = std::accumulate(times.begin(), times.end(), 0.0f) / times.size();
        float sq = 0; for (float t : times) sq += (t - mean) * (t - mean);
        float std_dev = sqrtf(sq / times.size());

        char config_name[64];
        snprintf(config_name, sizeof(config_name), "block_size_%d", bs);
        fprintf(fp, "%s,%d,%.4f,%.3f,%.3f,%d,%d,\n",
                config_name, N, eps, mean, std_dev, last_iters, last_conv ? 1 : 0);
        printf("  %.2f ± %.2f ms, iters=%d\n", mean, std_dev, last_iters);
    }

    fclose(fp);
    cudaFree(d_C); cudaFree(d_mu_dev); cudaFree(d_nu_dev);
    cudaFree(d_log_mu); cudaFree(d_log_nu);

    printf("\nResults written to experiments/data/ablation_blocksize.csv\n");
    return 0;
}
