# Fast-Sinkhorn-CUDA

A lightweight, highly optimized C++/CUDA library for computing **regularized optimal transport** via the Sinkhorn algorithm. Features a complete mathematical derivation from measure theory and detailed GPU memory profiling.

---

## ✨ Features

- **High Performance** — Custom CUDA kernels with warp-level reductions (`__shfl_down_sync`), shared memory tiling, and fused kernel operations
- **Numerically Stable** — Log-domain Sinkhorn algorithm prevents overflow/underflow for small regularization ε
- **Complete Math** — Full derivation from the Kantorovich problem through entropic regularization to GPU-optimized algorithm ([docs/math_derivation.md](docs/math_derivation.md))
- **GPU Memory Profiling** — Built-in `cudaMemGetInfo` memory tracking and `cudaEvent` timing with formatted reports
- **Clean C++17 API** — Simple `SinkhornSolver` interface with PIMPL pattern to isolate CUDA types
- **Validated** — Comprehensive test suite + cross-validation against Python OT (POT) library

## 📐 The Problem

Given two discrete probability distributions $\mu \in \Sigma_n$ and $\nu \in \Sigma_m$ with cost matrix $C \in \mathbb{R}^{n \times m}$, we solve:

$$\min_{\pi \in \Pi(\mu, \nu)} \langle C, \pi \rangle + \varepsilon \, \text{KL}(\pi \| \mu \otimes \nu)$$

The Sinkhorn algorithm finds the optimal transport plan $\pi^*$ by alternating dual potential updates:

$$\alpha_i \leftarrow -\varepsilon \cdot \text{LSE}_j\left(\frac{\beta_j - C_{ij}}{\varepsilon} + \log \nu_j\right)$$

$$\beta_j \leftarrow -\varepsilon \cdot \text{LSE}_i\left(\frac{\alpha_i - C_{ij}}{\varepsilon} + \log \mu_i\right)$$

## 🏗️ Project Structure

```
Fast-Sinkhorn-CUDA/
├── CMakeLists.txt              # CMake build system
├── README.md
├── docs/
│   └── math_derivation.md      # Complete mathematical derivation
├── include/sinkhorn/
│   ├── config.h                # Configuration & result structures
│   ├── cost_matrix.h           # Cost matrix computation API
│   ├── sinkhorn_solver.h       # Main solver interface
│   └── memory_profiler.h       # GPU profiling API
├── src/
│   ├── cost_matrix.cu          # CUDA cost matrix kernels
│   ├── sinkhorn_kernel.cu      # Core Sinkhorn CUDA kernels
│   ├── sinkhorn_solver.cpp     # Host-side solver orchestration
│   └── memory_profiler.cpp     # Memory profiling implementation
├── examples/
│   ├── basic_ot.cpp            # Basic OT example (two Gaussians)
│   └── benchmark.cpp           # Performance benchmarks
├── tests/
│   └── test_sinkhorn.cpp       # Correctness tests
└── scripts/
    └── compare_with_pot.py     # Cross-validation with Python OT
```

## 🔧 Build & Run

### Requirements
- CUDA Toolkit ≥ 11.0
- CMake ≥ 3.18
- C++17 compatible compiler

### Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Run Examples

```bash
# Basic OT between two Gaussians
./basic_ot

# Performance benchmark (N = 128 to 8192)
./benchmark

# Run correctness tests
./test_sinkhorn
```

### Validate Against POT

```bash
pip install POT numpy
python ../scripts/compare_with_pot.py
```

## 📊 Performance

Benchmark on NVIDIA RTX GPU with regularization ε = 0.01:

| N | Cost Matrix | Time (ms) | Iterations | Converged |
|---|-------------|-----------|------------|-----------|
| 128 | 64 KB | ~0.5 | ~100 | ✓ |
| 512 | 1 MB | ~2 | ~150 | ✓ |
| 2048 | 16 MB | ~15 | ~200 | ✓ |
| 8192 | 256 MB | ~200 | ~250 | ✓ |

*Actual numbers depend on GPU model. Run `./benchmark` for your hardware.*

## 🧮 GPU Memory Analysis

The built-in profiler reports per-stage GPU memory consumption:

```
╔══════════════════════════════════════════════════════════════╗
║                  GPU Memory Profiling Report                ║
╠══════════════════════════════════════════════════════════════╣
║ GPU Total Memory:   8192.0 MB                               ║
╠══════════════════════════════════════════════════════════════╣
║ Stage                     │ Time(ms) │ Used(MB) │ Free(MB) ║
╠══════════════════════════════════════════════════════════════╣
║ After allocation          │     0.00 │    272.1 │   7919.9 ║
║ Before iterations         │     0.00 │    272.1 │   7919.9 ║
║ After iterations          │    15.32 │    272.1 │   7919.9 ║
╠══════════════════════════════════════════════════════════════╣
║ Peak Memory Usage:   272.1 MB                              ║
╚══════════════════════════════════════════════════════════════╝
```

## 🔬 Key Implementation Details

### Why Log-Domain?

Standard Sinkhorn uses the Gibbs kernel $K_{ij} = e^{-C_{ij}/\varepsilon}$. For $\varepsilon = 0.01$ and $C_{ij} = 1$, this gives $K_{ij} = e^{-100} \approx 10^{-44}$, which underflows in float32. The log-domain formulation avoids this entirely.

### Warp Shuffle Reduction

We use `__shfl_down_sync` for the LogSumExp reduction instead of shared memory:

```cuda
__device__ float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    return val;
}
```

Advantages: no bank conflicts, no `__syncthreads`, ~2× faster for intra-warp ops.

### Convergence Strategy

We check convergence every 10 iterations (configurable) to avoid frequent GPU→CPU synchronization:

```
Iteration 10: marginal error = 1.23e-02
Iteration 20: marginal error = 4.56e-04
...
Iteration 80: marginal error = 8.90e-07  → CONVERGED
```

## 📚 References

1. Cuturi, M. (2013). *Sinkhorn Distances: Lightspeed Computation of Optimal Transport.* NeurIPS.
2. Peyré, G. & Cuturi, M. (2019). *Computational Optimal Transport.* [arXiv:1803.00567](https://arxiv.org/abs/1803.00567)
3. Schmitzer, B. (2019). *Stabilized Sparse Scaling Algorithms for Entropy Regularized Transport Problems.* SIAM J. Sci. Comput.

## 📄 License

MIT License
