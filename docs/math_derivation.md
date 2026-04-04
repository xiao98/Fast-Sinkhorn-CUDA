# Mathematical Derivation: Entropic Regularized Optimal Transport

*Complete derivation from measure theory to GPU-optimized algorithm.*

---

## 1. The Kantorovich Problem

### 1.1 Measure-Theoretic Foundation

Let $(X, \mu)$ and $(Y, \nu)$ be two probability spaces, where $\mu \in \mathcal{P}(X)$ and $\nu \in \mathcal{P}(Y)$ are probability measures. Let $c: X \times Y \to \mathbb{R}_+$ be a measurable cost function.

**Definition (Coupling).** A *coupling* of $\mu$ and $\nu$ is a joint probability measure $\pi \in \mathcal{P}(X \times Y)$ whose marginals are $\mu$ and $\nu$:

$$\Pi(\mu, \nu) = \left\{ \pi \in \mathcal{P}(X \times Y) \mid (P_X)_{\#}\pi = \mu, \quad (P_Y)_{\#}\pi = \nu \right\}$$

where $(P_X)_{\#}\pi$ denotes the pushforward measure.

**Definition (Kantorovich Problem).** The *optimal transport cost* between $\mu$ and $\nu$ is:

$$\mathcal{T}_c(\mu, \nu) = \inf_{\pi \in \Pi(\mu, \nu)} \int_{X \times Y} c(x, y) \, d\pi(x, y)$$

### 1.2 Discrete Setting

In the computational setting, we work with *discrete* measures:

$$\mu = \sum_{i=1}^{n} \mu_i \, \delta_{x_i}, \qquad \nu = \sum_{j=1}^{m} \nu_j \, \delta_{y_j}$$

where $\mu \in \Sigma_n = \{a \in \mathbb{R}_+^n : \sum_i a_i = 1\}$ is the probability simplex.

The coupling $\pi$ becomes a matrix $\pi \in \mathbb{R}_+^{n \times m}$ and the problem reduces to:

$$\min_{\pi \in \Pi(\mu, \nu)} \langle C, \pi \rangle_F = \sum_{i,j} C_{ij} \pi_{ij}$$

where $C_{ij} = c(x_i, y_j)$ and the feasible set is:

$$\Pi(\mu, \nu) = \left\{ \pi \in \mathbb{R}_+^{n \times m} \mid \pi \mathbf{1}_m = \mu, \quad \pi^\top \mathbf{1}_n = \nu \right\}$$

> **Remark.** This is a linear program (LP) in $nm$ variables with $n + m$ equality constraints. The computational complexity of exact solvers (e.g., network simplex) scales as $O(n^3 \log n)$, which is prohibitive for modern applications.

---

## 2. Entropic Regularization

### 2.1 Motivation

To overcome the computational bottleneck of exact OT, we add an **entropic regularization** term (Cuturi, 2013):

$$\mathcal{T}_c^\varepsilon(\mu, \nu) = \min_{\pi \in \Pi(\mu, \nu)} \langle C, \pi \rangle + \varepsilon \, \text{KL}(\pi \| \mu \otimes \nu)$$

where the Kullback-Leibler divergence is:

$$\text{KL}(\pi \| \mu \otimes \nu) = \sum_{i,j} \pi_{ij} \left( \log \frac{\pi_{ij}}{\mu_i \nu_j} - 1 \right) + 1$$

### 2.2 Properties of the Regularized Problem

**Proposition.** For $\varepsilon > 0$:
1. The objective is **strictly convex** (sum of a linear and strictly convex function).
2. There exists a **unique** minimizer $\pi^\varepsilon$.
3. As $\varepsilon \to 0^+$, we have $\pi^\varepsilon \to \pi^*$ (optimal solution of the original LP).
4. The regularized problem can be solved in $\tilde{O}(n^2 / \varepsilon)$ time.

### 2.3 Structure of the Optimal Solution

Writing the Lagrangian of the regularized problem:

$$\mathcal{L}(\pi, \alpha, \beta) = \langle C, \pi \rangle + \varepsilon \, \text{KL}(\pi \| \mu \otimes \nu) - \langle \alpha, \pi \mathbf{1}_m - \mu \rangle - \langle \beta, \pi^\top \mathbf{1}_n - \nu \rangle$$

Setting $\frac{\partial \mathcal{L}}{\partial \pi_{ij}} = 0$:

$$C_{ij} + \varepsilon \log \frac{\pi_{ij}}{\mu_i \nu_j} - \alpha_i - \beta_j = 0$$

Solving for $\pi_{ij}$:

$$\boxed{\pi_{ij}^\varepsilon = \mu_i \nu_j \exp\left(\frac{\alpha_i + \beta_j - C_{ij}}{\varepsilon}\right)}$$

This shows the optimal transport plan has a **Gibbs kernel** structure.

---

## 3. The Sinkhorn Algorithm

### 3.1 Derivation from KKT Conditions

Substituting the optimal form of $\pi$ into the marginal constraints:

**Row constraint** ($\pi \mathbf{1}_m = \mu$):

$$\sum_j \mu_i \nu_j \exp\left(\frac{\alpha_i + \beta_j - C_{ij}}{\varepsilon}\right) = \mu_i$$

$$\Rightarrow \quad \sum_j \nu_j \exp\left(\frac{\beta_j - C_{ij}}{\varepsilon}\right) = \exp\left(\frac{-\alpha_i}{\varepsilon}\right)$$

$$\Rightarrow \quad \alpha_i = -\varepsilon \log \sum_j \nu_j \exp\left(\frac{\beta_j - C_{ij}}{\varepsilon}\right)$$

**Column constraint** ($\pi^\top \mathbf{1}_n = \nu$) gives analogously:

$$\beta_j = -\varepsilon \log \sum_i \mu_i \exp\left(\frac{\alpha_i - C_{ij}}{\varepsilon}\right)$$

### 3.2 Fixed-Point Iteration

The Sinkhorn algorithm alternates between these two updates:

$$\boxed{\alpha_i^{(k+1)} = -\varepsilon \log \sum_j \nu_j \exp\left(\frac{\beta_j^{(k)} - C_{ij}}{\varepsilon}\right)}$$

$$\boxed{\beta_j^{(k+1)} = -\varepsilon \log \sum_i \mu_i \exp\left(\frac{\alpha_i^{(k+1)} - C_{ij}}{\varepsilon}\right)}$$

Starting from $\alpha^{(0)} = 0, \beta^{(0)} = 0$.

### 3.3 Convergence

**Theorem (Convergence Rate).** The Sinkhorn algorithm converges linearly:

$$\|\alpha^{(k)} - \alpha^*\|_\infty \leq \lambda^k \|\alpha^{(0)} - \alpha^*\|_\infty$$

where the contraction rate $\lambda = e^{-2R/\varepsilon}$ and $R = \max_{i,j} C_{ij} - \min_{i,j} C_{ij}$.

> **Key insight for GPU:** Smaller $\varepsilon$ → slower convergence ($\lambda \to 1$). This is the fundamental trade-off: approximation quality vs. computation cost.

---

## 4. Log-Domain Stabilization

### 4.1 The Numerical Problem

In the "standard" Sinkhorn (matrix scaling form):

$$u_i = \frac{\mu_i}{\sum_j K_{ij} v_j}, \qquad v_j = \frac{\nu_j}{\sum_i K_{ij} u_i}$$

where $K_{ij} = \exp(-C_{ij}/\varepsilon)$. For small $\varepsilon$, the kernel entries $K_{ij}$ range from near-zero to very large, causing **overflow/underflow** in floating-point arithmetic.

### 4.2 LogSumExp Trick

We work entirely in the log-domain. The key operation is the **numerically stable LogSumExp**:

$$\text{LSE}(x_1, \ldots, x_m) = \log \sum_{j=1}^m e^{x_j} = M + \log \sum_{j=1}^m e^{x_j - M}$$

where $M = \max_j x_j$.

**Why this works:** By subtracting the maximum, we ensure that:
- The largest exponent becomes $e^0 = 1$ (no overflow)
- All other exponents are $e^{x_j - M} \leq 1$ (no overflow)
- The sum is at least 1, so $\log(\cdot) \geq 0$ (no underflow)

### 4.3 Log-Domain Sinkhorn Updates

The Sinkhorn updates become:

$$\alpha_i = -\varepsilon \cdot \text{LSE}_j\left(\frac{\beta_j - C_{ij}}{\varepsilon} + \log \nu_j\right)$$

$$\beta_j = -\varepsilon \cdot \text{LSE}_i\left(\frac{\alpha_i - C_{ij}}{\varepsilon} + \log \mu_i\right)$$

### 4.4 Convergence Check

The marginal constraint error (our convergence criterion) is:

$$\text{err} = \sum_i |r_i - \mu_i|, \qquad r_i = \sum_j \pi_{ij}^\varepsilon$$

where the row marginal $r_i$ is computed in log-domain, avoiding materializing $\pi$ explicitly:

$$\log r_i = \log \mu_i + \text{LSE}_j\left(\frac{\alpha_i + \beta_j - C_{ij}}{\varepsilon} + \log \nu_j\right)$$

---

## 5. CUDA Implementation Strategy

### 5.1 Parallelization Structure

Each Sinkhorn half-iteration requires a LogSumExp reduction:
- **α update:** For each row $i$, reduce over all columns $j$ → **One CUDA block per row**
- **β update:** For each column $j$, reduce over all rows $i$ → **One CUDA block per column**

### 5.2 Warp-Level Reduction

The LogSumExp reduction within a block uses a two-pass approach:

```
Pass 1: max_val = warpReduceMax(thread_local_max)   // __shfl_down_sync
Pass 2: sum_val = warpReduceSum(exp(x - max_val))   // __shfl_down_sync
Result: max_val + log(sum_val)
```

Using `__shfl_down_sync` (warp shuffle) instead of shared memory provides:
- No shared memory bank conflicts
- No `__syncthreads()` barrier within a warp
- ~2× faster than shared memory reduction for warp-level operations

### 5.3 Memory Layout

| Buffer | Size | Type | Description |
|--------|------|------|-------------|
| `d_C` | $n \times m$ | float | Cost matrix (dominant memory) |
| `d_alpha` | $n$ | float | Source dual potential |
| `d_beta` | $m$ | float | Target dual potential |
| `d_log_mu` | $n$ | float | Log source weights |
| `d_log_nu` | $m$ | float | Log target weights |
| **Total** | $\approx nm + 2(n+m)$ | | **Linear in $nm$** |

> **Memory analysis:** For $n = m = 8192$, the cost matrix alone requires $8192^2 \times 4 = 256$ MB. The dual potentials add only $\approx 64$ KB — negligible. The cost matrix is the bottleneck.

### 5.4 Computational Complexity

| Operation | Per Iteration | Total ($K$ iterations) |
|-----------|--------------|----------------------|
| α update | $O(nm)$ | $O(Knm)$ |
| β update | $O(nm)$ | $O(Knm)$ |
| Convergence check | $O(nm)$ | $O(nm \cdot K/c)$ |
| **Total** | | **$O(Knm)$** |

where $c$ is the convergence check interval (we use $c = 10$ to amortize GPU→CPU transfers).

---

## 6. References

1. **Cuturi, M.** (2013). "Sinkhorn Distances: Lightspeed Computation of Optimal Transport." *NeurIPS*.
2. **Peyré, G. & Cuturi, M.** (2019). "Computational Optimal Transport." *Foundations and Trends in Machine Learning*, 11(5-6), 355-607. [arXiv:1803.00567](https://arxiv.org/abs/1803.00567)
3. **Schmitzer, B.** (2019). "Stabilized Sparse Scaling Algorithms for Entropy Regularized Transport Problems." *SIAM J. Sci. Comput.*
4. **Sinkhorn, R.** (1967). "Diagonal Equivalence to Matrices with Prescribed Row and Column Sums." *American Mathematical Monthly*, 74(4), 402-405.
5. **Villani, C.** (2003). *Topics in Optimal Transportation.* AMS Graduate Studies in Mathematics 58.
