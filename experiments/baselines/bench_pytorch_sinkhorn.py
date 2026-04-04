#!/usr/bin/env python3
"""
Baseline benchmark: log-domain Sinkhorn in pure PyTorch on GPU.

This is a naive GPU implementation of the Sinkhorn algorithm using
log-domain stabilisation.  It serves as a reference for what a
straightforward PyTorch port achieves without custom CUDA kernels.

Usage:
    python bench_pytorch_sinkhorn.py [--output-dir experiments/data]
"""

import argparse
import csv
import os
import time

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Experiment grid
# ---------------------------------------------------------------------------
SIZES = [256, 512, 1024, 2048, 4096, 8192]
EPSILONS = [0.1, 0.01, 0.001]

WARMUP_RUNS = 3
TIMED_RUNS = 10

MAX_ITER = 5000
THRESHOLD = 1e-9
CONVERGENCE_CHECK_INTERVAL = 10

# Distribution parameters
MU_CENTER = 0.3
NU_CENTER = 0.7
SIGMA = 0.08


# ---------------------------------------------------------------------------
# Log-domain Sinkhorn
# ---------------------------------------------------------------------------
def sinkhorn_log_pytorch(C, mu, nu, epsilon, max_iter=MAX_ITER, threshold=THRESHOLD):
    """Log-domain Sinkhorn in pure PyTorch on GPU.

    Parameters
    ----------
    C : torch.Tensor, shape (n, m)
        Cost matrix.
    mu : torch.Tensor, shape (n,)
        Source distribution (sums to 1).
    nu : torch.Tensor, shape (m,)
        Target distribution (sums to 1).
    epsilon : float
        Regularisation parameter.
    max_iter : int
        Maximum number of Sinkhorn iterations.
    threshold : float
        Convergence threshold on the L1 marginal error.

    Returns
    -------
    alpha : torch.Tensor, shape (n,)
        Dual potential for the source.
    beta : torch.Tensor, shape (m,)
        Dual potential for the target.
    iterations : int
        Number of iterations performed.
    converged : bool
        Whether the marginal constraint error dropped below *threshold*.
    """
    n, m = C.shape
    log_mu = torch.log(mu)
    log_nu = torch.log(nu)

    alpha = torch.zeros(n, device=C.device, dtype=C.dtype)
    beta = torch.zeros(m, device=C.device, dtype=C.dtype)

    for k in range(max_iter):
        # alpha update:  alpha_i = -eps * LSE_j( (beta_j - C_ij) / eps + log(nu_j) )
        M_alpha = (beta.unsqueeze(0) - C) / epsilon + log_nu.unsqueeze(0)
        alpha = -epsilon * torch.logsumexp(M_alpha, dim=1)

        # beta update:   beta_j = -eps * LSE_i( (alpha_i - C_ij) / eps + log(mu_i) )
        M_beta = (alpha.unsqueeze(1) - C) / epsilon + log_mu.unsqueeze(1)
        beta = -epsilon * torch.logsumexp(M_beta, dim=0)

        # Periodic convergence check
        if (k + 1) % CONVERGENCE_CHECK_INTERVAL == 0:
            # Row marginal in log-space
            log_pi_rowsum = log_mu + torch.logsumexp(
                (alpha.unsqueeze(1) + beta.unsqueeze(0) - C) / epsilon
                + log_nu.unsqueeze(0),
                dim=1,
            )
            row_marginal = torch.exp(log_pi_rowsum)
            err = (row_marginal - mu).abs().sum().item()
            if err < threshold:
                return alpha, beta, k + 1, True

    return alpha, beta, max_iter, False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_distributions(n, device):
    """Return (mu, nu, C) on *device* as float64 tensors."""
    x_np = np.linspace(0, 1, n, dtype=np.float64)

    # Un-normalised Gaussian densities
    mu_np = np.exp(-0.5 * ((x_np - MU_CENTER) / SIGMA) ** 2)
    nu_np = np.exp(-0.5 * ((x_np - NU_CENTER) / SIGMA) ** 2)
    mu_np /= mu_np.sum()
    nu_np /= nu_np.sum()

    mu = torch.tensor(mu_np, dtype=torch.float64, device=device)
    nu = torch.tensor(nu_np, dtype=torch.float64, device=device)

    # Cost matrix C[i,j] = (x_i - x_j)^2
    x = torch.tensor(x_np, dtype=torch.float64, device=device)
    C = (x[:, None] - x[None, :]) ** 2

    return mu, nu, C


def compute_transport_cost(alpha, beta, C, mu, nu, epsilon):
    """Compute the primal transport cost from dual potentials.

    Given optimal dual potentials (alpha, beta) the transport plan is
        pi_ij = mu_i * nu_j * exp( (alpha_i + beta_j - C_ij) / epsilon )
    and the transport cost is
        <C, pi> = sum_ij C_ij * pi_ij
    """
    log_mu = torch.log(mu)
    log_nu = torch.log(nu)
    log_pi = (
        log_mu.unsqueeze(1)
        + log_nu.unsqueeze(0)
        + (alpha.unsqueeze(1) + beta.unsqueeze(0) - C) / epsilon
    )
    pi = torch.exp(log_pi)
    cost = (C * pi).sum()
    return float(cost.item())


def run_benchmark(mu, nu, C, epsilon):
    """Run one Sinkhorn solve with synchronisation and timing.

    Returns (transport_cost, iterations, converged, elapsed_ms).
    """
    torch.cuda.synchronize()
    start = time.perf_counter()

    alpha, beta, iterations, converged = sinkhorn_log_pytorch(
        C, mu, nu, epsilon, max_iter=MAX_ITER, threshold=THRESHOLD
    )

    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    cost = compute_transport_cost(alpha, beta, C, mu, nu, epsilon)
    return cost, iterations, converged, elapsed_ms


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Benchmark pure-PyTorch log-domain Sinkhorn (GPU) on 1-D Gaussian OT problems."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/data",
        help="Directory for the output CSV file (default: experiments/data)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "baseline_pytorch.csv")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.  PyTorch GPU benchmark requires a GPU.")

    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)

    print("=" * 70)
    print("Pure-PyTorch log-domain Sinkhorn (GPU) baseline benchmark")
    print(f"  GPU: {gpu_name}")
    print("=" * 70)

    rows = []

    for n in SIZES:
        mu, nu, C = make_distributions(n, device)
        for epsilon in EPSILONS:
            print(f"\n  n={n:>5d}  m={n:>5d}  epsilon={epsilon:.3f}")

            # Warmup runs
            for _ in range(WARMUP_RUNS):
                run_benchmark(mu, nu, C, epsilon)

            # Timed runs
            times_ms = []
            last_cost = 0.0
            last_iters = 0
            last_converged = False

            for _ in range(TIMED_RUNS):
                cost, iters, conv, t_ms = run_benchmark(mu, nu, C, epsilon)
                times_ms.append(t_ms)
                last_cost = cost
                last_iters = iters
                last_converged = conv

            mean_ms = float(np.mean(times_ms))
            std_ms = float(np.std(times_ms))

            print(
                f"    time = {mean_ms:>10.3f} +/- {std_ms:>8.3f} ms  "
                f"iters = {last_iters:>5d}  "
                f"cost = {last_cost:.6f}  "
                f"converged = {last_converged}"
            )

            rows.append(
                {
                    "n": n,
                    "m": n,
                    "epsilon": epsilon,
                    "time_ms_mean": f"{mean_ms:.3f}",
                    "time_ms_std": f"{std_ms:.3f}",
                    "iterations": last_iters,
                    "transport_cost": f"{last_cost:.6f}",
                    "converged": 1 if last_converged else 0,
                    "gpu": gpu_name,
                }
            )

    # Write CSV
    fieldnames = [
        "n",
        "m",
        "epsilon",
        "time_ms_mean",
        "time_ms_std",
        "iterations",
        "transport_cost",
        "converged",
        "gpu",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults written to {output_path}")
    print("Done.")


if __name__ == "__main__":
    main()
