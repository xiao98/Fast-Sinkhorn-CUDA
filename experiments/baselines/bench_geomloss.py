#!/usr/bin/env python3
"""
Baseline benchmark: GeomLoss library with PyTorch GPU backend.

Computes the Sinkhorn divergence via geomloss.SamplesLoss for a pair of 1-D
Gaussian distributions over a range of problem sizes and regularisation
parameters.  Results are written as CSV in the same format consumed by
plot_baselines.py.

Note on the blur parameter
--------------------------
GeomLoss parameterises regularisation via ``blur`` (standard deviation in
feature space).  For squared-Euclidean cost the mapping is approximately
blur = sqrt(epsilon), which gives a Sinkhorn kernel exp(-C / epsilon) when
C = ||x-y||^2.

Usage:
    python bench_geomloss.py [--output-dir experiments/data]
"""

import argparse
import csv
import math
import os
import time

import numpy as np
import torch
from geomloss import SamplesLoss


# ---------------------------------------------------------------------------
# Experiment grid
# ---------------------------------------------------------------------------
SIZES = [256, 512, 1024, 2048, 4096, 8192]
EPSILONS = [0.1, 0.01, 0.001]

WARMUP_RUNS = 3
TIMED_RUNS = 10

# Distribution parameters
MU_CENTER = 0.3
NU_CENTER = 0.7
SIGMA = 0.08


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_distributions(n, device):
    """Return (mu_weights, nu_weights, mu_points, nu_points, C) on *device*.

    GeomLoss operates on weighted point clouds, so we return both the weights
    (probability masses) and the point coordinates as 2-D tensors of shape
    (1, n, 1) and (1, n) respectively.

    The cost matrix C is returned separately for transport-cost computation
    after timing.
    """
    x_np = np.linspace(0, 1, n, dtype=np.float64)

    # Un-normalised Gaussian densities
    mu_np = np.exp(-0.5 * ((x_np - MU_CENTER) / SIGMA) ** 2)
    nu_np = np.exp(-0.5 * ((x_np - NU_CENTER) / SIGMA) ** 2)
    mu_np /= mu_np.sum()
    nu_np /= nu_np.sum()

    # Convert to torch tensors on GPU (float32 for GPU efficiency)
    mu_weights = torch.tensor(mu_np, dtype=torch.float32, device=device)
    nu_weights = torch.tensor(nu_np, dtype=torch.float32, device=device)

    # Point coordinates -- shape (n, 1) for 1-D problem
    mu_points = torch.tensor(x_np, dtype=torch.float32, device=device).unsqueeze(1)
    nu_points = mu_points.clone()  # same grid

    # Cost matrix for later transport-cost evaluation
    C = (mu_points - nu_points.T) ** 2  # (n, n)

    return mu_weights, nu_weights, mu_points, nu_points, C


def run_geomloss(mu_weights, nu_weights, mu_points, nu_points, epsilon):
    """Run GeomLoss Sinkhorn and return (cost, elapsed_ms).

    GeomLoss does not directly expose iteration counts or convergence flags,
    so we report ``iterations=-1`` and ``converged=True`` as placeholders.
    The ``scaling`` parameter (multi-scale descent) defaults to 0.9.
    """
    blur = math.sqrt(epsilon)
    loss_fn = SamplesLoss(
        loss="sinkhorn",
        p=2,
        blur=blur,
        scaling=0.9,
        backend="tensorized",
    )

    # GeomLoss expects (batch, n, d) or (n, d) point clouds.  For weighted
    # samples we pass weights explicitly.
    torch.cuda.synchronize()
    start = time.perf_counter()

    cost = loss_fn(mu_weights, mu_points, nu_weights, nu_points)

    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    return float(cost.item()), elapsed_ms


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Benchmark GeomLoss Sinkhorn (GPU) on 1-D Gaussian OT problems."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/data",
        help="Directory for the output CSV file (default: experiments/data)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "baseline_geomloss.csv")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.  GeomLoss GPU benchmark requires a GPU.")

    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)

    print("=" * 70)
    print("GeomLoss Sinkhorn (GPU) baseline benchmark")
    print(f"  GPU: {gpu_name}")
    print("=" * 70)

    rows = []

    for n in SIZES:
        mu_w, nu_w, mu_pts, nu_pts, C = make_distributions(n, device)
        for epsilon in EPSILONS:
            print(f"\n  n={n:>5d}  m={n:>5d}  epsilon={epsilon:.3f}")

            # Warmup runs
            for _ in range(WARMUP_RUNS):
                run_geomloss(mu_w, nu_w, mu_pts, nu_pts, epsilon)

            # Timed runs
            times_ms = []
            last_cost = 0.0

            for _ in range(TIMED_RUNS):
                cost, t_ms = run_geomloss(mu_w, nu_w, mu_pts, nu_pts, epsilon)
                times_ms.append(t_ms)
                last_cost = cost

            mean_ms = float(np.mean(times_ms))
            std_ms = float(np.std(times_ms))

            print(
                f"    time = {mean_ms:>10.3f} +/- {std_ms:>8.3f} ms  "
                f"cost = {last_cost:.6f}"
            )

            # GeomLoss does not expose iteration count; use -1 as sentinel.
            rows.append(
                {
                    "n": n,
                    "m": n,
                    "epsilon": epsilon,
                    "time_ms_mean": f"{mean_ms:.3f}",
                    "time_ms_std": f"{std_ms:.3f}",
                    "iterations": -1,
                    "transport_cost": f"{last_cost:.6f}",
                    "converged": 1,
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
