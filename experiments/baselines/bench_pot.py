#!/usr/bin/env python3
"""
Baseline benchmark: POT (Python Optimal Transport) library on CPU.

Computes the regularised Sinkhorn transport cost using ot.sinkhorn2 for a
pair of 1-D Gaussian distributions over a range of problem sizes and
regularisation parameters.  Results are written as CSV in the same format
consumed by plot_baselines.py.

Usage:
    python bench_pot.py [--output-dir experiments/data]
"""

import argparse
import csv
import math
import os
import time

import numpy as np
import ot  # Python Optimal Transport


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
def make_distributions(n):
    """Return (mu, nu, C) for two 1-D Gaussians on a uniform grid [0, 1]."""
    x = np.linspace(0, 1, n)

    # Un-normalised Gaussian densities
    mu = np.exp(-0.5 * ((x - MU_CENTER) / SIGMA) ** 2)
    nu = np.exp(-0.5 * ((x - NU_CENTER) / SIGMA) ** 2)

    # Normalise to probability vectors
    mu /= mu.sum()
    nu /= nu.sum()

    # Squared Euclidean cost matrix C[i,j] = (x_i - x_j)^2
    C = (x[:, None] - x[None, :]) ** 2
    C = np.ascontiguousarray(C, dtype=np.float64)

    return mu, nu, C


def run_sinkhorn(mu, nu, C, epsilon):
    """Run POT Sinkhorn and return (cost, iterations, converged, elapsed_ms).

    POT's sinkhorn2 returns the transport cost directly.  We also pass
    ``log=True`` so that we can retrieve the number of iterations from
    the log dictionary.
    """
    start = time.perf_counter()
    cost, log_dict = ot.sinkhorn2(
        mu,
        nu,
        C,
        reg=epsilon,
        numItermax=5000,
        stopThr=1e-9,
        log=True,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    # Extract iteration count from log (POT stores the error list)
    iterations = len(log_dict.get("logu", log_dict.get("log", {}).get("logu", [])))
    # Fallback: use length of the error vector
    if iterations == 0 and "err" in log_dict:
        iterations = len(log_dict["err"])

    # Determine convergence: POT raises a warning but still returns a result.
    # If the final marginal error is below the threshold we call it converged.
    converged = True
    if "err" in log_dict and len(log_dict["err"]) > 0:
        converged = log_dict["err"][-1] < 1e-9

    return float(cost), iterations, converged, elapsed_ms


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Benchmark POT Sinkhorn (CPU) on 1-D Gaussian OT problems."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/data",
        help="Directory for the output CSV file (default: experiments/data)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "baseline_pot_cpu.csv")

    print("=" * 70)
    print("POT Sinkhorn (CPU) baseline benchmark")
    print("=" * 70)

    rows = []

    for n in SIZES:
        mu, nu, C = make_distributions(n)
        for epsilon in EPSILONS:
            print(f"\n  n={n:>5d}  m={n:>5d}  epsilon={epsilon:.3f}")

            # Warmup runs (discard results)
            for _ in range(WARMUP_RUNS):
                run_sinkhorn(mu, nu, C, epsilon)

            # Timed runs
            times_ms = []
            last_cost = 0.0
            last_iters = 0
            last_converged = False

            for _ in range(TIMED_RUNS):
                cost, iters, conv, t_ms = run_sinkhorn(mu, nu, C, epsilon)
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
                    "gpu": "N/A (CPU)",
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
