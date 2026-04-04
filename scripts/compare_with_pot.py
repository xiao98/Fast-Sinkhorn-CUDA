#!/usr/bin/env python3
"""
compare_with_pot.py — Validate FastSinkhornCUDA results against Python OT library.

Computes the same regularized optimal transport problem using the
well-established POT (Python Optimal Transport) library and compares
the results with our CUDA implementation.

Usage:
    pip install POT numpy
    python scripts/compare_with_pot.py

This script generates a reference CSV that can be compared with
the output of our C++/CUDA `basic_ot` example.
"""

import numpy as np

try:
    import ot
except ImportError:
    print("Please install POT: pip install POT")
    exit(1)


def make_gaussian(n: int, center: float, sigma: float) -> np.ndarray:
    """Generate a normalized Gaussian distribution on [0, 1]."""
    x = np.linspace(0, 1, n)
    dist = np.exp(-0.5 * ((x - center) / sigma) ** 2)
    return dist / dist.sum()


def main():
    print("=" * 60)
    print("  Validation: FastSinkhornCUDA vs POT (Python OT Library)")
    print("=" * 60)

    # Problem parameters (must match basic_ot.cpp)
    N = 128
    epsilon = 0.01
    center_mu = 0.3
    center_nu = 0.7
    sigma = 0.05

    # Create distributions
    mu = make_gaussian(N, center_mu, sigma)
    nu = make_gaussian(N, center_nu, sigma)

    # Cost matrix: squared Euclidean on [0, 1] grid
    x = np.linspace(0, 1, N)
    C = (x[:, None] - x[None, :]) ** 2
    C = C.astype(np.float64)

    print(f"\nProblem Setup:")
    print(f"  N = {N}")
    print(f"  ε = {epsilon}")
    print(f"  μ = N({center_mu}, {sigma}²)")
    print(f"  ν = N({center_nu}, {sigma}²)")

    # Solve with POT's Sinkhorn
    print(f"\nRunning POT Sinkhorn...")
    pi_pot = ot.sinkhorn(mu, nu, C, reg=epsilon, numItermax=5000, stopThr=1e-9)
    cost_pot = np.sum(pi_pot * C)
    sinkhorn_dist_pot = ot.sinkhorn2(mu, nu, C, reg=epsilon, numItermax=5000, stopThr=1e-9)

    # Marginal errors
    row_error = np.sum(np.abs(pi_pot.sum(axis=1) - mu))
    col_error = np.sum(np.abs(pi_pot.sum(axis=0) - nu))

    print(f"\nPOT Results:")
    print(f"  Transport cost (Σ C·π):    {cost_pot:.8f}")
    print(f"  Sinkhorn distance:         {float(sinkhorn_dist_pot):.8f}")
    print(f"  Row marginal L1 error:     {row_error:.2e}")
    print(f"  Col marginal L1 error:     {col_error:.2e}")

    # Save reference values for comparison
    print(f"\n--- Reference Values for C++ Comparison ---")
    print(f"  Expected transport cost ≈ {cost_pot:.8f}")
    print(f"  Acceptable tolerance:      1e-4 (relative)")

    # Also try different epsilon values
    print(f"\n--- ε Sweep (for Test 5 validation) ---")
    print(f"  {'ε':>8s}  {'Cost':>12s}  {'Sinkhorn Dist':>14s}")
    for eps in [1.0, 0.1, 0.01, 0.005]:
        sd = float(ot.sinkhorn2(mu, nu, C, reg=eps, numItermax=5000))
        pi = ot.sinkhorn(mu, nu, C, reg=eps, numItermax=5000)
        cost = float(np.sum(pi * C))
        print(f"  {eps:>8.3f}  {cost:>12.6f}  {sd:>14.6f}")

    # Save transport plan for detailed comparison
    np.savetxt("scripts/reference_transport_plan.csv", pi_pot, delimiter=",",
               header=f"Transport plan π* computed by POT (N={N}, ε={epsilon})")
    print(f"\n  Saved reference transport plan to scripts/reference_transport_plan.csv")
    print(f"\nDone!")


if __name__ == "__main__":
    main()
