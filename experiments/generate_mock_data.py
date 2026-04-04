#!/usr/bin/env python3
"""
Generate mock experimental data for testing plotting scripts.
Run this on a machine without GPU to verify all plotting scripts work correctly.
The mock data mimics realistic scaling behavior and performance characteristics.

Usage:
    python generate_mock_data.py [--output-dir experiments/data]
"""

import argparse
import os
import numpy as np
import csv


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def write_csv(filepath, header, rows):
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"  Written: {filepath}")


def generate_baseline_data(output_dir):
    """Generate mock baseline comparison data."""
    np.random.seed(42)
    sizes = [256, 512, 1024, 2048, 4096, 8192]
    epsilons = [0.1, 0.01, 0.001]

    methods = {
        "baseline_ours.csv": {
            "base_time": 0.3,  # ms for N=256
            "scale_exp": 2.05,  # slightly super-quadratic due to memory
            "gpu": "NVIDIA RTX 3090",
        },
        "baseline_pot_cpu.csv": {
            "base_time": 15.0,
            "scale_exp": 2.8,
            "gpu": "N/A (CPU)",
        },
        "baseline_geomloss.csv": {
            "base_time": 1.5,
            "scale_exp": 2.1,
            "gpu": "NVIDIA RTX 3090",
        },
        "baseline_pytorch.csv": {
            "base_time": 2.0,
            "scale_exp": 2.15,
            "gpu": "NVIDIA RTX 3090",
        },
    }

    for filename, params in methods.items():
        header = ["n", "m", "epsilon", "time_ms_mean", "time_ms_std",
                  "iterations", "transport_cost", "converged", "gpu"]
        rows = []
        for n in sizes:
            for eps in epsilons:
                scale = (n / 256) ** params["scale_exp"]
                eps_factor = (0.01 / eps) ** 0.3  # smaller eps -> more iters -> slower
                base = params["base_time"] * scale * eps_factor
                time_mean = base * (1 + np.random.normal(0, 0.05))
                time_std = time_mean * 0.03
                iters = int(100 * (0.01 / eps) ** 0.5 * (1 + np.random.normal(0, 0.1)))
                cost = 0.04 + np.random.normal(0, 0.002)
                converged = 1
                rows.append([n, n, eps, f"{time_mean:.3f}", f"{time_std:.3f}",
                            iters, f"{cost:.6f}", converged, params["gpu"]])
        write_csv(os.path.join(output_dir, filename), header, rows)


def generate_scaling_data(output_dir):
    """Generate mock scaling experiment data."""
    np.random.seed(43)
    sizes = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 12288, 16384]
    header = ["n", "m", "epsilon", "time_ms_mean", "time_ms_std",
              "iterations", "peak_memory_bytes", "cost_matrix_mb",
              "transport_cost", "converged"]
    rows = []
    for n in sizes:
        scale = (n / 256) ** 2.05
        time_mean = 0.3 * scale * (1 + np.random.normal(0, 0.03))
        time_std = time_mean * 0.02
        iters = int(180 + np.random.normal(0, 10))
        mem = n * n * 4 + n * 5 * 4  # cost matrix + buffers
        cost_mb = n * n * 4 / (1024 * 1024)
        cost = 0.04 + np.random.normal(0, 0.001)
        rows.append([n, n, 0.01, f"{time_mean:.3f}", f"{time_std:.3f}",
                    iters, mem, f"{cost_mb:.2f}", f"{cost:.6f}", 1])
    write_csv(os.path.join(output_dir, "scaling_ours.csv"), header, rows)


def generate_ablation_data(output_dir):
    """Generate mock ablation study data."""
    np.random.seed(44)
    n = 2048
    eps = 0.01
    base_time = 15.0  # ms for full system

    configs = [
        ("full_system", base_time, 180),
        ("shared_mem_only", base_time * 1.85, 180),
        ("standard_domain", base_time * 1.1, 200),
        ("block_size_64", base_time * 1.45, 180),
        ("block_size_128", base_time * 1.15, 180),
        ("block_size_256", base_time, 180),
        ("block_size_512", base_time * 1.08, 180),
        ("check_interval_1", base_time * 1.65, 180),
        ("check_interval_5", base_time * 1.12, 180),
        ("check_interval_10", base_time, 180),
        ("check_interval_20", base_time * 0.97, 180),
        ("check_interval_50", base_time * 0.95, 185),
    ]

    header = ["config", "n", "epsilon", "time_ms_mean", "time_ms_std",
              "iterations", "converged", "notes"]
    rows = []
    for config_name, time_val, iters in configs:
        noise = time_val * np.random.normal(0, 0.02)
        rows.append([config_name, n, eps, f"{time_val + noise:.3f}",
                    f"{time_val * 0.03:.3f}", iters, 1, ""])
    write_csv(os.path.join(output_dir, "ablation_results.csv"), header, rows)


def generate_stability_data(output_dir):
    """Generate mock numerical stability data."""
    np.random.seed(45)
    epsilons = [1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
    n = 512

    # Log-domain results
    header = ["method", "n", "epsilon", "time_ms", "iterations",
              "transport_cost", "marginal_error", "converged", "has_nan"]
    rows = []

    for eps in epsilons:
        # Log-domain: always works
        iters = int(50 / eps ** 0.4)
        cost = 0.04 - 0.005 * np.log10(eps)  # cost increases as eps -> 0
        error = 1e-7 * (1 + np.random.exponential(0.5))
        time_ms = 2.0 * (iters / 100) * (1 + np.random.normal(0, 0.05))
        rows.append(["log_domain", n, eps, f"{time_ms:.3f}", iters,
                    f"{cost:.6f}", f"{error:.2e}", 1, 0])

        # Standard-domain: fails for small eps
        if eps >= 0.01:
            iters_std = int(iters * 1.1)
            cost_std = cost + np.random.normal(0, 0.001)
            error_std = 1e-6 * (1 + np.random.exponential(1.0))
            time_std = time_ms * 1.05
            rows.append(["standard_domain", n, eps, f"{time_std:.3f}", iters_std,
                        f"{cost_std:.6f}", f"{error_std:.2e}", 1, 0])
        elif eps >= 0.005:
            rows.append(["standard_domain", n, eps, "5.0", 500,
                        "nan", "nan", 0, 1])
        else:
            rows.append(["standard_domain", n, eps, "nan", 0,
                        "nan", "nan", 0, 1])

    write_csv(os.path.join(output_dir, "stability_epsilon.csv"), header, rows)

    # Large cost stability
    max_costs = [1, 10, 100, 1000]
    header2 = ["method", "n", "epsilon", "max_cost", "converged", "has_nan",
               "transport_cost", "marginal_error"]
    rows2 = []
    for mc in max_costs:
        for eps in [0.1, 0.01, 0.001]:
            # Log-domain: robust
            cost = mc * 0.04
            rows2.append(["log_domain", n, eps, mc, 1, 0,
                         f"{cost:.4f}", "1e-7"])
            # Standard: fails for large cost + small eps
            if mc * 1.0 / eps > 80:
                rows2.append(["standard_domain", n, eps, mc, 0, 1, "nan", "nan"])
            else:
                rows2.append(["standard_domain", n, eps, mc, 1, 0,
                             f"{cost + 0.01:.4f}", "1e-5"])
    write_csv(os.path.join(output_dir, "stability_largecost.csv"), header2, rows2)


def generate_convergence_data(output_dir):
    """Generate mock convergence profile data."""
    np.random.seed(46)
    configs = [
        (256, 0.1), (256, 0.01), (256, 0.001),
        (1024, 0.1), (1024, 0.01), (1024, 0.001),
        (4096, 0.1), (4096, 0.01), (4096, 0.001),
    ]

    header = ["n", "epsilon", "iteration", "marginal_error"]
    rows = []
    for n, eps in configs:
        # Linear convergence rate on semi-log: error = err0 * lambda^k
        lam = np.exp(-2.0 / eps)  # contraction rate (cost range ~ 1)
        err0 = 1.0
        max_iter = min(int(2000 * (0.01 / eps) ** 0.5), 5000)
        check_interval = 10
        for k in range(0, max_iter, check_interval):
            err = err0 * (lam ** k)
            # Add some noise
            err *= (1 + np.random.normal(0, 0.05))
            err = max(err, 1e-10)
            rows.append([n, eps, k, f"{err:.2e}"])
            if err < 1e-9:
                break

    write_csv(os.path.join(output_dir, "convergence_profile.csv"), header, rows)


def main():
    parser = argparse.ArgumentParser(description="Generate mock data for plotting")
    parser.add_argument("--output-dir", default="experiments/data",
                        help="Output directory for CSV files")
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    print("Generating mock experimental data...")
    print()

    print("[1/5] Baseline comparisons:")
    generate_baseline_data(args.output_dir)
    print()

    print("[2/5] Scaling experiments:")
    generate_scaling_data(args.output_dir)
    print()

    print("[3/5] Ablation study:")
    generate_ablation_data(args.output_dir)
    print()

    print("[4/5] Numerical stability:")
    generate_stability_data(args.output_dir)
    print()

    print("[5/5] Convergence profiles:")
    generate_convergence_data(args.output_dir)
    print()

    print("Done! All mock data generated in:", args.output_dir)


if __name__ == "__main__":
    main()
