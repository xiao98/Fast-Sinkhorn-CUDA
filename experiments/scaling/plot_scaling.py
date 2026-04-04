#!/usr/bin/env python3
"""
Plot scaling experiment figures.

Generates:
  - paper/figures/fig_scaling.pdf  (3-panel: time, memory, iterations vs N)

Usage:
    python plot_scaling.py [--data-dir experiments/data] [--output-dir paper]
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


NEURIPS_RC = {
    "figure.figsize": (5.5, 2.0),
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 6.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "lines.linewidth": 1.5,
    "lines.markersize": 4,
    "font.family": "serif",
    "text.usetex": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="experiments/data")
    parser.add_argument("--output-dir", default="paper")
    args = parser.parse_args()

    fig_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    filepath = os.path.join(args.data_dir, "scaling_ours.csv")
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found. Run generate_mock_data.py first.")
        return

    df = pd.read_csv(filepath)
    df = df.sort_values("n")
    sizes = df["n"].values
    times = df["time_ms_mean"].astype(float).values
    memory = df["peak_memory_bytes"].astype(float).values / (1024 * 1024)  # MB
    iters = df["iterations"].astype(int).values

    plt.rcParams.update(NEURIPS_RC)
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.0))

    # (a) Time vs N
    ax = axes[0]
    ax.plot(sizes, times, "o-", color="#E63946", markersize=4)
    # O(n^2) reference
    ref_times = times[2] * (sizes / sizes[2]) ** 2
    ax.plot(sizes, ref_times, "--", color="gray", alpha=0.5, label="$O(n^2)$")
    ax.set_xlabel("$n$")
    ax.set_ylabel("Time (ms)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_title("(a) Computation time")
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)

    # (b) Memory vs N
    ax = axes[1]
    ax.plot(sizes, memory, "s-", color="#2A9D8F", markersize=4)
    ref_mem = 4 * sizes ** 2 / (1024 * 1024)
    ax.plot(sizes, ref_mem, "--", color="gray", alpha=0.5, label="$4n^2$ bytes")
    ax.set_xlabel("$n$")
    ax.set_ylabel("Memory (MB)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_title("(b) Peak GPU memory")
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)

    # (c) Iterations vs N
    ax = axes[2]
    ax.plot(sizes, iters, "^-", color="#457B9D", markersize=4)
    ax.axhline(y=np.mean(iters), color="gray", linestyle="--", alpha=0.5,
               label=f"Mean = {np.mean(iters):.0f}")
    ax.set_xlabel("$n$")
    ax.set_ylabel("Iterations")
    ax.set_xscale("log", base=2)
    ax.set_title("(c) Convergence iterations")
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(fig_dir, "fig_scaling.pdf")
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
