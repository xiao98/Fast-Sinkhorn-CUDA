#!/usr/bin/env python3
"""
Plot convergence profile figures.

Generates:
  - paper/figures/fig_convergence.pdf  (semi-log plot of error vs iteration)

Usage:
    python plot_convergence.py [--data-dir experiments/data] [--output-dir paper]
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


NEURIPS_RC = {
    "figure.figsize": (5.5, 2.8),
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 6.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "lines.linewidth": 1.2,
    "font.family": "serif",
    "text.usetex": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
}

COLORS = {
    0.1: "#E63946",
    0.01: "#457B9D",
    0.001: "#2A9D8F",
}

LINESTYLES = {
    256: "-",
    1024: "--",
    4096: ":",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="experiments/data")
    parser.add_argument("--output-dir", default="paper")
    args = parser.parse_args()

    fig_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    filepath = os.path.join(args.data_dir, "convergence_profile.csv")
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found. Run generate_mock_data.py first.")
        return

    df = pd.read_csv(filepath)

    plt.rcParams.update(NEURIPS_RC)
    fig, ax = plt.subplots(figsize=(5.5, 3.0))

    epsilons = sorted(df["epsilon"].unique())
    sizes = sorted(df["n"].unique())

    for eps in epsilons:
        for n in sizes:
            subset = df[(df["epsilon"] == eps) & (df["n"] == n)].copy()
            if len(subset) == 0:
                continue
            subset = subset.sort_values("iteration")
            iters = subset["iteration"].values
            errors = pd.to_numeric(subset["marginal_error"], errors="coerce").values

            color = COLORS.get(eps, "gray")
            ls = LINESTYLES.get(n, "-")
            label = f"$\\varepsilon={eps}$, $n={n}$"
            ax.plot(iters, errors, color=color, linestyle=ls, label=label)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Marginal error $\\|\\pi\\mathbf{1} - \\mu\\|_1$")
    ax.set_yscale("log")
    ax.set_title("Convergence profiles")
    ax.axhline(y=1e-6, color="gray", linestyle="-.", alpha=0.4, linewidth=0.8)
    ax.text(ax.get_xlim()[1] * 0.7, 1.5e-6, "threshold $= 10^{-6}$",
            fontsize=6, color="gray")
    ax.legend(ncol=3, loc="upper right", fontsize=5.5, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    output_path = os.path.join(fig_dir, "fig_convergence.pdf")
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
