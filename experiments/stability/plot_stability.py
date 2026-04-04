#!/usr/bin/env python3
"""
Plot numerical stability figures.

Generates:
  - paper/figures/fig_stability.pdf  (2-panel: cost vs eps + status heatmap)

Usage:
    python plot_stability.py [--data-dir experiments/data] [--output-dir paper]
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


NEURIPS_RC = {
    "figure.figsize": (5.5, 2.5),
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "lines.linewidth": 1.5,
    "lines.markersize": 5,
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

    # Load epsilon sweep data
    eps_path = os.path.join(args.data_dir, "stability_epsilon.csv")
    cost_path = os.path.join(args.data_dir, "stability_largecost.csv")

    if not os.path.exists(eps_path):
        print(f"Error: {eps_path} not found. Run generate_mock_data.py first.")
        return

    df_eps = pd.read_csv(eps_path)
    df_log = df_eps[df_eps["method"] == "log_domain"].copy()
    df_std = df_eps[df_eps["method"] == "standard_domain"].copy()

    plt.rcParams.update(NEURIPS_RC)
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.5))

    # (a) Transport cost vs epsilon
    ax = axes[0]

    # Log-domain: all valid
    eps_log = df_log["epsilon"].values
    costs_log = pd.to_numeric(df_log["transport_cost"], errors="coerce").values
    valid_log = ~np.isnan(costs_log)
    ax.plot(eps_log[valid_log], costs_log[valid_log], "o-", color="#E63946",
            label="Log-domain (ours)", markersize=4)

    # Standard-domain: some NaN
    eps_std = df_std["epsilon"].values
    costs_std = pd.to_numeric(df_std["transport_cost"], errors="coerce").values
    valid_std = ~np.isnan(costs_std)
    if np.any(valid_std):
        ax.plot(eps_std[valid_std], costs_std[valid_std], "s--", color="#457B9D",
                label="Standard domain", markersize=4)
    # Mark NaN points
    nan_std = np.isnan(costs_std)
    if np.any(nan_std):
        ax.scatter(eps_std[nan_std],
                   np.full(np.sum(nan_std), ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 0.1),
                   marker="x", color="red", s=30, zorder=5, label="NaN (failure)")

    ax.set_xlabel("$\\varepsilon$")
    ax.set_ylabel("Transport cost")
    ax.set_xscale("log")
    ax.set_title("(a) Cost vs. regularization")
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)

    # (b) Status heatmap for large cost experiments
    ax = axes[1]
    if os.path.exists(cost_path):
        df_cost = pd.read_csv(cost_path)
        # Build status grid
        max_costs = sorted(df_cost["max_cost"].unique())
        epsilons = sorted(df_cost["epsilon"].unique())

        # Focus on log-domain
        status_grid = np.zeros((len(max_costs), len(epsilons), 2))  # [log, std]
        for mi, mc in enumerate(max_costs):
            for ei, ep in enumerate(epsilons):
                for method_idx, method in enumerate(["log_domain", "standard_domain"]):
                    row = df_cost[(df_cost["max_cost"] == mc) &
                                  (df_cost["epsilon"] == ep) &
                                  (df_cost["method"] == method)]
                    if len(row) > 0:
                        status_grid[mi, ei, method_idx] = int(row["converged"].iloc[0])

        # Show standard domain status as heatmap
        cmap = ListedColormap(["#E63946", "#2A9D8F"])  # red=fail, green=converge
        im = ax.imshow(status_grid[:, :, 1], cmap=cmap, aspect="auto",
                       vmin=0, vmax=1)
        ax.set_xticks(range(len(epsilons)))
        ax.set_xticklabels([f"{e}" for e in epsilons], fontsize=6)
        ax.set_yticks(range(len(max_costs)))
        ax.set_yticklabels([str(int(mc)) for mc in max_costs])
        ax.set_xlabel("$\\varepsilon$")
        ax.set_ylabel("$\\max(C)$")
        ax.set_title("(b) Std-domain convergence")

        # Add text annotations
        for mi in range(len(max_costs)):
            for ei in range(len(epsilons)):
                val = status_grid[mi, ei, 1]
                text = "OK" if val == 1 else "NaN"
                color = "white" if val == 0 else "white"
                ax.text(ei, mi, text, ha="center", va="center",
                       fontsize=6, color=color, fontweight="bold")
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("(b) Std-domain convergence")

    plt.tight_layout()
    output_path = os.path.join(fig_dir, "fig_stability.pdf")
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
