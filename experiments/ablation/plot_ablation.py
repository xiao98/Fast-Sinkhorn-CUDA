#!/usr/bin/env python3
"""
Plot ablation study figures and generate LaTeX table.

Generates:
  - paper/figures/fig_ablation.pdf    (grouped bar chart)
  - paper/tables/tab_ablation.tex     (LaTeX table fragment)

Usage:
    python plot_ablation.py [--data-dir experiments/data] [--output-dir paper]
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
    "axes.titlesize": 9,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "font.family": "serif",
    "text.usetex": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
}

ABLATION_DISPLAY = {
    "full_system": "Full system (ours)",
    "shared_mem_only": "Shared-mem only\n(no warp shuffle)",
    "standard_domain": "Standard domain\n(no log-LSE)",
    "block_size_64": "Block size = 64",
    "block_size_256": "Block size = 256",
    "block_size_512": "Block size = 512",
    "check_interval_1": "Check every iter",
    "check_interval_10": "Check every 10",
    "check_interval_50": "Check every 50",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="experiments/data")
    parser.add_argument("--output-dir", default="paper")
    args = parser.parse_args()

    fig_dir = os.path.join(args.output_dir, "figures")
    tab_dir = os.path.join(args.output_dir, "tables")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tab_dir, exist_ok=True)

    filepath = os.path.join(args.data_dir, "ablation_results.csv")
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found. Run generate_mock_data.py first.")
        return

    df = pd.read_csv(filepath)

    # --- Bar chart: key ablations ---
    key_configs = ["full_system", "shared_mem_only", "standard_domain",
                   "block_size_64", "check_interval_1"]
    df_key = df[df["config"].isin(key_configs)].copy()
    df_key = df_key.set_index("config").loc[key_configs].reset_index()

    base_time = float(df[df["config"] == "full_system"]["time_ms_mean"].iloc[0])

    plt.rcParams.update(NEURIPS_RC)
    fig, ax = plt.subplots(figsize=(5.5, 2.5))

    labels = [ABLATION_DISPLAY.get(c, c) for c in key_configs]
    times = df_key["time_ms_mean"].astype(float).values
    slowdowns = times / base_time

    colors = ["#E63946" if c == "full_system" else "#457B9D" for c in key_configs]

    bars = ax.barh(range(len(labels)), times, color=colors, edgecolor="white", height=0.6)

    # Add slowdown annotations
    for i, (t, s) in enumerate(zip(times, slowdowns)):
        label = f"{t:.1f} ms" if s == 1.0 else f"{t:.1f} ms ({s:.2f}$\\times$)"
        ax.text(t + 0.5, i, label, va="center", fontsize=6.5)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.set_xlabel("Time (ms)")
    ax.set_title("Ablation study ($n = 2048$, $\\varepsilon = 0.01$)")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    output_path = os.path.join(fig_dir, "fig_ablation.pdf")
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")

    # --- LaTeX table ---
    table_configs = ["full_system", "shared_mem_only", "standard_domain",
                     "block_size_64", "block_size_512", "check_interval_1"]
    lines = []
    lines.append(r"\begin{tabular}{l r r}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Configuration} & \textbf{Time (ms)} & \textbf{Slowdown} \\")
    lines.append(r"\midrule")

    for config in table_configs:
        row = df[df["config"] == config]
        if len(row) > 0:
            t = float(row["time_ms_mean"].iloc[0])
            slowdown = t / base_time
            display = ABLATION_DISPLAY.get(config, config).replace("\n", " ")
            lines.append(f"{display} & {t:.1f} & {slowdown:.2f}$\\times$ \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    tab_path = os.path.join(tab_dir, "tab_ablation.tex")
    with open(tab_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved: {tab_path}")


if __name__ == "__main__":
    main()
