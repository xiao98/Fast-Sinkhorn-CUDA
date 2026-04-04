#!/usr/bin/env python3
"""
Plot baseline comparison figures and generate LaTeX table.

Generates:
  - paper/figures/fig_baseline_bars.pdf    (grouped bar chart)
  - paper/figures/fig_baseline_scaling.pdf  (log-log scaling plot)
  - paper/tables/tab_baseline.tex          (LaTeX table fragment)

Usage:
    python plot_baselines.py [--data-dir experiments/data] [--output-dir paper]
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# NeurIPS figure style
NEURIPS_RC = {
    "figure.figsize": (5.5, 3.0),
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
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

METHOD_LABELS = {
    "baseline_ours.csv": "Ours (CUDA)",
    "baseline_pot_cpu.csv": "POT (CPU)",
    "baseline_geomloss.csv": "GeomLoss (GPU)",
    "baseline_pytorch.csv": "PyTorch Sinkhorn (GPU)",
}

METHOD_COLORS = {
    "Ours (CUDA)": "#E63946",
    "POT (CPU)": "#457B9D",
    "GeomLoss (GPU)": "#2A9D8F",
    "PyTorch Sinkhorn (GPU)": "#E9C46A",
}

METHOD_MARKERS = {
    "Ours (CUDA)": "o",
    "POT (CPU)": "s",
    "GeomLoss (GPU)": "^",
    "PyTorch Sinkhorn (GPU)": "D",
}


def load_data(data_dir):
    """Load all baseline CSV files into a dict of DataFrames."""
    data = {}
    for filename, label in METHOD_LABELS.items():
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            data[label] = df
        else:
            print(f"  Warning: {filepath} not found, skipping {label}")
    return data


def plot_bars(data, output_path, eps_filter=0.01):
    """Grouped bar chart of time vs N for a fixed epsilon."""
    plt.rcParams.update(NEURIPS_RC)
    fig, ax = plt.subplots()

    methods = list(data.keys())
    sizes = sorted(data[methods[0]]["n"].unique())

    x = np.arange(len(sizes))
    width = 0.8 / len(methods)

    for i, method in enumerate(methods):
        df = data[method]
        df_eps = df[np.isclose(df["epsilon"], eps_filter)]
        times = []
        for n in sizes:
            row = df_eps[df_eps["n"] == n]
            times.append(float(row["time_ms_mean"].iloc[0]) if len(row) > 0 else 0)

        offset = (i - len(methods) / 2 + 0.5) * width
        bars = ax.bar(x + offset, times, width * 0.9,
                      label=method, color=METHOD_COLORS[method],
                      edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Problem size $n = m$")
    ax.set_ylabel("Time (ms)")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_title(f"Baseline comparison ($\\varepsilon = {eps_filter}$)")
    ax.grid(axis="y", alpha=0.3)

    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_scaling(data, output_path, eps_filter=0.01):
    """Log-log scaling plot."""
    plt.rcParams.update(NEURIPS_RC)
    fig, ax = plt.subplots()

    for method, df in data.items():
        df_eps = df[np.isclose(df["epsilon"], eps_filter)]
        df_eps = df_eps.sort_values("n")
        sizes = df_eps["n"].values
        times = df_eps["time_ms_mean"].astype(float).values

        ax.plot(sizes, times, marker=METHOD_MARKERS[method],
                color=METHOD_COLORS[method], label=method)

    # Reference O(n^2) line
    sizes_ref = np.array([256, 8192])
    t_ref = 0.3 * (sizes_ref / 256) ** 2
    ax.plot(sizes_ref, t_ref, "--", color="gray", alpha=0.5, label="$O(n^2)$ reference")

    ax.set_xlabel("Problem size $n = m$")
    ax.set_ylabel("Time (ms)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_title(f"Scaling comparison ($\\varepsilon = {eps_filter}$)")
    ax.grid(True, alpha=0.3)

    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def generate_table(data, output_path, eps_filter=0.01):
    """Generate LaTeX table fragment."""
    methods = list(data.keys())
    sizes = sorted(data[methods[0]]["n"].unique())
    # Filter to common sizes for table
    table_sizes = [s for s in sizes if s in [256, 512, 1024, 2048, 4096]]

    lines = []
    lines.append(r"\begin{tabular}{l " + " ".join(["r"] * len(table_sizes)) + "}")
    lines.append(r"\toprule")
    header = r"\textbf{Method} & " + " & ".join([f"$n{{=}}{s}$" for s in table_sizes]) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    ours_times = {}
    for method in methods:
        df = data[method]
        df_eps = df[np.isclose(df["epsilon"], eps_filter)]
        vals = []
        for s in table_sizes:
            row = df_eps[df_eps["n"] == s]
            if len(row) > 0:
                t = float(row["time_ms_mean"].iloc[0])
                vals.append(f"{t:.1f}")
                if method == "Ours (CUDA)":
                    ours_times[s] = t
            else:
                vals.append("--")
        lines.append(f"{method} & " + " & ".join(vals) + r" \\")

    # Speedup row
    lines.append(r"\midrule")
    pot_label = "POT (CPU)"
    if pot_label in data:
        df_pot = data[pot_label]
        df_pot_eps = df_pot[np.isclose(df_pot["epsilon"], eps_filter)]
        speedups = []
        for s in table_sizes:
            row = df_pot_eps[df_pot_eps["n"] == s]
            if len(row) > 0 and s in ours_times:
                speedup = float(row["time_ms_mean"].iloc[0]) / ours_times[s]
                speedups.append(f"{speedup:.0f}$\\times$")
            else:
                speedups.append("--")
        lines.append(r"\textbf{Speedup vs.\ POT} & " + " & ".join(speedups) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="experiments/data")
    parser.add_argument("--output-dir", default="paper")
    args = parser.parse_args()

    fig_dir = os.path.join(args.output_dir, "figures")
    tab_dir = os.path.join(args.output_dir, "tables")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tab_dir, exist_ok=True)

    print("Loading baseline data...")
    data = load_data(args.data_dir)
    if not data:
        print("No data found. Run generate_mock_data.py first.")
        return

    print("Generating figures...")
    plot_bars(data, os.path.join(fig_dir, "fig_baseline_bars.pdf"))
    plot_scaling(data, os.path.join(fig_dir, "fig_baseline_scaling.pdf"))
    generate_table(data, os.path.join(tab_dir, "tab_baseline.tex"))


if __name__ == "__main__":
    main()
