#!/bin/bash
# ============================================================================
# run_all.sh — Master script to run all experiments and generate figures
# ============================================================================
#
# Prerequisites:
#   1. Build the project: mkdir build && cd build && cmake .. && make -j
#   2. Install Python deps: pip install numpy pandas matplotlib seaborn pot geomloss torch
#
# Usage:
#   cd Fast-Sinkhorn-CUDA
#   bash experiments/run_all.sh [build_dir]
#
# ============================================================================

set -e

BUILD_DIR="${1:-build}"
DATA_DIR="experiments/data"
PAPER_DIR="paper"

echo "============================================"
echo "  Fast-Sinkhorn-CUDA: Full Experiment Suite"
echo "============================================"
echo ""

# Record environment info
echo "=== Environment ===" | tee "$DATA_DIR/environment.txt"
date | tee -a "$DATA_DIR/environment.txt"
nvidia-smi | tee -a "$DATA_DIR/environment.txt"
nvcc --version | tee -a "$DATA_DIR/environment.txt"
echo "" | tee -a "$DATA_DIR/environment.txt"

mkdir -p "$DATA_DIR"

# ============================================================================
# 1. Baseline comparisons
# ============================================================================
echo ""
echo "[1/7] Running baseline: ours (C++)"
"$BUILD_DIR/bench_ours"

echo ""
echo "[2/7] Running baseline: POT (CPU)"
python experiments/baselines/bench_pot.py --output-dir "$DATA_DIR"

echo ""
echo "[3/7] Running baseline: GeomLoss (GPU)"
python experiments/baselines/bench_geomloss.py --output-dir "$DATA_DIR" || echo "  (GeomLoss not installed, skipping)"

echo ""
echo "[4/7] Running baseline: PyTorch Sinkhorn (GPU)"
python experiments/baselines/bench_pytorch_sinkhorn.py --output-dir "$DATA_DIR" || echo "  (PyTorch not available, skipping)"

# ============================================================================
# 2. Scaling experiment
# ============================================================================
echo ""
echo "[5/7] Running scaling experiment"
"$BUILD_DIR/scaling_experiment"

# ============================================================================
# 3. Ablation studies
# ============================================================================
echo ""
echo "[6/7] Running ablation studies"
"$BUILD_DIR/ablation_warp_shuffle"
"$BUILD_DIR/ablation_log_domain"
"$BUILD_DIR/ablation_block_size"
"$BUILD_DIR/ablation_check_interval"

# Merge all ablation CSVs into one
echo "Merging ablation results..."
python3 -c "
import pandas as pd, glob, os
dfs = []
for f in glob.glob('$DATA_DIR/ablation_*.csv'):
    dfs.append(pd.read_csv(f))
if dfs:
    merged = pd.concat(dfs, ignore_index=True)
    merged.to_csv('$DATA_DIR/ablation_results.csv', index=False)
    print(f'  Merged {len(dfs)} files -> ablation_results.csv')
"

# ============================================================================
# 4. Stability experiment
# ============================================================================
echo ""
echo "[7/7] Running stability experiment"
"$BUILD_DIR/stability_epsilon"

# ============================================================================
# 5. Convergence experiment
# ============================================================================
echo ""
echo "[8/7] Running convergence experiment"
"$BUILD_DIR/convergence_profile"

# ============================================================================
# 6. Generate all figures
# ============================================================================
echo ""
echo "=== Generating figures ==="
python experiments/baselines/plot_baselines.py --data-dir "$DATA_DIR" --output-dir "$PAPER_DIR"
python experiments/scaling/plot_scaling.py --data-dir "$DATA_DIR" --output-dir "$PAPER_DIR"
python experiments/ablation/plot_ablation.py --data-dir "$DATA_DIR" --output-dir "$PAPER_DIR"
python experiments/stability/plot_stability.py --data-dir "$DATA_DIR" --output-dir "$PAPER_DIR"
python experiments/convergence/plot_convergence.py --data-dir "$DATA_DIR" --output-dir "$PAPER_DIR"

echo ""
echo "============================================"
echo "  All experiments complete!"
echo "  Data:    $DATA_DIR/"
echo "  Figures: $PAPER_DIR/figures/"
echo "  Tables:  $PAPER_DIR/tables/"
echo "============================================"
