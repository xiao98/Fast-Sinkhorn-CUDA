#!/usr/bin/env python3
"""
Application: 3D point cloud matching via Optimal Transport.

Generates two 3D point clouds (one is a rotated/translated version of the other)
and computes OT correspondences between them.

Generates:
  - paper/figures/fig_point_cloud.pdf

Usage:
    python point_cloud_matching.py [--output-dir paper/figures]
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


NEURIPS_RC = {
    "figure.figsize": (5.5, 2.8),
    "font.size": 8,
    "axes.labelsize": 7,
    "axes.titlesize": 8,
    "font.family": "serif",
    "text.usetex": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
}


def generate_bunny_like_points(n_points, seed=42):
    """Generate a synthetic 3D point cloud resembling a simple shape."""
    rng = np.random.RandomState(seed)

    # Create a sphere with some deformations
    phi = rng.uniform(0, 2 * np.pi, n_points)
    costheta = rng.uniform(-1, 1, n_points)
    theta = np.arccos(costheta)

    r = 1.0 + 0.3 * np.sin(3 * theta) * np.cos(2 * phi)  # Deformed sphere

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    points = np.stack([x, y, z], axis=1).astype(np.float64)
    return points


def rotation_matrix(angle_x, angle_y, angle_z):
    """Create a 3D rotation matrix from Euler angles (in radians)."""
    cx, sx = np.cos(angle_x), np.sin(angle_x)
    cy, sy = np.cos(angle_y), np.sin(angle_y)
    cz, sz = np.cos(angle_z), np.sin(angle_z)

    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])

    return Rz @ Ry @ Rx


def compute_ot_matching(source, target, epsilon=0.05):
    """Compute OT matching between two point clouds.

    Returns the transport plan matrix.
    """
    n = len(source)
    m = len(target)
    mu = np.ones(n) / n
    nu = np.ones(m) / m

    # Squared Euclidean cost matrix
    C = np.sum((source[:, None, :] - target[None, :, :]) ** 2, axis=2)

    try:
        import ot
        P = ot.sinkhorn(mu, nu, C, reg=epsilon, numItermax=2000)
    except ImportError:
        print("Warning: POT not installed. Using random matching as placeholder.")
        P = np.outer(mu, nu)  # Independent coupling as fallback

    return P


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="paper/figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    n_points = 200  # Keep small for visualization

    print("Generating point clouds...")
    source = generate_bunny_like_points(n_points, seed=42)

    # Create target by rotation + translation
    R = rotation_matrix(0.3, 0.5, 0.2)
    t = np.array([0.5, -0.3, 0.2])
    target = (source @ R.T) + t

    # Add some noise to target
    rng = np.random.RandomState(43)
    target += rng.normal(0, 0.05, target.shape)

    print("Computing OT matching...")
    P = compute_ot_matching(source, target, epsilon=0.05)

    # Find best matching: for each source point, find the target with max transport
    matching = P.argmax(axis=1)

    # Plot
    plt.rcParams.update(NEURIPS_RC)
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.8),
                              subplot_kw={"projection": "3d"})

    # (a) Point clouds
    ax = axes[0]
    ax.scatter(source[:, 0], source[:, 1], source[:, 2],
               c="#E63946", s=8, alpha=0.7, label="Source")
    ax.scatter(target[:, 0], target[:, 1], target[:, 2],
               c="#457B9D", s=8, alpha=0.7, label="Target")
    ax.set_title("(a) Input point clouds")
    ax.legend(fontsize=5, loc="upper left")
    ax.tick_params(labelsize=5)

    # (b) OT correspondences
    ax = axes[1]
    ax.scatter(source[:, 0], source[:, 1], source[:, 2],
               c="#E63946", s=8, alpha=0.7)
    ax.scatter(target[:, 0], target[:, 1], target[:, 2],
               c="#457B9D", s=8, alpha=0.7)

    # Draw matching lines (subsample for clarity)
    n_lines = min(50, n_points)
    line_indices = np.linspace(0, n_points - 1, n_lines, dtype=int)
    for idx in line_indices:
        j = matching[idx]
        ax.plot([source[idx, 0], target[j, 0]],
                [source[idx, 1], target[j, 1]],
                [source[idx, 2], target[j, 2]],
                color="#2A9D8F", alpha=0.3, linewidth=0.5)

    ax.set_title("(b) OT correspondences")
    ax.tick_params(labelsize=5)

    plt.tight_layout()
    output_path = os.path.join(args.output_dir, "fig_point_cloud.pdf")
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
