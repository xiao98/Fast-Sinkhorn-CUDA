#!/usr/bin/env python3
"""
Application: Image color transfer via Optimal Transport.

Transfers the color palette of a target image onto a source image using
the OT plan computed by our Sinkhorn solver (via subprocess) or POT as fallback.

Generates:
  - paper/figures/fig_color_transfer.pdf

Usage:
    python color_transfer.py [--output-dir paper/figures] [--solver-binary build/bench_ours]
"""

import argparse
import os
import subprocess
import tempfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


NEURIPS_RC = {
    "figure.figsize": (5.5, 2.5),
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "font.family": "serif",
    "text.usetex": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
}


def generate_synthetic_image(shape, palette, seed=42):
    """Generate a synthetic image with a given color palette.

    Creates a smooth gradient image in the given palette.
    """
    rng = np.random.RandomState(seed)
    h, w = shape[:2]

    # Create smooth spatial patterns
    x = np.linspace(0, 2 * np.pi, w)
    y = np.linspace(0, 2 * np.pi, h)
    xx, yy = np.meshgrid(x, y)

    # Generate mixing weights from spatial patterns
    w1 = (np.sin(xx) * np.cos(yy) + 1) / 2
    w2 = (np.cos(xx * 0.7 + 1) * np.sin(yy * 1.3) + 1) / 2
    w3 = 1.0 - 0.5 * w1 - 0.5 * w2
    w3 = np.clip(w3, 0, 1)

    # Normalize weights
    total = w1 + w2 + w3
    w1, w2, w3 = w1 / total, w2 / total, w3 / total

    # Mix palette colors
    img = np.zeros((h, w, 3))
    for c in range(3):
        img[:, :, c] = (w1 * palette[0][c] + w2 * palette[1][c] + w3 * palette[2][c])

    # Add some noise
    img += rng.normal(0, 0.02, img.shape)
    img = np.clip(img, 0, 1)

    return img.astype(np.float32)


def sample_pixels(image, n_samples, seed=42):
    """Randomly sample pixel RGB values from an image."""
    rng = np.random.RandomState(seed)
    h, w = image.shape[:2]
    indices = rng.choice(h * w, size=n_samples, replace=False)
    pixels = image.reshape(-1, 3)[indices]
    return pixels


def ot_color_transfer(source_img, target_img, n_samples=512, epsilon=0.01):
    """Transfer colors from target to source using OT.

    Uses POT library as the solver (since this runs without GPU).
    """
    try:
        import ot
    except ImportError:
        print("Warning: POT not installed. Using simple histogram matching instead.")
        return simple_color_transfer(source_img, target_img)

    # Sample pixels from both images
    source_pixels = sample_pixels(source_img, n_samples, seed=42)
    target_pixels = sample_pixels(target_img, n_samples, seed=43)

    # Uniform distributions
    mu = np.ones(n_samples, dtype=np.float64) / n_samples
    nu = np.ones(n_samples, dtype=np.float64) / n_samples

    # Cost matrix: squared Euclidean distance in RGB space
    C = np.sum((source_pixels[:, None, :].astype(np.float64) -
                target_pixels[None, :, :].astype(np.float64)) ** 2, axis=2)

    # Compute OT plan
    P = ot.sinkhorn(mu, nu, C, reg=epsilon, numItermax=2000)

    # Apply barycentric mapping: for each source pixel, compute weighted average
    # of target pixels according to the transport plan
    # Normalize transport plan rows
    P_normalized = P / (P.sum(axis=1, keepdims=True) + 1e-10)
    transferred_samples = P_normalized @ target_pixels

    # Apply transfer to full image using nearest-neighbor in source sample space
    h, w = source_img.shape[:2]
    all_pixels = source_img.reshape(-1, 3).astype(np.float64)

    # Find nearest source sample for each pixel
    from scipy.spatial import cKDTree
    tree = cKDTree(source_pixels)
    _, nearest_idx = tree.query(all_pixels)

    # Map each pixel to its corresponding transferred color
    transferred_all = transferred_samples[nearest_idx]
    transferred_img = transferred_all.reshape(h, w, 3).astype(np.float32)
    transferred_img = np.clip(transferred_img, 0, 1)

    return transferred_img


def simple_color_transfer(source_img, target_img):
    """Simple mean/std color transfer as fallback."""
    result = source_img.copy()
    for c in range(3):
        s_mean, s_std = source_img[:, :, c].mean(), source_img[:, :, c].std()
        t_mean, t_std = target_img[:, :, c].mean(), target_img[:, :, c].std()
        if s_std > 0:
            result[:, :, c] = (source_img[:, :, c] - s_mean) * (t_std / s_std) + t_mean
    return np.clip(result, 0, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="paper/figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Generating synthetic images...")

    # Warm palette (source)
    warm_palette = [
        [0.9, 0.4, 0.1],  # Orange
        [0.8, 0.2, 0.2],  # Red
        [1.0, 0.8, 0.3],  # Yellow
    ]
    source_img = generate_synthetic_image((128, 192), warm_palette, seed=42)

    # Cool palette (target)
    cool_palette = [
        [0.1, 0.4, 0.8],  # Blue
        [0.2, 0.7, 0.6],  # Teal
        [0.5, 0.3, 0.7],  # Purple
    ]
    target_img = generate_synthetic_image((128, 192), cool_palette, seed=43)

    print("Computing OT color transfer...")
    transferred_img = ot_color_transfer(source_img, target_img, n_samples=512)

    # Plot results
    plt.rcParams.update(NEURIPS_RC)
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.0))

    axes[0].imshow(source_img)
    axes[0].set_title("Source")
    axes[0].axis("off")

    axes[1].imshow(target_img)
    axes[1].set_title("Target palette")
    axes[1].axis("off")

    axes[2].imshow(transferred_img)
    axes[2].set_title("Transferred")
    axes[2].axis("off")

    plt.tight_layout()
    output_path = os.path.join(args.output_dir, "fig_color_transfer.pdf")
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
