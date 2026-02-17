"""
Generate uncertainty calibration figures from ensemble analysis results.
Reuses ensemble_results.npz — no model inference needed.

Usage:
    python plot_uncertainty_figures.py \
        --npz results/ensemble/analysis/ensemble_results.npz \
        --output_dir results/ensemble/analysis/figures
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def plot_scatter_clean(disagreement, error, output_dir):
    """Clean scatter plot: disagreement vs error, no category coloring."""
    r, p = stats.pearsonr(disagreement, error)
    n = len(disagreement)

    fig, ax = plt.subplots(figsize=(7, 5.5))

    ax.scatter(
        disagreement, error,
        c="#4A90D9", alpha=0.45, s=18, edgecolors="none",
    )

    # Trend line
    z = np.polyfit(disagreement, error, 1)
    x_line = np.linspace(disagreement.min(), disagreement.max(), 100)
    ax.plot(x_line, np.polyval(z, x_line), color="#D94A4A", linewidth=2,
            label=f"r = {r:.3f} (p < {max(p, 1e-10):.1e})")

    ax.set_xlabel("Ensemble Disagreement (mean std of predicted embeddings)", fontsize=12)
    ax.set_ylabel("Prediction Error (MSE in DINO embedding space)", fontsize=12)
    fig.suptitle("Dynamics Model Uncertainty Correlates with Prediction Error",
                 fontsize=13, fontweight="bold", y=1.02)
    ax.set_title(f"K=5 ensemble on CALVIN-D  |  {n} validation demos  |  DINOv2 ViT-S/14 encoder",
                 fontsize=9, color="#777777", pad=10)
    ax.legend(fontsize=12, loc="upper left")
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "scatter_clean.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, "scatter_clean.pdf"), bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir}/scatter_clean.png")


def plot_calibration(disagreement, error, output_dir, n_bins=5):
    """Calibration plot: bin by disagreement quintiles, show mean error per bin."""
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(disagreement, percentiles)

    bin_centers = []
    bin_mean_errors = []
    bin_std_errors = []
    bin_counts = []

    for i in range(n_bins):
        if i < n_bins - 1:
            mask = (disagreement >= bin_edges[i]) & (disagreement < bin_edges[i + 1])
        else:
            mask = (disagreement >= bin_edges[i]) & (disagreement <= bin_edges[i + 1])

        if np.sum(mask) == 0:
            continue

        bin_centers.append(disagreement[mask].mean())
        bin_mean_errors.append(error[mask].mean())
        bin_std_errors.append(error[mask].std() / np.sqrt(np.sum(mask)))  # SEM
        bin_counts.append(np.sum(mask))

    bin_centers = np.array(bin_centers)
    bin_mean_errors = np.array(bin_mean_errors)
    bin_std_errors = np.array(bin_std_errors)

    fig, ax = plt.subplots(figsize=(7, 5.5))

    ax.bar(
        range(len(bin_centers)), bin_mean_errors,
        yerr=bin_std_errors, capsize=5,
        color=["#6BAED6", "#4292C6", "#2171B5", "#08519C", "#08306B"][:len(bin_centers)],
        edgecolor="white", linewidth=0.8, alpha=0.9,
    )

    # Add count labels
    for i, (mean_e, count) in enumerate(zip(bin_mean_errors, bin_counts)):
        ax.text(i, mean_e + bin_std_errors[i] + 0.01, f"n={count}",
                ha="center", va="bottom", fontsize=9, color="#555555")

    quintile_labels = [f"Q{i+1}\n(lowest)" if i == 0
                       else f"Q{i+1}\n(highest)" if i == len(bin_centers) - 1
                       else f"Q{i+1}"
                       for i in range(len(bin_centers))]
    ax.set_xticks(range(len(bin_centers)))
    ax.set_xticklabels(quintile_labels, fontsize=11)
    ax.set_xlabel("Ensemble Disagreement Quintile", fontsize=12)
    ax.set_ylabel("Mean Prediction Error (MSE)", fontsize=12)
    n_total = sum(bin_counts)
    fig.suptitle("Higher Disagreement = Higher Prediction Error",
                 fontsize=13, fontweight="bold", y=1.02)
    ax.set_title(f"K=5 ensemble on CALVIN-D  |  {n_total} validation demos  |  DINOv2 ViT-S/14 encoder",
                 fontsize=9, color="#777777", pad=10)
    ax.grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "calibration.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, "calibration.pdf"), bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir}/calibration.png")


def main():
    parser = argparse.ArgumentParser(description="Generate uncertainty calibration figures from saved results")
    parser.add_argument("--npz", type=str, required=True, help="Path to ensemble_results.npz")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save figures")
    parser.add_argument("--bins", type=int, default=5, help="Number of bins for calibration plot")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    data = np.load(args.npz, allow_pickle=True)
    disagreement = data["disagreement"]
    error = data["error"]

    print(f"Loaded {len(disagreement)} data points")
    print(f"Disagreement: [{disagreement.min():.4f}, {disagreement.max():.4f}]")
    print(f"Error: [{error.min():.4f}, {error.max():.4f}]")

    r, p = stats.pearsonr(disagreement, error)
    print(f"Pearson r = {r:.4f}, p = {p:.2e}")

    plot_scatter_clean(disagreement, error, args.output_dir)
    plot_calibration(disagreement, error, args.output_dir, n_bins=args.bins)

    print(f"\nAll figures saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
