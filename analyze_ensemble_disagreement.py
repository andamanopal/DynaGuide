"""
Ensemble Disagreement vs Prediction Error Analysis
====================================================
Loads K dynamics models trained with different seeds, runs forward passes
on labeled validation data, and produces the key scatter plot:

    X-axis: Ensemble disagreement (std of predicted embeddings across models)
    Y-axis: Actual prediction error (distance from mean prediction to ground truth)

Each point = one (state, action) pair from the validation set.
Points are colored by behavior category to show that OOD data (MovableObjects)
clusters in the high-disagreement / high-error region.

Usage:
    python analyze_ensemble_disagreement.py \
        --checkpoints results/ensemble/seed_42/6000.pth \
                      results/ensemble/seed_123/6000.pth \
                      results/ensemble/seed_999/6000.pth \
        --test_hdf5 dataset/CalvinDD_betterseg/data.hdf5 \
        --output_dir results/ensemble/analysis
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm

from core.dynamics_models import FinalStatePredictionDino
from core.embedder_datasets import MultiviewDataset

MAIN_CAMERA = "third_person"

# Behavior categories
ARTICULATED_BEHAVIORS = {
    "switch_on", "switch_off",
    "button_on", "button_off",
    "drawer_open", "drawer_close",
    "door_left", "door_right",
}
MOVABLE_BEHAVIORS = {
    "red_displace", "blue_displace", "pink_displace",
    "red_lift", "blue_lift", "pink_lift",
}


def load_models(checkpoint_paths, action_dim, action_chunk_length, cameras, proprio, proprio_dim, device):
    models = []
    for path in checkpoint_paths:
        model = FinalStatePredictionDino(
            action_dim, action_chunk_length,
            cameras=cameras, reconstruction=True,
            proprio=proprio, proprio_dim=proprio_dim,
        )
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        model.eval()
        models.append(model)
        print(f"Loaded: {path}")
    return models


def prepare(data, device="cuda"):
    if isinstance(data, dict):
        return {k: (torch.tensor(np.array(v)) if not isinstance(v, torch.Tensor) else v).to(device).to(torch.float32)
                for k, v in data.items()}
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(np.array(data))
    return data.to(device).to(torch.float32)


def categorize_behavior(label):
    label = str(label)
    if label in ARTICULATED_BEHAVIORS:
        return "Articulated"
    if label in MOVABLE_BEHAVIORS:
        return "Movable"
    return "Other"


def analyze(models, dataset, device, max_samples=2000):
    disagreements = []
    prediction_errors = []
    labels = []
    categories = []

    idx = 0
    sample_count = 0
    action_chunk_len = dataset.action_chunk_length

    for demo_idx, length in enumerate(tqdm.tqdm(dataset.lengths_list, desc="Processing demos")):
        demo_start = idx
        idx += length

        # Skip demos too short to have a meaningful action chunk
        if length < action_chunk_len + 1:
            continue

        # Use a mid-demo timestep where the action chunk covers actual behavior
        # One action chunk before the end gives us meaningful actions AND a known future
        mid_idx = demo_start + max(0, length - action_chunk_len - 1)
        last_idx = demo_start + length - 1

        if mid_idx >= len(dataset) or last_idx >= len(dataset):
            continue

        # Get state + action at mid-demo point (where actions are meaningful)
        mid_sample = dataset.get_labeled_item(mid_idx)
        state_raw, action_raw, label = mid_sample[0], mid_sample[1], mid_sample[2]

        # Get the actual end state (ground truth for what the dynamics model should predict)
        last_sample = dataset.get_labeled_item(last_idx)
        last_state_raw = last_sample[0]

        state = prepare(state_raw, device)
        action = prepare(action_raw, device)
        last_state = prepare(last_state_raw, device)
        state = {k: torch.unsqueeze(v, dim=0) for k, v in state.items()}
        last_state = {k: torch.unsqueeze(v, dim=0) for k, v in last_state.items()}
        action = torch.unsqueeze(action, dim=0)

        with torch.no_grad():
            # Ground truth: embedding of the actual end state of this demo
            gt_embedding = models[0].state_embedding(last_state, normalize=False).flatten(start_dim=1)
            embed_dim = gt_embedding.shape[-1]

            # Predicted future embeddings from each ensemble member
            pred_embeddings = []
            for model in models:
                z_hat = model.state_action_embedding(state, action, normalize=False).flatten(start_dim=1)
                pred_embeddings.append(z_hat)

            pred_stack = torch.stack(pred_embeddings, dim=0)  # (K, 1, D)

            # Ensemble mean prediction
            pred_mean = pred_stack.mean(dim=0)  # (1, D)

            # Disagreement: mean std across embedding dimensions
            pred_std = pred_stack.std(dim=0)  # (1, D)
            disagreement = pred_std.mean().item()

            # Prediction error: normalized MSE (mean squared error per dimension)
            error = torch.mean((pred_mean - gt_embedding) ** 2).item()

        disagreements.append(disagreement)
        prediction_errors.append(error)
        labels.append(str(label))
        categories.append(categorize_behavior(label))

        sample_count += 1
        if sample_count >= max_samples:
            break

    return {
        "disagreement": np.array(disagreements),
        "error": np.array(prediction_errors),
        "label": labels,
        "category": categories,
    }


def plot_scatter(results, output_dir):
    """Main scatter plot: disagreement vs prediction error, colored by category."""
    fig, ax = plt.subplots(figsize=(8, 6))

    category_colors = {
        "Articulated": "#2196F3",
        "Movable": "#F44336",
        "Other": "#9E9E9E",
    }
    category_order = ["Other", "Articulated", "Movable"]  # Movable on top

    for cat in category_order:
        mask = np.array(results["category"]) == cat
        if not np.any(mask):
            continue
        ax.scatter(
            results["disagreement"][mask],
            results["error"][mask],
            c=category_colors[cat],
            label=f"{cat} (n={np.sum(mask)})",
            alpha=0.5,
            s=20,
            edgecolors="none",
        )

    ax.set_xlabel("Ensemble Disagreement (mean std of predicted embeddings)", fontsize=12)
    ax.set_ylabel("Prediction Error (L2 to ground truth)", fontsize=12)
    ax.set_title("Dynamics Model: Ensemble Disagreement vs Prediction Error", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "disagreement_vs_error.png"), dpi=300)
    plt.savefig(os.path.join(output_dir, "disagreement_vs_error.pdf"))
    plt.close()
    print(f"Saved scatter plot to {output_dir}/disagreement_vs_error.png")


def plot_box_by_category(results, output_dir):
    """Box plots showing disagreement distribution per category."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    category_order = ["Articulated", "Movable", "Other"]
    category_colors = ["#2196F3", "#F44336", "#9E9E9E"]

    # Disagreement by category
    data_disagree = []
    data_error = []
    tick_labels = []
    colors = []
    for cat, color in zip(category_order, category_colors):
        mask = np.array(results["category"]) == cat
        if not np.any(mask):
            continue
        data_disagree.append(results["disagreement"][mask])
        data_error.append(results["error"][mask])
        tick_labels.append(cat)
        colors.append(color)

    bp1 = ax1.boxplot(data_disagree, labels=tick_labels, patch_artist=True)
    for patch, color in zip(bp1["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax1.set_ylabel("Ensemble Disagreement")
    ax1.set_title("Disagreement by Behavior Category")
    ax1.grid(True, alpha=0.3)

    bp2 = ax2.boxplot(data_error, labels=tick_labels, patch_artist=True)
    for patch, color in zip(bp2["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax2.set_ylabel("Prediction Error (L2)")
    ax2.set_title("Prediction Error by Behavior Category")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "category_boxplots.png"), dpi=300)
    plt.savefig(os.path.join(output_dir, "category_boxplots.pdf"))
    plt.close()
    print(f"Saved box plots to {output_dir}/category_boxplots.png")


def print_statistics(results):
    """Print correlation and per-category statistics."""
    corr = np.corrcoef(results["disagreement"], results["error"])[0, 1]
    print(f"\n{'='*60}")
    print(f"Pearson correlation (disagreement vs error): {corr:.4f}")
    print(f"{'='*60}")

    for cat in ["Articulated", "Movable", "Other"]:
        mask = np.array(results["category"]) == cat
        if not np.any(mask):
            continue
        d = results["disagreement"][mask]
        e = results["error"][mask]
        print(f"\n{cat} (n={np.sum(mask)}):")
        print(f"  Disagreement: mean={d.mean():.4f}, std={d.std():.4f}")
        print(f"  Pred Error:   mean={e.mean():.4f}, std={e.std():.4f}")

    print(f"\n{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Analyze ensemble disagreement vs prediction error")
    parser.add_argument("--checkpoints", nargs="+", required=True, help="Paths to dynamics model checkpoints (one per seed)")
    parser.add_argument("--test_hdf5", type=str, required=True, help="Labeled validation HDF5 (mixed behaviors)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save plots")
    parser.add_argument("--action_dim", type=int, default=7)
    parser.add_argument("--action_chunk_length", type=int, default=16)
    parser.add_argument("--proprio_dim", type=int, default=15)
    parser.add_argument("--max_samples", type=int, default=2000, help="Max demo endpoints to evaluate")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading {len(args.checkpoints)} dynamics models...")
    models = load_models(
        args.checkpoints,
        action_dim=args.action_dim,
        action_chunk_length=args.action_chunk_length,
        cameras=[MAIN_CAMERA],
        proprio="proprio",
        proprio_dim=args.proprio_dim,
        device=args.device,
    )

    print(f"Loading validation dataset from {args.test_hdf5}...")
    dataset = MultiviewDataset(
        args.test_hdf5,
        action_chunk_length=args.action_chunk_length,
        cameras=[MAIN_CAMERA],
        padding=True,
        pad_mode="zeros",
        proprio="proprio",
    )
    print(f"  {len(dataset.lengths_list)} demos, {len(dataset)} total samples")

    print("Running ensemble analysis...")
    results = analyze(models, dataset, args.device, max_samples=args.max_samples)

    print_statistics(results)
    plot_scatter(results, args.output_dir)
    plot_box_by_category(results, args.output_dir)

    # Save raw data for further analysis
    np.savez(
        os.path.join(args.output_dir, "ensemble_results.npz"),
        disagreement=results["disagreement"],
        error=results["error"],
        label=results["label"],
        category=results["category"],
    )
    print(f"Saved raw results to {args.output_dir}/ensemble_results.npz")


if __name__ == "__main__":
    main()
