#!/usr/bin/env python3
"""Visualize GALI spectral analysis results.

Generates plots for:
  1. Per-layer projection spectrum (P_final vs singular direction index)
  2. Cross-round consistency heatmap
  3. Energy centroid distribution across layers
  4. Comparison with W's singular value spectrum

Usage:
    python visualize_projections.py \
        --projection_dir ./gali_data/projections \
        --output_dir ./gali_data/plots
"""
import argparse
import os

import torch


def main():
    parser = argparse.ArgumentParser(description="GALI: Visualize projection spectra")
    parser.add_argument("--projection_dir", default="./gali_data/projections")
    parser.add_argument("--output_dir", default="./gali_data/plots")
    parser.add_argument("--top_n_layers", type=int, default=8, help="Plot top N most interesting layers")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available. Install it for visualization: pip install matplotlib")
        print("Generating text summary instead.\n")
        _text_summary(args)
        return

    # Load all projections
    files = sorted([f for f in os.listdir(args.projection_dir) if f.endswith(".pt")])
    if not files:
        print(f"No projection files in {args.projection_dir}")
        return

    data = {}
    for f in files:
        name = f.replace(".pt", "")
        d = torch.load(os.path.join(args.projection_dir, f), map_location="cpu", weights_only=True)
        data[name] = d

    # ---- Plot 1: Per-layer projection spectrum ----
    # Select layers with highest variance in importance (most "interesting")
    layer_scores = {}
    for name, d in data.items():
        imp = d["importance"]
        # Score by how non-uniform the distribution is (KL from uniform)
        uniform = torch.ones_like(imp) / imp.shape[0]
        kl = (imp * (imp / uniform + 1e-12).log()).sum().item()
        layer_scores[name] = kl

    top_layers = sorted(layer_scores, key=layer_scores.get, reverse=True)[:args.top_n_layers]

    fig, axes = plt.subplots(2, (len(top_layers) + 1) // 2, figsize=(20, 10))
    axes = axes.flatten()
    for idx, name in enumerate(top_layers):
        ax = axes[idx]
        d = data[name]
        imp = d["importance"]
        S_w = d["S_w"]
        x = torch.arange(len(imp))

        ax.bar(x.numpy(), imp.numpy(), alpha=0.6, label="GALI importance", color="steelblue")
        ax2 = ax.twinx()
        ax2.plot(x.numpy(), (S_w / S_w.max()).numpy(), "r-", alpha=0.5, label="σ(W) normalized")
        ax2.set_ylabel("σ(W) / σ_max", color="red")

        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Singular direction index")
        ax.set_ylabel("GALI importance")
        if idx == 0:
            ax.legend(loc="upper right", fontsize=7)
            ax2.legend(loc="center right", fontsize=7)

    for idx in range(len(top_layers), len(axes)):
        axes[idx].set_visible(False)
    plt.suptitle("GALI Importance vs W Singular Values (top layers by non-uniformity)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "projection_spectra.png"), dpi=150)
    plt.close()
    print(f"Saved projection_spectra.png")

    # ---- Plot 2: Energy centroid histogram ----
    centroids = []
    names = []
    for name, d in data.items():
        imp = d["importance"]
        dim = imp.shape[0]
        centroid = (torch.arange(dim, dtype=torch.float32) * imp).sum().item() / dim
        centroids.append(centroid)
        names.append(name)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(centroids, bins=30, edgecolor="black", alpha=0.7)
    ax.axvline(0.5, color="red", linestyle="--", label="Uniform centroid (0.5)")
    ax.set_xlabel("Energy centroid (0=head, 1=tail)")
    ax.set_ylabel("Count (layers)")
    ax.set_title("GALI importance energy centroid distribution")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "energy_centroids.png"), dpi=150)
    plt.close()
    print(f"Saved energy_centroids.png")

    # ---- Plot 3: Cross-round P_all heatmap for a representative layer ----
    rep_layer = top_layers[0]
    P_all = data[rep_layer]["P_all"]  # [K, d]
    fig, ax = plt.subplots(figsize=(14, 4))
    im = ax.imshow(P_all.abs().numpy(), aspect="auto", cmap="viridis")
    ax.set_xlabel("Singular direction index")
    ax.set_ylabel("Exploration round")
    ax.set_title(f"Cross-round projection magnitudes: {rep_layer}")
    plt.colorbar(im, ax=ax, label="|P_i^(k)|")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "crossround_heatmap.png"), dpi=150)
    plt.close()
    print(f"Saved crossround_heatmap.png")

    # ---- Plot 4: Consistency distribution ----
    all_consistency = []
    for name, d in data.items():
        all_consistency.append(d["consistency"])
    all_consistency = torch.cat(all_consistency)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(all_consistency.numpy(), bins=50, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Consistency score (mean/std)")
    ax.set_ylabel("Count (directions across all layers)")
    ax.set_title("Cross-round consistency distribution")
    ax.axvline(1.0, color="red", linestyle="--", label="Threshold=1.0")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "consistency_distribution.png"), dpi=150)
    plt.close()
    print(f"Saved consistency_distribution.png")

    print(f"\nAll plots saved to {args.output_dir}")


def _text_summary(args):
    """Fallback text summary when matplotlib is unavailable."""
    files = sorted([f for f in os.listdir(args.projection_dir) if f.endswith(".pt")])
    for f in files:
        name = f.replace(".pt", "")
        d = torch.load(os.path.join(args.projection_dir, f), map_location="cpu", weights_only=True)
        imp = d["importance"]
        dim = imp.shape[0]
        centroid = (torch.arange(dim, dtype=torch.float32) * imp).sum().item() / dim
        top5 = imp.topk(5)
        print(f"{name}: centroid={centroid:.3f}  top5_dirs={top5.indices.tolist()}  "
              f"top5_imp={[f'{v:.4f}' for v in top5.values.tolist()]}")


if __name__ == "__main__":
    main()
