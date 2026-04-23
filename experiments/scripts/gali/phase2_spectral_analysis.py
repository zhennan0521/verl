#!/usr/bin/env python3
"""Phase 2: Spectral projection analysis on W's SVD basis (CPU).

Projects each round's δW onto W's singular directions, then aggregates
across rounds with consistency-weighted denoising.

Usage:
    python phase2_spectral_analysis.py \
        --svd_dir ./gali_data/svd_cache \
        --delta_dir ./gali_data/delta_w \
        --output_dir ./gali_data/projections \
        --aggregation consistency

Output per layer:
    projections/layer{l}_{module}.pt:
        "P_final":     [d]  -- denoised projection strength per singular direction
        "importance":  [d]  -- normalized importance distribution
        "P_all":       [K, d] -- per-round raw projections (for visualization)
        "consistency": [d]  -- cross-round consistency score
"""
import argparse
import os
import time

import torch


AGGREGATION_METHODS = ["median", "trimmed_mean", "consistency"]


def compute_projection_lowrank(
    U_w: torch.Tensor,   # [m, d]
    Vh_w: torch.Tensor,  # [d, n]
    B: torch.Tensor,     # [m, k]  (low-rank factor of δW)
    A: torch.Tensor,     # [k, n]  (low-rank factor of δW)
) -> torch.Tensor:
    """Compute projection of δW = B @ A onto W's SVD basis.

    P_i = u_i^T @ (B @ A) @ v_i

    Efficient: O(d * m * k + d * n * k) instead of O(d * m * n).

    Returns: P [d] -- projection strength per singular direction.
    """
    # alpha = U_w^T @ B : [d, k]
    alpha = U_w.T @ B.float()
    # beta = A @ Vh_w^T : [k, d]
    beta = A.float() @ Vh_w.T
    # P_i = sum_j alpha[i,j] * beta[j,i] = row-wise dot product
    P = (alpha * beta.T).sum(dim=1)  # [d]
    return P


def aggregate_median(P_all: torch.Tensor) -> torch.Tensor:
    return P_all.abs().median(dim=0).values


def aggregate_trimmed_mean(P_all: torch.Tensor) -> torch.Tensor:
    K = P_all.shape[0]
    if K <= 2:
        return P_all.abs().mean(dim=0)
    P_abs = P_all.abs().sort(dim=0).values  # [K, d], sorted per direction
    # Drop min and max, average the rest
    return P_abs[1:-1].mean(dim=0)


def aggregate_consistency(P_all: torch.Tensor, threshold: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Consistency-weighted aggregation (recommended).

    Directions with strong signal in ALL rounds → high weight (true signal).
    Directions with strong signal in FEW rounds → low weight (noise).
    """
    P_abs = P_all.abs()
    P_mean = P_abs.mean(dim=0)     # [d]
    P_std = P_abs.std(dim=0)       # [d]
    eps = 1e-8

    consistency = P_mean / (P_std + eps)  # signal-to-noise ratio
    weight = torch.sigmoid(consistency - threshold)
    P_final = P_mean * weight

    return P_final, consistency


def main():
    parser = argparse.ArgumentParser(description="GALI Phase 2: Spectral projection analysis")
    parser.add_argument("--svd_dir", default="./gali_data/svd_cache")
    parser.add_argument("--delta_dir", default="./gali_data/delta_w")
    parser.add_argument("--output_dir", default="./gali_data/projections")
    parser.add_argument("--aggregation", default="consistency", choices=AGGREGATION_METHODS)
    parser.add_argument("--consistency_threshold", type=float, default=1.0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Find rounds
    round_dirs = sorted(
        [d for d in os.listdir(args.delta_dir) if d.startswith("round_")],
        key=lambda x: int(x.split("_")[1]),
    )
    K = len(round_dirs)
    if K == 0:
        raise FileNotFoundError(f"No round directories in {args.delta_dir}")
    print(f"Rounds: {K}, Aggregation: {args.aggregation}")

    # Find layer files from first round to enumerate (layer, module) pairs
    first_round = os.path.join(args.delta_dir, round_dirs[0])
    layer_files = sorted([f for f in os.listdir(first_round) if f.endswith(".pt")])
    print(f"Layer×module count: {len(layer_files)}")

    t0 = time.time()
    stats_log = []

    for layer_file in layer_files:
        name = layer_file.replace(".pt", "")  # e.g. "layer0_q_proj"

        # Load SVD of W
        svd_path = os.path.join(args.svd_dir, layer_file)
        if not os.path.exists(svd_path):
            print(f"  WARNING: SVD not found for {name}, skipping")
            continue
        svd = torch.load(svd_path, map_location="cpu", weights_only=True)
        U_w = svd["U"]    # [m, d]
        Vh_w = svd["Vh"]  # [d, n]
        S_w = svd["S"]    # [d]
        d = S_w.shape[0]

        # Collect projections from all rounds
        P_all = []
        for round_name in round_dirs:
            delta_path = os.path.join(args.delta_dir, round_name, layer_file)
            if not os.path.exists(delta_path):
                print(f"  WARNING: {delta_path} not found, skipping round")
                continue
            delta = torch.load(delta_path, map_location="cpu", weights_only=True)
            B = delta["B"]  # [m, k]
            A = delta["A"]  # [k, n]

            P_k = compute_projection_lowrank(U_w, Vh_w, B, A)
            P_all.append(P_k)

        if len(P_all) < 2:
            print(f"  WARNING: Only {len(P_all)} rounds available for {name}, need ≥2")
            continue

        P_all = torch.stack(P_all)  # [K, d]

        # Aggregate
        if args.aggregation == "median":
            P_final = aggregate_median(P_all)
            consistency = torch.zeros(d)
        elif args.aggregation == "trimmed_mean":
            P_final = aggregate_trimmed_mean(P_all)
            consistency = torch.zeros(d)
        elif args.aggregation == "consistency":
            P_final, consistency = aggregate_consistency(P_all, args.consistency_threshold)

        # Normalize to importance distribution
        importance = P_final / (P_final.sum() + 1e-12)

        # Save
        torch.save(
            {
                "P_final": P_final,
                "importance": importance,
                "P_all": P_all,
                "consistency": consistency,
                "S_w": S_w,
            },
            os.path.join(args.output_dir, layer_file),
        )

        # Stats
        top10_idx = importance.topk(10).indices.tolist()
        energy_centroid = (torch.arange(d, dtype=torch.float32) * importance).sum().item() / d
        stats_log.append(f"  {name}: centroid={energy_centroid:.3f}  "
                         f"top10_dirs={top10_idx}  "
                         f"P_final_max={P_final.max():.4f}")

    for line in stats_log:
        print(line)

    print(f"\nDone. Projections saved to {args.output_dir} ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
