#!/usr/bin/env python3
"""Phase 1.5: Extract weight deltas (δW) from exploration checkpoints.

For each exploration round, loads the merged HF checkpoint (saved by verl with
checkpoint.contents.save=[model,hf_model]) and diffs against the original model
to compute δW per layer.

Also extracts low-rank factors via SVD(δW, rank=2r) for memory-efficient
storage and faster Phase 2 analysis.

Usage:
    python phase1_extract_delta.py \
        --model_path /path/to/DeepSeek-R1-Distill-Qwen-1.5B \
        --explore_dir ./gali_data/explorations \
        --output_dir ./gali_data/delta_w \
        --lora_rank 32 \
        --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj
"""
import argparse
import gc
import json
import os
import time

import torch
from safetensors import safe_open


ATTN_MODULES = {"q_proj", "k_proj", "v_proj", "o_proj"}
MLP_MODULES = {"gate_proj", "up_proj", "down_proj"}


def weight_key(layer_idx: int, module: str) -> str:
    if module in ATTN_MODULES:
        return f"model.layers.{layer_idx}.self_attn.{module}.weight"
    elif module in MLP_MODULES:
        return f"model.layers.{layer_idx}.mlp.{module}.weight"
    else:
        raise ValueError(f"Unknown module: {module}")


def build_key_to_file_map(model_path: str) -> dict[str, str]:
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        return {k: os.path.join(model_path, v) for k, v in index["weight_map"].items()}

    single = os.path.join(model_path, "model.safetensors")
    if os.path.exists(single):
        with safe_open(single, framework="pt") as f:
            return {k: single for k in f.keys()}

    raise FileNotFoundError(f"No safetensors files found in {model_path}")


def load_weight(key: str, key_to_file: dict[str, str]) -> torch.Tensor:
    filepath = key_to_file[key]
    with safe_open(filepath, framework="pt", device="cpu") as f:
        return f.get_tensor(key)


def find_hf_checkpoint(explore_round_dir: str) -> str:
    """Find the huggingface/ subdirectory inside the exploration checkpoint."""
    # verl saves to: {explore_dir}/global_step_{N}/actor/huggingface/
    for root, dirs, files in os.walk(explore_round_dir):
        if os.path.basename(root) == "huggingface" and any(
            f.endswith(".safetensors") for f in files
        ):
            return root
        # Also check for config.json as indicator
        if "config.json" in files and any(f.endswith(".safetensors") for f in files):
            return root
    return None


def main():
    parser = argparse.ArgumentParser(description="GALI Phase 1.5: Extract δW from exploration checkpoints")
    parser.add_argument("--model_path", required=True, help="Path to original HF model")
    parser.add_argument("--explore_dir", default="./gali_data/explorations", help="Directory with exploration rounds")
    parser.add_argument("--output_dir", default="./gali_data/delta_w")
    parser.add_argument("--lora_rank", type=int, default=32, help="LoRA rank (used for low-rank approx of δW)")
    parser.add_argument(
        "--target_modules",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    parser.add_argument("--num_layers", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Detect num_layers
    if args.num_layers is None:
        with open(os.path.join(args.model_path, "config.json")) as f:
            cfg = json.load(f)
        num_layers = cfg["num_hidden_layers"]
    else:
        num_layers = args.num_layers

    # Find exploration rounds
    round_dirs = sorted(
        [d for d in os.listdir(args.explore_dir) if d.startswith("gali_explore_round_")],
        key=lambda x: int(x.split("_")[-1]),
    )
    if not round_dirs:
        raise FileNotFoundError(f"No exploration rounds found in {args.explore_dir}")

    print(f"Found {len(round_dirs)} exploration rounds: {round_dirs}")
    print(f"Model: {args.model_path}, Layers: {num_layers}")

    # Build original model weight map
    orig_key_to_file = build_key_to_file_map(args.model_path)

    t0 = time.time()
    rank_factor = 2 * args.lora_rank  # save rank-2r approximation of δW

    for round_name in round_dirs:
        round_dir = os.path.join(args.explore_dir, round_name)
        round_idx = int(round_name.split("_")[-1])
        round_output = os.path.join(args.output_dir, f"round_{round_idx}")

        if os.path.exists(round_output) and len(os.listdir(round_output)) > 0:
            print(f"  Round {round_idx}: already extracted, skipping")
            continue

        # Find HF checkpoint
        hf_ckpt_path = find_hf_checkpoint(round_dir)
        if hf_ckpt_path is None:
            print(f"  WARNING: Round {round_idx}: no HF checkpoint found in {round_dir}, skipping")
            continue

        print(f"  Round {round_idx}: loading from {hf_ckpt_path}")
        ckpt_key_to_file = build_key_to_file_map(hf_ckpt_path)

        os.makedirs(round_output, exist_ok=True)

        for layer_idx in range(num_layers):
            for module in args.target_modules:
                key = weight_key(layer_idx, module)

                W0 = load_weight(key, orig_key_to_file).float()
                W_merged = load_weight(key, ckpt_key_to_file).float()

                delta_W = W_merged - W0  # [m, n]

                # Low-rank approximation: SVD(δW) → keep top rank_factor components
                # This captures the LoRA update (which is at most rank r) plus noise
                k = min(rank_factor, min(delta_W.shape))
                U_dw, S_dw, Vh_dw = torch.linalg.svd(delta_W, full_matrices=False)
                # Truncate to rank k
                U_k = U_dw[:, :k]   # [m, k]
                S_k = S_dw[:k]      # [k]
                Vh_k = Vh_dw[:k, :] # [k, n]

                # Save: B_eff = U_k * sqrt(S_k), A_eff = sqrt(S_k) * Vh_k
                # so δW ≈ B_eff @ A_eff
                sqrt_S = S_k.sqrt()
                B_eff = U_k * sqrt_S.unsqueeze(0)  # [m, k]
                A_eff = sqrt_S.unsqueeze(1) * Vh_k  # [k, n]

                torch.save(
                    {
                        "B": B_eff.to(torch.bfloat16),  # [m, k]
                        "A": A_eff.to(torch.bfloat16),  # [k, n]
                        "S": S_k,                        # [k], full precision for analysis
                        "delta_norm": delta_W.norm().item(),
                        "shape": list(delta_W.shape),
                    },
                    os.path.join(round_output, f"layer{layer_idx}_{module}.pt"),
                )

                del W0, W_merged, delta_W, U_dw, S_dw, Vh_dw
                gc.collect()

        print(f"  Round {round_idx}: extracted {num_layers * len(args.target_modules)} layers")

    elapsed = time.time() - t0
    print(f"\nDone. δW saved to {args.output_dir} ({elapsed:.1f}s)")
    print(f"Low-rank factor: {rank_factor} (= 2 × lora_rank={args.lora_rank})")


if __name__ == "__main__":
    main()
