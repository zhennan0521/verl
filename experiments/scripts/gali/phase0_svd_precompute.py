#!/usr/bin/env python3
"""Phase 0: Precompute SVD of W for all LoRA target layers (CPU, offline).

Usage:
    python phase0_svd_precompute.py \
        --model_path /path/to/DeepSeek-R1-Distill-Qwen-1.5B \
        --output_dir ./gali_data/svd_cache \
        --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj

Output structure:
    svd_cache/
        layer0_q_proj.pt   # {"U": [m, k], "S": [k], "Vh": [k, n], "shape": [m, n]}
        layer0_k_proj.pt
        ...
        layer27_down_proj.pt
"""
import argparse
import gc
import os
import time

import torch
from safetensors import safe_open


# ---------- weight key helpers ----------
ATTN_MODULES = {"q_proj", "k_proj", "v_proj", "o_proj"}
MLP_MODULES = {"gate_proj", "up_proj", "down_proj"}


def weight_key(layer_idx: int, module: str) -> str:
    if module in ATTN_MODULES:
        return f"model.layers.{layer_idx}.self_attn.{module}.weight"
    elif module in MLP_MODULES:
        return f"model.layers.{layer_idx}.mlp.{module}.weight"
    else:
        raise ValueError(f"Unknown module: {module}")


# ---------- weight loading ----------
def build_key_to_file_map(model_path: str) -> dict[str, str]:
    """Build mapping from weight key -> safetensors file that contains it."""
    import json

    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        return {k: os.path.join(model_path, v) for k, v in index["weight_map"].items()}

    # Single file
    single = os.path.join(model_path, "model.safetensors")
    if os.path.exists(single):
        with safe_open(single, framework="pt") as f:
            return {k: single for k in f.keys()}

    raise FileNotFoundError(f"No safetensors files found in {model_path}")


def load_weight(key: str, key_to_file: dict[str, str]) -> torch.Tensor:
    filepath = key_to_file[key]
    with safe_open(filepath, framework="pt", device="cpu") as f:
        return f.get_tensor(key)


# ---------- main ----------
def main():
    parser = argparse.ArgumentParser(description="GALI Phase 0: SVD precomputation")
    parser.add_argument("--model_path", required=True, help="Path to HF model directory")
    parser.add_argument("--output_dir", default="./gali_data/svd_cache")
    parser.add_argument(
        "--target_modules",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    parser.add_argument("--num_layers", type=int, default=None, help="Override num_hidden_layers (auto-detected)")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    compute_dtype = torch.float64 if args.dtype == "float64" else torch.float32

    # Detect num_layers from config
    if args.num_layers is None:
        import json

        with open(os.path.join(args.model_path, "config.json")) as f:
            cfg = json.load(f)
        num_layers = cfg["num_hidden_layers"]
    else:
        num_layers = args.num_layers

    print(f"Model: {args.model_path}")
    print(f"Layers: {num_layers}, Target modules: {args.target_modules}")
    print(f"SVD dtype: {compute_dtype}")

    key_to_file = build_key_to_file_map(args.model_path)

    total = num_layers * len(args.target_modules)
    done = 0
    t0 = time.time()

    for layer_idx in range(num_layers):
        for module in args.target_modules:
            out_path = os.path.join(args.output_dir, f"layer{layer_idx}_{module}.pt")
            if os.path.exists(out_path):
                done += 1
                continue

            key = weight_key(layer_idx, module)
            W = load_weight(key, key_to_file).to(compute_dtype)  # [out_features, in_features]

            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            # U: [m, k], S: [k], Vh: [k, n] where k = min(m, n)

            torch.save(
                {
                    "U": U.to(torch.float32),
                    "S": S.to(torch.float32),
                    "Vh": Vh.to(torch.float32),
                    "shape": list(W.shape),
                },
                out_path,
            )

            done += 1
            elapsed = time.time() - t0
            eta = elapsed / done * (total - done)
            print(f"  [{done}/{total}] layer{layer_idx}_{module}  shape={list(W.shape)}  "
                  f"S_max={S[0]:.2f} S_min={S[-1]:.4f}  ({elapsed:.1f}s elapsed, ~{eta:.0f}s remaining)")

            del W, U, S, Vh
            gc.collect()

    print(f"\nDone. SVD cache saved to {args.output_dir} ({time.time() - t0:.1f}s total)")


if __name__ == "__main__":
    main()
