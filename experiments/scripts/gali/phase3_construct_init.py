#!/usr/bin/env python3
"""Phase 3: Construct GALI initialization (CPU).

Reads the importance distribution from Phase 2 and W's SVD from Phase 0,
then constructs:
  1. A PEFT-compatible adapter directory (A_init, B_init)
  2. A modified base model directory (W_frozen = W_0 - α/r * B_init @ A_init)

Usage:
    python phase3_construct_init.py \
        --model_path /path/to/DeepSeek-R1-Distill-Qwen-1.5B \
        --svd_dir ./gali_data/svd_cache \
        --projection_dir ./gali_data/projections \
        --output_dir ./gali_data/gali_init \
        --lora_rank 32 --lora_alpha 64 --eta 0.7

Output:
    gali_init/
        adapter/           # PEFT adapter (adapter_config.json + adapter_model.safetensors)
        base_model/        # Modified base model with W_frozen weights
"""
import argparse
import os
import sys
import time

# Add peft to path so we can import gali modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "peft", "src"))

import torch
from peft.tuners.lora.gali_init import construct_gali_adapter


def main():
    parser = argparse.ArgumentParser(description="GALI Phase 3: Construct initialization")
    parser.add_argument("--model_path", required=True, help="Path to original HF model")
    parser.add_argument("--svd_dir", default="./gali_data/svd_cache")
    parser.add_argument("--projection_dir", default="./gali_data/projections")
    parser.add_argument("--output_dir", default="./gali_data/gali_init")
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--eta", type=float, default=0.7, help="Concentration param (0=uniform, 1=pure importance)")
    parser.add_argument("--method", default="sampling", choices=["sampling", "deterministic"])
    parser.add_argument("--weighted_amplitude", action="store_true", help="Weight init amplitude by importance")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--target_modules",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    parser.add_argument("--num_layers", type=int, default=None)
    parser.add_argument("--save_dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = parser.parse_args()

    adapter_dir = os.path.join(args.output_dir, "adapter")
    base_model_dir = os.path.join(args.output_dir, "base_model")
    save_dtype = getattr(torch, args.save_dtype)

    print(f"Model: {args.model_path}")
    print(f"Rank: {args.lora_rank}, Alpha: {args.lora_alpha}, η: {args.eta}")
    print(f"Method: {args.method}, Weighted amplitude: {args.weighted_amplitude}")
    print(f"Seed: {args.seed}")

    t0 = time.time()

    construct_gali_adapter(
        model_path=args.model_path,
        svd_dir=args.svd_dir,
        projection_dir=args.projection_dir,
        output_adapter_dir=adapter_dir,
        output_base_model_dir=base_model_dir,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        eta=args.eta,
        method=args.method,
        weighted_amplitude=args.weighted_amplitude,
        seed=args.seed,
        num_layers=args.num_layers,
        save_dtype=save_dtype,
    )

    elapsed = time.time() - t0
    print(f"\nDone ({elapsed:.1f}s)")
    print(f"  Adapter:    {adapter_dir}")
    print(f"  Base model: {base_model_dir}")
    print(f"\nFor Phase 4 training, set:")
    print(f"  actor_rollout_ref.model.path={base_model_dir}")
    print(f"  actor_rollout_ref.model.lora_adapter_path={adapter_dir}")


if __name__ == "__main__":
    main()
