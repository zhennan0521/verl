"""
Analyze LoRA weight evolution across checkpoints to diagnose weight_decay suppression.

Usage:
    python analyze_lora_weights.py \
        --ckpt_dir /mnt/.../experiments/results/lora_rlvr/<exp_name> \
        --out_csv /tmp/lora_analysis.csv

Hypothesis:
    weight_decay=0.1 pulls lora_A/lora_B toward zero. If true, ||lora_A||, ||lora_B||,
    and ||ΔW|| = ||B@A|| will stay tiny or grow very slowly across steps.

How it works:
    - verl saves FSDP v1 SHARDED_STATE_DICT via torch.save on each rank.
    - Each rank's .pt file contains a dict: {param_name: ShardedTensor or Tensor}.
    - LoRA A/B matrices are usually small enough to be fully-replicated or one-shard;
      for FSDP1 flat-param sharding, they are slices of a larger flat param, but
      peft wraps them into separate Linear submodules, and FSDP wraps per
      transformer block, so lora params live inside a larger sharded flat param.

    To stay robust against sharding specifics, we compute the squared Frobenius
    norm directly from the tensor values we can reach on each rank:
      ||W||_F^2 = sum over disjoint shards of ||shard||_F^2
    This is exact for any disjoint partition (FSDP1 flat / FSDP2 per-param).

Outputs:
    - CSV: step, layer, param_name, sqnorm
    - Aggregates printed: global ||A||, ||B||, ||A||*||B|| bound on ||ΔW||, ratio to base.
"""

import argparse
import json
import math
import os
import re
from collections import defaultdict
from glob import glob

import torch


def _tensor_sqnorm(t):
    if t is None:
        return 0.0
    if hasattr(t, "_local_shards"):
        # ShardedTensor
        s = 0.0
        for shard in t._local_shards:
            s += float(shard.tensor.detach().to(torch.float32).pow(2).sum().item())
        return s
    if hasattr(t, "to_local"):
        # DTensor (FSDP2)
        local = t.to_local()
        return float(local.detach().to(torch.float32).pow(2).sum().item())
    if isinstance(t, torch.Tensor):
        return float(t.detach().to(torch.float32).pow(2).sum().item())
    return 0.0


def _tensor_numel(t):
    if t is None:
        return 0
    if hasattr(t, "_local_shards"):
        return sum(int(s.tensor.numel()) for s in t._local_shards)
    if hasattr(t, "to_local"):
        return int(t.to_local().numel())
    if isinstance(t, torch.Tensor):
        return int(t.numel())
    return 0


def classify_param(name: str):
    """
    Return (kind, layer_idx, module_key) for categorizing params.
    kind in {'lora_A', 'lora_B', 'base', 'other'}
    """
    # normalize
    n = name
    # peft LoRA markers
    if "lora_A" in n:
        kind = "lora_A"
    elif "lora_B" in n:
        kind = "lora_B"
    elif "lora_magnitude_vector" in n or "lora_magnitude" in n:
        kind = "lora_mag"
    elif "base_layer.weight" in n or re.search(r"\.(q|k|v|o|gate|up|down)_proj\.weight$", n):
        kind = "base"
    else:
        kind = "other"

    m = re.search(r"layers\.(\d+)\.", n)
    layer_idx = int(m.group(1)) if m else -1

    # module key: q_proj / k_proj / ...
    m2 = re.search(r"\.(\w+_proj)\.", n)
    module_key = m2.group(1) if m2 else "?"
    return kind, layer_idx, module_key


def aggregate_step(step_dir: str):
    """
    Load all rank shards for a single global_step and aggregate sqnorms.
    Returns dict: param_name -> {"sqnorm": float, "numel": int}
    """
    rank_files = sorted(glob(os.path.join(step_dir, "model_world_size_*_rank_*.pt")))
    if not rank_files:
        raise FileNotFoundError(f"No model shard files in {step_dir}")

    acc = defaultdict(lambda: {"sqnorm": 0.0, "numel": 0})
    for f in rank_files:
        sd = torch.load(f, map_location="cpu", weights_only=False)
        if not isinstance(sd, dict):
            # some versions wrap in {'model': ...}
            sd = getattr(sd, "state_dict", lambda: sd)()
        for name, t in sd.items():
            try:
                sq = _tensor_sqnorm(t)
                nl = _tensor_numel(t)
            except Exception as e:
                print(f"  [warn] skip {name}: {e}")
                continue
            acc[name]["sqnorm"] += sq
            acc[name]["numel"] += nl
        del sd
    return dict(acc)


def find_step_dirs(ckpt_dir: str):
    """Return list of (step_int, actor_dir) sorted by step."""
    steps = []
    for d in glob(os.path.join(ckpt_dir, "global_step_*")):
        m = re.search(r"global_step_(\d+)$", d)
        if not m:
            continue
        actor = os.path.join(d, "actor")
        if os.path.isdir(actor):
            steps.append((int(m.group(1)), actor))
    steps.sort()
    return steps


def summarize(acc, step):
    """Summarize per-step: total A-norm, B-norm, ΔW bound, base norm."""
    sumA = sumB = sumMag = sumBase = sumOther = 0.0
    numelA = numelB = numelMag = numelBase = 0
    # also per-(layer, module) for later
    per_module = defaultdict(lambda: {"A": 0.0, "B": 0.0, "base": 0.0,
                                       "A_numel": 0, "B_numel": 0, "base_numel": 0})
    for name, stats in acc.items():
        kind, layer, mod = classify_param(name)
        key = (layer, mod)
        if kind == "lora_A":
            sumA += stats["sqnorm"]; numelA += stats["numel"]
            per_module[key]["A"] += stats["sqnorm"]; per_module[key]["A_numel"] += stats["numel"]
        elif kind == "lora_B":
            sumB += stats["sqnorm"]; numelB += stats["numel"]
            per_module[key]["B"] += stats["sqnorm"]; per_module[key]["B_numel"] += stats["numel"]
        elif kind == "lora_mag":
            sumMag += stats["sqnorm"]; numelMag += stats["numel"]
        elif kind == "base":
            sumBase += stats["sqnorm"]; numelBase += stats["numel"]
            per_module[key]["base"] += stats["sqnorm"]; per_module[key]["base_numel"] += stats["numel"]
        else:
            sumOther += stats["sqnorm"]

    normA = math.sqrt(sumA)
    normB = math.sqrt(sumB)
    normBase = math.sqrt(sumBase)
    # rms per-element (scale-free):
    rmsA = math.sqrt(sumA / numelA) if numelA else 0.0
    rmsB = math.sqrt(sumB / numelB) if numelB else 0.0
    rmsBase = math.sqrt(sumBase / numelBase) if numelBase else 0.0
    bound_dW = normA * normB  # upper bound on ||ΔW||_F = ||B@A||_F <= ||B||_F * ||A||_F (loose but useful trend)

    return {
        "step": step,
        "normA": normA, "normB": normB, "normBase": normBase,
        "rmsA": rmsA, "rmsB": rmsB, "rmsBase": rmsBase,
        "bound_dW": bound_dW,
        "ratio_bound_dW_to_base": (bound_dW / normBase) if normBase > 0 else 0.0,
        "numelA": numelA, "numelB": numelB, "numelBase": numelBase,
        "per_module": per_module,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True, help="dir containing global_step_* subdirs")
    ap.add_argument("--out_csv", default=None)
    ap.add_argument("--per_layer_csv", default=None)
    ap.add_argument("--steps", default=None,
                    help="comma-separated step numbers to analyze; default=all")
    args = ap.parse_args()

    steps = find_step_dirs(args.ckpt_dir)
    if args.steps:
        wanted = set(int(s) for s in args.steps.split(","))
        steps = [s for s in steps if s[0] in wanted]
    if not steps:
        raise SystemExit(f"no checkpoints found under {args.ckpt_dir}")

    print(f"analyzing {len(steps)} checkpoints in {args.ckpt_dir}")
    summaries = []
    per_layer_rows = []
    for step, actor in steps:
        print(f"  [step {step}] loading shards from {actor}")
        acc = aggregate_step(actor)
        s = summarize(acc, step)
        summaries.append(s)
        print(
            f"    step={step:>5}  ||A||={s['normA']:.4e}  ||B||={s['normB']:.4e}  "
            f"||A||*||B||={s['bound_dW']:.4e}  ||base||={s['normBase']:.4e}  "
            f"ratio={s['ratio_bound_dW_to_base']:.2e}"
        )
        print(
            f"                rmsA={s['rmsA']:.4e}  rmsB={s['rmsB']:.4e}  rmsBase={s['rmsBase']:.4e}"
        )
        for (layer, mod), d in s["per_module"].items():
            per_layer_rows.append({
                "step": step, "layer": layer, "module": mod,
                "sqnorm_A": d["A"], "sqnorm_B": d["B"], "sqnorm_base": d["base"],
                "numel_A": d["A_numel"], "numel_B": d["B_numel"], "numel_base": d["base_numel"],
            })

    print("\n=== summary trend ===")
    print(f"{'step':>6} {'||A||':>12} {'||B||':>12} {'||A||*||B||':>14} "
          f"{'||base||':>12} {'dW/base':>12} {'rmsA':>10} {'rmsB':>10}")
    for s in summaries:
        print(f"{s['step']:>6d} {s['normA']:>12.4e} {s['normB']:>12.4e} "
              f"{s['bound_dW']:>14.4e} {s['normBase']:>12.4e} "
              f"{s['ratio_bound_dW_to_base']:>12.2e} {s['rmsA']:>10.2e} {s['rmsB']:>10.2e}")

    if args.out_csv:
        import csv
        with open(args.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "step", "normA", "normB", "normBase", "rmsA", "rmsB", "rmsBase",
                "bound_dW", "ratio_bound_dW_to_base",
                "numelA", "numelB", "numelBase",
            ])
            w.writeheader()
            for s in summaries:
                row = {k: s[k] for k in w.fieldnames}
                w.writerow(row)
        print(f"wrote {args.out_csv}")

    if args.per_layer_csv:
        import csv
        with open(args.per_layer_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(per_layer_rows[0].keys()))
            w.writeheader()
            w.writerows(per_layer_rows)
        print(f"wrote {args.per_layer_csv}")

    # ---- diagnosis heuristic ----
    print("\n=== diagnosis ===")
    if len(summaries) >= 2:
        s0, sN = summaries[0], summaries[-1]
        growth_A = sN["normA"] / max(s0["normA"], 1e-12)
        growth_B = sN["normB"] / max(s0["normB"], 1e-12)
        growth_dW = sN["bound_dW"] / max(s0["bound_dW"], 1e-12)
        print(f"  ||A||  step {s0['step']} -> {sN['step']}: x{growth_A:.3f}")
        print(f"  ||B||  step {s0['step']} -> {sN['step']}: x{growth_B:.3f}  "
              f"(B starts ~0; this should be >> 1 if learning)")
        print(f"  ||A||*||B|| growth: x{growth_dW:.3f}")
        print(f"  final dW/base ratio: {sN['ratio_bound_dW_to_base']:.2e}")
        print()
        if sN["ratio_bound_dW_to_base"] < 1e-3:
            print("  ⚠️  ΔW bound is <0.1% of ||base||. adapter is NOT meaningfully updating base model.")
        if growth_B < 2.0 and sN["step"] > 64:
            print("  ⚠️  ||lora_B|| barely grew. Consistent with weight_decay pulling B toward 0.")
        if sN["normB"] < 1e-4:
            print("  ⚠️  ||lora_B|| is near zero. Adapter is effectively inactive.")
    else:
        print("  need >=2 checkpoints for trend analysis")


if __name__ == "__main__":
    main()
