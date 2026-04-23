#!/usr/bin/env bash
# =============================================================================
# GALI End-to-End Pipeline
#
# Runs all phases sequentially: SVD precompute → Exploration → Extraction →
# Spectral analysis → Construct init → Full training
#
# Usage:
#   bash run_pipeline.sh
#
# All intermediate data is stored under GALI_DATA_DIR (default: ./gali_data/)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../../.."  # repo root

# ---------- Configuration ----------
export MODEL_PATH="${MODEL_PATH:-/mnt/llm-train-5p/shenzhennan/models/DeepSeek-R1-Distill-Qwen-1.5B}"
export GALI_DATA_DIR="${GALI_DATA_DIR:-$(pwd)/gali_data}"
export K="${K:-5}"           # Exploration rounds
export N="${N:-20}"          # Steps per round
export NNODES="${NNODES:-4}"

LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
ETA="${ETA:-0.7}"            # Concentration parameter
METHOD="${METHOD:-sampling}" # sampling | deterministic
SEED="${SEED:-42}"

SVD_DIR="${GALI_DATA_DIR}/svd_cache"
DELTA_DIR="${GALI_DATA_DIR}/delta_w"
PROJ_DIR="${GALI_DATA_DIR}/projections"
INIT_DIR="${GALI_DATA_DIR}/gali_init"
PLOT_DIR="${GALI_DATA_DIR}/plots"

echo "============================================"
echo "  GALI Pipeline"
echo "  Model:  ${MODEL_PATH}"
echo "  Rank:   ${LORA_RANK}, Alpha: ${LORA_ALPHA}"
echo "  K=${K} rounds × N=${N} steps"
echo "  η=${ETA}, method=${METHOD}"
echo "  Data:   ${GALI_DATA_DIR}"
echo "============================================"

# ---------- Phase 0: SVD Precomputation (CPU) ----------
echo ""
echo "=== Phase 0: SVD Precomputation ==="
python3 "${SCRIPT_DIR}/phase0_svd_precompute.py" \
    --model_path "${MODEL_PATH}" \
    --output_dir "${SVD_DIR}"

# ---------- Phase 1: Exploration (GPU) ----------
echo ""
echo "=== Phase 1: Multi-round Exploration (K=${K}, N=${N}) ==="
bash "${SCRIPT_DIR}/phase1_explore.sh"

# ---------- Phase 1.5: Extract δW (CPU) ----------
echo ""
echo "=== Phase 1.5: Extract Weight Deltas ==="
python3 "${SCRIPT_DIR}/phase1_extract_delta.py" \
    --model_path "${MODEL_PATH}" \
    --explore_dir "${GALI_DATA_DIR}/explorations" \
    --output_dir "${DELTA_DIR}" \
    --lora_rank "${LORA_RANK}"

# ---------- Phase 2: Spectral Analysis (CPU) ----------
echo ""
echo "=== Phase 2: Spectral Projection Analysis ==="
python3 "${SCRIPT_DIR}/phase2_spectral_analysis.py" \
    --svd_dir "${SVD_DIR}" \
    --delta_dir "${DELTA_DIR}" \
    --output_dir "${PROJ_DIR}" \
    --aggregation consistency

# ---------- Visualization (CPU, optional) ----------
echo ""
echo "=== Visualization ==="
python3 "${SCRIPT_DIR}/visualize_projections.py" \
    --projection_dir "${PROJ_DIR}" \
    --output_dir "${PLOT_DIR}" || echo "(visualization skipped, matplotlib not available)"

# ---------- Phase 3: Construct Initialization (CPU) ----------
echo ""
echo "=== Phase 3: Construct GALI Initialization ==="
python3 "${SCRIPT_DIR}/phase3_construct_init.py" \
    --model_path "${MODEL_PATH}" \
    --svd_dir "${SVD_DIR}" \
    --projection_dir "${PROJ_DIR}" \
    --output_dir "${INIT_DIR}" \
    --lora_rank "${LORA_RANK}" \
    --lora_alpha "${LORA_ALPHA}" \
    --eta "${ETA}" \
    --method "${METHOD}" \
    --seed "${SEED}"

# ---------- Phase 4: Full Training (GPU) ----------
echo ""
echo "=== Phase 4: Full RL Training with GALI Init ==="
bash "${SCRIPT_DIR}/phase4_train.sh"

echo ""
echo "============================================"
echo "  GALI Pipeline Complete"
echo "============================================"
