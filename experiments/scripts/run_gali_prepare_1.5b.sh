#!/usr/bin/env bash
# =============================================================================
# GALI Preparation: Phase 0 → 1 → 1.5 → 2 → 3
#
# Step 1 (GPU): K rounds of short LoRA-RL exploration, each N steps, saves
#   merged HF checkpoints so we can diff δW = W_merged - W_0.
#
# Step 2 (CPU): SVD precompute, δW extraction, spectral projection analysis,
#   construct GALI adapter + frozen base model.
#
# After this finishes, run run_gali_train_1.5b.sh for full training.
#
# Usage:
#   bash run_gali_prepare_1.5b.sh                # defaults: K=5 rounds × N=20 steps
#   K=3 N=10 bash run_gali_prepare_1.5b.sh       # override exploration config
# =============================================================================
set -xeuo pipefail

export HYDRA_FULL_ERROR=1

# ==================== Configurable parameters ====================
# GALI exploration
K=${K:-5}                       # Number of exploration rounds
N=${N:-20}                      # Steps per exploration round

# LoRA hyperparams (same as baseline)
LORA_RANK=${LORA_RANK:-32}
LORA_ALPHA=${LORA_ALPHA:-64}

# GALI hyperparams
ETA=${ETA:-0.7}                 # Concentration param (0=uniform, 1=pure importance)
METHOD=${METHOD:-sampling}      # sampling | deterministic
SEED=${SEED:-42}

# Cluster
NNODES=${NNODES:-4}

# Paths
MODEL_PATH="/mnt/llm-train-5p/shenzhennan/models/DeepSeek-R1-Distill-Qwen-1.5B"
TRAIN_FILE="/mnt/llm-train-5p/shenzhennan/datasets/dapo-math-17k-profilled-1.5b-0.7/train.parquet"
TEST_FILE="/mnt/llm-train-5p/shenzhennan/datasets/aime-2024-boxed/aime-2024.parquet"

GALI_DATA_DIR="/mnt/llm-train-5p/shenzhennan/verl/experiments/results/gali_data"
SVD_DIR="${GALI_DATA_DIR}/svd_cache"
EXPLORE_DIR="${GALI_DATA_DIR}/explorations"
DELTA_DIR="${GALI_DATA_DIR}/delta_w"
PROJ_DIR="${GALI_DATA_DIR}/projections"
INIT_DIR="${GALI_DATA_DIR}/gali_init"
PLOT_DIR="${GALI_DATA_DIR}/plots"

mkdir -p "${GALI_DATA_DIR}"

# Env
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GALI_SCRIPT_DIR="${SCRIPT_DIR}/gali"
source "${SCRIPT_DIR}/../../.env" 2>/dev/null || true
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy 2>/dev/null || true

if [ -n "${WANDB_API_KEY}" ] && [ -n "${WANDB_BASE_URL}" ]; then
    wandb login --relogin --host "${WANDB_BASE_URL}" "${WANDB_API_KEY}" 2>/dev/null || true
fi

TARGET_MODULES="q_proj k_proj v_proj o_proj gate_proj up_proj down_proj"

# Training config (mirrors lora baseline)
max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 28))
train_prompt_bsz=64
gen_prompt_bsz=$((train_prompt_bsz * 2))
n_resp_per_prompt=8
train_prompt_mini_bsz=16
gen_tp=1

echo "============================================"
echo "  GALI Preparation Pipeline"
echo "  Model:  ${MODEL_PATH}"
echo "  Rank:   ${LORA_RANK}, Alpha: ${LORA_ALPHA}"
echo "  K=${K} rounds × N=${N} steps"
echo "  η=${ETA}, method=${METHOD}, seed=${SEED}"
echo "  Data:   ${GALI_DATA_DIR}"
echo "============================================"

# =====================================================================
# Phase 0: SVD Precomputation (CPU, ~5min for 1.5B)
# =====================================================================
echo ""
echo "========== Phase 0: SVD Precomputation =========="
python3 "${GALI_SCRIPT_DIR}/phase0_svd_precompute.py" \
    --model_path "${MODEL_PATH}" \
    --output_dir "${SVD_DIR}" \
    --target_modules ${TARGET_MODULES}

# =====================================================================
# Phase 1: Multi-round LoRA exploration (GPU)
#   Each round: random LoRA init → N steps GRPO → save merged HF checkpoint
# =====================================================================
echo ""
echo "========== Phase 1: Multi-round Exploration (K=${K}, N=${N}) =========="
mkdir -p "${EXPLORE_DIR}"

for k in $(seq 1 $K); do
    exp_name="gali_explore_round_${k}"
    ROUND_DIR="${EXPLORE_DIR}/${exp_name}"
    mkdir -p "${ROUND_DIR}"

    echo ""
    echo "-------- Exploration Round ${k}/${K} --------"
    echo "  Steps: ${N}, Output: ${ROUND_DIR}"

    python3 -m recipe.dapo.main_dapo \
        data.train_files="${TRAIN_FILE}" \
        data.val_files="${TEST_FILE}" \
        data.prompt_key=prompt \
        data.truncation='left' \
        data.max_prompt_length=${max_prompt_length} \
        data.max_response_length=${max_response_length} \
        data.gen_batch_size=${gen_prompt_bsz} \
        data.train_batch_size=${train_prompt_bsz} \
        actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
        algorithm.adv_estimator=grpo \
        algorithm.use_kl_in_reward=False \
        algorithm.kl_ctrl.kl_coef=0.0 \
        actor_rollout_ref.actor.use_kl_loss=False \
        actor_rollout_ref.actor.kl_loss_coef=0.0 \
        actor_rollout_ref.actor.clip_ratio_low=0.2 \
        actor_rollout_ref.actor.clip_ratio_high=0.28 \
        actor_rollout_ref.actor.clip_ratio_c=10.0 \
        algorithm.filter_groups.enable=True \
        algorithm.filter_groups.max_num_gen_batches=5 \
        algorithm.filter_groups.metric=acc \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.use_dynamic_bsz=True \
        actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
        actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((max_prompt_length + max_response_length)) \
        actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$((max_prompt_length + max_response_length)) \
        actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$((max_prompt_length + max_response_length)) \
        actor_rollout_ref.model.path="${MODEL_PATH}" \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.model.lora_rank=${LORA_RANK} \
        actor_rollout_ref.model.lora_alpha=${LORA_ALPHA} \
        actor_rollout_ref.model.lora.merge=True \
        actor_rollout_ref.actor.optim.lr=1e-5 \
        actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
        actor_rollout_ref.actor.optim.weight_decay=0.1 \
        actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
        actor_rollout_ref.actor.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
        actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.actor.grad_clip=1.0 \
        actor_rollout_ref.actor.loss_agg_mode=token-mean \
        actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.80 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
        actor_rollout_ref.rollout.enable_chunked_prefill=True \
        actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
        actor_rollout_ref.rollout.temperature=1.0 \
        actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
        actor_rollout_ref.rollout.top_p=1.0 \
        actor_rollout_ref.rollout.top_k=-1 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
        reward.reward_manager.name=dapo \
        reward.reward_kwargs.overlong_buffer_cfg.enable=False \
        +reward.reward_kwargs.max_resp_len=${max_response_length} \
        actor_rollout_ref.actor.checkpoint.save_contents='[model,hf_model]' \
        trainer.logger='["console","file"]' \
        trainer.project_name="gali_explore" \
        trainer.experiment_name="${exp_name}" \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes="${NNODES}" \
        trainer.val_before_train=False \
        trainer.test_freq=99999 \
        trainer.save_freq=${N} \
        trainer.total_epochs=100 \
        trainer.total_training_steps=${N} \
        trainer.default_local_dir="${ROUND_DIR}" \
        trainer.resume_mode=disable \
        2>&1 | tee "${ROUND_DIR}/train_log.txt"

    echo "  Round ${k} done."
done
echo ""
echo "All ${K} exploration rounds complete."

# =====================================================================
# Phase 1.5: Extract δW from merged checkpoints (CPU)
# =====================================================================
echo ""
echo "========== Phase 1.5: Extract Weight Deltas =========="
python3 "${GALI_SCRIPT_DIR}/phase1_extract_delta.py" \
    --model_path "${MODEL_PATH}" \
    --explore_dir "${EXPLORE_DIR}" \
    --output_dir "${DELTA_DIR}" \
    --lora_rank "${LORA_RANK}" \
    --target_modules ${TARGET_MODULES}

# =====================================================================
# Phase 2: Spectral Projection Analysis (CPU)
# =====================================================================
echo ""
echo "========== Phase 2: Spectral Projection Analysis =========="
python3 "${GALI_SCRIPT_DIR}/phase2_spectral_analysis.py" \
    --svd_dir "${SVD_DIR}" \
    --delta_dir "${DELTA_DIR}" \
    --output_dir "${PROJ_DIR}" \
    --aggregation consistency

# =====================================================================
# Visualization (optional, won't fail the pipeline)
# =====================================================================
echo ""
echo "========== Visualization =========="
python3 "${GALI_SCRIPT_DIR}/visualize_projections.py" \
    --projection_dir "${PROJ_DIR}" \
    --output_dir "${PLOT_DIR}" \
    2>&1 || echo "(visualization skipped — matplotlib not available)"

# =====================================================================
# Phase 3: Construct GALI Initialization (CPU)
# =====================================================================
echo ""
echo "========== Phase 3: Construct GALI Initialization =========="
python3 "${GALI_SCRIPT_DIR}/phase3_construct_init.py" \
    --model_path "${MODEL_PATH}" \
    --svd_dir "${SVD_DIR}" \
    --projection_dir "${PROJ_DIR}" \
    --output_dir "${INIT_DIR}" \
    --lora_rank "${LORA_RANK}" \
    --lora_alpha "${LORA_ALPHA}" \
    --eta "${ETA}" \
    --method "${METHOD}" \
    --seed "${SEED}"

echo ""
echo "============================================"
echo "  GALI Preparation Complete!"
echo "============================================"
echo "  Adapter:    ${INIT_DIR}/adapter"
echo "  Base model: ${INIT_DIR}/base_model"
echo ""
echo "  Next: bash run_gali_train_1.5b.sh"
echo "============================================"
