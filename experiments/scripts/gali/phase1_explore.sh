#!/usr/bin/env bash
# =============================================================================
# GALI Phase 1: Multi-round LoRA exploration
#
# Runs K short RL training rounds (N steps each) with different random LoRA
# initializations. Each round saves an HF-format merged checkpoint that
# phase1_extract_delta.py will diff against the original model.
#
# Usage:
#   K=5 N=20 bash phase1_explore.sh
#
# Requires:
#   - Ray cluster already running
#   - Environment variables set (WANDB_*, model/data paths)
# =============================================================================
set -xeuo pipefail

# ---------- GALI exploration hyperparams ----------
K=${K:-5}             # Number of exploration rounds
N=${N:-20}            # Steps per round

# ---------- Paths (edit these) ----------
MODEL_PATH="${MODEL_PATH:-/mnt/llm-train-5p/shenzhennan/models/DeepSeek-R1-Distill-Qwen-1.5B}"
TRAIN_FILE="${TRAIN_FILE:-/mnt/llm-train-5p/shenzhennan/datasets/dapo-math-profilled-boxed/train.parquet}"
TEST_FILE="${TEST_FILE:-/mnt/llm-train-5p/shenzhennan/datasets/aime-2024-boxed/aime-2024.parquet}"
GALI_DATA_DIR="${GALI_DATA_DIR:-$(pwd)/gali_data}"
EXPLORE_DIR="${GALI_DATA_DIR}/explorations"
mkdir -p "${EXPLORE_DIR}"

# ---------- Cluster ----------
NNODES=${NNODES:-4}

# ---------- Env ----------
export HYDRA_FULL_ERROR=1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../../.env" 2>/dev/null || true
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
if [ -n "${WANDB_API_KEY}" ] && [ -n "${WANDB_BASE_URL}" ]; then
    wandb login --relogin --host "${WANDB_BASE_URL}" "${WANDB_API_KEY}" 2>/dev/null || true
fi

# ---------- Training config (mirrors baseline) ----------
max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 28))
train_prompt_bsz=64
gen_prompt_bsz=$((train_prompt_bsz * 2))
n_resp_per_prompt=8
train_prompt_mini_bsz=16
gen_tp=1

# ---------- Run K exploration rounds ----------
for k in $(seq 1 $K); do
    exp_name="gali_explore_round_${k}"
    CKPT_DIR="${EXPLORE_DIR}/${exp_name}"
    mkdir -p "${CKPT_DIR}"

    echo "========================================"
    echo "  GALI Exploration Round ${k}/${K}"
    echo "  Steps: ${N}, Output: ${CKPT_DIR}"
    echo "========================================"

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
        actor_rollout_ref.model.lora_rank=32 \
        actor_rollout_ref.model.lora_alpha=64 \
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
        trainer.default_local_dir="${CKPT_DIR}" \
        trainer.resume_mode=disable \
        2>&1 | tee "${CKPT_DIR}/train_log.txt"

    echo "  Round ${k} done. Checkpoint at ${CKPT_DIR}/"
done

echo ""
echo "All ${K} exploration rounds complete."
echo "Run phase1_extract_delta.py to extract weight deltas."
