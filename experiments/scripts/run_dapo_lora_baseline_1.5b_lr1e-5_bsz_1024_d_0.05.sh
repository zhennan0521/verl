#!/usr/bin/env bash
# Based on: recipe/dapo/run_dapo_qwen2.5_32b.sh + test_dapo_7b_math_lora.sh
# Changes: model→Qwen3-8B-sft-dolci-think, +LoRA r32/a64, vllm tp=2, 4nodes, aime@32
set -xeuo pipefail

export HYDRA_FULL_ERROR=1
# # Ray memory monitor sees cgroup limit (512GB) not real memory (2TB).
# # Disable it to prevent false OOM kills.
# export RAY_memory_monitor_refresh_ms=0

project_name='lora_rlvr'
exp_name="dapo-distilled-qwen-1.5b-sft-dolci-lora-r32-lr1e-5-d0.05-bsz1024_$(date +%m%d_%H%M)"

adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 28))
enable_overlong_buffer=False
overlong_buffer_len=$((1024 * 4))
overlong_penalty_factor=1.0

loss_agg_mode="token-mean"

enable_filter_groups=True
filter_groups_metric=acc
max_num_gen_batches=5
train_prompt_bsz=128
gen_prompt_bsz=$((train_prompt_bsz * 3))
n_resp_per_prompt=8
train_prompt_mini_bsz=128

# Cluster
NNODES=${NNODES:-4}

# Paths
MODEL_PATH="/mnt/llm-train-5p/shenzhennan/models/DeepSeek-R1-Distill-Qwen-1.5B"
TRAIN_FILE="/mnt/llm-train-5p/shenzhennan/datasets/dapo-math-17k-profilled-1.5b-0.7/train.parquet"
TEST_FILE="/mnt/llm-train-5p/shenzhennan/datasets/aime-2024-boxed/aime-2024.parquet"
CKPTS_DIR="/mnt/llm-train-5p/shenzhennan/verl/experiments/results/${project_name}/${exp_name}"
mkdir -p "${CKPTS_DIR}"

# Env
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../.env" 2>/dev/null || true
export WANDB_API_KEY="${WANDB_API_KEY}"
export WANDB_BASE_URL="${WANDB_BASE_URL}"
export WANDB_ENTITY="${WANDB_ENTITY}"
unset HTTP_PROXY
unset HTTPS_PROXY
unset http_proxy
unset https_proxy

# # vLLM + LoRA: disable torch.compile to avoid flashinfer workspace init error during CUDA graph capture
# export VLLM_USE_V1=1
# export VLLM_TORCH_COMPILE_LEVEL=0

# Wandb login (writes to ~/.netrc, so Ray workers can pick it up)
wandb login --relogin --host "${WANDB_BASE_URL}" "${WANDB_API_KEY}" 2>/dev/null || true

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1
val_top_p=0.7

# Performance
sp_size=1
use_dynamic_bsz=True
actor_ppo_max_token_len=$((max_prompt_length + max_response_length))
infer_ppo_max_token_len=$((max_prompt_length + max_response_length))
offload=True
gen_tp=1

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
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=64 \
    +actor_rollout_ref.model.lora_dropout=0.05 \
    actor_rollout_ref.model.lora.merge=True \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.80 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=32 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    reward.reward_manager.name=dapo \
    reward.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    reward.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    +reward.reward_kwargs.max_resp_len=${max_response_length} \
    trainer.validation_data_dir="${CKPTS_DIR}/val_generations" \
    trainer.logger='["console","wandb","tensorboard","file"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=True \
    trainer.test_freq=64 \
    trainer.save_freq=64 \
    trainer.total_epochs=100 \
    trainer.total_training_steps=1024 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    trainer.log_val_generations=10 \
    trainer.validation_data_dir="${CKPTS_DIR}/val_generations" \
    2>&1 | tee "${CKPTS_DIR}/train_log_$(date +%Y%m%d_%H%M%S).txt"
