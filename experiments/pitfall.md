# Pitfalls & Fixes

记录实验过程中遇到的坑和解决方案。

---

## 1. sglang LoRA adapter 模式 KeyError: 'weights'

**现象**：训练第一步 `update_weights` 时 sglang scheduler 报 `KeyError: 'weights'` 然后 abort，后续全部 CUDA error。

**根因**：

verl 的 colocate 架构中，推理引擎在 actor 训练时会 "sleep"（释放显存）。LoRA adapter 模式下 (`sleep_level=1`)，sleep 只释放 kv_cache，base weights 常驻 GPU 不动。但 `engine_workers.py:update_weights()` 里无条件调用了 `rollout.resume(tags=["weights"])`，sglang 那边发现 `offload_tags` 里根本没有 `'weights'`（因为从没被 offload 过），直接 KeyError。

**调用链**：
```
engine_workers.py:update_weights()
  → rollout.resume(tags=["weights"])
  → sglang_rollout.resume() → engine.resume_memory_occupation(tags=["weights"])
  → scheduler_update_weights_mixin.py:163: self.offload_tags.remove('weights')  ← KeyError
```

**细节**：
- 第一次 `update_weights` 时 `sleep_level` 还是默认的 2（weights 确实被 offload 了），没问题
- 第一次之后 `sleep_level` 被设成 1（701 行），后续 `release()` 只 offload kv_cache
- 第二次 `update_weights` 时再 resume weights 就炸了

**修复** (`verl/workers/engine_workers.py`):

```python
# Before (line 685-687):
if self.config.rollout.free_cache_engine:
    await self.rollout.resume(tags=["weights"])

# After:
if self.config.rollout.free_cache_engine and getattr(self.rollout, "sleep_level", 2) != 1:
    await self.rollout.resume(tags=["weights"])
```

**影响范围**：所有 LoRA `merge=False`（adapter 模式）+ colocate 架构 + sglang/vllm 的组合。

---

## 2. sglang LoRA target_modules="all-linear" 不支持

**现象**：`NotImplementedError: get_hidden_dim not implemented for n/e/-/l/i`

**根因**：sglang 把 `"all-linear"` 字符串当 iterable 遍历了每个字符。

**修复**：显式指定 target_modules 列表：
```
actor_rollout_ref.model.target_modules=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]
```

---

## 3. reward config 新旧 API 不兼容

**现象**：`KeyError` 或 hydra 报 config key 不存在。

**根因**：verl 有新旧两套 reward config：
- 旧：`reward_model.reward_manager=dapo` + `+reward_model.reward_kwargs.*`
- 新（recipe/dapo 用的）：`reward.reward_manager.name=dapo` + `reward.reward_kwargs.*`

`recipe/dapo/main_dapo.py` 会调用 `migrate_legacy_reward_impl()` 做桥接，但如果混用两套 key 就会出问题。

**修复**：统一用新 API（参考 `recipe/dapo/config/dapo_trainer.yaml`）。

---

## 4. wandb 认证在 Ray workers 上失败

**现象**：`wandb: ERROR Error while calling W&B API: user is not logged in`

**根因**：Ray workers 不继承主进程的环境变量，只 `export WANDB_API_KEY` 不够。

**修复**：在脚本开头加 `wandb login`，写到 `~/.netrc` 让所有 worker 共享：
```bash
wandb login --relogin --host "${WANDB_BASE_URL}" "${WANDB_API_KEY}" 2>/dev/null || true
```

---

## 5. vLLM + LoRA adapter 模式 base weight 全部丢失（模型输出乱码）

**现象**：模型输出全是乱码（多语言 Unicode 混合），val_before_train 阶段 acc=0%，DAPO filter_groups 找不到 mixed accuracy group 报错。

**根因**：

LoRA `merge=False` + `load_format=dummy` 时，vLLM 先加载随机权重，然后通过 `update_weights` 分两步同步：
1. **Base sync**（`base_sync_done=False`）：发送 399 个基座权重
2. **Adapter sync**（`base_sync_done=True`）：发送 LoRA adapter 权重

问题出在 Base sync：`transformer_impl.py:get_per_tensor_param()` 调用 `replace_lora_wrapper()` 把 weight key 从标准 HF 格式改成了带 `.base_layer.` 的格式：

```
原始：model.layers.0.self_attn.q_proj.weight
改后：model.layers.0.self_attn.q_proj.base_layer.weight
```

这是给 SGLang 设计的（SGLang LoRA 模型内部用 `.base_layer.` 命名）。但 vLLM 的 `model.load_weights()` 期望标准 HF key，不认 `.base_layer.`，静默跳过了 252 个核心权重（所有 attention + MLP proj），这些层保持 dummy 随机权重。

**修复** (`verl/workers/rollout/vllm_rollout/utils.py`):

在 vLLM 的 `_update_weights` 非 adapter 路径中 strip `.base_layer.`：
```python
weights = [(name.replace(".base_layer.", "."), tensor) for name, tensor in weights]
```

同时修复了 `vLLMColocateWorkerExtension` 和 `vLLMOmniColocateWorkerExtension` 两个类。

**影响范围**：所有 vLLM + LoRA `merge=False` + `load_format=dummy`（colocate 架构默认配置）。

---

## 6. 训练/评测数据缺少 instruction prefix 导致模型输出不按格式、打分全错

**现象**：val accuracy 极低（~0.83%），模型输出有推理过程但不输出 `Answer: \boxed{...}` 格式。

**根因**：

SFT 模型是用带 instruction prefix 的数据训练的（`"Solve the following math problem step by step. The last line of your response should be of the form Answer: \boxed{$Answer}..."`），但 DAPO 训练/评测数据的 prompt 是纯数学题，没有这个前缀。模型没看到指令就不按格式输出，打分函数提取不到答案。

**修复**：

转换数据集，在 prompt 前加上 instruction prefix：
- 训练集：`/mnt/llm-train-5p/shenzhennan/datasets/dapo-math-17k-boxed/train.parquet`
- 评测集：`/mnt/llm-train-5p/shenzhennan/datasets/aime-2024-boxed/aime-2024.parquet`

修复后 val accuracy 从 0.83% → 43.3%。

---

## 7. compute_score 截断 + 多策略 fallback 导致打分不准

**现象**：原始 `math_dapo.compute_score` 只看最后 300 字符（`solution_str[-300:]`），长 thinking 响应的 `Answer: \boxed{73}` 在更前面就被截掉了。

**根因**：

1. **`[-300:]` 硬截断**：长 CoT 响应里答案不在最后 300 字符内
2. **多策略 fallback 不合理**：如果 `\boxed{}` 提取到但答案错了，不应该再 fallback 去 `Answer:` 正则碰运气

**修复** (`verl/utils/reward_score/math_dapo.py`):

改为单一策略，与 deepscaler 一致：
```python
# 1. 用 </think> 分割取答案部分
if "</think>" in solution_str:
    answer_part = solution_str.split("</think>")[-1]
else:
    answer_part = solution_str

# 2. 从中提取 \boxed{}，提取不到就是错
boxed_str = last_boxed_only_string(answer_part)
# 3. normalize 比较
```

没有 fallback 链，一条路走到底。

---

## 8. compute_score 返回 `pred: None` 导致 validation metrics 崩溃

**现象**：`TypeError: unsupported operand type(s) for +: 'NoneType' and 'str'` in `metric_utils.py`

**根因**：

`compute_score` 在找不到 `\boxed{}` 时返回 `{"pred": None}`。`metric_utils.py:process_validation_metrics` 遍历所有 key 计算统计量，640 行用 `isinstance(var_vals[0], str)` 跳过字符串类型，但 `None` 不是 str，没被跳过，进入 `np.mean([None, ...])` 崩溃。

**修复**：`pred` 永远返回字符串：
```python
# 找不到 \boxed{}
return {"score": 0.0, "acc": False, "pred": "[NO_BOXED]"}
# 解析失败
return {"score": 0.0, "acc": False, "pred": "[PARSE_ERR]"}
```

---

## 9. DoRA magnitude vector dtype 不匹配导致 FSDP 训练崩溃

**现象**：两个阶段分别报错：
1. `ValueError: Requires uniform dtype across all gradients but got {torch.float32, torch.bfloat16}`（clip_grad_norm_）
2. `RuntimeError: Tensors of the same index must be on the same device and the same dtype`（Adam optimizer.step()）

**根因**：

DoRA 在每个 target module 上额外创建一个 `lora_magnitude_vector`（`nn.Parameter`），值通过 `torch.linalg.norm(W, dim=1)` 计算。`linalg.norm` 会将 bf16 权重提升到 fp32 再算范数，结果 magnitude vector 以 fp32 初始化。而 FSDP mixed precision 要求所有参数同 dtype（bf16），clip_grad_norm_ 和 optimizer 在遇到 fp32/bf16 混合时直接报错。

**修复** (`verl/workers/engine/fsdp/transformer_impl.py`):

在 `_build_model_optimizer` 中，LoRA module 构建完成后、FSDP wrap 之前，强制转换 dtype：
```python
if self._is_lora:
    module = self._build_lora_module(module)
    if self.model_config.use_dora:
        module = module.to(torch.bfloat16)
```

同时需要在 `_build_lora_module` 的 `LoraConfig` 中传入 `use_dora`：
```python
lora_config = {
    ...
    "use_dora": self.model_config.use_dora,
}
```

以及在 config 中新增字段 (`verl/workers/config/model.py`):
```python
use_dora: bool = False
```

**DoRA 前向公式**：
$$W' = m \cdot \frac{W_0 + \alpha BA}{\|W_0 + \alpha BA\|_c}$$

其中 $m$ 是可学习的 magnitude vector，$BA$ 是 LoRA 低秩矩阵，$\|.\|_c$ 是列方向 L2 范数。方向由 $W_0 + BA$ 控制，幅度由 $m$ 单独控制，两者解耦学习。

**注意**：`module.to(bf16)` 会把 magnitude vector 的精度从 fp32 降到 bf16。理论上 bf16 有效精度约 7 位，对于 norm 值通常足够，但如果发现训练不稳定可考虑用 fp32 magnitude + 手动梯度 cast 的方案。

**影响范围**：所有 DoRA + FSDP mixed precision 的组合。
