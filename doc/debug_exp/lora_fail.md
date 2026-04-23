# LoRA DAPO 训练失败 Debug 记录

> Model: DeepSeek-R1-Distill-Qwen-1.5B  
> Task: DAPO (GRPO) on dapo-math-17k  
> Cluster: 4×8 H200  
> Framework: verl (FSDP + vLLM rollout)  
> Date: 2026-04-23

## 1. 现象

所有 LoRA lr=1e-5 实验在 **wall-step ~128** 附近出现 **entropy 起飞 + ppo_kl 激增**，随后训练崩溃。  
LoRA lr=1e-6 稳定但 eval 涨势差，远不如同配置 full FT。

## 2. 实验矩阵

| # | 方法 | lr | bsz/mini | α/r | dropout | wd | filter_groups | clip_high | 结果 |
|---|---|---|---|---|---|---|---|---|---|
| E1 | LoRA | 1e-5 | 64/16 | 64/32 | 0 | 0.1 | True | 0.28 | ~128 步崩，ppo_kl 激增 |
| E2 | LoRA | 1e-6 | 64/16 | 64/32 | 0 | 0.1 | True | 0.28 | 600 步稳定，eval 涨势差 |
| E3 | LoRA | 1e-5 | 128/128 | 64/32 | 0.05 | 0.1 | True | 0.28 | ~128 步崩，entropy 起飞 |
| E4 | **Full FT** | 1e-6 | 64/16 | — | — | 0.1 | True | 0.28 | **正常，reward+eval 涨势好** |
| E5 | LoRA | 1e-5 | 64/16 | 64/32 | 0.05 | **0.0** | True | 0.28 | 🔄 Running |

脚本路径：
- E1: `experiments/scripts/run_dapo_lora_baseline_1.5b.sh`
- E2: E1 手动改 lr=1e-6
- E3: `experiments/scripts/run_dapo_lora_baseline_1.5b_lr1e-5_bsz_1024_d_0.05.sh`
- E4: `experiments/scripts/run_dapo_full_baseline_1.5b.sh`
- E5: `experiments/scripts/run_dapo_lora_baseline_1.5b_lr1e-5_bsz_512_d_0.05_wd_0.sh`

## 3. 排除的假设

### 3.1 merge=True 数值漂移 ❌ 已排除

**假设**: `lora.merge=True` 每步 merge/restore 在 bf16 下累积误差，导致 vLLM rollout 和 actor forward 分布渐行渐远。

**验证**: 在 pod 内加载 step 64 checkpoint（32 shards），重建完整权重并计算 merge-path vs forward-path 的 bf16 权重差异：

```
Base W norm:     81.46
LoRA delta norm: 0.021    (delta/base = 0.026%)
vLLM vs Actor bf16 weight diff:
  Mean abs:  1.1e-8   (< 1 bf16 ULP)
  Max abs:   2.4e-4
```

**结论**: backup/restore 在 fp32 下执行（checkpoint 存 fp32），merge 误差远小于 bf16 量化噪声。**不是原因。**

### 3.2 filter_groups 数据分布偏移 ❌ 可能性低

**假设**: filter_groups 过滤全对/全错 prompt 后训练分布渐变，导致 advantage 方差增大 → 训练不稳定。

**反驳**: Full FT (E4) 使用**完全相同的** filter_groups 设置（enable=True, metric=acc, max_gen=5），训练正常。如果 filter_groups 有问题，full FT 也应崩。

### 3.3 batch size / mini_batch 配置 ❌ 已排除

E1 和 E3 的 batch 配置差异很大：

| | E1 | E3 |
|---|---|---|
| train_prompt_bsz | 64 | 128 |
| mini_bsz | 16 | 128 |
| optim steps/rollout | 4 | 1 |
| 累积 optim steps @step128 | 512 | 128 |

两者都在 wall-step ~128 崩 → 和 optim step 数无关，和 wall-step 相关。

### 3.4 lora_dropout ❌ 无关

E1 (dropout=0) 和 E3 (dropout=0.05) 都崩。dropout=0.05 是 PeRL 论文推荐值。

## 4. 当前主要假设：LoRA 有效更新幅度过大

### 核心对比

```
Full FT (E4):  lr=1e-6, 梯度分散到 1.5B 参数, 无 scaling
LoRA (E1/E3): lr=1e-5, 梯度集中到 37M LoRA 参数, scaling=α/r=64/32=2
有效更新幅度比: (1e-5 × 2) / 1e-6 = 20x
```

LoRA 对输出的影响是 full FT 的 ~20 倍/step。配合 DAPO `clip_ratio_high=0.28`（比标准 PPO 的 0.2 更激进），policy 更新过猛。

### 支撑证据

1. Full FT 同配置（filter_groups=True, clip_high=0.28）完全正常 → 问题是 LoRA 特有的
2. LoRA lr=1e-6（有效 lr=2e-6）稳定但太慢 → 有效 lr 在 2e-6~2e-5 之间有一个稳定性阈值
3. PeRL 论文用 lr=1e-5 + α=64 + r=32 在 TRL 上工作正常，但 TRL 用标准 PPO clip=0.2，不用 DAPO 的 clip-higher
4. LoRA 梯度自带 scaling 倍数：`grad_LoRA ∝ scaling × grad_output` → 梯度本身就是 2x

### 为什么两个 lr=1e-5 实验都在 wall-step ~128 崩

模型在 ~128 rollout 后学到足够多 → LoRA 权重增长到某个阈值 → 单步 policy 变化（lr × scaling × grad）超出 clip 能约束的范围 → importance ratio 偏离 → ppo_kl 爆炸 → entropy 起飞。

lr=1e-6 的 LoRA 增长慢 10x，600 步时仍未到达该阈值。

## 5. 待验证实验

| # | 方案 | 改动 | 预期 | 状态 |
|---|---|---|---|---|
| E5 | 去 weight_decay | wd=0.0, 其余同 E1+dropout=0.05 | 和 PeRL 对齐 (PeRL wd=0.0) | 🔄 Running |
| E6 | 降 α | α=32 (scaling=1), 保持 lr=1e-5 | 有效 lr 降一半，梯度 scaling 也减半 | TODO |
| E7 | 降 lr | lr=5e-6, 保持 α=64 | 有效 lr=1e-5，比 E1 的 2e-5 低 | TODO |
| E8 | 收紧 clip | clip_ratio_high=0.2, 保持 lr=1e-5/α=64 | 限制单步 policy 更新上限，和 PeRL 对齐 | TODO |
| E9 | 关 filter | filter_groups=False, 保持其余 | 对照实验 | TODO |

**推荐优先级**: E5(running) → E6(降α) → E8(收紧clip) → E7(降lr) → E9

### weight_decay 分析

**PeRL 没有设 weight_decay**（TRL 默认 = 0.0），verl 唯一的 LoRA 实际训练脚本（flowgrpo）用 0.0001。
我们的脚本照搬 full FT 的 0.1。

AdamW weight_decay 对 LoRA 的效应：
- weight_decay 把参数往 0 拉（即往初始 SFT 状态回退）
- RL 梯度把参数往 reward 方向推
- 在 LoRA 的 rank-32 低维子空间里，这两个力方向相反 → 可能造成震荡
- full FT 有 1.5B 参数分散这个拉锯力，LoRA 只有 37M 参数承受

### PeRL clip_ratio_high 确认

PeRL 的 `epsilon_high=0.2`（标准 PPO），**不是 DAPO 的 0.28**。
这是 PeRL 能用 lr=1e-5 + α=64 + r=32 跑稳的重要因素之一。

### 降 α=32 (E6) 推荐理由
- 同时降低梯度 scaling 和输出 scaling，效果最彻底
- 不改 lr（方便和 E1/E2 对比）
- 保留 DAPO clip-higher 特性

## 6. 参考

- PeRL 论文 setting: lr=1e-5, r=32, α=64, dropout=0.05, **wd=0.0**, bsz=128, grad_accum=8, **clip=0.2 (标准 PPO)**
- verl 官方 LoRA test 脚本 (`recipe/dapo/test_dapo_7b_math_lora.sh`): lr=1e-6, r=8, wd=0.1, 不开 filter_groups
- verl flowgrpo LoRA 脚本: **wd=0.0001**（唯一低 wd 的 LoRA 例子）
- DAPO 论文: clip_ratio_low=0.2, clip_ratio_high=0.28 (asymmetric clipping)

## 7. 技术细节

### Checkpoint 格式
- FSDP DTensor, Shard(dim=0), world_size=32
- 权重存储为 **fp32**（optimizer master copy）
- LoRA 占比: ~2% 参数量 (37M / 1.5B)
- Step 64 时 LoRA B norm: ~0.003 (absmax ~0.0003), 极小

### verl merge=True 路径
```
get_per_tensor_param():
  1. backup_base_model_weights() → clone fp32 base to CPU (exact)
  2. fsdp_merge_unmerge(merge=True) → base += scaling*B@A (fp32, inside summon_full_params)
  3. state_dict() → get merged weights
  4. cast to bf16 → send to vLLM
  5. restore_base_model_weights() → copy fp32 backup back (exact)
```

### verl ref policy 路径
```
compute_log_prob(is_lora=True):
  with model.disable_adapter():  # 关掉 LoRA，用 base 算 ref_log_prob
      ref_log_prob = forward(data)
  # 但 kl_coef=0, use_kl_loss=False → ref 不参与 loss
```
