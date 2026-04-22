# RL场景下LoRA最优初始化方法：从GASD分析到梯度引导的自适应初始化

## 目录

1. [背景与动机](#1-背景与动机)
2. [GASD分析的核心发现：RL训练的谱动力学](#2-gasd分析的核心发现)
3. [LoRA初始化实验：现有方法在RL中的表现](#3-lora初始化实验)
4. [问题诊断：为什么现有初始化在RL中失败](#4-问题诊断)
5. [相关工作：LoRA-GA及其局限性](#5-相关工作lora-ga)
6. [我们的方法：梯度引导的自适应LoRA初始化（GALI）](#6-我们的方法)
7. [详细算法](#7-详细算法)
8. [理论分析](#8-理论分析)
9. [实验计划](#9-实验计划)
10. [总结](#10-总结)

---

## 1. 背景与动机

### 1.1 LoRA在RL中的必要性

大模型的RL训练（GRPO/PPO/DAPO等）的显存开销远超SFT：需要同时维护policy模型、reference模型、value模型，外加rollout的KV cache。全参训练在中大规模模型（7B+）上对显存要求极高。LoRA通过将权重更新限制在低秩子空间 $W' = W_0 + \frac{\alpha}{r} BA$ 内，大幅降低可训练参数量和优化器状态的显存占用，是RL场景下的刚需。

### 1.2 核心矛盾

LoRA的低秩约束意味着它只能在一个 rank-r 的子空间内更新。**初始化决定了这个子空间的"起点"**——虽然训练过程中A和B会调整，但初始子空间的选择对收敛速度和最终性能有决定性影响。

在SFT场景下，这个问题已经被较好地研究（PiSSA、LoRA-GA等）。但**RL场景下的最优初始化问题几乎未被探索**，而且我们的实验表明RL与SFT的最优初始化策略截然不同。

---

## 2. GASD分析的核心发现

### 2.1 Three-Gate Theory与谱动力学

通过对Qwen3-8B在DAPO/GRPO训练过程中的逐步权重分析，我们建立了GASD（Geometry-Aware Steepest Descent）分析框架，核心发现如下：

**发现1：主成分子空间极其稳定。**

所有优化器（Adam、Muon、GASD）下，W的top-k主成分子空间在整个RL训练过程中几乎不变：
- 主成分角度偏移 < 2 度
- 子空间重叠度 > 0.998
- NSS（Normalized Spectral Shift）~ 0.0001

这意味着RL训练并不修改W的"骨架结构"，而是在非主成分方向上做微调。

**发现2：RL的有效更新主要发生在非主成分（off-principal）方向。**

Wedin的 $\sin\Theta$ 定理给出了直觉：
$$\sin\Theta_k \leq \frac{\|\Delta W\|_2}{\gamma_k}, \quad \gamma_k = |\sigma_k(W_0) - \sigma_{k+1}(W_0)|$$

- 大gap方向（principal）：子空间旋转被强约束，"代价"高
- 小gap方向（off-principal）：约束弱，更新空间大

实验验证：在所有优化器下，ΔW的能量重心（energy centroid）都偏向tail方向（centroid > 0.5），Spearman相关系数为负（ΔW能量与W0奇异值负相关）。

**发现3：RL梯度本质上是高噪声的。**

RL梯度 = 微弱信号 + 大噪声：
- 真实信号强度：$O(1/\text{batch\_size})$
- 采样噪声强度：$O(1)$
- 学习通过N步累积实现：信号 $O(N)$，噪声 $O(\sqrt{N})$

表现为：连续两步梯度的cosine similarity为负（-0.05 ~ -0.20），这不是bug，而是RL梯度的固有特征。所有方向的漂移都呈 $\sqrt{t}$ 增长（噪声主导），不存在纯信号方向。

**发现4：GASD优化器验证了谱调制的有效性。**

GASD优化器通过 $\Delta W = -(WW^\top + \epsilon I)^{-1} G$ 实现谱感知更新：
- 前~100步：尾部有未开发信号，GASD的 $1/\epsilon$ 放大快速收割 → reward涨得快
- 后期：尾部信号耗尽，主成分被 $1/\sigma^2$ 压制 → 停滞

这证明了RL训练中确实存在"先收割off-principal信号，再缓慢积累principal信号"的动态过程。

### 2.2 对LoRA初始化的启示

以上发现对LoRA初始化的直接推论：

1. **LoRA的rank-r子空间应该覆盖off-principal方向**——因为RL的有效更新主要在那里
2. **不能只覆盖off-principal**——主成分方向的微弱信号虽然每步很小，但通过长期累积也很重要；只覆盖off-principal会导致主成分方向完全无法学习
3. **最优覆盖是task-dependent的**——具体哪些off-principal方向最重要取决于RL任务和优化器
4. **由于RL梯度噪声极大，单步梯度信息不足以确定最优子空间**——需要多次采样来去噪

---

## 3. LoRA初始化实验

### 3.1 实验设置

- 模型：DeepSeek-R1-Distill-Qwen-1.5B
- 任务：GRPO on math reasoning
- LoRA配置：rank=16/32，target=all linear layers
- 对比方法：

| 方法 | 初始化策略 | 初始子空间 |
|------|-----------|-----------|
| LoRA (vanilla) | A=Kaiming, B=0 | 随机，训练中自由演化 |
| DoRA | LoRA + 方向/幅度解耦 | 同LoRA |
| PiSSA | A,B = W的top-r SVD | W的主成分方向 |
| MiLoRA | A,B = W的bottom-r SVD | W的尾部方向 |
| Full | 全参训练 | 无约束（上界参考） |

### 3.2 关键实验观察

**观察1：LoRA（随机初始化）在RL中表现尚可，PiSSA和MiLoRA不稳定。**

- LoRA：能正常训练，reward稳步上升，虽然比Full慢
- PiSSA：在RL中容易崩溃（reward振荡或下降）
- MiLoRA：同样不稳定，在某些配置下崩溃

这与SFT场景形成鲜明对比——在SFT中PiSSA通常优于vanilla LoRA。

**观察2：ΔW在W0谱基下的能量分布揭示了差异的根源。**

通过 `lora_solution_analysis.py` 和 `rl_lora_init_analysis.py` 的分析：

- Full训练的ΔW的能量重心偏向tail（centroid > 0.5），与W0奇异值负相关
- LoRA（随机初始化）的ΔW在W0谱基下分布相对均匀——**这正是它能work的原因**
- PiSSA的ΔW被限制在W0的top方向——而RL需要在off-principal方向更新，产生冲突
- MiLoRA的ΔW被限制在W0的bottom方向——虽然方向对了，但完全缺失对principal方向的覆盖

**观察3：不同初始化的ΔW最终不收敛到同一子空间。**

Pairwise cosine similarity和子空间重叠度分析显示，不同初始化策略的LoRA训练到最后的ΔW方向差异很大。这说明初始化不仅影响收敛速度，还影响最终解的质量。

### 3.3 核心结论

> **RL场景下LoRA初始化的关键不是"选W的哪些方向"，而是"选RL梯度实际需要的方向"。**
> 随机初始化之所以work，是因为它无差别地覆盖了所有方向（包括RL需要的off-principal方向），虽然不是最优但至少不会系统性地错过关键方向。PiSSA/MiLoRA之所以崩，是因为它们基于W的静态结构做选择，而这个选择与RL的动态需求不匹配。

---

## 4. 问题诊断

### 4.1 为什么PiSSA在RL中失败

PiSSA初始化 $A = V_{[1:r]}^\top$, $B = U_{[1:r]}$，将LoRA的全部rank-r表达能力集中在W的主成分方向。

在SFT中这是合理的：SFT的梯度高度低秩（论文LoRA-GA验证），且梯度的top方向与W的top方向有较大重叠。

在RL中这是致命的：
- RL的有效更新主要在off-principal方向（GASD发现）
- PiSSA的子空间与RL需要的更新方向几乎正交
- 训练过程中A和B试图旋转到正确的子空间，但这个旋转本身就需要梯度信号，而RL梯度噪声极大，导致旋转过程不稳定 → 崩溃

### 4.2 为什么MiLoRA在RL中也不稳定

MiLoRA取W的bottom-r方向，看似与RL的off-principal偏好一致，但：
- W的最小奇异值方向对应的是W中"最不重要"的结构，不等于RL最想修改的方向
- 完全缺失对principal方向的覆盖，而RL虽然主要更新off-principal，但也需要微量的principal方向调整
- 类比：GASD优化器也是放大off-principal方向，但后期因为完全压制principal方向而停滞——MiLoRA的问题本质相同

### 4.3 为什么随机初始化"还行"

随机初始化的AB的列空间均匀分布在所有方向上。这意味着：
- 对每个奇异方向都有 $O(r/d)$ 的覆盖
- 虽然每个方向的覆盖很弱，但不会系统性遗漏任何方向
- RL的梯度信号通过多步累积逐渐将子空间拉向正确方向

但随机初始化的问题是：
- rank-r的表达能力被均匀"撒"在所有方向上，对每个方向的表达能力都很弱
- 收敛速度慢——大量训练步被浪费在"子空间搜索"上
- 最终性能不如Full finetune

### 4.4 理想的初始化应该是什么样的

综合以上分析，理想的LoRA初始化应该同时满足：

1. **信号集中性**：将rank-r的大部分表达能力集中在RL实际需要更新的方向上
2. **频谱覆盖性**：对其余方向保留少量覆盖，避免系统性遗漏
3. **任务自适应性**：不依赖W的静态结构，而是根据RL任务的实际梯度信号来选择方向
4. **显存友好**：分析过程不能比LoRA本身更耗显存

---

## 5. 相关工作：LoRA-GA

### 5.1 LoRA-GA的方法

LoRA-GA（Low-Rank Adaptation with Gradient Approximation, NeurIPS 2024）的核心思想与我们的方向一致：用梯度信息指导初始化。

具体方法：
1. 做一次forward-backward，得到全量梯度 $\nabla_W \mathcal{L}$
2. 对梯度做SVD：$\nabla_W \mathcal{L} = USV^\top$
3. 用梯度的top-2r奇异向量初始化A和B：
   $$A_{\text{init}} = \frac{\sqrt[4]{d_{out}}}{\gamma} V_{[1:r]}^\top, \quad B_{\text{init}} = \frac{\sqrt[4]{d_{out}}}{\gamma} U_{[r+1:2r]}$$
4. 调整frozen权重：$W_{\text{frozen}} = W_0 - \eta B_{\text{init}} A_{\text{init}}$

关键技巧：逐层反向传播+立即释放梯度（hook into PyTorch backward），使得显存 = $O(\text{单层梯度})$，而不是 $O(\text{所有层梯度})$。

### 5.2 LoRA-GA在RL场景下的局限性

**局限1：SFT梯度低秩 ≠ RL梯度低秩。**

LoRA-GA的理论基础是梯度矩阵高度低秩（论文Figure 2验证了SFT梯度的singular value急剧下降）。在RL场景下，由于梯度噪声 $O(1)$ 远大于信号 $O(1/\text{batch})$，梯度的singular value衰减更慢，top-2r覆盖率远低于SFT。

**局限2：LoRA-GA取梯度的top方向，在RL中可能是错误的。**

LoRA-GA初始化的是 $\nabla W$ 的top-r奇异向量。但在RL中：
- 梯度的top方向可能被噪声主导（因为噪声 >> 信号）
- 真正有用的更新方向（off-principal of W）可能不在梯度的top中
- 类似PiSSA取W的top方向在RL中崩，LoRA-GA取梯度的top方向在RL中也可能崩

**局限3：未考虑MUON优化器的影响。**

LoRA-GA假设SGD或AdamW，其中 $\Delta W \propto \nabla_W \mathcal{L}$。但在使用MUON优化器时，梯度经过正交化变换，实际更新方向 $\neq$ 原始梯度方向。对原始梯度做SVD得到的初始化在MUON下不是最优的。

**局限4：单步梯度在RL中噪声太大。**

LoRA-GA只用一次forward-backward的梯度。在SFT中这够了（梯度低秩且稳定），但在RL中，单步梯度几乎被噪声淹没。

### 5.3 LoRA-GA的可借鉴之处

尽管LoRA-GA不能直接用于RL，但以下技术组件非常有价值：

1. **逐层反向传播+释放的显存控制技巧**：证明了可以用 $O(\text{单层})$ 显存获取全量梯度信息
2. **Scale Stability理论**：Theorem 2推导的缩放因子不依赖梯度方向，在RL场景同样适用
3. **梯度驱动初始化的整体思路**：用动态信息（梯度）而非静态信息（W的SVD）来指导初始化

---

## 6. 我们的方法

### 6.1 核心思路（一句话）

**通过多轮低成本LoRA探索收集RL梯度信号，在CPU上做GASD式谱分析（投影到W的SVD空间去噪），然后按投影谱的加权分布初始化A和B——既集中表达能力到RL需要的方向，又通过概率采样保留频谱覆盖性。**

### 6.2 方法概览

```
Phase 0 [CPU, 离线]:  W 的 SVD 预计算
Phase 1 [GPU, 低成本]: K 轮探索（随机初始化 LoRA → 跑 N 步 RL+MUON → 收集梯度/δW）
Phase 2 [CPU, 分析]:   K 组投影谱聚合去噪 → 得到 RL 偏好的奇异方向分布
Phase 3 [CPU, 构造]:   按分布加权采样 W 的奇异向量 → 构造最终 A_init, B_init
Phase 4 [GPU, 训练]:   用新初始化正式训练
```

### 6.3 为什么需要多轮探索

**单轮探索的根本缺陷：观测窗口偏差。**

每轮随机初始化的AB限定了一个rank-r的子空间。在这个子空间内跑几步RL得到的δW，只是梯度在该子空间内的投影，而不是梯度的完整信息。如果初始子空间恰好不覆盖某个重要方向，你根本观测不到那个方向的信号。

多轮探索的作用：
- 每轮用不同的随机AB → 不同的"观测窗口"
- K轮覆盖了 $K \times r$ 维的子空间（假设随机选择近似正交），观测到更完整的梯度分布
- **交叉验证**：跨轮一致出现的强信号方向才是真信号，单轮独有的可能是噪声

**类比**：这类似于compressed sensing中的多次随机投影测量——每次投影只看到部分信息，多次投影后可以恢复完整信号。

**为什么不简单增加单轮的步数**：
- 增加步数只能降低该子空间内的噪声（通过 $\sqrt{N}$ 累积），但无法获得子空间外的信息
- 而且步数太多时模型已经偏移，分析出的方向不再代表"初始阶段的需求"

### 6.4 与LoRA-GA的本质区别

| | LoRA-GA | 本方法（GALI） |
|---|---------|--------------|
| SVD对象 | $\nabla_W \mathcal{L}$（原始梯度） | $W$ 本身（权重） |
| 选方向依据 | 梯度的top奇异向量 | W的奇异向量，按RL梯度投影加权 |
| 优化器假设 | SGD / AdamW | MUON-aware（观测post-MUON的δW） |
| 适用场景 | SFT（梯度低秩） | RL（梯度高噪声，更新在off-principal） |
| 覆盖性 | 无（只取top，可能在RL中崩） | 有（概率采样保留覆盖性） |
| 去噪机制 | 无（单步梯度） | 多轮探索 + 跨轮聚合 |
| 显存开销 | $O(\text{单层梯度})$ | $O(\text{标准LoRA})$（探索阶段就是正常LoRA训练） |

---

## 7. 详细算法

### 7.1 Phase 0：W的SVD预计算（CPU，离线）

对每个需要加LoRA的权重矩阵 $W_l \in \mathbb{R}^{m \times n}$：

```python
# 在CPU上完成，一次性预计算
U_w, Sigma_w, V_w = torch.linalg.svd(W_l.float().cpu(), full_matrices=False)
# U_w: [m, min(m,n)], Sigma_w: [min(m,n)], V_w: [min(m,n), n]
# 保存到磁盘，后续直接加载
torch.save({"U": U_w, "S": Sigma_w, "V": V_w}, f"svd_cache/layer_{l}.pt")
```

**复杂度**：$O(mn \cdot \min(m,n))$ per layer。对于Qwen-1.5B (hidden=1536, intermediate=8960)，单层 SVD 约几秒；对于8B模型 (hidden=4096)，约几十秒。总共几十层，全部预计算在CPU上几分钟到几十分钟完成。

**优化**：如果full SVD太慢，可以用randomized SVD只计算前k和后k个奇异向量（k > r即可），中间部分不需要精确值。

### 7.2 Phase 1：多轮LoRA探索（GPU）

```
Input:
  - K: 探索轮数（推荐 3~5）
  - N: 每轮探索步数（推荐 10~30）
  - r: LoRA rank
  - RL训练配置（含MUON优化器）

Output:
  - 每层 l 的 K 组更新量 {δW_l^(k)}, k = 1..K

for k = 1 to K:
    # 1. 随机初始化 LoRA
    for each layer l:
        A_l^(k) ~ Kaiming_uniform(shape=[r, n])
        B_l^(k) = zeros(shape=[m, r])

    # 2. 跑 N 步 RL + MUON
    run_rl_training(model, steps=N, optimizer=MUON)

    # 3. 记录更新量
    for each layer l:
        δA_l^(k) = A_l^(k)_current - A_l^(k)_init
        δB_l^(k) = B_l^(k)_current - B_l^(k)_init
        # δW_l^(k) = B_current * A_current - B_init * A_init
        # 但不需要展开为全量矩阵，只存 A_init, B_init, A_current, B_current
        save_to_cpu(A_l^(k)_init, B_l^(k)_init, A_l^(k)_current, B_l^(k)_current)

    # 4. 重置模型到初始状态（重新加载W0）
    reset_model()
```

**显存开销**：每轮 = 标准LoRA训练的显存，无任何额外开销。A和B是小矩阵（rank r），存到CPU的成本可忽略。

**总GPU成本**：$K \times N$ 步RL训练。K=5, N=20 时为 100 步，通常是整个训练（几百到几千步）的5%~10%。

### 7.3 Phase 2：GASD式谱投影分析（CPU）

这是整个方法的核心分析步骤。

#### 7.3.1 计算投影谱

对每层 $l$，每轮 $k$：

```python
# 加载预计算的 W 的 SVD
U_w, S_w, V_w = load_svd(layer_l)

# 计算该轮的有效更新 δW = B_cur * A_cur - B_init * A_init
# 注意：δW 是 rank ≤ 2r 的矩阵，不需要展开为 [m,n]

# 对每个 W 的奇异方向 i，计算投影强度
# P_i^(k) = u_i^T @ δW^(k) @ v_i
#         = u_i^T @ (B_cur @ A_cur - B_init @ A_init) @ v_i
#         = (u_i^T @ B_cur)(A_cur @ v_i) - (u_i^T @ B_init)(A_init @ v_i)

d = len(S_w)  # min(m, n)
P = torch.zeros(d)  # 投影谱

for i in range(d):
    u_i = U_w[:, i]  # [m]
    v_i = V_w[i, :]  # [n]

    # 高效计算，每个方向 O(m*r + n*r) = O(d*r)
    term_cur = (u_i @ B_cur) @ (A_cur @ v_i)    # scalar
    term_init = (u_i @ B_init) @ (A_init @ v_i)  # scalar
    P[i] = term_cur - term_init

# 实际实现：向量化，一次矩阵乘法完成所有方向
# alpha = U_w^T @ B_cur    # [d, r]
# beta  = A_cur @ V_w^T    # [r, d]
# P_cur = (alpha * beta^T).sum(dim=1)  即逐行点积
# 同理 P_init，最终 P = P_cur - P_init

alpha_cur = U_w.T @ B_cur      # [d, r]
beta_cur  = A_cur @ V_w.T      # [r, d]  → 转置后 [d, r]
P_cur = (alpha_cur * beta_cur.T).sum(dim=1)  # [d]

alpha_init = U_w.T @ B_init
beta_init  = A_init @ V_w.T
P_init = (alpha_init * beta_init.T).sum(dim=1)

P_k = P_cur - P_init  # 第 k 轮的投影谱
```

**复杂度**：$O(d \cdot r)$ 的矩阵乘法 × 4次 = $O(d \cdot r)$ per layer per round。在CPU上极快。

#### 7.3.2 跨轮聚合去噪

```python
# 收集 K 轮的投影谱
P_all = stack([P_1, P_2, ..., P_K])  # [K, d]

# 方法1：取绝对值的中位数（对异常值鲁棒）
P_robust = median(|P_all|, dim=0)  # [d]

# 方法2：截断均值（去掉最大最小后取平均）
P_sorted = sort(|P_all|, dim=0)
P_trimmed = mean(P_sorted[1:-1], dim=0)  # 去掉每个方向的最大最小值

# 方法3（推荐）：一致性加权
# 如果某个方向在多轮中都有强信号 → 高权重（真信号）
# 如果只在少数轮中有强信号 → 低权重（噪声或子空间偏差）
P_mean = mean(|P_all|, dim=0)
P_std  = std(|P_all|, dim=0)
consistency = P_mean / (P_std + eps)  # 信号一致性（SNR of projection）
P_final = P_mean * sigmoid(consistency - threshold)  # 一致性加权
```

$P_{\text{final}}(i)$ 的物理含义：**RL+MUON训练在W的第i个奇异方向上的有效更新强度**，经过多轮去噪后的鲁棒估计。

### 7.4 Phase 3：构造最优初始化（CPU）

#### 7.4.1 加权采样方案（推荐）

```python
# 将 P_final 归一化为概率分布
importance = P_final / P_final.sum()  # [d], 每个奇异方向的重要性权重

# 混合分布：η * importance + (1-η) * uniform
# η 控制 "集中性 vs 覆盖性" 的权衡
eta = 0.7  # 超参数，推荐 0.5~0.8
mixed = eta * importance + (1 - eta) * torch.ones(d) / d

# 无放回采样 r 个方向
selected_indices = torch.multinomial(mixed, num_samples=r, replacement=False)
selected_indices = sort(selected_indices)  # 排序

# 用选中方向的 W 的奇异向量构造 A 和 B
# 借鉴 LoRA-GA 的 scale stability (Theorem 2)
scale = (d_out ** 0.25) / gamma  # gamma 为超参数

A_init = scale * V_w[selected_indices, :]      # [r, n]
B_init = scale * U_w[:, selected_indices]       # [m, r]

# 调整 frozen 权重
W_frozen = W_0 - (alpha / r) * B_init @ A_init
```

#### 7.4.2 确定性方案（备选）

```python
# 不做采样，直接取 top-r_main + 均匀散布 r_cover 个方向
r_main = int(0.7 * r)   # 70% rank 给最重要方向
r_cover = r - r_main     # 30% rank 给覆盖性

# 主方向：取 importance 最大的 r_main 个
top_indices = torch.topk(importance, r_main).indices

# 覆盖方向：在剩余方向中均匀采样
remaining = set(range(d)) - set(top_indices.tolist())
cover_indices = random.sample(remaining, r_cover)

selected_indices = sort(top_indices.tolist() + cover_indices)

# 构造 A, B 同上
```

#### 7.4.3 加权幅度方案（进阶）

不仅选方向，还根据重要性调整每个方向的初始幅度：

```python
selected_importance = importance[selected_indices]
# 归一化使得总能量不变
weights = sqrt(selected_importance / selected_importance.sum() * r)

A_init = scale * diag(weights) @ V_w[selected_indices, :]  # [r, n]
B_init = scale * U_w[:, selected_indices] @ diag(weights)  # [m, r]
```

这样不仅子空间方向正确，而且重要方向的初始幅度更大，加速收敛。

### 7.5 Phase 4：正式训练（GPU）

```python
# 用 Phase 3 构造的 A_init, B_init 初始化 LoRA
model = load_pretrained(W_frozen)  # 注意是调整后的 W_frozen
apply_lora(model, A_init, B_init, rank=r, alpha=alpha)

# 正常 RL 训练
train_rl(model, optimizer=MUON, ...)
```

### 7.6 完整算法伪代码

```
Algorithm: GALI — Gradient-Aligned LoRA Initialization for RL

Input:
  model M with weights {W_l}, RL training config
  r: LoRA rank, α: LoRA alpha
  K: exploration rounds (default 5)
  N: steps per round (default 20)
  η: concentration parameter (default 0.7)
  γ: scale factor (default same as LoRA-GA)

# === Phase 0: Precompute W's SVD (CPU, offline) ===
for each target layer l:
    U_w^l, Σ_w^l, V_w^l ← SVD(W_l)

# === Phase 1: Multi-round exploration (GPU) ===
for k = 1 to K:
    Initialize LoRA with random A^(k), B^(k) = 0
    Run N steps of RL with MUON optimizer
    Store (A_init^(k), B_init^(k), A_final^(k), B_final^(k)) to CPU
    Reset model to W_0

# === Phase 2: Spectral projection analysis (CPU) ===
for each target layer l:
    for k = 1 to K:
        Compute projection spectrum P^(k) via:
        P_i^(k) = u_i^T(B_final^(k) A_final^(k) - B_init^(k) A_init^(k))v_i
        using efficient rank-r computation (no full-matrix expansion)

    Aggregate: P_final ← ConsistencyWeightedMean(|P^(1)|, ..., |P^(K)|)

# === Phase 3: Construct initialization (CPU) ===
for each target layer l:
    importance ← P_final^l / sum(P_final^l)
    mixed ← η · importance + (1-η) · Uniform(d)
    selected ← MultinomialSample(mixed, r, replacement=False)
    scale ← d_out^(1/4) / γ
    A_init^l ← scale · V_w^l[selected, :]
    B_init^l ← scale · U_w^l[:, selected]
    W_frozen^l ← W_l - (α/r) · B_init^l · A_init^l

# === Phase 4: Full RL training (GPU) ===
Load model with {W_frozen^l}
Apply LoRA with {A_init^l, B_init^l}
Train with MUON optimizer until convergence
```

---

## 8. 理论分析

### 8.1 为什么在W的SVD空间下分析

LoRA-GA在 $\nabla_W \mathcal{L}$ 的SVD空间下分析。我们选择在W的SVD空间下分析，原因如下：

1. **GASD的核心发现**：RL的优化偏好（哪些方向更新多、哪些少）主要由W的谱结构决定，与具体的数据集和RL算法关系较弱。因此W的SVD基是分析RL动力学的"自然坐标系"。

2. **稳定性**：W在训练过程中变化很小（主成分角度偏移 < 2度），所以W的SVD基在整个训练过程中几乎不变。用一个不变的基底来分析变化的梯度，比用变化的梯度自身的SVD基更稳定。

3. **可解释性**：在W的SVD空间下，我们可以直接区分"principal方向"和"off-principal方向"，这与GASD理论的框架一致。

### 8.2 多轮探索的去噪效果

单轮探索得到的投影谱 $P^{(k)}$ 是真实投影谱 $P^*$ 的有噪声估计：

$$P^{(k)} = \Pi_{S_k} P^* + \text{noise}^{(k)}$$

其中 $\Pi_{S_k}$ 是到第k轮LoRA子空间 $S_k$ 的投影算子。

- 单轮的问题：$\Pi_{S_k}$ 是rank-r投影，会系统性地丢失 $S_k$ 之外的信号
- 多轮的优势：不同轮的 $S_k$ 近似正交，$\bigcup_k S_k$ 覆盖了约 $Kr$ 维子空间
- 当 $Kr \sim d$ 时，联合观测理论上可以恢复完整的 $P^*$

实际上不需要 $Kr = d$，因为 $P^*$ 本身是稀疏或近似稀疏的（只有少数方向有强信号），所以 $K = 3 \sim 5$ 轮就足够了。

### 8.3 概率采样保证覆盖性

混合分布 $p = \eta \cdot \text{importance} + (1-\eta) \cdot \text{uniform}$ 的性质：

- 当 $\eta = 0$：退化为随机初始化（完全覆盖，无集中）
- 当 $\eta = 1$：退化为纯importance采样（最大集中，可能崩溃）
- 中间值 $\eta$：每个方向被选中的概率 $\geq (1-\eta)/d$，保证最低覆盖

这从理论上避免了PiSSA/MiLoRA的崩溃模式——任何方向都有被选中的非零概率。

### 8.4 MUON兼容性

我们的方法天然兼容MUON，因为：
- Phase 1中LoRA训练使用MUON优化器，收集的δW已经包含了MUON正交化的效果
- 投影分析看的是post-MUON的实际更新，而不是raw gradient
- 因此不需要在CPU上"模拟MUON"——MUON的效果已经编码在δW中

---

## 9. 实验计划

### 9.1 第一阶段：小规模验证

**模型**：DeepSeek-R1-Distill-Qwen-1.5B

**任务**：GRPO on math reasoning（与现有LoRA实验一致）

**对比方法**：
| 方法 | 说明 |
|------|------|
| LoRA (vanilla) | 基线：Kaiming + Zero |
| PiSSA | W的top-r SVD |
| MiLoRA | W的bottom-r SVD |
| LoRA-GA | 梯度的top-2r SVD |
| GALI (ours) | 多轮探索 + 谱投影 + 加权采样 |
| Full finetune | 上界参考 |

**配置**：
- rank = 16, 32
- K = 3, 5（探索轮数）
- N = 10, 20, 30（探索步数）
- η = 0.5, 0.7, 0.9（集中度）

**评估指标**：
- 收敛速度（达到X% reward所需步数）
- 最终性能（reward、pass@1）
- 训练稳定性（是否崩溃、reward方差）

### 9.2 第二阶段：消融实验

| 消融维度 | 变体 |
|---------|------|
| 探索轮数K | K=1 vs K=3 vs K=5 vs K=10 |
| 每轮步数N | N=1 vs N=10 vs N=30 vs N=50 |
| 集中度η | η=0, 0.3, 0.5, 0.7, 0.9, 1.0 |
| 去噪方式 | 中位数 vs 截断均值 vs 一致性加权 |
| 采样 vs 确定性 | 概率采样 vs top-r + scatter |
| W-SVD空间 vs 梯度SVD空间 | 在W的基下分析 vs 在∇W的基下分析 |

**关键消融**：
- K=1 vs K>1：验证多轮去噪的必要性
- η=0 (=随机) vs η=1 (=纯importance)：验证混合策略的价值
- W-SVD vs 梯度-SVD：验证在W的基下分析是否优于LoRA-GA式的梯度基分析

### 9.3 第三阶段：分析实验

**需要运行的分析脚本**：

1. `rl_lora_init_analysis.py`（已有）：各优化器的ΔW在W0谱基下的分布、各初始化策略的捕获率
2. 新增：`gali_projection_analysis.py`：可视化多轮探索得到的投影谱，验证跨轮一致性
3. 新增：`gali_init_quality.py`：对比GALI初始化的子空间与Oracle（Full finetune的ΔW top-r SVD）的重叠度

**核心验证问题**：
- GALI选出的r个方向，与Full ΔW的top-r子空间重叠度是否显著高于随机？
- 投影谱 $P_{\text{final}}$ 的分布是否与GASD理论一致（off-principal方向更强）？
- 多轮去噪后的投影谱是否比单轮更稳定（跨随机种子的方差更小）？

### 9.4 第四阶段：规模化验证

**模型**：Qwen3-8B 或 Llama-3-8B

**任务**：DAPO / GRPO on math reasoning

**重点**：
- CPU预计算SVD的时间开销（8B模型）
- 探索阶段的总GPU时间占比
- 与Full finetune的性能差距
- 不同优化器（Adam vs MUON）下GALI的表现

### 9.5 成本分析

以Qwen-1.5B, rank=32, K=5, N=20为例：

| 阶段 | 设备 | 时间估计 | 显存 |
|------|------|---------|------|
| Phase 0: SVD | CPU | ~2分钟（28层） | ~8GB RAM |
| Phase 1: 探索 | GPU | 100步 ≈ 正常训练5~10% | = 标准LoRA |
| Phase 2: 分析 | CPU | ~30秒 | ~4GB RAM |
| Phase 3: 构造 | CPU | ~1秒 | 可忽略 |
| Phase 4: 训练 | GPU | 正常训练 | = 标准LoRA |

**额外开销**：~100步GPU + ~3分钟CPU，换取潜在的2-4x收敛加速。

---

## 10. 总结

### 10.1 逻辑链

```
GASD分析发现RL更新主要在off-principal方向
    ↓
现有LoRA初始化（PiSSA/MiLoRA）基于W的静态谱分析，与RL动态不匹配
    ↓
需要基于RL的实际梯度信号来选择初始化方向
    ↓
但RL梯度噪声极大（信号O(1/batch), 噪声O(1)），单步梯度不可靠
    ↓
解决方案：多轮低秩探索 + 在W的SVD空间下投影去噪
    ↓
投影谱揭示RL真正需要更新的方向 → 按此分布加权采样初始化AB
    ↓
保留概率覆盖性防止崩溃，集中表达能力加速收敛
```

### 10.2 方法的核心创新点

1. **首次系统研究RL场景下的LoRA初始化问题**，揭示SFT与RL的最优策略截然不同
2. **将GASD的谱分析框架应用于LoRA初始化**，在W的SVD空间（而非梯度SVD空间）下分析
3. **多轮探索去噪机制**，解决RL梯度高噪声下单步信息不可靠的问题
4. **概率采样的混合策略**，在信号集中性和频谱覆盖性之间取得平衡
5. **全程GPU显存 = 标准LoRA**，所有额外计算在CPU上完成，符合"用CPU换GPU"的原则

### 10.3 潜在风险与缓解

| 风险 | 可能性 | 缓解措施 |
|------|--------|---------|
| 探索步数N不够，信号被噪声淹没 | 中 | 增大N或K，通过投影谱的信噪比自适应判断 |
| 探索步数N太多，模型偏移导致分析过时 | 低 | 控制N ≤ 30，且重置模型后重新训练 |
| K轮的子空间恰好都不覆盖某关键方向 | 极低 | K=5时覆盖5r维，对于r=32已覆盖160维/4096维 |
| η选择不当 | 中 | 通过消融实验确定，或设为可调超参 |
| CPU上SVD太慢（超大模型） | 低 | 改用randomized SVD，只需top-k和bottom-k |
