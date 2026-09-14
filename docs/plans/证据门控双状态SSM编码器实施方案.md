# 证据门控双状态 SSM 编码器实施方案

_研究与工程实施方案 · 基于当前 DeMo actor-only / TrajImpute direct 管线 · 2026-09-14_

> **文档定位**：本方案只描述“如何改造历史编码器，使其真正利用缺失结构与 SSM 状态递推优势”。它是对《缺失感知模型构建与实验验证方案》的独立方法层补充，不覆盖、不改写既有 TrajImpute 数据协议、DeMo 解码器和结果统计规则。
>
> **核心原则**：先实现一个能证明状态更新语义正确的最小版本，再增加双状态、社会证据和双向扫描。不要一开始同时引入 BTS、双向 Mamba、社会证据和新的损失，否则即使结果变化也无法归因。

---

## 1. 方法目标与最小主张

### 1.1 目标

当前历史编码器把缺失帧置零并附加 mask，Mamba 仍会把每个缺失帧作为一次普通状态更新。目标是把两类情况明确区分：

```text
有效观测：写入新的运动证据
缺失观测：不写入伪观测，只推进/衰减已有状态
```

进一步将历史状态拆成：

```text
h_motion  : focal 自身近期运动状态
h_context : 长期趋势与社会/场景上下文
```

缺失时：

- `h_motion` 较快衰减，因为速度、转向等局部证据容易过时；
- `h_context` 较慢衰减，因为方向趋势、群体结构等上下文可保持更久；
- 若 focal 缺失但邻居同期可见，只允许社会证据优先写入 `h_context`，不直接伪造 `h_motion`。

### 1.2 第一版可 claim 的最小方法

第一版只主张：

> **提出缺失条件化的证据门控 SSM 状态转移，使有效观测写入状态、缺失区间仅进行时间推进，并在相同训练协议下改善高缺失历史的预测或未来分布诊断。**

以下内容暂不作为第一版主张：

- “提出新的 Mamba 架构”；
- “首次将 Mamba 用于轨迹预测”；
- “解决轨迹插补”；
- “恢复真实心理意图或物理状态”；
- “双向 Mamba 一定优于单向 Mamba”。

UniTraj 已有双向 Temporal Mamba 与 BTS 缺失缩放设计，MambaPTP 已研究行人轨迹中的门控双向 Mamba，Social-Mamba 已研究将无序社会交互组织为 SSM 扫描。因此本方案的新增点必须落在**状态转移级的缺失门控和状态角色解耦**，而不是简单替换 backbone。

---

## 2. 现有代码路径与改动边界

### 2.1 已确认的现有路径

| 文件 | 当前职责 | 本方案改动 |
|---|---|---|
| `src/model/model_forecast.py` | 构造历史输入、调用历史 Mamba、取 actor token、调用 Scene Transformer | 增加可配置的证据门控历史编码器；保留旧路径作为 M0 |
| `src/model/layers/mamba/vim_mamba.py` | 封装 `mamba_ssm` 的 block、norm、残差和 `create_block` | 优先复用；只有无法实现状态门控时才增加独立 wrapper |
| `src/datamodule/missing_features.py` | 从历史 mask 生成 gap、motion run、missing summary | 增加反向 gap / 双向缺失统计量时扩展纯函数 |
| `src/datamodule/trajimpute_dataset.py` | TrajImpute 官方 release 适配、跨缺口运动量、collate | 只补充必要字段；不改变未来 target 和现有 mask 语义 |
| `src/model/layers/time_decoder.py` | DeMo State/Mode/Hybrid 解码 | 第一阶段不改；第二阶段只接收 encoder 输出，不重写 DeMo 查询逻辑 |
| `conf/model/missing_aware_ethucy_model_forecast.yaml` | 缺失模型配置 | 增加显式开关和默认值 |
| `tests/` | 数据、M2、模型接口回归 | 增加状态门控、等价性、mask 不变性和梯度测试 |

### 2.2 不应修改的内容

- 官方 TrajImpute pkl；
- 未来轨迹 target；
- Clean/ Easy-direct / Hard-direct 的数据协议；
- 既有 M0 默认行为；
- DeMo decoder 的 Mode/State 原始定义；
- 既有结果输出目录中的历史代码快照；
- `.env`、凭据和无关实验脚本。

### 2.3 运行环境

所有验收和 smoke test 使用项目指定环境，避免默认 Python 3.13 加载用户级 CUDA 扩展：

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=. /home/lbh/.conda/envs/DeMo/bin/python
```

正式实现前先确认该环境能导入当前 `mamba_ssm`。若导入失败，先记录环境阻塞，不切换到默认 Python 伪造测试结果。

---

## 3. 总体架构

```text
历史位置/差分/速度
        +
历史有效掩码 mask
        +
距最近有效观测 gap
        +
跨缺口运动证据
        +
邻居同期证据（后续阶段）
        │
        ▼
Evidence Token / Feature Builder
        │
        ├── Intent-Context SSM
        │       长期趋势、社会上下文、未来分支证据
        │
        └── Dynamic-State SSM
                速度、转向、局部运动、证据新鲜度
        │
        ▼
Reliability-Gated Fusion
        │
        ▼
Scene Context Transformer
        │
        ▼
原有 DeMo Mode / State / Hybrid Decoder
```

### 3.1 核心状态转移

普通 SSM 可抽象为：

```text
h_t = A_t h_{t-1} + B_t x_t
```

证据门控版本应满足：

```text
有效帧：
    h_t = observed_update(h_{t-1}, x_t)

缺失帧：
    h_t = time_propagation(h_{t-1}, delta_t)
```

工程实现不使用 Python 分支逐样本处理，而使用连续门控：

```text
h_t = write_gate_t * observed_update
    + propagate_gate_t * decayed_state
```

其中：

- `write_gate_t` 由 `mask_t` 和观测可靠性产生；
- `propagate_gate_t` 由 `1-mask_t` 与 `gap_t` 产生；
- `gap_t` 控制状态推进时间，不等价于新的位置观测；
- 所有门控值必须有限，且缺失时不能把零填充值当成真实位移写入状态。

### 3.2 双状态语义

```text
h_motion_t  = motion-specific SSM state
h_context_t = context-specific SSM state
```

建议初始版本使用两套独立 `create_block`，而不是立即修改底层 CUDA selective scan：

```text
motion_input  -> Mamba_motion -> motion_state
context_input -> Mamba_context -> context_state
```

状态融合：

```text
h_actor = gate * h_motion + (1 - gate) * h_context
```

`gate` 由历史有效率、最近有效观测间隔、运动证据质量和邻居可见率产生。第一阶段没有邻居可见率时，先使用 focal 自身字段。

---

## 4. 分阶段实施步骤

> **详细级别规则**：第 4.1–4.4 节是关键步骤，写明接口、张量形状和验收；第 4.5 节以后属于简单步骤，只列执行动作和验收目标。

### 4.1 阶段 A：冻结 M0 并建立等价性基线【关键步骤，详细】

#### 目的

确保新编码器关闭时不改变当前 DeMo，避免后续收益来自随机初始化、输入顺序或配置漂移。

#### 实施

1. 在 `ModelForecast.__init__` 增加配置开关，建议：

```text
use_evidence_ssm: false
use_dual_state_ssm: false
use_bidirectional_evidence: false
use_social_state_write: false
```

2. 保持 `use_evidence_ssm=false` 时原有：

```text
hist_embed_mlp
→ hist_embed_mamba
→ norm
→ 取最后时间步
```

3. 新编码器不得覆盖原模块名和 checkpoint key；建议使用独立模块名：

```text
hist_embed_mamba_legacy
hist_evidence_encoder
```

4. 用相同随机种子构造 legacy model 和新 model；通过复制 legacy state dict 或严格控制初始化测试关闭开关时的前向等价性。不要仅在同一个 seed 下分别构造两个结构不同的模型，因为额外模块会改变 RNG 消耗顺序。

#### 验收

- 关闭所有新开关时输出 shape 不变；
- 同一权重、同一输入下输出 bitwise equal 或在明确容差内一致；
- 旧 checkpoint 可以加载；
- 旧 M0 smoke 和既有测试全部通过；
- 新模块参数在关闭时不参与 forward 和 loss。

---

### 4.2 阶段 B：实现 Evidence-Gated Temporal SSM【关键步骤，详细】

#### 目的

先验证“缺失帧不应作为普通观测写入 SSM 状态”这一核心假设，暂时不加入双状态和社会证据。

#### 输入张量

历史数据按当前路径处理：

```text
hist_feat        [B, N, T, D_in]
hist_valid_mask  [B, N, T]       # True = 有效观测
x_gap_steps      [B, N, T]
x_motion_valid   [B, N, T]
```

展平真实 actor 后：

```text
[B*N, T, D_in]
[B*N, T, 1]  mask
[B*N, T, 1]  gap
```

padding actor 必须在进入 SSM 前过滤或保持零状态，不能把 padding 当成缺失轨迹参与统计。

#### 最小状态门控设计

建议优先实现 wrapper，而非改写 `mamba_ssm` CUDA kernel：

```text
z_t = Mamba(x_t)                         # 原始候选更新
r_t = exp(-softplus(W_gap * gap_t + b))  # 缺失后的保留/衰减因子
w_t = mask_t * sigmoid(W_write e_t)      # 有效观测写入门
h_t = mask_t * z_t + (1-mask_t) * r_t * h_{t-1}
```

如果当前 Mamba block 只能返回整段输出，先实现“输入侧证据门控”作为工程 baseline：

```text
x'_t = mask_t * x_t + (1-mask_t) * decay_t * x_last_valid
```

但该版本只能作为过渡对照，不得直接命名为最终 Evidence-Gated SSM，因为它还没有控制内部状态更新。

#### 必须明确的语义

- 缺失帧的 `x_positions_diff`、速度、转向仍然为占位值，不参与 observed update；
- `gap_t` 只表示时间推进长度，不是额外运动观测；
- `mask_t=1` 时模型可以写入当前证据；
- `mask_t=0` 时模型只能传播已有状态或按学习到的速率衰减。

#### 单元测试

1. **全有效退化**：mask 全 True 时，新编码器应退化为普通 temporal encoder 或在预定义容差内一致。
2. **无效值不敏感**：mask 为 False 的帧位置任意扰动，输出不应改变，除非只改变显式 gap/mask 字段。
3. **缺口长度敏感**：相同可见端点、不同缺口长度应产生不同状态，但变化方向不能由占位坐标决定。
4. **末帧缺失**：最终 actor representation 必须来自“传播后的状态”，不能错误读取未观测末帧坐标。
5. **padding 隔离**：padding actor 的位置和 mask 任意扰动不影响真实 actor 输出。
6. **梯度存在**：门控参数、Mamba 参数和输入投影均有梯度，且无 NaN/Inf。

#### 阶段门槛

至少在一个 Easy 和一个 Hard 场景完成：

```text
M0 legacy
M0 + mask feature
M0 + input decay
M0 + evidence-gated temporal SSM
```

如果 Evidence-Gated SSM 无法超过简单 mask/input decay，不进入双状态阶段，先检查状态语义和训练稳定性。

---

### 4.3 阶段 C：实现 Intent / Dynamic 双状态 SSM【关键步骤，详细】

#### 目的

让不同类型的历史证据拥有不同的记忆和缺失衰减规律。

#### 输入划分

```text
motion_input：
    x_positions_diff
    x_velocity_diff
    x_velocity / x_accel / x_turn_rate
    x_motion_valid

context_input：
    x_positions_diff
    x_valid_mask
    x_gap_steps
    x_missing_summary broadcast 或时间级 gap
```

不要把所有字段无区分地复制到两个分支后称为双状态。两个分支必须有不同输入和不同门控。

#### 状态更新

```text
h_motion_t  = motion_write_t * motion_update_t
             + (1-motion_write_t) * motion_decay_t * h_motion_{t-1}

h_context_t = context_write_t * context_update_t
             + (1-context_write_t) * context_decay_t * h_context_{t-1}
```

建议初始化：

- `motion_decay` 较快；
- `context_decay` 较慢；
- 新增融合层最后一层零初始化，使双状态模块初始不破坏 M0；
- `initialize_weights()` 之后重新检查零初始化，防止全局 init 覆盖。

#### 融合输出

```text
reliability = MLP([missing_rate, gap, motion_valid, last_motion_run])
g = sigmoid(reliability)
h_actor = g * h_motion + (1-g) * h_context
```

这里的 `g` 只表示内部证据融合权重，不能直接解释为真实意图概率或物理状态概率。

#### 验收对照

至少包含：

```text
M0 legacy
单路 evidence-gated SSM
双路但共享 decay
双路独立 decay
双路 + reliability fusion
```

必须报告：

- `minADE20`、`minFDE20`、MR、b-minFDE20；
- 速度/加速度/jerk 连续性；
- mode diversity 或 endpoint coverage；
- 按缺失帧数和最长缺口分组的结果。

#### Go 条件

双状态模型只有在以下至少一项成立时保留：

- Hard-direct 主指标稳定改善；
- 在状态连续性或模态覆盖上有独立改善；
- `h_motion` 与局部运动指标、`h_context` 与邻居/场景条件呈现可重复的分组规律。

如果双状态只改善平均误差而没有任何机制诊断，不把“motion/context”写成论文贡献，只保留为工程结构。

---

### 4.4 阶段 D：接入社会证据写入 context state【关键步骤，详细】

#### 目的

把已有 M2' 从“最终 focal token 的普通 cross-attention 残差”升级为：

```text
focal 缺失时，邻居同期观测作为 context state 的外部证据写入
```

#### 前置条件

必须先确认：

- TrajImpute sample 保留场景内多 actor；
- focal 与邻居处于同一局部坐标系；
- `seq_start_end` 分组有效；
- 邻居 `x_valid_mask` 被保留；
- 当前 `tests/test_m2_social_evidence.py` 的 mask 和梯度测试通过。

#### 社会 token

邻居帧级 token 建议包含：

```text
relative_position       [2]
relative_velocity       [1 或 2]
neighbor_valid           [1]
neighbor_gap             [1]
time_index / timestamp   [1]
interaction_weight       [1，可选]
```

形状：

```text
neighbor_tokens [B, (N-1)*T, D]
neighbor_valid  [B, (N-1)*T]
```

#### 写入规则

```text
social_write_t = focal_missing_t * neighbor_evidence_reliability_t
h_context_t'  = h_context_t + social_write_t * social_state_t
```

关键边界：

- focal 完整时不能无条件依赖补偿路径；
- 邻居无效或 padding 时不能写入；
- 邻居信息只作为 context/evidence，不直接生成 focal 历史坐标；
- 不能用未来轨迹计算 interaction weight。

#### 主要对照

```text
普通 Scene Transformer
Scene Transformer + mask
当前 M2' cross-attention
context-state social write
时间打乱邻居
随机替换邻居
focal/neighbor 同时缺失
```

#### 机制门槛

只有当收益符合以下规律，才把它称为社会证据写入：

```text
focal 缺失更严重
+ 邻居同期可见率更高
+ 交互更强
→ 补偿收益更大
```

时间打乱、邻居共同缺失和无关邻居应显著削弱收益。否则收缩为普通社会编码增强，不使用“证据补偿”命名。

---

### 4.5 阶段 E：双向 evidence scan / BTS 对照【简单步骤，粗略】

- 从 `x_valid_mask` 计算 forward gap 和 reverse gap；
- 做一个独立 BTS/双向缩放对照，不直接合并到主模型；
- 对比：单向 gap、BTS feature scaling、evidence-gated state transition；
- 记录双向版本对当前“编码器固定单向”主链裁定的偏离；
- 若 BTS 已能解释全部收益，不把它和 evidence-gated SSM 合并成一个复杂模型。

UniTraj 的 BTS 已有相近设计，相关代码可作为复现基线，不作为未经改造的原创模块。

### 4.6 阶段 F：配置、checkpoint 与训练入口【简单步骤，粗略】

- 在 `conf/model/missing_aware_ethucy_model_forecast.yaml` 增加所有新开关、默认值和版本名；
- 为每个方法保存完整 Hydra config；
- 新模块参数使用独立 key，避免旧 checkpoint 静默覆盖；
- `strict=False` 加载时打印 missing/unexpected keys 并写入日志；
- 先跑 2 train batches + 1 val/test batch smoke，再跑单场景正式筛选；
- 不修改原始 M0 配置默认值。

### 4.7 阶段 G：正式实验和统计【简单步骤，粗略】

沿用现有正式协议：

```text
Easy-direct：方法筛选与机制诊断
Hard-direct：高缺失泛化与退化分析
Clean：完整历史独立参考，不用 Easy 零缺失替代
```

执行顺序：

1. M0 legacy / M0 evidence-gated；
2. 单状态 / 双状态；
3. 双状态 + 社会证据写入；
4. 通过门槛后再做 BTS、双向和额外校准；
5. 三种子、多场景和 Hard-direct 验证；
6. 生成唯一汇总，不把历史 v1/v2/v3 数字与正式 direct 主表混排。

---

## 5. 测试与验收清单

### 5.1 单元测试【关键】

- [ ] 全有效 mask 下新模块与 M0 等价或差异在预设容差内；
- [ ] 缺失位置占位值扰动不影响输出；
- [ ] gap 改变会影响状态传播，但不能引入 NaN；
- [ ] 末帧缺失不会读取帧 7 的伪坐标；
- [ ] padding actor 不影响真实 actor；
- [ ] focal 缺失时邻居有效证据可影响 context state；
- [ ] 邻居无效、padding、时间打乱的行为符合预期；
- [ ] 所有新参数有梯度；
- [ ] AMP/bf16 或项目正式精度下无 NaN/Inf；
- [ ] 输出形状保持 `y_hat/new_y_hat [B,K,12,2]`、`pi/new_pi [B,K]`、`scal/scal_new [B,K,12,2]`。

### 5.2 工程 smoke【简单】

- [ ] 项目指定 Python 环境可导入 Mamba；
- [ ] 2 个 train batch 前向/反向通过；
- [ ] 1 个验证 batch 输出指标；
- [ ] checkpoint 可保存和重新加载；
- [ ] Easy/Hard 各抽取真实 batch 通过；
- [ ] 配置组合和命令行 override 正确。

### 5.3 正式 Go/No-Go

**进入下一阶段的 Go 条件：**

1. 当前阶段没有破坏 M0；
2. 至少一个连续高缺失分组主指标改善；
3. complete 条件平均 `minFDE20` 退化不超过 3%；
4. 多个场景/种子方向一致；
5. 机制对照能排除简单 mask、参数量和邻居顺序解释。

**停止条件：**

- Evidence-Gated SSM 不超过简单 input decay；
- 双状态只增加参数，不改善高缺失、覆盖或连续性；
- 社会证据收益在时间打乱邻居后不下降；
- 收益只来自单一场景、seed 或测试调参；
- 新模块导致 complete 明显退化且无法由机制解释；
- 默认 Mamba 环境无法通过真实项目环境验收。

停止后应收缩为“缺失历史的系统性退化与失败边界分析”，不继续堆叠 SSM 模块。

---

## 6. 推荐实验矩阵

| 版本 | 历史编码 | 双状态 | 社会证据写入 | 作用 |
|---|---|---:|---:|---|
| M0 | legacy Mamba | 否 | 否 | 原始基线 |
| E1 | evidence-gated temporal SSM | 否 | 否 | 验证状态门控 |
| E2 | evidence-gated temporal SSM | 是 | 否 | 验证状态角色解耦 |
| E3 | E2 | 是 | 是 | 完整候选方法 |
| C1 | mask feature + Mamba | 否 | 否 | 简单特征对照 |
| C2 | input decay / GRU-D-style | 否 | 否 | 衰减对照 |
| C3 | BTS / bidirectional scaling | 否 | 否 | UniTraj 风格对照 |
| C4 | 当前 M2' cross-attention | 否 | 是 | 现有社会补偿对照 |
| N1 | 时间打乱邻居 | 视对应版本 | 是 | 社会证据负对照 |
| N2 | 邻居共同缺失 | 视对应版本 | 是 | 证据消失负对照 |

所有版本必须固定：

- 数据协议；
- fold/seed；
- `num_modes=20`；
- checkpoint 选择规则；
- 训练预算；
- 指标和聚合口径。

---

## 7. 论文贡献边界

若 E1–E3 通过机制验收，可以考虑以下表述：

1. **方法问题**：将缺失历史预测中的核心困难形式化为“观测证据写入与状态传播”的区分问题；
2. **编码器方法**：提出缺失条件化、状态角色解耦的 Evidence-Gated Dual-State SSM；
3. **社会证据扩展**：将同期邻居观测作为 context state 的可控外部证据，而不是恢复 focal 历史；
4. **分析贡献**：建立缺失程度、证据新鲜度、状态衰减和未来分布质量之间的可重复关系。

不能写成：

- 重新提出 DeMo 的意图/状态分解；
- 首次使用 Mamba 预测轨迹；
- 首次处理缺失轨迹；
- 仅凭内部 hidden state 将其命名为真实意图或真实物理状态；
- 仅凭 ADE/FDE 平均提升证明社会证据转移成立。

---

## 8. 参考实现与借鉴边界

| 来源 | 借鉴内容 | 不能直接 claim 的内容 |
|---|---|---|
| UniTraj / Sports-Traj [24][25] | GSM、双向 Temporal Mamba、BTS 缺失缩放、代码组织 | 双向 Mamba/BTS 本身 |
| GRU-D [30] | mask + time interval、输入/隐状态衰减 | 普通可学习 decay |
| TIDES [28][29] | 物理时间步与选择性状态转移解耦 | 直接声称提出 irregular-time SSM |
| Social-Mamba [26][27] | 无序社会交互的扫描组织、social gate | 普通 social scan / Cycle Mamba |
| MambaPTP [31] | 行人轨迹中的门控双向 Mamba 对照 | 纯 Mamba 行人预测 backbone |
| TIMBA / SSD-TS [32][33] | 缺失时间序列中的双向 SSM 和 mask 处理 | 直接把插补方法当作未来预测创新 |
| MambaTS [34] | 扫描顺序、变量关系和 permutation 对照 | 直接复制 VAST 或变量扫描 |

---

## 9. 最终推荐执行顺序

```text
A. 冻结 M0，补等价性测试
   ↓
B. Evidence-Gated Temporal SSM
   ↓ 通过高缺失/无效值不敏感/梯度门槛
C. Intent-State 双状态 SSM
   ↓ 通过状态分支和连续性诊断
D. 社会证据写入 context state
   ↓ 通过时间对齐和共同缺失负对照
E. BTS/双向扫描作为独立对照
   ↓
F. Easy-direct 确认性实验
   ↓
G. Hard-direct、三种子、跨场景和统计汇总
```

**第一优先级不是实现完整 E3，而是验证 B：缺失帧是否真的应该跳过观测写入，只进行状态传播。** 如果这个最小假设不能超过 mask/input-decay 对照，后续双状态和社会证据都会失去清晰归因基础。
