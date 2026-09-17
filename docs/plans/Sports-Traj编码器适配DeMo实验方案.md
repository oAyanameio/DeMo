# Sports-Traj 编码器适配 DeMo 实验方案

## 一、实验目的

验证 Sports-Traj 的缺失感知编码思想能否提升当前 DeMo 在**缺失历史轨迹条件下直接预测未来轨迹**任务上的效果。

本轮实验不替换 DeMo 的多模态解码器，先把 Sports-Traj 作为历史编码器增强来源，保持未来预测目标、输出接口和评估口径不变。

## 二、当前正式基线

### 2.1 任务

- 数据：TrajImpute 官方 release。
- 任务：Easy-direct / Hard-direct；输入历史轨迹含缺失，直接预测未来，不经过独立插补器。
- 场景：ETH-M、HOTEL-M、UNIV-M、ZARA1-M、ZARA2-M。
- 历史长度：8 帧。
- 未来长度：12 帧。
- 每个场景独立从头训练，不使用零样本 checkpoint。

### 2.2 模型与训练

- 模型：当前 DeMo `ModelForecast`。
- 历史编码器：4 层单向 Mamba（`bimamba=false`）。
- 场景交互：Scene Context Transformer，8 heads。
- embedding dimension：128。
- decoder：当前 `TimeDecoder`，正式默认 20 个 mode query。
- decoder 内部保留现有 state/mode/hybrid 结构及其 Mamba 实现。
- `dt=0.4`，batch size=64，训练 100 epochs，warmup 5 epochs。
- optimizer：learning rate=1e-3，weight decay=1e-4。
- precision：bf16；gradient clipping=5，norm。
- seed：2024。
- checkpoint：按 `val_minFDE20` 选择最佳 epoch。
- 测试：K=20，报告现有 ADE/FDE 等指标。

### 2.3 默认 M0

```text
use_observation_features = false
use_missing_summary      = false
use_motion_features      = false
use_evidence_clock       = false
use_gap_condition        = false
bimamba                  = false
num_modes                = 20
```

正式入口与默认配置：

```text
conf/config_missing_aware_trajimpute.yaml
scripts/训练与评估/run_trajimpute_experiments.py
```

## 三、可借鉴内容

Sports-Traj 的核心借鉴不是直接搬运 CVAE 解码器，而是将输入统一表示为：

```text
masked trajectory + visibility mask + missingness structure
```

本项目优先借鉴以下三点：

### 3.1 缺失模式显式编码

输入中区分：

- 可见位置；
- 缺失位置；
- 缺失持续时间；
- 距最近有效观测的时间间隔。

缺失位置不能只作为零值输入。模型需要知道零值是占位符而不是实际运动状态。

### 3.2 BTS 的缺失距离调制思想

对每个 agent 和历史时刻计算缺失距离：

```text
mask: 1 1 1 0 0 0 1
 gap:  0 0 0 1 2 3 0
```

再将缺失距离映射为 temporal feature 的调制系数：

```text
h_gap = h * alpha(gap)
alpha(gap) = exp(-MLP(gap))
```

其含义是：连续缺失越长，相关历史表征的可靠性越低。

本项目先采用单向、历史内生的版本，不直接复制 Sports-Traj 的完整双向 temporal generation 结构。

### 3.3 两种不同层级的缺失条件

Sports-Traj 的 GSM 在**每个时间步**汇总所有 agent 的可见性，并作为额外 token 参与空间交互；这是场景级、时刻级的缺失建模。

当前模块一实现的不是 GSM，也不建模“某个邻居在当前时刻是否缺失”。它是每个 actor 对其**整个历史窗口**计算 6 维缺失统计量：

```text
missing rate / longest gap / missing prefix
/ tail gap / gap area / valid rate
    -> per-actor missing-summary embedding
    -> 加到该 actor token
    -> Scene Context Transformer
```

它应称为 **per-actor missing-summary conditioning（逐 actor 缺失摘要条件化）**。其作用是让当前场景 Transformer 得知每个 actor 的历史证据完整度；这与 BTS 的 gap scaling 互补，但不能宣称为 Sports-Traj GSM 复现。

后续若模块一有效，再考虑单独实现严格的 scene-time ghost token 或 visibility-aware attention bias；这些不属于当前模块一。

## 四、与当前 DeMo decoder 的适配关系

Sports-Traj 编码器通常保留时间维度，输出近似：

```text
[B, N, T_hist, D]
```

当前 DeMo `TimeDecoder` 接收：

```text
[B, N, D]
```

因此采用以下接口适配：

```text
Sports-Traj-inspired history encoder
    -> mask-aware temporal summary
    -> [B, N, D]
    -> current Scene Context Transformer
    -> current 20-mode TimeDecoder
```

历史 summary 不固定取最后一帧。优先比较：

1. 最后有效观测时刻特征；
2. 有效历史 mean pooling；
3. 最后有效特征与有效 mean 特征拼接后投影。

当前 decoder、20 个 mode query、state/mode/hybrid coupling、GMM/Laplace 输出和现有损失先保持不变。

## 五、模块化实验方案（一次实现一个模块）

原则：**一次实现一个模块、一次训练回答一个问题**。相关性高的设计合并进同一模块一起实现、一起训练；模块内部组件不单独训练。下表是候选路线，不表示所有模块都必须执行；每一轮只在上一轮胜出后继续。

| 模块 | 合并内容 | 训练轮次 | 回答的问题 |
|---|---|---|---|
| 模块一：缺失感知条件化包 | gap-conditioned temporal scaling（E1，已实现）+ 逐 actor 缺失摘要条件化 | R1 | 缺失证据条件化能否稳定超过 M0 |
| 模块二：时序编码器升级 | Sports-style temporal encoder + mask-aware pooling（最后有效帧特征 ⊕ 有效均值特征 → 投影回 embed_dim） | R2 | 编码器结构升级能否在 R1 胜者之上再提升 |
| 模块三：辅助重建头 | 共享编码器 + 历史缺失重建辅助 head；`L = L_future + λ·L_hist_missing` | R3 | 联合训练目标能否进一步提升未来预测 |

执行规则：

1. **递进叠加**：R2 在 R1 胜出配置之上训练；R1 判负则停止本路线，不自动拆分模块一内部组件。
2. **先筛选后确认**：若已有 screening 预算，先跑 1–2 个场景；只有筛选结果达到门槛才跑五场景确认轮。若直接承担五场景训练成本，则该轮本身作为筛选与确认，不再重复训练。
3. **零初始化起步**：模块内新增组件全部零初始化加性设计，与上一轮基线等价起步。
4. **消融推迟**：模块内组件不单独消融；胜出模块的组件归因留到论文补充阶段。
5. 预算上限由用户在每轮结束后决定；默认不预设必须完成三轮。

### 模块一实现要点（当前轮）

E1 的 `use_gap_scaling` 已实现并通过链路检查；逐 actor 缺失摘要条件化与其同属"缺失感知条件化"一个机制族，合并进模块一一次实现：

```text
S1 = M0 + gap scaling + per-actor missing-summary conditioning（零初始化加性）
```

判负则整个模块一作为组合方案判负，不据此分别断言 gap scaling 或缺失摘要单组件无效；组件归因留到后续受限消融。当前实现的缺失摘要不是 Sports-Traj GSM，结果不得表述为 GSM 已验证。

### R1 判负后的修正路线（2026-09-17 更新）

R1 首轮判负，但训后诊断发现两个实现混杂因素（gap gate 无界尺度失控：α(0) 训至 8千~1万倍；summary 零缺失非零注入），且 M0/S1 从头训练时 RNG 流分叉（178/430 共享 tensor 不同）——本轮判"当前实现负"，不判"缺失条件化思想负"。因此不直接废弃模块一，先修正后低成本复筛：

1. gap gate 锚定：有效帧（gap=0）强制 α=1；log α 有界；每 epoch 记录 α(g) 曲线
2. summary 零缺失中性：注入形式改 r_i = missing_rate_i · MLP(s_i)，零缺失注入恒为零（MLP bias 非零也不泄漏）
3. 配对初始化：保存 M0 初始 backbone state_dict，S1 显式加载（新分支零初始化），保证共同参数逐位相同
4. 复筛协议：仅 UNIV Easy 单场景，M0/S1 配对（同初始权重）；通过再五场景
5. 修正版仍判负，则模块一整体废弃，模块二基于 M0

结果明细与注意事项见 docs/results/实验总汇总.md §8。

当前实现边界：

- `gap_steps`：每个 actor、每个历史时刻距最近有效观测的步数；有效帧为 0，前缀无有效观测时按 `t+1` 计。
- gap scaling：将 `gap_steps / obs_len` 经 `gap_scale_mlp` 映射为 `alpha=exp(-MLP(.))`，作用在历史 MLP 输出、4 层 history Mamba 之前。
- missing summary：每个 actor 一个 6 维窗口级向量 `[missing_rate, longest_gap, missing_prefix, tail_gap, gap_area, valid_rate]`，经 MLP 后加到 actor token。
- padding actor：summary 条件乘 `x_key_valid_mask`，不参与 Scene Context Transformer 的有效交互。
- 新分支：均为默认关闭；新增 MLP 输出层零初始化，初始化时不改变上一版本前向。
- 不包含：严格 scene-time GSM token、visibility-aware attention bias、双向历史生成器、CVAE 和历史重建 head。

代码入口：

```text
src/datamodule/missing_features.py
src/datamodule/trajimpute_dataset.py
src/model/model_forecast.py
conf/model/missing_aware_ethucy_model_forecast.yaml
scripts/训练与评估/run_trajimpute_experiments.py
scripts/结果分析/evaluate_trajimpute_direct.py
```

正式运行名称：`S1-module1`；其 runner 开关为 `use_gap_scaling=true` 与 `use_missing_summary=true`，K 固定为 20。

进入正式训练前的最低验收：

1. mask-only 特征测试通过，覆盖完整、前缀缺失、中间缺口、尾部缺失和全缺失 padding actor；
2. 模块一前向 smoke 通过，输出 finite，loss/backward 无 NaN；
3. 新分支缺字段时明确报错；
4. 零初始化权重复制测试确认与上一版本前向等价；
5. Hydra train/eval 两端解析到相同的 `S1-module1` 开关和 K=20。

## 六、训练矩阵

| 顺序 | 配置 | 训练范围 | 决策 |
|---|---|---|---|
| 0 | 现有 M0 | 复用已有正式结果；无精确匹配才补训 | 固定基线 |
| 1 | R1 = M0 + 模块一（gap scaling + per-actor missing summary） | Easy-direct，按预算选择筛选或五场景 | 模块一判胜/判负 |
| 2 | R2 = R1 胜者 + 模块二 | Easy-direct，按预算选择筛选或五场景 | 模块二判胜/判负 |
| 3 | R3 = R2 胜者 + 模块三 | Easy-direct，按预算选择筛选或五场景 | 模块三判胜/判负 |
| 4 | 最终累计胜者 | Hard-direct 五场景 | Easy→Hard 迁移确认 |
| 5 | 最终模型去组件版（-模块 k） | 代表场景或受限筛选 | 论文消融 |

Hard 只跑最终累计胜者一次，不在中间轮重复确认。每个后续模块是否执行，取决于上一轮的实际收益和剩余训练预算。

## 七、评估要求

所有训练轮次使用同一套 K=20 direct 评估。除总体指标外，至少保留：

- 五个场景的逐场景 ADE/FDE；
- Easy / Hard；
- 短缺失与长缺失分组；
- 平均结果与最差场景结果。

判胜不采用单一场景或单一指标的偶然下降。默认判定依据为：五场景平均 ADE/FDE 相对 M0 有改善，且多数场景不恶化；若总体提升来自单一场景，记为不稳定，不进入下一模块。长缺失分组用于诊断，不替代总体主指标。

当前模块一只回答两个问题：

```text
1. gap scaling 与逐 actor 缺失摘要的组合，是否改善总体及长缺失条件下的未来预测？
2. 收益是否在多个场景同向，而不是由单一场景驱动？
```

变量隔离保持不变：

- decoder、20 个 mode query、loss 和评估口径不变；
- R1–R3 只改变历史 encoder、缺失条件或辅助目标；
- 不同时改变训练 mask 分布和模型结构；
- 不用历史重建指标替代未来 ADE/FDE。

## 八、论文方法定位

若模块一及后续模块获得稳定收益，最终方法可表述为：

> 面向部分可观测历史的缺失感知行人轨迹预测。模型显式编码可见性、缺失持续时间和不完整社会交互，并将其作为条件输入当前 DeMo 的结构化多模态未来解码器，在不进行独立轨迹插补的情况下直接预测未来轨迹。

与 Sports-Traj 的差异：

- Sports-Traj 面向体育场景的统一完整轨迹生成；
- 本方法面向行人场景的缺失历史直接未来预测；
- 本方法保留 DeMo 的 20-mode 结构化未来 decoder；
- 重点研究缺失证据如何影响历史上下文和未来分布，而不是简单移植 Sports-Traj 的完整 CVAE 生成框架。

## 九、当前结论

当前已实现的 Sports-Traj 借鉴方案与 DeMo decoder **适配，但不是即插即用**。最佳迁移方式是：

```text
借鉴 Sports-Traj 的 mask-aware temporal encoding 思想
保留当前 DeMo Scene Context Transformer 与 20-mode TimeDecoder
先验证 gap scaling 与逐 actor 缺失摘要条件化的组合收益
```

第一实验目标不是复现整篇 Sports-Traj，而是确认：

> **缺失持续时间与历史证据完整度的联合条件化，能否在当前 DeMo 多模态解码器下稳定改善直接未来预测。**
