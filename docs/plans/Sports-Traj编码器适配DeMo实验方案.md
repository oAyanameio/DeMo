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

### 3.3 GSM / visibility-aware social interaction

Sports-Traj 用 ghost masking embedding 汇总每个时间步的 agent 可见性，再用于空间交互编码。

本项目可将其改造成：

```text
agent visibility + missing duration
    -> missingness summary / ghost condition
    -> Scene Context Transformer
```

更进一步可以在 attention 中加入 visibility/gap bias，使缺失严重的邻居对社会交互的影响降低。

注意：当前 DeMo 的 `encoding[:, 0]` 具有 focal actor 语义，不能直接把 ghost token 放到第 0 位而不修改 decoder。优先使用加到 actor token 的全局条件向量，或单独改造 token 索引并做消融。

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

原则：**一次实现一个模块、一次训练回答一个问题**。相关性高的设计合并进同一模块一起实现、一起训练；模块内部组件不单独训练。

| 模块 | 合并内容 | 训练轮次 | 回答的问题 |
|---|---|---|---|
| 模块一：缺失感知条件化包 | gap-conditioned temporal scaling（E1，已实现）+ 场景级缺失摘要条件（GSM 改造版：汇总可见性/缺失时长注入 actor token，不动 `encoding[:,0]` 的 focal 语义） | R1 | 显式缺失感知条件化能否稳定超过 M0 |
| 模块二：时序编码器升级 | Sports-style temporal encoder + mask-aware pooling（最后有效帧特征 ⊕ 有效均值特征 → 投影回 embed_dim） | R2 | 编码器结构升级能否在 R1 胜者之上再提升 |
| 模块三：辅助重建头 | 共享编码器 + 历史缺失重建辅助 head；`L = L_future + λ·L_hist_missing` | R3 | 联合训练目标能否进一步提升未来预测 |

执行规则：

1. **递进叠加**：R2 在 R1 胜出配置之上训练；R1 判负则模块一整体废弃，R2 直接基于 M0。
2. **先筛选后确认**：每模块先用 runner `--screening` 跑 1–2 个场景，胜出才跑五场景确认轮。
3. **零初始化起步**：模块内新增组件全部零初始化加性设计，与上一轮基线等价起步。
4. **消融推迟**：模块内组件不单独消融；胜出模块的组件归因留到论文补充阶段。
5. 预算上限 = 3 轮 × 5 场景（外加 screening 小轮）。

### 模块一实现要点（当前轮）

E1 的 `use_gap_scaling` 已实现并通过链路检查；GSM-lite summary 与其同属"缺失感知条件化"一个机制族，合并进模块一一次实现：

```text
S1 = M0 + gap scaling + GSM-lite missingness summary（零初始化加性）
```

判负则整个模块一废弃，不再拆开追问是哪个组件无效。

## 六、训练矩阵

| 顺序 | 配置 | 训练范围 | 决策 |
|---|---|---|---|
| 0 | 现有 M0 | 复用已有正式结果；无精确匹配才补训 | 固定基线 |
| 1 | S1 = M0 + 模块一（gap scaling + GSM-lite） | Easy-direct 五场景，seed 2024 | 模块一判胜/判负 |
| 2 | S2 = R1 胜者 + 模块二 | Easy-direct 五场景 | 模块二判胜/判负 |
| 3 | S3 = R2 胜者 + 模块三 | Easy-direct 五场景 | 模块三判胜/判负 |
| 4 | 累计胜者 | Hard-direct 五场景 | Easy→Hard 迁移确认 |
| 5 | 最终模型去组件版（-模块k） | 代表场景或受限筛选 | 论文消融 |

Hard 只跑累计胜者一次，不在中间轮消耗预算；首轮即模块一，最多一次新的 Easy 全量训练。

## 七、评估要求

阶段 1 和阶段 2 使用同一套 K=20 direct 评估。除总体指标外，至少保留：

- 五个场景的逐场景 ADE/FDE；
- Easy / Hard；
- 短缺失与长缺失分组；
- 平均结果与最差场景结果。

首轮决策只回答三个问题：

```text
1. gap scaling 是否改善长缺失下的未来预测？
2. 加入可见性社会条件后，是否在多数场景继续改善？
3. 收益能否从 Easy 迁移到 Hard？
```

变量隔离保持不变：

- decoder、20 个 mode query、loss 和评估口径不变；
- S1/S2 只改变历史 encoder/社会条件；
- 不同时改变训练 mask 分布和模型结构；
- 不用历史重建指标替代未来 ADE/FDE。

## 八、论文方法定位

最终方法可表述为：

> 面向部分可观测历史的缺失感知行人轨迹预测。模型显式编码可见性、缺失持续时间和不完整社会交互，并将其作为条件输入当前 DeMo 的结构化多模态未来解码器，在不进行独立轨迹插补的情况下直接预测未来轨迹。

与 Sports-Traj 的差异：

- Sports-Traj 面向体育场景的统一完整轨迹生成；
- 本方法面向行人场景的缺失历史直接未来预测；
- 本方法保留 DeMo 的 20-mode 结构化未来 decoder；
- 重点研究缺失证据如何影响历史上下文和未来分布，而不是简单移植 Sports-Traj 的完整 CVAE 生成框架。

## 九、当前结论

Sports-Traj 与当前 DeMo decoder **适配，但不是即插即用**。最佳迁移方式是：

```text
借鉴 Sports-Traj 的 mask-aware temporal/spatial encoding
保留当前 DeMo Scene Context Transformer 与 20-mode TimeDecoder
先验证 gap scaling 和 visibility-aware interaction 的独立收益
```

第一实验目标不是复现整篇 Sports-Traj，而是确认：

> **缺失持续时间调制历史表征，能否在当前 DeMo 多模态解码器下稳定改善直接未来预测。**
