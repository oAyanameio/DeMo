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

## 五、实验阶段

### 阶段 A：缺失距离调制

在当前历史 encoder 上加入 Sports-Traj/BTS 风格的 gap-conditioned temporal scaling：

```text
M0
 -> M0 + gap-conditioned temporal scaling
```

目标：验证缺失持续时间是否比普通 mask channel 提供额外信息。

### 阶段 B：Sports-style temporal encoder

比较当前历史 Mamba 与 Sports-Traj 风格 temporal encoder：

```text
M0
 -> M0 + alternative temporal encoder
```

先保持单向历史输入与当前 decoder 不变。

### 阶段 C：可见性社会交互

在阶段 A/B 中加入以下候选之一：

```text
ghost missing summary
visibility-aware attention bias
```

不要同时加入，分别进行消融。

### 阶段 D：空间—时间双分支

候选结构：

```text
per-time spatial Transformer
+ per-agent temporal Mamba
-> fusion
-> actor-level summary
-> current DeMo decoder
```

由于当前 DeMo 已经具有 Scene Context Transformer，必须比较：

1. 新 temporal encoder + 当前 Scene Context Transformer；
2. 新 spatial-temporal encoder + 当前 Scene Context Transformer；
3. 新 spatial-temporal encoder 替代当前 Scene Context Transformer。

### 阶段 E：联合历史缺失恢复辅助任务

仅在编码器增强确认有效后考虑：

```text
shared missing-aware encoder
    ├── history missing reconstruction head
    └── current future prediction decoder
```

辅助损失：

```text
L = L_future + lambda * L_history_missing
```

历史恢复结果与未来预测结果必须分开报告，不能用历史重建收益代替未来预测收益。

### 阶段 F：生成式 latent 扩展

CVAE 或 diffusion 不是第一阶段内容。只有在确定性/现有 20-mode decoder 上确认缺失编码有效后，才考虑将 latent 作为额外 decoder condition。

## 六、推荐最小实验矩阵

| 编号 | 变化 | 目的 |
|---|---|---|
| E0 | 当前正式 M0 | 基线 |
| E1 | M0 + gap-conditioned temporal scaling | 验证缺失距离调制 |
| E2 | M0 + Sports-style temporal encoder | 验证 temporal encoder 迁移 |
| E3 | E2 + ghost missing summary | 验证 GSM 思路 |
| E4 | E2 + visibility-aware attention bias | 验证可见性感知社会交互 |
| E5 | E4 + history missing reconstruction auxiliary head | 验证联合恢复/预测 |
| E6 | E5 + latent conditioning | 后续生成式扩展 |

首轮优先运行：

```text
E0 -> E1 -> E2 -> E4
```

不在首轮同时引入 CVAE、双向完整生成器和新的解码器。

## 七、评估要求

除总体指标外，至少按以下条件拆分：

- Easy / Hard；
- 五个场景；
- 短缺失、中等缺失、长缺失；
- 缺失比例；
- 连续缺失长度；
- 单 agent 缺失与多 agent 缺失；
- ADE 与 FDE；
- K=1 与 K=20（如评估脚本支持）。

核心观察是：

```text
缺失连续长度增加时，FDE 是否改善；
缺失 agent 增加时，社会交互模块是否仍然有效；
提升是否在多个场景稳定出现，而不是单一场景或单一种子收益。
```

所有实验必须保持以下变量隔离：

- encoder 变化与 decoder 变化分开；
- 模型变化与 missing protocol 变化分开；
- 训练 mask 分布变化与测试协议变化分开；
- 共享基础模型增强与缺失感知方法增强分开。

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
