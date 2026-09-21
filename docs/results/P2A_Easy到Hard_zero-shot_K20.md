# P2-A：Easy train → Hard test zero-shot 诊断

> **状态**：完成
>
> **实验类型**：zero-shot，不重训。
>
> **目的**：分离 M0 高缺失退化中的训练分布失配效应与模型结构效应。
>
> **训练 checkpoint**：每个场景已有的 `M0-current_<scene>_easy-direct_seed2024_uni` 最佳 Easy checkpoint。
>
> **测试数据**：同一场景 TrajImpute 官方 Hard `data_test.pkl`。
>
> **配置**：K=20、`bimamba=false`、direct evaluator、seed=2024。

---

## 1. 执行与产物

输出目录：

```text
outputs/p2_zero_shot_easy_to_hard_seed2024/
```

五个场景均完成：

| 场景 | 状态 | n |
|---|---|---:|
| ETH-M | complete | 724 |
| HOTEL-M | complete | 4212 |
| UNIV-M | complete | 97336 |
| ZARA1-M | complete | 9012 |
| ZARA2-M | complete | 23332 |
| **合计** | **5/5** | **134616** |

执行过程中首个启动脚本曾因 checkpoint 解析子命令引号错误退出；该次没有产生评估结果。修正后重新执行，五场景结果均已生成并完成读取。该启动错误不涉及模型、数据或评估逻辑。

---

## 2. Zero-shot 结果

### 2.1 场景明细

| 场景 | minADE20 | minFDE20 | ADE@1 | FDE@1 | MR |
|---|---:|---:|---:|---:|---:|
| ETH-M | 1.638 | 2.260 | 2.160 | 3.307 | 0.311 |
| HOTEL-M | 0.552 | 0.828 | 0.864 | 1.418 | 0.115 |
| UNIV-M | 0.696 | 1.017 | 1.237 | 2.099 | 0.148 |
| ZARA1-M | 1.076 | 1.642 | 1.682 | 2.802 | 0.237 |
| ZARA2-M | 0.486 | 0.677 | 0.971 | 1.637 | 0.097 |
| **宏平均** | **0.890** | **1.285** | **1.383** | **2.253** | **0.182** |
| **微平均** | **0.686** | **1.001** | **1.214** | **2.051** | **0.145** |

### 2.2 与 Hard train-adapt M0 对比

Hard train-adapt 使用同一场景的 Hard train/val 从头训练，作为 P0 已审计基线。

| 口径 | minADE20 | minFDE20 | ADE@1 | FDE@1 | MR |
|---|---:|---:|---:|---:|---:|
| Easy→Hard zero-shot，宏平均 | 0.890 | 1.285 | 1.383 | 2.253 | 0.182 |
| Hard train-adapt，宏平均 | 0.432 | 0.640 | 1.277 | 2.107 | 0.074 |
| Zero-shot 相对 train-adapt | +105.8% | +100.7% | +8.3% | +6.9% | +146.4% |

按五场景宏平均，Easy checkpoint 直接测试 Hard 时：

- minADE20 约恶化 **106%**；
- minFDE20 约恶化 **101%**；
- MR 约恶化 **146%**。

这说明训练分布失配是 M0 高缺失退化的主要组成部分，但不是全部原因：Hard train-adapt 后仍保留明显的 m=7 高缺失尾部退化。

---

## 3. 场景级 zero-shot 增量

相对于同场景 Hard train-adapt，zero-shot 的相对变化为：

| 场景 | minADE20 | minFDE20 | ADE@1 | FDE@1 | MR |
|---|---:|---:|---:|---:|---:|
| ETH-M | +107.5% | +105.8% | +16.4% | +11.0% | +81.5% |
| HOTEL-M | +51.4% | +50.5% | +7.7% | +9.0% | +43.6% |
| UNIV-M | +52.5% | +43.5% | **−2.2%** | **−2.8%** | +101.5% |
| ZARA1-M | +238.9% | +242.0% | +28.1% | +28.2% | +1013.0% |
| ZARA2-M | +108.4% | +86.0% | **−15.5%** | **−14.3%** | +335.5% |

这里出现了重要的指标分裂：UNIV 和 ZARA2 的 zero-shot ADE@1/FDE@1 略优于 Hard train-adapt，但 MR 明显恶化。这进一步说明：

```text
top-1 平均误差偶尔改善
≠
未来分布可靠性改善
```

MR 是本轮不可忽略的保护指标。

---

## 4. 诊断结论

### 4.1 训练分布失配已被确认

P2-A 证明：

```text
Easy train → Hard test
```

相对于：

```text
Hard train → Hard test
```

出现大幅 zero-shot 退化，尤其是：

- minADE20/minFDE20 约翻倍；
- MR 约增长 1.5 倍以上；
- ZARA1 的 zero-shot 退化最严重；
- UNIV/ZARA2 再次显示 top-1 与 MR 可能分裂。

因此，后续任何模型结构实验都不能只报告 train-adapt。必须同时报告 zero-shot 或 Mixed training 泛化。

### 4.2 训练适应不能解释全部高缺失失败

Hard train-adapt 后，宏平均仍为：

```text
minADE20/minFDE20 = 0.432/0.640
MR = 0.074
```

而 D0 已显示 Hard m=7：

```text
minADE20/minFDE20 = 0.886/1.271
ADE@1/FDE@1 = 3.138/4.752
MR = 0.187
```

因此即使训练分布匹配，valid_count=1、长 forecast gap 和长连续缺口仍然造成结构性困难。P2 不能替代 evidence-gated temporal state 或 reliability-conditioned decoder。

### 4.3 当前路线裁定

P2-A **通过，作为训练分布失配证据**，但不是模型改进结果。

P2-B Mixed 已完成并已补齐 Easy/Hard 交叉测试；完整结果见《[缺失分布交叉泛化结果](缺失分布交叉泛化结果.md)》。当前裁定：naive Easy+Hard 混合训练与 Hard-specific training 整体接近，但没有稳定优势；高缺失尾部仍然存在。

---

## 5. 产物与复现

每个场景结果保存于：

```text
outputs/p2_zero_shot_easy_to_hard_seed2024/
  M0-Easy2Hard_ETH-M/
  M0-Easy2Hard_HOTEL-M/
  M0-Easy2Hard_UNIV-M/
  M0-Easy2Hard_ZARA1-M/
  M0-Easy2Hard_ZARA2-M/
```

每个结果 JSON 均记录：

- Hard difficulty；
- Easy checkpoint 路径；
- K=20；
- direct evaluator；
- checkpoint missing/unexpected keys；
- scene、n 和 missing-count 分组。

本轮未修改模型代码、训练 checkpoint 或原有 M0 结果。
