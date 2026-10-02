# C3 与 M0、C1-A 对照结果

> 2026-09-26 更新｜TrajImpute Hard-direct，单种子 screening；C3 原始产物位于 `outputs/c3_hard_screening_seed2024/`，汇总产物位于 `outputs/c3_hard_screening_seed2024/summary/`。

## 1. 评估协议

- 数据：TrajImpute 官方 release；历史 8 帧，未来 12 帧；缺失历史直接输入，不经过外部插补。
- 场景：`ETH-M / HOTEL-M / UNIV-M / ZARA1-M / ZARA2-M`，5/5 个 `results.json` 完整。
- 训练：Hard train-adapt；seed=2024；100 epochs；batch size 64；K=20；单向 Mamba；`val_minFDE20` 选点。
- 模型变量：
  - M0：`readout_mode=last`、`evidence_state_mode=none`；
  - C1-A：`readout_mode=last_valid`、`evidence_state_mode=none`；
  - C3：`readout_mode=last`、`evidence_state_mode=observed_write_gap`。
- C3 仅增加 observed-write / gap-propagation 状态机制；原 Mamba、Scene Transformer、decoder、loss、K=20 和评估器保持不变。
- 指标：`minADE20`、`minFDE20`、`ADE@1`、`FDE@1`、`MR`；同时报告五场景宏平均和样本加权微平均。

## 2. 五场景宏平均

| 模型 | minADE20 | minFDE20 | ADE@1 | FDE@1 | MR |
|---|---:|---:|---:|---:|---:|
| M0 | 0.4664 | 0.6968 | 1.3179 | 2.1649 | 0.0879 |
| C1-A | **0.4355** | **0.6538** | **1.2862** | **2.1300** | **0.0755** |
| C3 | 0.4423 | 0.6533 | 1.3022 | 2.1467 | 0.0781 |

### 相对 M0

| 模型 | ΔminADE20 | ΔminFDE20 | ΔADE@1 | ΔFDE@1 | ΔMR |
|---|---:|---:|---:|---:|---:|
| C3 vs M0 | −5.16% | −6.23% | −1.19% | −0.84% | −11.11% |

### 相对 C1-A

| 模型 | ΔminADE20 | ΔminFDE20 | ΔADE@1 | ΔFDE@1 | ΔMR |
|---|---:|---:|---:|---:|---:|
| C3 vs C1-A | +1.56% | −0.07% | +1.25% | +0.79% | +3.46% |

## 3. 五场景微平均

| 模型 | n | minADE20 | minFDE20 | ADE@1 | FDE@1 | MR |
|---|---:|---:|---:|---:|---:|---:|
| M0 | 134,616 | 0.5280 | 0.8385 | 1.3712 | 2.2672 | 0.1156 |
| C1-A | 134,616 | **0.4123** | **0.6416** | **1.2049** | **2.0328** | **0.0644** |
| C3 | 134,616 | 0.4216 | 0.6531 | 1.2739 | 2.1511 | 0.0653 |

相对 M0，C3 微平均改善 `minADE20 −20.15%`、`minFDE20 −22.11%`、`ADE@1 −7.09%`、`FDE@1 −5.12%`、`MR −43.52%`；相对 C1-A 则五项均未改善，分别为 `+2.27%`、`+1.80%`、`+5.73%`、`+5.82%`、`+1.37%`。

## 4. 逐场景结论

- C3 相对 M0：UNIV-M 改善最大；ZARA2-M 的 best-of-K 指标改善；ETH-M/HOTEL-M 的 best-of-K 指标回归但 top-1 略有改善。
- C3 相对 C1-A：C3 在 ZARA1-M/ZARA2-M 的 best-of-K 指标更好，在 ETH-M 的 top-1 略好；但 UNIV-M 和 HOTEL-M 的整体指标不如 C1-A。
- 因此，C3 相对 M0 有明确 screening 信号，但相对当前最强 control C1-A 没有稳定跨场景优势。

## 5. Hard 分组归因

以下为五场景按样本数加权的 C3 相对变化，负值表示 C3 更优。

| 分组 | ΔminFDE20 vs M0 | ΔminFDE20 vs C1-A | ΔFDE@1 vs M0 | ΔFDE@1 vs C1-A | ΔMR vs C1-A |
|---|---:|---:|---:|---:|---:|
| missing=4 / valid=4 | −30.47% | −0.04% | −10.63% | +2.62% | −36.42% |
| missing=5 / valid=3 | −28.41% | +0.26% | −13.20% | +1.58% | −32.08% |
| missing=6 / valid=2 | −27.76% | −0.84% | −19.08% | +1.42% | −28.84% |
| missing=7 / valid=1 | −13.37% | **+4.05%** | +5.57% | **+9.70%** | **+13.22%** |

`anchor_lag` / `forecast_gap` 的趋势一致：C3 相对 M0 在短 gap 和中等 gap 上改善明显，但相对 C1-A 的差距在长 gap 处扩大；`anchor_lag=7` 时 C3 的 minFDE20 约差 `+4.53%`，FDE@1 约差 `+5.27%`，MR 约差 `+11.23%`。

## 6. 裁定

1. **C3 超过 M0，但未超过 C1-A。** 单种子 Hard screening 不能把 C3 写成已验证主方法。
2. **C3 的 observed-write / gap-propagation 机制尚未证明具有独立增量。** 预注册的 C2 gap-only control 尚未完成，不能把 C3 相对 M0 的收益全部归因于 observed-write。
3. **当前主要失败区仍是 `valid_count=1` / `missing=7` / 长 gap。** 这一区间相对 C1-A 发生 top-1 和 MR 回归，说明仅改历史状态传播仍不足以解决未来 mode 排序可靠性。
4. **下一步顺序固定为：** 先完成 C2 gap-only control；若 C3 超过 C2，再实现 C4 reliability-conditioned mode logit；若 C3 不超过 C2，停止继续添加 dual state 或社会模块。
5. 当前论文定位保持为：C3 是有正向信号的模型候选，贡献等级不超过 `L1/L2 之间`；需要 C2、C4、多种子和 unseen-pattern 验证后才能升级。
