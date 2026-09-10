# P0：M0-current 完整历史基础能力（Clean-direct）

> 时间：2026-09-08/09 ｜ 原始数据：`outputs/clean_ethucy/M0-current_k20/M0-current/complete/seed_2024/`（results.json/csv、各折 fold_result.json、manifest、ckpt）

## 评估协议

- 数据：`data/ETHUCY_benchmark_v1`（原始完整 ETH/UCY，8 帧历史 → 12 帧未来，无缺失）
- 划分：五折留一场景（LOSO：ETH / HOTEL / UNIV / ZARA1 / ZARA2 轮流作测试集），各折使用官方 train/val/test 目录
- 模型：DeMo M0-current（actor-only，bimamba=true；use_observation_features / use_missing_summary / use_motion_features 全 false，无缺失感知模块、无插补）
- 训练：100 epochs，batch 64，lr 1e-3，wd 1e-4，bf16，seed=2024；checkpoint 按 val_minFDE20 选择
- 指标：K=20（minADE20/minFDE20 同组 20 条取 min；ADE@1/FDE@1 为最高概率模式；MR 阈值 2.0m；b-minFDE20 为最佳概率模式 FDE），米制、12 帧全程

## 五折结果

| fold | checkpoint epoch | minADE20 | minFDE20 | MR | b-minFDE20 | ADE@1 | FDE@1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ETH | 67 | 0.532 | 0.908 | 0.0962 | 1.805 | 1.030 | 2.059 |
| HOTEL | 76 | 0.145 | 0.223 | 0.0017 | 1.039 | 0.408 | 0.783 |
| UNIV | 97 | 0.226 | 0.413 | 0.0116 | 1.301 | 0.613 | 1.330 |
| ZARA1 | 89 | 0.159 | 0.285 | 0.0051 | 1.168 | 0.425 | 0.923 |
| ZARA2 | 91 | 0.119 | 0.217 | 0.0080 | 1.066 | 0.311 | 0.700 |
| **均值** | — | **0.236** | **0.409** | **0.0245** | **1.276** | **0.557** | **1.159** |
| 标准差 | — | 0.170 | 0.290 | 0.0402 | 0.313 | 0.286 | 0.558 |

- 五折 status 全部 ok，无失败折
- 单种子（2024）首轮；候选确认种子（2025/2026）未跑
- 协议为 LOSO 跨场景泛化，与 TrajImpute 场景内划分的 Clean 不可直接同表比较
