# DeMo-direct 主协议首轮结果：M0-current × TrajImpute Easy/Hard（K=20）

> 时间：2026-09-09/10 ｜ 原始数据：`outputs/trajimpute_retrain/{easy,hard}/M0-current_<场景>_direct_seed2024/`（results.json/manifest/ckpt 各场景独立目录）

## 评估协议

- 数据：TrajImpute 官方 release（`/home/lbh/TrajImpute/dataset/TrajImpute`，ETH/UCY 五场景，官方 train/val/test 划分；Easy = 0–4 帧缺失，Hard = 4–7 帧缺失）
- 模型：DeMo M0-current（actor-only，bimamba=true；use_observation_features / use_missing_summary / use_motion_features 全 false；无缺失感知模块、无插补；掩码拼输入为 M0 内置）
- 训练：每场景独立 train_adapt 重训（official train 全量混合缺失样本，zero_missing_only=false），100 epochs，batch 64，lr 1e-3，wd 1e-4，bf16，seed=2024，checkpoint 按 val_minFDE20 选
- 指标：K=20（minADE20/minFDE20 同组 20 条取 min；ADE@1/FDE@1 最高概率模式；MR=全部 20 模式终点偏差>2.0；micro-average 全 focal 样本），米制、12 帧全程
- 执行记录：ETH-M 于 GPU0 双臂并行完成；方案 C 拆链（GPU0: easy-UNIV/ZARA1；GPU1: hard-UNIV/ZARA1；GPU3: easy-HOTEL/ZARA2 + hard-HOTEL/ZARA2）加速其余场景

## Easy-direct 结果（minADE20 / minFDE20 / MR / ADE@1 / FDE@1）

| 场景 | n | minADE20 | minFDE20 | MR | ADE@1 | FDE@1 |
|---|---:|---:|---:|---:|---:|---:|
| ETH-M | 905 | 0.562 | 0.871 | 0.135 | 0.970 | 1.868 |
| HOTEL-M | 5265 | 0.140 | 0.214 | 0.002 | 0.358 | 0.661 |
| UNIV-M | 121670 | 0.249 | 0.450 | 0.014 | 0.753 | 1.543 |
| ZARA1-M | 11265 | 0.179 | 0.309 | 0.009 | 0.477 | 1.002 |
| ZARA2-M | 29165 | 0.136 | 0.241 | 0.011 | 0.370 | 0.793 |
| **均值** | — | **0.253** | **0.417** | **0.034** | 0.582 | 1.373 |

## Hard-direct 结果（minADE20 / minFDE20 / MR / ADE@1 / FDE@1）

| 场景 | n | minADE20 | minFDE20 | MR | ADE@1 | FDE@1 |
|---|---:|---:|---:|---:|---:|---:|
| ETH-M | 724 | 0.789 | 1.101 | 0.169 | 1.912 | 3.080 |
| HOTEL-M | 4212 | 0.361 | 0.548 | 0.077 | 0.832 | 1.347 |
| UNIV-M | 97336 | 0.457 | 0.708 | 0.070 | 1.357 | 2.298 |
| ZARA1-M | 9012 | 0.319 | 0.489 | 0.027 | 1.349 | 2.269 |
| ZARA2-M | 23332 | 0.237 | 0.355 | 0.017 | 1.155 | 1.905 |
| **均值** | — | **0.433** | **0.640** | **0.072** | 1.321 | 2.180 |

## 与外部方法对比（ADE/FDE 均值，按 minFDE 升序）

### Easy

| 方法 | 均值 | 口径 |
|---|---:|---|
| LBEBM-ET (NeurIPS24 表3) | 0.232/0.376 | impute |
| **DeMo-direct M0-current（本文）** | **0.253/0.417** | direct |
| SGCN-ET | 0.248/0.424 | impute |
| MoFlow-direct（重训） | 0.265/0.406 | direct |
| GPGraph | 0.272/0.424 | impute |
| GraphTern | 0.318/0.428 | impute |
| EQmotion | 0.422/0.576 | impute |
| TUTR | 0.530/0.732 | impute |

### Hard

| 方法 | 均值 | 口径 |
|---|---:|---|
| EQmotion | 0.446/0.612 | impute |
| **DeMo-direct M0-current（本文）** | **0.433/0.640** | direct |
| MoFlow-direct（重训） | 0.475/0.680 | direct |
| GPGraph | 0.856/0.784 | impute |
| GraphTern | 0.858/0.878 | impute |
| LBEBM-ET | 1.088/1.448 | impute |
| TUTR | 1.180/1.520 | impute |
| SGCN-ET | 1.214/1.634 | impute |

## 口径与边界说明

- 论文基线为 impute 口径（SAITS 插补后训练+测试），本文与 MoFlow-direct 为 direct 口径（缺失直接输入不插补）；两口径输入信息不同，严格同口径仅 DeMo-direct 与 MoFlow-direct 互比
- Clean 参照：P0（Clean-direct LOSO 五折，minFDE20 均值 0.409，见 [P0_Clean-direct_M0-current_K20.md](P0_Clean-direct_M0-current_K20.md)）为不同协议（跨场景泛化 vs 场景内 train_adapt），不与本文同表
- 单种子（2024）首轮；确认性种子未跑
- 分组明细（按缺失帧数 0–4 / 4–7）见各场景 eval 目录 results.json 的 by_group 字段
