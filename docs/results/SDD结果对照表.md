# SDD 结果对照表（minADE20 / minFDE20，像素制，K=20，8 obs → 12 pred）

> 生成：2026-08-29。训练：单卡 RTX 5880 Ada（单向 GPU0 / 双向 GPU1，各 ~45 min，100 epochs）。
> 数据：`data/sdd/original/{sdd_train,sdd_test}.pkl`（与 MoFlow 官方 `/home/lbh/MoFlow/data/sdd` 同源，TUTR 版本）。
> held-out test：sdd_test.pkl，N=2829。train/val = sdd_train.pkl 确定性 90/10 划分（seed=2024），两臂完全一致。
> 汇总文件：`outputs/config_moflow_sdd_sdd_summary.txt`、`outputs/config_moflow_sdd_bimamba_sdd_summary.txt`。

## 1. 单向 vs 双向（核心对比）

| 臂 | 选点 epoch | val_minFDE20 | test_minADE20 | test_minFDE20 |
|---|---|---|---|---|
| DeMo-单向 (config_moflow_sdd) | 66 | 12.812 | **8.308** | **13.659** |
| DeMo-双向 (config_moflow_sdd_bimamba) | 24 | 12.981 | 9.839 | 17.547 |

| 对比 | minADE20 | minFDE20 |
|---|---|---|
| 双向 vs 单向 | +18.4% | +28.5%（正=双向更差） |
| val→test 泛化 gap | 单向 +6.6% | 双向 +35.2% |
| 单向 vs MoFlow 论文(教师) | +10.8% | +14.2% |
| 双向 vs MoFlow 论文(教师) | +31.2% | +46.7% |

## 2. SDD 基线对照（MoFlow 论文 project page，像素）

| 方法 | minADE20 | minFDE20 |
|---|---|---|
| ET+HighGraph | 7.81 | 11.09 |
| TUTR (AAAI23) | 7.76 | 12.69 |
| **MoFlow 教师 (CVPR25)** | **7.50** | 11.96 |
| MoFlow-IMLE | 7.85 | 12.86 |
| LED (CVPR23) | 8.48 | 11.66 |
| EigenTraj (CVPR23) | 8.05 | 13.25 |
| NPSN | 8.56 | 11.85 |
| **DeMo-单向（本文）** | **8.31** | **13.66** |
| **DeMo-双向（本文）** | **9.84** | **17.55** |
| SocialVAE | 8.88 | 14.81 |
| MID (WACV22) | 9.73 | 15.32 |
| MemoNet | 9.50 | 14.78 |
| CAGN | 9.42 | 15.93 |
| GroupNet | 9.31 | 16.11 |
| Y-net | 11.49 | 20.23 |

DeMo-单向在 SDD 上处于 EigenTraj 档（ADE 优于 NPSN/LED，FDE 略逊于 EigenTraj），距 MoFlow 教师 +10.8%/+14.2%——比 ETH/UCY 上的差距（+14.7%/+20.7%）更近。

## 3. 要点

1. **SDD 复证了 ETH/UCY 的结论：单向 mamba 更优**，且惩罚幅度更大（ETH/UCY 双向 +5.5%/+6.9%，SDD +18.4%/+28.5%）。双向在两个数据集上无一平均占优。
2. **双向的失败模式是泛化而非拟合**：两臂 val 几乎打平（12.812 vs 12.981，仅差 1.3%），但 test 上双向崩开（泛化 gap 单向 +6.6% vs 双向 +35.2%）。双向额外参数在训练分布上能追平，离开分布后明显更脆——这是「双向不是稳定增益」的跨数据集证据。
3. 双向最优 epoch 仅 24/100（单向 66/100），早期即过拟合，与第 2 点一致。
4. 协议对齐：单智能体（MoFlow 官方 SDD `agents: 1`）、8/12 帧、K=20 best-of-20、像素制、translate+rotate 刚性变换不改变 L2。选点 val_minFDE20，test 从未参与训练/选点。

## 4. 注意（与论文数字对比的口径）

- 单/双向两臂内部完全公平：同一数据划分（seed=2024）、同一训练配方（lr=1e-3, wd=1e-4, bs=64, 100 epochs, bf16）、唯一变量 bimamba 开关。
- MoFlow 官方 SDD 配方不同（150 epochs, lr=1e-4, AdamW wd=0.01, bs=128, fm 训练目标），故「DeMo vs 论文数字」的差距含训练配方差异，与 ETH/UCY 报告中的口径说明相同。
- SDD 无官方 val，90/10 划分是本仓库约定；引用时需注明。

## 5. 复现

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n DeMo python -u scripts/训练与评估/run_sdd.py config_moflow_sdd 100
CUDA_VISIBLE_DEVICES=1 conda run -n DeMo python -u scripts/训练与评估/run_sdd.py config_moflow_sdd_bimamba 100
```

最优 checkpoint 仍在（save_top_k=3）：
- 单向 `outputs/moflow_sdd/20260829-173529/checkpoints/epoch=66.ckpt`
- 双向 `outputs/moflow_sdd_bimamba/20260829-174637/checkpoints/epoch=24.ckpt`

## 相关文件

- ETH/UCY 主报告：`MoFlow与DeMo留出测试对比.md`
- ETH/UCY 对照表：`ETHUCY结果对照表.md`
- 论文数字来源：https://moflow-imle.github.io/（SDD 表）
