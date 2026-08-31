# ETH/UCY: MoFlow 论文 vs DeMo(单向 mamba / bimamba) held-out 测试公平对比

> 时间:2026-08-27
> 硬件:CUDA_VISIBLE_DEVICES=2(RTX 5880 Ada,单卡)。
> 结论:真实留出测试集上,DeMo 单向平均 minADE20/minFDE20 = **0.232 / 0.389**,双向 = **0.244 / 0.416**;MoFlow 论文教师模型 = **0.20 / 0.32**。DeMo 尚未胜出,且双向在平均值与 ETH 上比单向更差。

---

## 1. 最终结果(横向对比)

指标:minADE20 / minFDE20(米制,越低越好)。MoFlow 数字取自论文 project page 官方教师模型。

### minADE20

| 数据集 | MoFlow(教师) | MoFlow-IMLE | DeMo-单向 | DeMo-双向 |
|---|---|---|---|---|
| eth | 0.400 | 0.400 | 0.486 | 0.559 |
| hotel | 0.110 | 0.120 | 0.147 | 0.150 |
| univ | 0.230 | 0.230 | 0.239 | 0.227 |
| zara1 | 0.150 | 0.160 | 0.164 | 0.164 |
| zara2 | 0.120 | 0.130 | 0.123 | 0.121 |
| **平均** | **0.202** | **0.208** | **0.232** | **0.244** |

### minFDE20

| 数据集 | MoFlow(教师) | MoFlow-IMLE | DeMo-单向 | DeMo-双向 |
|---|---|---|---|---|
| eth | 0.570 | 0.580 | 0.767 | 0.928 |
| hotel | 0.170 | 0.180 | 0.217 | 0.215 |
| univ | 0.390 | 0.390 | 0.441 | 0.420 |
| zara1 | 0.260 | 0.260 | 0.293 | 0.293 |
| zara2 | 0.220 | 0.220 | 0.226 | 0.224 |
| **平均** | **0.322** | **0.326** | **0.389** | **0.416** |

---

## 2. 领域 SOTA 基线(平均 min20ADE/min20FDE,同表常见)

| 方法 | minADE20 | minFDE20 |
|---|---|---|
| MID | 0.21 | 0.38 |
| TUTR | 0.21 | 0.36 |
| EqMotion | 0.21 | 0.35 |
| EigenTraj | 0.21 | 0.34 |
| LED | 0.21 | 0.33 |
| SingularTraj | 0.21 | 0.32 |
| MoFlow(CVPR25) | 0.20 | 0.32 |
| DeMo-单向(本文) | 0.232 | 0.389 |
| DeMo-双向(本文) | 0.244 | 0.416 |

### 单向 vs 双向 逐场景相对变化（+ 为双向更差）

| 场景 | minADE20 | minFDE20 |
|---|---|---|
| eth | +15.2% | +21.0% |
| hotel | +2.6% | -1.1% |
| univ | -5.1% | -4.9% |
| zara1 | +0.0% | -0.0% |
| zara2 | -1.3% | -1.1% |
| 平均 | +5.5% | +6.9% |

场景间标准差（跨 5 折）：单向 ADE/FDE = 0.133/0.205，双向 = 0.161/0.266 —— 双向不仅均值更差，跨场景离散度也更大（主要被 ETH 拖累）。详细数值表：`MoFlow与DeMo留出测试数据.csv`。

数据来源与取证：`research/ETHUCY公开资料.md`、`audits/留出测试复核日志.log`。

---

## 3. 评估协议对齐(逐项核实,确保可直接比)

下面每一行都对照 MoFlow 官方源码/配置(`/home/lbh/MoFlow`)与 DeMo 侧核实:

| 协议项 | MoFlow 官方 | DeMo(本实验) | 一致 |
|---|---|---|---|
| 数据与切分 | SocialGAN Leave-One-Out,`<subset>_test.pkl` 为留出场景 | 直接读取 MoFlow 同一 pkl(Loo 划分) | ✓ |
| 预测时域 | past 8 / future 12 帧(cfg/eth_ucy/cor_fm.yml) | obs_len 8 / pred_len 12 | ✓ |
| 采样数 K | denoising_head_preds = 20(best-of-20) | num_modes = 20 | ✓ |
| minADE 选样 | 距离时序平均 → 对 K 取 min(`compute_ADE_FDE`) | 同(independent best-of-20) | ✓ |
| minFDE 选样 | 末帧距离 → 对 K 取 min | 同 | ✓ |
| 单位 | 米(用 `*_original_scale`,输入才做 min_max) | 米(局部系保持尺度) | ✓ |
| 坐标参考 | 仅平移到初始点 | 平移+旋转到朝向 | L2 刚性不变,可比 |
| 选点 | 训练期在 `<subset>_val.pkl`(其余 4 场景 val 划分)上,取 `val_minFDE20` 最优 epoch 的 ckpt | 同(monitor=`val_minFDE20`,save_top_k=3) | ✓ |

### 注意(重要)

- **本次所有 DeMo 数字均为 held-out 测试(<subset>_test.pkl,`test=true`)重算的结果**,不是训练期验证指标。
- 历史文件 `outputs/moflow_protocol_summary.txt` / `moflow_bimamba_protocol_summary.txt` 里记录的是 **`val_new_minADE20/minFDE20` 验证集指标**(在其余 4 场景的 val 划分上测),数值偏低且五折扁平,**不能与 MoFlow 论文直接比**,勿直接引用。
- DeMo 重算命令见第 5 节。

---

## 4. 对比实验配置

### 4.1 DeMo 训练(两臂,唯一变量 = 单向/双向 mamba)

| 配置项 | 单向 mamba | 双向 mamba(bimamba) |
|---|---|---|
| Hydra 配置 | `conf/config_moflow_ethucy.yaml` | `conf/config_moflow_ethucy_bimamba.yaml` |
| 模型 | `ModelForecast`(moflow_ethucy_model_forecast) | 同上,`bimamba=True`(moflow_ethucy_bimamba_model_forecast) |
| 主干 | 单向 mamba(vim_mamba) + Transformer blocks + decoupled-query time decoder | 双向 mamba(`bimamba=True`) |
| 输出 | `num_modes=20` 条轨迹 + 概率 `pi` | 同 |
| 训练超参 | lr=0.001, weight_decay=0.0001, clip=5(norm), bs=64, epochs=100, warmup=5, bf16 | 同 |
| 数据 | `/home/lbh/MoFlow/data/eth_ucy/original`,LOO 每折一个 subset | 同 |
| 输出目录 | `outputs/moflow_<subset>/` | `outputs/moflow_bimamba_<subset>/` |
| 训练命令 | `conda run -n DeMo python train.py --config-name config_moflow_ethucy datamodule.target.subset=<subset> epochs=100` | 同,`--config-name config_moflow_ethucy_bimamba` |
| 批量脚本 | `scripts/训练与评估/run_moflow_protocol.py` | `scripts/训练与评估/run_moflow_protocol_bimamba.py` |

每折 100 epochs,共 5 折(eth/hotel/univ/zara1/zara2),约 15–16 小时/臂(单卡 RTX 5880 Ada)。

### 4.2 MoFlow 论文数字来源

- ETH/UCY:min20ADE/min20FDE(米),取论文官方 project page(https://moflow-imle.github.io/)教师模型表。
- 官方代码:`/home/lbh/MoFlow`,指标函数 `trainer/denoising_model_trainers.py:compute_ADE_FDE`、数据 `data/dataloader_eth_ucy.py`(metric 用 `*_original_scale`,输入才 min_max)。

---

## 5. 评测复现方法(DeMo held-out test 重算)

用每折训练出的 best ckpt(monitor=`val_minFDE20`),在留出测试场景上重算:

```bash
# 单向 (config_moflow_ethucy), 双向用 config_moflow_ethucy_bimamba
CUDA_VISIBLE_DEVICES=2 conda run -n DeMo python -u eval.py \
    --config-name config_moflow_ethucy \
    gpus=1 test=true \
    datamodule.target.subset=<eth|hotel|univ|zara1|zara2> \
    checkpoint=outputs/moflow_<subset>/<ts>/checkpoints/epoch=<best>.ckpt
```

- 批量脚本:`scripts/训练与评估/run_heldout_test.py`(会自动为各折创建无 `=` 的 symlink 以规避 Hydra 解析,输出每折 `N` / `test_minADE20` / `test_minFDE20`)。
- 测试样本量(held-out `<subset>_test.pkl`):eth=181, hotel=1053, univ=24334, zara1=2253, zara2=5833。
- 读取 `TEST METRICS` 中的 `test_minADE20` / `test_minFDE20`。

---

## 6. 结论要点

1. 协议对齐下,DeMo 单向(0.232/0.389)、双向(0.244/0.416)均落在 SOTA(minADE20 0.20–0.21,minFDE20 0.32–0.38)下方/贴近,**尚未超越 MoFlow(0.20/0.32)**。
2. **双向 mamba 不是稳定增益**:平均更差(0.244/0.416 vs 0.232/0.389),ETH 退化显著(单向 0.486/0.767 → 双向 0.559/0.928,ADE/FDE 均变差)。双向仅在 univ 的 minADE20(0.227)及个别 FDE 上略优。
3. ETH 是最大短板(DeMo 0.486/0.767 vs 论文 0.40/0.57),是后续可优化重点之一。
4. 写论文引用此表时,须注明「同一 LOO 数据、同一评测管线,仅主干不同」。

---

## 7. 可达性审查附录(数据泄漏 / 选点 / K / 坐标尺度)

逐项核查代码(2026-08-27):

1. **数据泄漏**: `moflow_ethucy_dataset.py` 严格按 split 读 `<subset>_{train,val,test}.pkl` — train/val 来自其余 4 场景,test 为留出场景,focal 局部坐标变换只用 obs 段信息(`positions[0, obs_len-1]` 与前一帧朝向),不接触 future → **无泄漏**。
2. **测试集参与选点**: `setup()` 仅当 val pkl 缺失才回退 test pkl;MoFlow 官方 pkl 五折均有 val,且实际 val 指标 ≠ test 指标(见历史 summary 与本次重跑差异),证明选点走的是 `val_minFDE20`(其余 4 场景 val),test 场景从未参与训练/选点 → **合规**。
3. **K=20 best-of-20**: `src/metrics/min_ade.py:36-40` — `sort_predictions` 按概率取 top-K 后对 K 条独立取 min(ADE 按 mean-over-time、FDE 按 endpoint,`min_fde.py` 同构),与 MoFlow `compute_ADE_FDE` 的 independent best-of-20 等价 → **一致**。另 minADE 选样口径(按 minADE 自身取 min)与 MoFlow 相同。
4. **坐标尺度**: pkl 存的是世界坐标(米),DeMo 不做 min_max 归一化,局部系仅平移+旋转(刚性),L2 距离不变;指标直接在米制局部系计算,等价于 MoFlow 的 `*_original_scale` → **一致**。
5. **遗留注意**: 该回退分支(val→test)是潜在坑——若换用无 val pkl 的数据,选点将泄漏到 test;MoFlow 协议下未触发,但建议后续训练脚本加 assert 阻断。
