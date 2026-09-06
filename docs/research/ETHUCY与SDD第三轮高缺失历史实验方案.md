# ETH/UCY 与 SDD 第三轮高缺失历史实验方案

_实验执行方案 · 制定日期：2026-09-05 · 数据协议：`missing_history_v3_noguard`_

## 1. 实验定位

第三轮实验研究**历史末端观测也可能缺失**时的行人轨迹预测问题。与第二轮 v2 固定历史帧 6、7 可见不同，v3 将缺失候选范围扩展到完整历史窗口 0–7，并以最后一个可见位置作为局部坐标锚点。

第三轮主要回答三个问题：

1. 完整历史训练的模型在末端观测缺失时如何退化；
2. 训练期接触无保护缺失后，模型能否适应锚点滞后和观测间隔变化；
3. 显式建模缺失位置、有效运动关系和最后观测间隔，能否缓解高缺失预测退化。

本轮不把“缺失帧数越多”作为唯一难度变量，而是同时分析：

```text
缺失帧数
+ 缺失是否连续
+ 最后有效观测距历史窗口末端的间隔
+ 最后有效观测距未来首帧的间隔
```

## 2. 数据协议

### 2.1 数据目录

```text
ETH/UCY : data/ETHUCY_missing_v3_noguard/
SDD     : data/SDD_missing_v3_noguard/
```

历史长度为 8 帧，未来长度为 12 帧，时间间隔为 `0.4s`。未来轨迹保持完整，缺失只发生在历史窗口。

### 2.2 缺失条件

| 条件 | 缺失规则 | 缺失比例 | 用途 |
|---|---|---:|---|
| `random_fixed3_ng` | 从历史帧 0–7 随机缺失 3 帧 | 37.5% | 中等离散缺失 |
| `random_fixed4_ng` | 从历史帧 0–7 随机缺失 4 帧 | 50% | 高比例离散缺失 |
| `random_block3_ng` | 随机位置连续缺失 3 帧 | 37.5% | 中等连续遮挡 |
| `random_block4_ng` | 随机位置连续缺失 4 帧 | 50% | 长连续遮挡 |
| `random_block6_ng` | 随机位置连续缺失 6 帧 | 75% | 极端连续缺失 |
| `uniform_hard_ng` | 每个 actor 随机缺失 4、5、6 或 7 帧 | 50%–87.5% | 混合困难条件 |

每个真实 actor 至少保留一个可见历史帧。

### 2.3 时间间隔变量

记最后一个可见历史帧为 `last_valid_idx`：

```text
anchor_lag_steps   = 7 - last_valid_idx
forecast_gap_steps = 8 - last_valid_idx
```

- `anchor_lag_steps` 描述最后有效观测距离历史窗口末端有多远；
- `forecast_gap_steps` 描述最后有效观测距离未来首帧有多远。

第三轮的主要分层变量采用 `anchor_lag_steps`，`forecast_gap_steps` 作为等价时间解释。

## 3. 实验总体结构

第三轮分为三部分：

```text
阶段 A：旧模型零样本基线
阶段 B：旧模型训练适应基线
阶段 C：缺失感知模型对照
```

三部分独立汇报，不能将训练适应与零样本结果混合平均。

## 4. 阶段 A：零样本基线

### 4.1 目的

使用完整历史数据训练得到的 DeMo checkpoint，直接评估 v3 无保护测试集，不进行重新训练。该阶段测量模型面对末端历史缺失和锚点变化时的直接鲁棒性。

### 4.2 ETH/UCY 实验矩阵

| 训练数据 | 测试数据 | 模型 | Fold |
|---|---|---|---|
| v1 `complete` | v3 六个条件 | DeMo 单向 | ETH、HOTEL、UNIV、ZARA1、ZARA2 |
| v1 `complete` | v3 六个条件 | DeMo 双向 | ETH、HOTEL、UNIV、ZARA1、ZARA2 |

总计：

```text
2 个模型 × 6 个条件 × 5 个 fold = 60 次评估
```

执行入口：

```bash
cd /home/lbh/DeMo

bash scripts/训练与评估/run_zeroshot_v3.sh 0 \
  random_fixed3_ng random_fixed4_ng \
  random_block3_ng random_block4_ng random_block6_ng \
  uniform_hard_ng
```

现有脚本同时评估单向和双向模型。结果输出到：

```text
outputs/ethucy_zeroshot_v3_<direction>_<condition>/
outputs/missing_v3_zeroshot_logs/
```

### 4.3 SDD 实验矩阵

| 训练数据 | 测试数据 | 模型 | Seed |
|---|---|---|---:|
| v1 `complete` | v3 六个条件 | DeMo 单向 | 2024 |
| v1 `complete` | v3 六个条件 | DeMo 双向 | 2024 |

总计：

```text
2 个模型 × 6 个条件 = 12 次评估
```

SDD 使用完整历史模型自身验证集选出的 checkpoint。建议新增独立入口：

```text
scripts/训练与评估/run_zeroshot_v3_sdd.py
```

该脚本应完成：

```text
读取 complete checkpoint
→ 切换 data_root 到 data/SDD_missing_v3_noguard
→ 依次评估六个 condition
→ 输出每个 condition 的结构化 JSON/CSV
```

## 5. 阶段 B：训练适应基线

### 5.1 目的

在每个 v3 condition 的 train/val 上重新训练，并在相同 condition 的 test 上评估，测量普通 DeMo 在已知无保护缺失分布下的适应能力。

### 5.2 第一批代表条件

第一批只训练三个代表条件：

| 条件 | 选择原因 |
|---|---|
| `random_fixed4_ng` | 代表 50% 离散缺失 |
| `random_block6_ng` | 代表 75% 极端连续缺失 |
| `uniform_hard_ng` | 代表缺失率和缺失位置混合变化 |

第一批完成后，再根据结果决定是否补齐 `fixed3/block3/block4`。

### 5.3 ETH/UCY 矩阵

```text
模型：DeMo 单向、DeMo 双向
条件：fixed4_ng、block6_ng、uniform_hard_ng
Fold：5
Seed：2024
```

总计：

```text
2 × 3 × 5 = 30 次训练与测试
```

单向示例：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
python scripts/训练与评估/run_ethucy_benchmark.py \
  --config-name config_ethucy_benchmark \
  --data-root data/ETHUCY_missing_v3_noguard/random_fixed4_ng \
  --output-root outputs/ethucy_v3_uni_random_fixed4_ng \
  --gpu 0 \
  --seed 2024
```

双向示例：

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. \
python scripts/训练与评估/run_ethucy_benchmark.py \
  --config-name config_ethucy_benchmark_bimamba \
  --data-root data/ETHUCY_missing_v3_noguard/random_fixed4_ng \
  --output-root outputs/ethucy_v3_bi_random_fixed4_ng \
  --gpu 1 \
  --seed 2024
```

其余条件只替换 `data-root` 和 `output-root`。

### 5.4 SDD 矩阵

```text
模型：DeMo 单向、DeMo 双向
条件：fixed4_ng、block6_ng、uniform_hard_ng
Seed：2024
```

单向示例：

```bash
CUDA_VISIBLE_DEVICES=0 \
python scripts/训练与评估/run_sdd_missing.py \
  random_fixed4_ng 100 data/SDD_missing_v3_noguard
```

双向示例：

```bash
CUDA_VISIBLE_DEVICES=1 \
python scripts/训练与评估/run_sdd_missing_bimamba.py \
  random_fixed4_ng 100 data/SDD_missing_v3_noguard
```

## 6. 阶段 C：缺失感知模型实验

阶段 C 在 M0/M1/M2 模型接口完成后启动。

### 6.1 模型定义

| 模型 | 输入和条件化方式 | 研究问题 |
|---|---|---|
| `M0_base` | 当前 DeMo 历史输入 | 基准性能 |
| `M1_obs` | 增加逐时间步缺失间隔和有效运动特征 | 显式时间步缺失信息是否有效 |
| `M2_history` | 在 M1 上增加历史级缺失摘要 | 整体缺失结构是否提供额外价值 |

State Query、Mode Query 和 Hybrid Coupling 条件化留到 M1/M2 结果明确后再开展。

### 6.2 第一轮筛选矩阵

第一轮使用双向骨干，原因是现有 v2 零样本实验已经表明双向模型在高缺失下更稳定。

```text
模型：M0_base、M1_obs、M2_history
条件：random_fixed4_ng、random_block6_ng、uniform_hard_ng
Fold：ETH/UCY 五折
Seed：2024
```

比较顺序：

```text
M1_obs - M0_base
M2_history - M1_obs
```

若 M1/M2 在至少一个连续高缺失条件上形成稳定改善，再进入 Query 和 Hybrid 条件化实验。

### 6.3 混合条件训练

缺失特征模型不能只在 complete 数据上训练，因为 complete 中缺失特征基本为常数。正式方法需要增加混合条件训练：

```text
训练条件：
complete
+ random_fixed3_ng
+ random_fixed4_ng
+ random_block3_ng
+ random_block4_ng

已见条件测试：
fixed3_ng、fixed4_ng、block3_ng、block4_ng

未见困难条件测试：
block6_ng、uniform_hard_ng
```

该实验用于判断模型是否学习了可泛化的缺失处理方式，而不是记住单一掩码分布。

## 7. 评价指标

### 7.1 主指标

ETH/UCY：

```text
minADE6
minFDE6
MR
b-minFDE6
```

SDD：

```text
minADE20
minFDE20
MR
b-minFDE20
```

主要终点采用 `minFDE`，MR 用于判断多模态预测是否覆盖真实未来。

### 7.2 三类相对变化

每个结果同时计算：

```text
1. 相对 complete 的退化；
2. v3 相对同缺失率 v2 的额外退化；
3. 缺失感知模型相对 M0_base 的改善。
```

## 8. 分层分析

### 8.1 按锚点滞后分层

按 focal actor 的 `anchor_lag_steps` 分组：

```text
lag = 0, 1, 2, ..., 7
```

每层报告：

```text
样本数
minADE
minFDE
MR
相对 lag=0 的变化
```

主图：

```text
anchor_lag_steps → minFDE
anchor_lag_steps → MR
```

该分析用于区分“总缺失量的影响”和“最近观测过旧的影响”。

### 8.2 随机缺失与连续缺失

同缺失率配对比较：

```text
random_fixed3_ng vs random_block3_ng
random_fixed4_ng vs random_block4_ng
```

分别报告整体结果，并在相同 `anchor_lag_steps` 层内再次比较。

### 8.3 `uniform_hard_ng` 分层

按照每个 focal actor 的实际缺失帧数分为：

```text
m=4
m=5
m=6
m=7
```

除混合均值外，必须报告四个子组的样本数和指标。

## 9. 结果表格

第三轮最终形成四张主要结果表。

### 表 1：零样本整体结果

```text
数据集 × 模型 × condition × minADE/minFDE/MR
```

### 表 2：训练适应结果

```text
数据集 × 模型 × condition × complete 相对变化
```

### 表 3：缺失感知模型消融

```text
M0_base / M1_obs / M2_history
× 三个代表条件
× 五折均值与配对差值
```

### 表 4：时间间隔和缺失结构分析

```text
anchor_lag 分层
fixed vs block
uniform_hard 的 m=4/5/6/7 分层
```

## 10. 执行顺序

```text
1. 完成 ETH/UCY 六条件零样本结果；
2. 完成 SDD 六条件零样本结果；
3. 实现 anchor_lag 和缺失帧数分层评估；
4. 汇总第三轮零样本基线；
5. 训练三个代表条件的单向/双向适应基线；
6. 完成 M0/M1/M2 后进行缺失感知模型筛选；
7. 运行混合条件训练和未见条件测试；
8. 根据 M1/M2 结果决定是否进入 Query/Hybrid 条件化。
```

## 11. 结论目标

第三轮最终应回答：

> 当最后有效历史观测逐渐远离预测起点时，DeMo 的多模态预测性能如何退化；显式建模观测间隔、有效运动关系和整体缺失结构，能否在训练适应与未见缺失条件下缓解这种退化。

第三轮不是单独追求某个 condition 的最低误差，而是建立以下关系：

```text
观测缺失结构
→ 最后有效观测间隔
→ 历史运动证据质量
→ 多模态未来预测退化
→ 缺失感知建模的改善幅度
```

