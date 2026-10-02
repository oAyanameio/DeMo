# TrajGap-Bench：统一混合缺失历史直接预测 Benchmark 设计

> 文档性质：Benchmark（基准）设计与当前实施协议
>
> 当前状态：统一 Mixed 方案，暂不加入其他扩展轨道
>
> 数据集名称：**TrajGap**（TrajImpute-derived Mixed Trajectory Dataset）
>
> Benchmark 名称：**TrajGap-Bench**（Mixed Missing-History Direct Forecasting Benchmark）
>
> 底层数据：TrajImpute 官方 Easy/Hard release
>
> 研究任务：缺失历史条件下的直接未来轨迹预测（direct / imputation-free trajectory forecasting）

## 1. 当前协议定位

TrajGap-Bench 对外不再区分 Easy、Hard 或 Clean。TrajImpute 仍是底层数据来源，但 Easy/Hard 目录只作为文件来源；训练、验证和测试均由 TrajGap 适配器合并为一个 Mixed 数据集。

本阶段只落实以下设想：

1. 将官方 Easy/Hard 的同一 split 合并为 Mixed；
2. 训练、验证和测试均使用 Mixed 口径；
3. 验证集只使用一个 Mixed 验证集进行 checkpoint 选点；
4. 主评估只报告一个 Mixed 结果和一个综合总榜分数；
5. 暂不加入 unseen-pattern、额外数据轨道、Clean 主分支或额外评价指标。

TrajGap 不修改官方 pkl，不重新切窗，也不重新生成未来目标；只在适配层进行样本级合并。官方 impute-then-predict 数字不进入 TrajGap-Bench direct 总榜。

## 2. 数据集组织：统一 Mixed split

对每个场景 `scene ∈ {ETH-M, HOTEL-M, UNIV-M, ZARA1-M, ZARA2-M}`，TrajGap 使用官方 release 的同一 split 合并：

```text
TrajGap/<scene>/train = Easy/data_train.pkl + Hard/data_train.pkl
TrajGap/<scene>/val   = Easy/data_val.pkl   + Hard/data_val.pkl
TrajGap/<scene>/test  = Easy/data_test.pkl  + Hard/data_test.pkl
```

这里的 `+` 是 actor-centric 样本级拼接，不是重新切窗，也不是去重后重采样。Easy/Hard 的来源标签保留在内部审计信息中，对外协议名称统一为 `Mixed`。

### 2.1 训练集

```text
Mixed-train = Easy/train + Hard/train
```

当前采用自然拼接（natural Mixed），不使用 severity-balanced 或 evidence-balanced 采样。两个底层数据源沿用官方场景训练边界，合并不改变场景划分。

### 2.2 验证集

```text
Mixed-val = Easy/val + Hard/val
```

验证集不再拆成 Easy-val 和 Hard-val，也不对两部分分别选 checkpoint。训练期间只计算一个 `val_minFDE20`，根据合并后的 Mixed-val 选择最佳 checkpoint。

这样可以保证：

1. checkpoint 选择目标与 Mixed 主测试口径一致；
2. 不因分别查看 Easy/Hard 验证结果而产生多重选点；
3. 不使用测试集决定 Easy/Hard 权重或 checkpoint；
4. 缺失程度差异只作为测试结果的诊断分组，不改变主验证协议。

### 2.3 测试集

```text
Mixed-test = Easy/test + Hard/test
```

评估器只输出一套 Mixed 主结果。`missing_count`、`valid_count`、`anchor_lag`、`forecast_gap` 等字段仍保留用于内部诊断，但不再输出 Easy 与 Hard 两个主榜。

当前已审计的 focal 样本规模如下：

| 场景 | Mixed train | Mixed val | Mixed test |
|---|---:|---:|---:|
| ETH-M | 59,618 | 10,698 | 1,629 |
| HOTEL-M | 58,304 | 10,272 | 9,477 |
| UNIV-M | 18,462 | 5,416 | 219,006 |
| ZARA1-M | 56,020 | 10,236 | 20,277 |
| ZARA2-M | 51,014 | 8,346 | 52,497 |

这些数量是官方 Easy/Hard 文件的样本级相加；测试集中的缺失块重复结构保持不变，不将其误称为独立原始轨迹数量。

### 2.4 数据适配规则

- 读取官方 Easy/Hard pkl，不修改源文件；
- 历史缺失坐标仍由 `missing_mask` 定义；
- 未来 `pred_traj` 保持完整，只用于监督和评估；
- 对 Easy/Hard 文件分别修复已知的 `seq_start_end` block offset 问题，再进行样本合并；
- focal actor、邻居分组、局部坐标和跨缺口运动特征沿用原适配器；
- 对外 `scene_id` 使用 `scene-Mixed-split-*`；
- 不把 Easy 的零缺失副本定义为 Clean；本阶段不建立独立 Clean 主分支。

## 3. 训练、验证、测试流程

```text
Easy/data_train.pkl ─┐
                     ├─> TrajGap Mixed-train ─┐
Hard/data_train.pkl ─┘                       │
                                             ├─> 模型训练
Easy/data_val.pkl   ─┐                       │
                     ├─> TrajGap Mixed-val ──┘
Hard/data_val.pkl   ─┘                       │
                                             └─> val_minFDE20 选 checkpoint

Easy/data_test.pkl  ─┐
                     ├─> TrajGap Mixed-test ─> 单一主结果 / 单一总榜
Hard/data_test.pkl  ─┘
```

训练、验证和测试使用相同的 8 帧历史、12 帧未来、direct 输入和 `K=20` 设置。测试集不参与 checkpoint 选择。

## 4. 评估指标与单一总榜

当前主榜只纳入三个指标：

```text
minFDE20
minADE20
MR
```

三者均为越低越好：

| 指标 | 含义 |
|---|---|
| `minFDE20` | 20 条候选未来中最接近 GT 的终点误差 |
| `minADE20` | 20 条候选未来中最接近 GT 的平均位移误差 |
| `MR` | 20 条候选均未达到终点阈值的比例 |

Benchmark 采用一个综合总榜（single composite leaderboard），不拆分多个排行榜。总榜规则：

1. 三个指标来自同一 Mixed-test、同一 checkpoint、同一场景划分和同一 `K=20`；
2. 总榜同时展示三个原始指标，综合分数不能替代原始结果；
3. 至少报告五场景 macro average、micro average、场景明细和 worst-scene；
4. 若不同方法综合分数接近，检查三项指标是否分裂，不把微小差异解释为稳定优势。

在正式发布综合分数前，必须冻结归一化参考范围，不能根据测试结果临时选择上下界。对方法 `m` 和指标 `j`，建议使用预先冻结的 min-max 归一化：

\[
z_{m,j}=\frac{x_{m,j}-x^{best}_{j}}
              {x^{worst}_{j}-x^{best}_{j}},
\qquad j\in\{\text{minFDE20},\text{minADE20},\text{MR}\}.
\]

综合分数为：

\[
S_m=\frac{1}{3}\left(z_{m,\text{minFDE20}}+
z_{m,\text{minADE20}}+z_{m,\text{MR}}\right).
\]

`S_m` 越低越好，排行榜按 `S_m` 升序排列。

`ADE@1`、`FDE@1`、`b-minFDE20`、Energy Score、coverage、calibration 等暂不进入当前总榜，也不作为本阶段新增协议要求。

## 5. 当前代码对应关系

| 功能 | 实现 |
|---|---|
| Mixed 数据集 | `src/datamodule/trajimpute_dataset.py::TrajGapDataset` |
| Mixed 训练/验证 | `TrajImputeDataModule` 接收 `train_difficulties=[Mixed]`、`val_difficulties=[Mixed]` |
| Mixed 测试 | `TrajGapDataset(..., split="test")` |
| 统一训练入口 | `scripts/训练与评估/run_trajimpute_experiments.py --protocol mixed-direct` |
| 统一 direct 评估 | `scripts/结果分析/evaluate_trajimpute_direct.py --difficulty Mixed` |
| checkpoint 选点 | 合并后的 Mixed-val 上的 `val_minFDE20` |

推荐入口：

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=. \
/home/lbh/.conda/envs/DeMo/bin/python \
scripts/训练与评估/run_trajimpute_experiments.py \
  --protocol mixed-direct \
  --variant M0 \
  --scenes ETH-M HOTEL-M UNIV-M ZARA1-M ZARA2-M \
  --seed 2024 \
  --gpu 3 \
  --output-root outputs/trajgap_bench_m0_seed2024
```

## 6. 当前范围边界

本阶段只实现统一 Mixed 数据组织和单一 Mixed 验证/测试口径，暂不实现：

- Easy/Hard 独立主榜；
- Clean 主训练或主测试分支；
- unseen-pattern 数据扩展；
- severity-balanced 或 evidence-balanced 主训练；
- 额外 direct/impute 双轨排行榜；
- 除 `minFDE20`、`minADE20`、`MR` 外的总榜指标。

当前最准确的定位是：

> **TrajGap 将 TrajImpute 官方 Easy/Hard 的同一 train/val/test split 在适配层统一合并为 Mixed，并以 Mixed-val 的单一 `val_minFDE20` 选点、以 Mixed-test 的 `minFDE20`、`minADE20`、`MR` 进行单一总榜评估。**
