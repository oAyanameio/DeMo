# ETH/UCY 与 SDD 第一轮缺失历史数据集制作说明

_研究实验数据协议 · 适配 DeMo actor-only、无地图模型 · v1 · 2026-08-29_

---

## 1. 目标与边界

本文只规定 ETH/UCY 和 SDD 第一轮缺失历史数据集的制作方式，不修改模型、训练脚本或现有完整数据集。

第一轮实验回答以下问题：

> 在未来轨迹保持完整的条件下，DeMo 是否能够使用带有历史缺失掩码的输入完成基本训练和预测？

本轮数据集的定位是**缺失退化基线**，不是最终的可靠性建模数据集。因此：

- 只对历史 8 帧施加缺失；
- 未来 12 帧始终保留完整，作为监督和评测目标；
- 不插值、不使用未来信息恢复历史；
- 不加入 `gap`、衰减、可靠性权重或运动学辅助标签；
- 不使用地图输入；
- ETH/UCY 保留多行人轨迹，用于保留社会交互；
- SDD 按当前仓库协议作为单行人外部验证数据，不宣称包含社会交互；
- 第一轮只制作 `complete`、`random_single`、`random_block2` 三种条件；37.5% 及以上的高缺失条件另见[第二轮高缺失历史实验方案](ETHUCY与SDD第二轮高缺失历史实验方案.md)，不属于 v1。

## 2. 整体流程

完整数据先完成固定切分，再生成历史掩码。缺失位置只作为无效输入保存，所有 DeMo 特征必须在读取时依据掩码重新计算。

```mermaid
flowchart LR
    accTitle: Missing History Dataset Workflow
    accDescr: The workflow starts from complete trajectories, applies deterministic masks only to the observed history, rebuilds valid model features, and saves audited samples with manifests.

    source_complete([Complete trajectories]) --> fixed_split[Preserve fixed split]
    fixed_split --> mask_history[Generate deterministic history mask]
    mask_history --> hide_history[Hide history only]
    hide_history --> derive_features[Rebuild masked features]
    derive_features --> save_samples[(Save canonical samples)]
    save_samples --> audit_dataset[Audit data and manifest]

    classDef process fill:#dbeafe,stroke:#2563eb,stroke-width:2px,color:#1e3a5f
    classDef data fill:#dcfce7,stroke:#16a34a,stroke-width:2px,color:#14532d
    classDef check fill:#fef9c3,stroke:#ca8a04,stroke-width:2px,color:#713f12

    class fixed_split,mask_history,hide_history,derive_features process
    class source_complete,save_samples data
    class audit_dataset check
```

## 3. 统一实验协议

### 3.1 固定参数

| 项目 | ETH/UCY | SDD |
|---|---:|---:|
| 历史长度 | 8 帧 | 8 帧 |
| 未来长度 | 12 帧 | 12 帧 |
| 总长度 | 20 帧 | 20 帧 |
| 时间间隔 | 0.4 秒 | 0.4 秒 |
| 采样频率 | 2.5 Hz | 2.5 Hz |
| 坐标单位 | 米 | 像素 |
| 地图输入 | 不使用 | 不使用 |
| 行人交互 | 保留 | 当前协议为单行人 |
| 缺失范围 | 历史 8 帧 | 历史 8 帧 |
| 未来目标 | 完整 | 完整 |

ETH/UCY 和 SDD 不在同一个训练任务中混合。两者坐标单位不同，指标也分别在米制和像素制下报告。

### 3.2 缺失条件

`history_mask=True` 表示该历史帧可见，`False` 表示该历史帧缺失。

| 条件 | 每个有效行人的历史掩码 | 标称缺失率 | 用途 |
|---|---|---:|---|
| `complete` | 8 帧全部可见 | 0% | 完整历史基线 |
| `random_single` | 随机缺失 1 帧 | 12.5% | 轻度缺失 |
| `random_block2` | 随机连续缺失 2 帧 | 25% | 短连续缺失 |

为兼容当前 DeMo 的局部坐标构造，v1 中所有有效行人的第 6、7 帧必须可见，缺失位置只从第 0 至第 5 帧中选择。这里的帧编号从 0 开始。

该约束是第一轮工程兼容条件，不是对真实缺失分布的假设。后续若改为“最后有效帧锚定”，再单独制作允许第 6、7 帧缺失的 v2 数据集。

注意：本节只定义已完成的 v1 条件。`random_fixed3`（37.5%）、`random_fixed4`（50%）、`random_block4`（50%）和 `random_block6`（75%）等高缺失条件已经纳入下一轮方案，但目前尚未生成数据、修改脚本或执行训练。

### 3.3 随机性与复现

- 缺失掩码只在数据制作阶段生成一次；
- 训练、验证和测试读取相同的已保存掩码；
- 不得在 `__getitem__`、每个 epoch 或每个 batch 中重新随机缺失；
- 缺失掩码使用固定 `mask_seed=42`；
- 掩码随机数应由以下字段共同确定：

```text
dataset + fold + split + scene_id + source_index + source_file + focal_id + condition + mask_seed
```

- 同一来源样本在不同条件下可以使用不同掩码；
- `complete` 条件不调用随机数生成器；
- 生成完成后必须在 `manifest.json` 中记录种子和实际缺失率。

## 4. 数据来源与切分

### 4.1 ETH/UCY 来源

ETH/UCY 应从 DeMo 当前完整 benchmark 数据读取：

```text
data/ETHUCY_benchmark_v1/
    manifest.json
    fold_ETH/
    fold_HOTEL/
    fold_UNIV/
    fold_ZARA1/
    fold_ZARA2/
```

该目录中的 `.pt` 样本已经包含完整的 `positions`、`agent_ids`、`focal_id`、`scene_id` 和 `valid_mask` 字段。正式制作时必须保留其 fold、split 和样本顺序。

每个 fold 的目录语义为：

```text
fold_<held_out_scene>/
    train/
    val/
    test/
```

不得重新混合场景，不得用缺失数据重新划分 train、val、test。生成缺失数据只改变输入历史，不改变样本归属。

当前完整 benchmark 的制作和字段约定见：

- [ETH/UCY benchmark 预处理器](/home/lbh/DeMo/src/datamodule/ethucy_benchmark_preprocess.py)
- [ETH/UCY benchmark Dataset](/home/lbh/DeMo/src/datamodule/ethucy_benchmark_dataset.py)
- [ETH/UCY benchmark 重建方案](/home/lbh/DeMo/docs/plans/ETHUCY基准重建与DeMo重训练方案.md)

### 4.2 SDD 来源

SDD 使用当前 DeMo 仓库中的完整数据：

```text
data/sdd/original/sdd_train.pkl
data/sdd/original/sdd_test.pkl
```

每条记录为：

```text
(past[8, 2], future[12, 2], sequence[20, 0, 2])
```

SDD 当前协议只有训练文件和测试文件，没有官方验证集。应保持现有约定：

- `sdd_train.pkl` 使用固定 `split_seed=2024` 划分为 90% train 和 10% val；
- `sdd_test.pkl` 原样作为 held-out test；
- train、val、test 的划分先完成，再生成缺失掩码；
- 测试集不得用于选择 checkpoint、缺失率或训练轮数；
- 同一个 `split_seed` 必须用于所有条件和所有模型对照实验。

SDD 当前完整数据管线见：

- [SDD Dataset](/home/lbh/DeMo/src/datamodule/moflow_sdd_dataset.py)
- [SDD DataModule](/home/lbh/DeMo/src/datamodule/moflow_sdd_datamodule.py)
- [SDD 配置](/home/lbh/DeMo/conf/datamodule/moflow_sdd.yaml)

## 5. Canonical 样本格式

### 5.1 保存原则

缺失数据集保存**原始坐标和有效掩码**，不保存已经计算好的 DeMo 历史差分特征作为唯一来源。这样可以避免在缺失位置被遮挡后继续使用完整历史预先计算的速度、方向或差分。

建议每个样本保存为一个 `.pt` 文件，格式与当前 ETH/UCY benchmark 样本一致。无效历史坐标统一填充为 `0.0`，但该值不是观测值，任何计算都必须先使用 `valid_mask`。

注意：缺失历史位置在 Dataset 的局部坐标变换（旋转+平移）后**不再保持数值 0**，它们是「原点系下的占位坐标」，仅保证与有效坐标可区分地存在于张量中，不携带任何观测语义；所有依赖掩码的派生量（diff/velocity/velocity_diff/angles/theta 锚点）已按 §5.3 置零或跳过，模型输入通道也拼接了 `hist_valid_mask`，因此模型不使用这些占位值。任何下游新特征工程必须以 `history_mask` 为准。

### 5.2 样本字段

| 字段 | 类型与形状 | 说明 |
|---|---|---|
| `positions` | `FloatTensor[A, 20, 2]` | 原始坐标，ETH/UCY 为米，SDD 为像素 |
| `valid_mask` | `BoolTensor[A, 20]` | 历史由缺失掩码决定，未来 12 帧全部为 `True` |
| `history_mask` | `BoolTensor[A, 8]` | 历史掩码副本，便于审计 |
| `agent_ids` | `LongTensor[A]` | 行人 ID；SDD 使用 `[0]` |
| `focal_id` | `int` | focal 行人 ID；SDD 使用 `0` |
| `scene_id` | `str` | `ETH`、`HOTEL`、`UNIV`、`ZARA1`、`ZARA2` 或 `SDD` |
| `fold` | `str` 或 `None` | ETH/UCY 为 held-out fold，SDD 为 `None` |
| `split` | `str` | `train`、`val` 或 `test` |
| `condition` | `str` | `complete`、`random_single` 或 `random_block2` |
| `source_file` | `str` | 原始样本来源文件 |
| `source_index` | `int` | 原始文件或原始列表中的稳定索引 |
| `sample_index` | `int` | 输出数据集中的稳定索引 |
| `mask_seed` | `int` | 本样本使用的掩码种子 |
| `obs_len` | `int` | 固定为 `8` |
| `pred_len` | `int` | 固定为 `12` |
| `dt` | `float` | 固定为 `0.4` |
| `unit` | `str` | ETH/UCY 为 `meter`，SDD 为 `pixel` |

`frame_ids`、`frame_ids_raw` 等原始元数据应在来源样本中存在时一并保留，但不作为模型输入。

### 5.3 DeMo 输入字段的生成规则

Dataset 读取 canonical 样本后，按照 focal 行人的局部坐标系生成以下模型字段：

| DeMo 字段 | 生成规则 |
|---|---|
| `x_positions` | 局部坐标系下的历史位置 |
| `x_positions_diff` | 只有相邻两帧都有效时才计算位移，否则置零 |
| `x_velocity` | 只有相邻两帧都有效时才计算速度，否则置零 |
| `x_velocity_diff` | 基于有效速度计算，涉及缺失帧的项置零 |
| `x_valid_mask` | 直接使用历史 8 帧的 `history_mask` |
| `x_key_valid_mask` | `x_valid_mask.any(-1)` |
| `x_centers` | 每个行人最后一个有效历史位置 |
| `x_angles` | 只使用有效相邻位移计算方向 |
| `target` | 局部坐标系下的完整未来 12 帧 |
| `target_mask` | 未来有效性掩码；v1 中 focal 和所有目标均为有效 |

局部坐标变换必须遵循以下顺序：

1. 先依据 focal 的第 6、7 帧确定原点和朝向；
2. 再只对 `valid_mask=True` 的位置做旋转和平移；
3. 缺失历史不参与原点、朝向、位移、速度和角度计算；
4. 未来坐标使用同一个 focal 局部坐标系；
5. 不使用未来位置反推历史原点或朝向。

## 6. ETH/UCY 制作规则

### 6.1 保留多行人上下文

每个 ETH/UCY 样本保留完整 benchmark 样本中的所有有效行人。缺失掩码按行人独立生成，而不是对整个场景使用一个同步掩码。

这样可以区分两类情况：

- focal 行人历史缺失，但邻居仍提供社会上下文；
- focal 和邻居在不同时间步发生缺失，模型需要使用剩余交互证据。

v1 仍要求每个有效行人的第 6、7 帧可见，以保证当前 actor-only DeMo 的末帧位置和朝向输入稳定。

### 6.2 输出目录

每个条件使用独立根目录，目录内部完全复制完整 benchmark 的 fold/split/scene 结构：

```text
data/ETHUCY_missing_v1/
    manifest.json
    complete/
        fold_ETH/
            train/<scene>/*.pt
            val/<scene>/*.pt
            test/<scene>/*.pt
        fold_HOTEL/
        fold_UNIV/
        fold_ZARA1/
        fold_ZARA2/
    random_single/
        fold_ETH/
        fold_HOTEL/
        fold_UNIV/
        fold_ZARA1/
        fold_ZARA2/
    random_block2/
        fold_ETH/
        fold_HOTEL/
        fold_UNIV/
        fold_ZARA1/
        fold_ZARA2/
```

运行某个条件时，Dataset 的根目录指向对应条件目录，例如：

```text
data/ETHUCY_missing_v1/random_single/
```

每个条件的样本数必须与 `data/ETHUCY_benchmark_v1` 对应 fold 和 split 完全一致。

## 7. SDD 制作规则

### 7.1 单行人样本

当前 SDD pkl 数据每条记录只有一个行人，因此构造：

```text
positions = concatenate(past, future, axis=0)[None, ...]  # [1, 20, 2]
agent_ids = [0]
focal_id = 0
```

SDD 的历史缺失只影响该行人的 8 帧历史，不改变未来 12 帧。

SDD 数据不提供 ETH/UCY 式的多行人交互，因此 SDD 结果只能用于验证缺失历史编码在不同数据分布和坐标尺度下的可迁移性，不能用于验证社会交互模块。

### 7.2 输出目录

```text
data/SDD_missing_v1/
    manifest.json
    complete/
        train/*.pt
        val/*.pt
        test/*.pt
    random_single/
        train/*.pt
        val/*.pt
        test/*.pt
    random_block2/
        train/*.pt
        val/*.pt
        test/*.pt
```

SDD 的 `train`、`val`、`test` 文件清单必须先固定，再分别生成三个条件。三个条件的样本数量和样本顺序必须一致。

### 7.3 坐标和指标

SDD 保留像素坐标，不转换为米，不使用 ETH/UCY 的归一化统计。平移和旋转只用于 DeMo 的局部输入构造，最终误差指标仍按当前 SDD 协议在像素单位下计算。

## 8. Manifest 内容

每个数据集根目录保存一个总 `manifest.json`，至少包含以下信息：

```json
{
  "version": "missing_history_v1",
  "dataset": "ETHUCY",
  "obs_len": 8,
  "pred_len": 12,
  "dt": 0.4,
  "unit": "meter",
  "use_map": false,
  "mask_seed": 42,
  "split_seed": null,
  "conditions": [
    "complete",
    "random_single",
    "random_block2"
  ],
  "history_visibility_rule": "frames_6_and_7_visible",
  "future_policy": "complete",
  "coordinate_policy": "raw_scene_coordinates",
  "feature_policy": "recompute_from_visible_history",
  "source_root": "data/ETHUCY_benchmark_v1"
}
```

SDD 的 `split_seed` 固定为 `2024`，`unit` 为 `pixel`，`source_root` 为 `data/sdd/original`。

Manifest 还应记录每个 condition、fold、split 的：

- 样本数量；
- 有效行人数或单行人数；
- 实际历史缺失帧数；
- 实际缺失率；
- 源文件和源样本索引范围；
- 输出文件校验值；
- 被拒绝样本数量及原因。

## 9. 制作步骤

### 9.1 ETH/UCY

1. 读取 `data/ETHUCY_benchmark_v1/manifest.json`，确认 5 个 fold 均存在。
2. 逐个读取 fold、split、scene 下的完整 `.pt` 样本。
3. 检查 `positions.shape == [A, 20, 2]`，且来源样本 `valid_mask` 全部为 `True`。
4. 按样本和行人生成固定历史掩码。
5. 强制所有有效行人的第 6、7 帧可见。
6. 将掩码为 `False` 的历史位置写为 `0.0`。
7. 将 `valid_mask[:, :8]` 替换为生成的历史掩码，保持 `valid_mask[:, 8:]` 全为 `True`。
8. 保留 `agent_ids`、`focal_id`、`scene_id`、fold、split 和来源索引。
9. 按原目录结构保存到对应 condition 目录。
10. 更新 manifest 并执行完整性检查。

### 9.2 SDD

1. 读取 `data/sdd/original/sdd_train.pkl` 和 `sdd_test.pkl`。
2. 使用 `split_seed=2024` 将训练列表划分为 train 和 val。
3. 将每条记录转换为 `[1, 20, 2]` 的 canonical `positions`。
4. 按原始列表索引和 condition 生成固定历史掩码。
5. 强制唯一行人的第 6、7 帧可见。
6. 只将历史缺失位置写为 `0.0`，未来保持原值。
7. 写入 `history_mask`、`valid_mask` 和 SDD 元数据。
8. 按 condition 和 split 保存 `.pt` 文件。
9. 更新 manifest 并执行完整性检查。

### 9.3 约定的制作命令接口

以下命令用于约定未来制作脚本的接口，当前文档不新增脚本，也不保证命令已经可执行：

```bash
python scripts/数据集构建/build_missing_history_dataset.py \
    --dataset ethucy \
    --source-root data/ETHUCY_benchmark_v1 \
    --output-root data/ETHUCY_missing_v1 \
    --conditions complete random_single random_block2 \
    --mask-seed 42
```

```bash
python scripts/数据集构建/build_missing_history_dataset.py \
    --dataset sdd \
    --source-root data/sdd \
    --output-root data/SDD_missing_v1 \
    --conditions complete random_single random_block2 \
    --split-seed 2024 \
    --mask-seed 42
```

## 10. 质量检查与验收标准

### 10.1 结构检查

- [ ] ETH/UCY 的 5 个 fold、train、val、test 目录均存在
- [ ] SDD 的 train、val、test 目录均存在
- [ ] 每个 condition 的样本数量与 complete 条件一致
- [ ] ETH/UCY 的 fold 和 split 样本数量与完整 benchmark 一致
- [ ] SDD 的 train/val 划分在所有 condition 中完全一致
- [ ] 每个样本都能被 `torch.load` 读取

### 10.2 掩码检查

- [ ] `history_mask.shape == [A, 8]`
- [ ] `valid_mask.shape == [A, 20]`
- [ ] `valid_mask[:, :8]` 与 `history_mask` 完全一致
- [ ] `valid_mask[:, 8:].all()` 为 `True`
- [ ] `complete` 条件的历史掩码全部为 `True`
- [ ] `random_single` 每个有效行人恰好缺失 1 帧
- [ ] `random_block2` 每个有效行人恰好缺失连续 2 帧
- [ ] 所有条件下第 6、7 帧均为可见
- [ ] 缺失位置只出现在历史 8 帧内

### 10.3 信息泄漏检查

- [ ] 缺失位置生成不读取未来 12 帧
- [ ] 缺失位置生成不读取未来速度或未来方向
- [ ] 缺失位置生成不改变未来目标
- [ ] 缺失位置不参与历史差分、速度、角度和局部坐标锚点计算
- [ ] 测试样本不参与 train/val 划分
- [ ] 归一化统计不从缺失测试集计算；v1 直接使用原始单位，不做新增归一化

### 10.4 模型接口检查

每个 condition 至少抽取一个 ETH/UCY 样本和一个 SDD 样本，经过 Dataset 后确认：

```text
x_positions_diff : [B, A, 8, 2]
x_velocity_diff  : [B, A, 8]
x_valid_mask     : [B, A, 8]
x_key_valid_mask : [B, A]
target           : [B, A, 12, 2]
target_mask      : [B, A, 12]
```

并确认：

- focal 行人的历史有效掩码不为空；
- focal 第 6、7 帧有效；
- `x_centers` 来自最后一个有效历史位置；
- 含缺失帧的相邻差分为零；
- forward 不产生 `NaN` 或 `Inf`；
- ETH/UCY 保留邻居 token；
- SDD 的 agent 维度为 1。

## 11. 第一轮实验使用方式

第一轮只比较同一模型在完整和缺失输入下的性能变化：

| 实验 | 训练条件 | 测试条件 | 目的 |
|---|---|---|---|
| 完整基线 | `complete` | `complete` | 复核完整数据性能 |
| 轻度缺失 | `random_single` | `random_single` | 测量轻度缺失退化 |
| 连续缺失 | `random_block2` | `random_block2` | 测量短连续缺失退化 |
| 可选鲁棒性测试 | `complete` | `random_single`、`random_block2` | 测量训练分布外缺失输入的脆弱性 |

正式比较时，train、val、test 必须使用同一 condition。最后一项只作为额外诊断，不替代同条件评测。

第一轮不应据此声称模型已经完成缺失感知建模。结果只能说明：

1. 当前 DeMo 数据接口能否运行缺失历史输入；
2. 缺失程度增加时预测性能如何变化；
3. ETH/UCY 与 SDD 的缺失退化是否具有一致趋势。

## 12. 禁止直接复用的输入

不得将以下数据直接作为当前 DeMo 的输入：

```text
/home/lbh/MoFlow/data/eth_ucy/missing_stage2/
```

该目录中的 `past_traj [N, A, 8, 6]` 是 MoFlow 专用的已派生、已归一化特征，且采用每个行人自己的锚点和旋转协议。它可以作为缺失数据生成参考，但不能替代本说明定义的 DeMo canonical 数据。

正确做法是：

```text
完整原始坐标
    -> 固定 train/val/test
    -> 生成历史掩码
    -> 置空历史位置
    -> 保存 raw positions + valid masks
    -> Dataset 内按 DeMo 协议重新构造特征
```

## 13. 后续扩展边界

第一轮之后的高缺失条件已经单独写入[第二轮高缺失历史实验方案](ETHUCY与SDD第二轮高缺失历史实验方案.md)，包括：

- 随机缺失 3 帧（37.5%）和 4 帧（50%）；
- 连续缺失 4 帧（50%）和 6 帧（75%）；
- 为比较同一缺失率下的连续性，建议增加连续缺失 3 帧（37.5%）；
- 可选的随机缺失 5 帧（62.5%）。

第二轮仍然保持第 6、7 帧可见，并使用独立的 v2 输出根目录；它不会覆盖 v1 数据或改写第一轮结果。允许第 6、7 帧缺失、使用最后有效帧锚定，仍属于更后续的独立协议。

以下内容仍不属于 v1 或第二轮高缺失基线：

- 基于真实跟踪丢失模式的掩码；
- `gap`、时间间隔和可学习衰减；
- 方向、曲率、加速度等运动学证据；
- 可靠性加权运动证据编码；
- 缺失历史与完整历史之间的知识蒸馏；
- 转向、停止和避让等行为分层评测。

v1 的任务是建立一个**切分固定、未来完整、掩码可复现、坐标协议与 DeMo 一致**的轻度缺失数据基础；第二轮在此基础上测试更高缺失程度，不改变 v1 的历史记录。
