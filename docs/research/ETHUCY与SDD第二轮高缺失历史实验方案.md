# ETH/UCY 与 SDD 第二轮高缺失历史实验方案

_研究实验数据协议 · 适配 DeMo actor-only、无地图模型 · missing_history_v2_high · 2026-08-31_

> **状态（2026-08-31 更新）：数据已生成并通过全量审计。** §8 步骤 1–4 已完成——掩码生成器、审计脚本、SDD reader 均已扩展；`data/ETHUCY_missing_v2_high/`（5 条件 × 181,542 样本）与 `data/SDD_missing_v2_high/`（5 条件 × 11,814 样本）已构建，`--full` 全量审计 0 失败（ETH 6,353,979 项 / SDD 413,500 项），模型接口冒烟 10/10 通过。审计记录见 [缺失历史数据集 v2 高缺失审计](../audits/缺失历史数据集v2高缺失审计.md)。步骤 5 已于 2026-09-01 启动：零样本臂（§6.2，v1 complete ckpt × uni/bi × 5 条件）与训练期适应臂（§6.1，5 条件 × uni/bi，GPU0=单向链 / GPU3=零样本+双向链）后台运行中；complete 基线复用第一轮 v1 结果（数据一致、协议相同）。

## 1. 目标与边界

第一轮 `missing_history_v1` 只覆盖 0%、12.5% 和 25% 的轻度缺失。本轮扩展到 37.5% 及以上，回答以下问题：

> 当历史窗口中缺失的帧数增加，尤其形成较长连续缺口时，DeMo 的预测误差、最优模态命中率和场景间稳定性如何变化？

本轮仍然是**缺失退化与训练适应实验**，不是最终的缺失感知方法验证。因此：

- 只对历史 8 帧施加缺失，未来 12 帧保持完整；
- 不插值、不读取未来信息恢复历史；
- 不新增 `gap`、时间衰减、可靠性权重或运动学辅助标签；
- 不使用地图输入；
- ETH/UCY 保留多行人轨迹和社会交互，SDD 继续作为单行人外部验证；
- 继续使用当前 DeMo 的局部坐标和 actor-only 接口；
- v1 数据集、v1 掩码和第一轮结果保持不变，不覆盖、不重写；
- 新条件建议使用独立输出根目录 `data/ETHUCY_missing_v2_high/` 和 `data/SDD_missing_v2_high/`。

本轮的主变量是**缺失帧数和连续性**。模型结构、数据切分、未来目标和评测口径应保持一致。

## 2. 固定数据协议

### 2.1 基本参数

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
| 缺失范围 | 历史帧 0–5 | 历史帧 0–5 |
| 强制可见 | 帧 6、7 | 帧 6、7 |
| 未来目标 | 完整 | 完整 |

帧编号从 0 开始。所有条件都必须保证第 6、7 帧可见，缺失只从第 0–5 帧中产生。这样做是为了继续使用当前模型的末帧位置锚点和第 6→7 帧朝向锚点，同时把本轮的变量限定为缺失程度，而不是同时改变坐标锚定协议。

### 2.2 新旧条件

下表中的“缺失率”按每个有效行人的历史 8 帧计算。二维坐标的 x、y 作为同一个时间帧整体缺失，不单独缺一个坐标维度。

| 条件 | 掩码规则 | 缺失帧数 | 缺失率 | 本轮用途 |
|---|---|---:|---:|---|
| `complete` | 全部可见 | 0 | 0% | 完整基线 |
| `random_single` | 随机缺 1 个候选帧 | 1 | 12.5% | v1 对照 |
| `random_block2` | 随机连续缺 2 帧 | 2 | 25% | v1 对照 |
| `random_fixed3` | 从候选帧中随机抽 3 个不同帧 | 3 | 37.5% | 新增随机缺失 |
| `random_fixed4` | 从候选帧中随机抽 4 个不同帧 | 4 | 50% | 新增随机缺失 |
| `random_block3` | 随机连续缺 3 帧 | 3 | 37.5% | 区分缺失连续性，建议纳入 |
| `random_block4` | 随机连续缺 4 帧 | 4 | 50% | 新增中高难度连续缺失 |
| `random_block6` | 连续缺帧 0–5 | 6 | 75% | 极端连续缺失 |

`random_fixed3` 和 `random_fixed4` 是本轮必须加入的 37.5% 和 50% 随机缺失条件；`random_block4` 和 `random_block6` 是本轮必须加入的连续缺失条件。`random_block3` 用于在同一缺失率下比较随机性与连续性，建议一并生成。

可选的更高随机缺失条件为 `random_fixed5`：从候选帧 0–5 中随机抽 5 帧，缺失率为 62.5%。它不是本轮最小必做条件，只有在需要补齐随机缺失曲线时才加入，不能替代 `random_block6`。

## 3. 掩码生成规则

### 3.1 候选帧与连续块

所有新条件的候选帧集合固定为：

```text
C = {0, 1, 2, 3, 4, 5}
```

随机缺失条件按行人独立从 `C` 中无放回抽取指定数量的帧：

- `random_fixed3`：抽取 3 帧；
- `random_fixed4`：抽取 4 帧；
- `random_fixed5`：抽取 5 帧，仅作为可选扩展。

连续缺失条件按行人独立抽取连续块起点 `s`，缺失集合为 `{s, ..., s+m-1}`：

- `random_block3`：`s ∈ {0, 1, 2, 3}`；
- `random_block4`：`s ∈ {0, 1, 2}`；
- `random_block6`：唯一块 `{0, 1, 2, 3, 4, 5}`，因此不需要在多个起点之间随机选择。

生成后再次确认 `mask[:, 6]` 和 `mask[:, 7]` 为 `True`。不能通过事后强制可见来改变已经抽取的缺失数量；正确做法是从候选集合 `C` 中抽取。

### 3.2 确定性与身份键

沿用 v1 的确定性规则，掩码随机数由以下字段共同确定：

```text
dataset|fold|split|scene_id|source_index|source_file|focal_id|condition|mask_seed
```

其中 `mask_seed` 继续固定为 `42`。因此：

- 相同源样本在不同条件下可以拥有不同掩码；
- 相同条件在不同进程数和不同遍历顺序下必须生成相同掩码；
- `complete` 不调用随机数生成器；
- `random_block6` 虽然只有一个合法块，也必须保留 condition 和 seed 元数据；
- train、val、test 的掩码均在制作期固定，不能在 `__getitem__`、epoch 或 batch 内重新生成。

### 3.3 样本内容

保存格式继续采用 v1 canonical 样本：

- `positions[A, 20, 2]`：历史缺失位置写为 `0.0`，未来保持源值；
- `history_mask[A, 8]`：缺失处为 `False`，可见处为 `True`；
- `valid_mask[A, 20]`：前 8 帧等于 `history_mask`，未来 12 帧全为 `True`；
- 保留 `agent_ids`、`focal_id`、`scene_id`、`fold`、`split`、`source_file`、`source_index` 和 `mask_seed`；
- 每个新 condition 的源样本数量、样本顺序和 train/val/test 归属与完整源数据一致。

缺失位置的占位坐标不具有观测语义。Dataset 重新构造局部坐标、位移、速度、速度变化和角度时，必须以 `history_mask` 为准。

## 4. ETH/UCY 与 SDD 来源

### 4.1 ETH/UCY

从已修复并通过审计的 `data/ETHUCY_benchmark_v1/` 读取，不重新切分、不混合 fold、不使用旧的 overwritten benchmark。每个新 condition 保持 5 个 leave-one-out fold，以及各 fold 的 train、val、test 划分。

新数据建议单独保存为：

```text
data/ETHUCY_missing_v2_high/
    manifest.json
    random_fixed3/
    random_fixed4/
    random_block3/
    random_block4/
    random_block6/
```

v1 的 `complete`、`random_single`、`random_block2` 继续使用已审计的 `data/ETHUCY_missing_v1/`，不要求为了本轮重复复制。若后续实现需要单一根目录，必须在 manifest 中明确哪些 condition 来自 v1、哪些 condition 来自 v2，不能只靠目录名推断。

### 4.2 SDD

从当前 `data/sdd/original/sdd_train.pkl` 和 `sdd_test.pkl` 读取，沿用 `split_seed=2024` 的 train/val 划分和原始 test。每条样本仍为单行人 `A=1`，新条件只改变该行人的历史 8 帧掩码。

新数据建议单独保存为：

```text
data/SDD_missing_v2_high/
    manifest.json
    random_fixed3/
    random_fixed4/
    random_block3/
    random_block4/
    random_block6/
```

SDD 的新条件不得改变 v1 的像素坐标、样本顺序和 90/10 train/val 划分。

## 5. DeMo 特征与当前接口兼容性

本轮不允许因缺失率升高而悄悄改变特征语义：

| 字段 | 高缺失条件下的规则 |
|---|---|
| `x_positions` | 局部坐标中的历史位置，占位位置仅由 mask 标记无效 |
| `x_positions_diff` | 相邻两帧有任一缺失时置零 |
| `x_velocity` | 相邻两帧有任一缺失时置零 |
| `x_velocity_diff` | 涉及缺失位移步时置零 |
| `x_angles` | 涉及缺失帧的相邻步方向置零 |
| `x_valid_mask` | 直接使用 `history_mask` |
| `x_key_valid_mask` | `x_valid_mask.any(-1)` |
| `x_centers` | 每个行人的最后一个有效历史位置 |
| `target` | 完整未来 12 帧 |
| `target_mask` | v2 未来目标全有效 |

由于帧 6、7 始终可见：

- 局部坐标原点仍可取 focal 的帧 7；
- focal 朝向仍可由帧 6→7 的有效位移确定，近静止时沿用 `theta=0` 的退化规则；
- `random_block6` 下每个行人仍有 2 帧有效历史，不会出现空历史；
- ETH/UCY 仍保留邻居 actor token，SDD 的 actor 维度仍为 1。

本轮不允许放开帧 6、7 的缺失。若要研究“末帧也缺失”或“只剩一个观测帧”，必须另建允许最后有效帧锚定的 v3 协议，不能把坐标锚点变化与 v2 的缺失率效果混在一起。

## 6. 第二轮实验设计

### 6.1 主实验：训练期适应

每个 condition 使用同一 condition 的 train、val、test，保持与第一轮相同的训练和选点口径：

| 实验 | 训练条件 | 测试条件 | 目的 |
|---|---|---|---|
| 完整基线 | `complete` | `complete` | 固定模型和数据基准 |
| 随机 37.5% | `random_fixed3` | `random_fixed3` | 测量缺 3 帧的适应能力 |
| 随机 50% | `random_fixed4` | `random_fixed4` | 测量缺 4 帧的适应能力 |
| 连续 37.5% | `random_block3` | `random_block3` | 分离缺失连续性影响 |
| 连续 50% | `random_block4` | `random_block4` | 测量连续 4 帧的影响 |
| 连续 75% | `random_block6` | `random_block6` | 测量仅保留末两帧的极端情形 |

所有 condition 应使用相同的模型配置、训练轮数、优化器、主随机种子、checkpoint 选择规则和评测脚本。不同条件之间只改变输入数据的掩码。

### 6.2 补充实验：零样本缺失鲁棒性

将 `complete` 模型直接用于新 condition 的 held-out test，不使用新 condition 的 test 指标选择 checkpoint、训练轮数或超参数。至少报告：

```text
complete -> random_fixed3
complete -> random_fixed4
complete -> random_block3
complete -> random_block4
complete -> random_block6
```

该实验回答的是“完整历史训练的模型能否直接处理高缺失输入”，与 6.1 的“训练期适应”不能合并成一个数字。

### 6.3 报告维度

ETH/UCY 继续按 5 个 held-out fold 分别报告并计算宏平均；SDD 继续按像素单位报告，单一固定 seed 的结果只能作为外部探索性验证。每个条件至少报告：

- `minADE`、`minFDE` 和 Miss Rate；
- 场景或 fold 分项结果；
- 相对于 `complete` 的绝对差和相对变化；
- 随缺失帧数变化的曲线；
- 随连续缺失长度变化的曲线。

不要把单次训练的正负小差异直接解释为单调规律。高缺失条件的主要判断是退化是否稳定、是否集中在连续缺失、以及是否集中在交互密集场景。

## 7. 数据审计与验收

新数据生成后，除复用 v1 的结构、泄漏和接口检查外，必须增加以下条件检查：

### 7.1 掩码检查

- [ ] `random_fixed3` 每个有效行人恰好缺 3 帧，且缺失只在 0–5；
- [ ] `random_fixed4` 每个有效行人恰好缺 4 帧，且缺失只在 0–5；
- [ ] `random_block3` 每个有效行人的缺失集合是唯一连续 3 帧；
- [ ] `random_block4` 每个有效行人的缺失集合是唯一连续 4 帧；
- [ ] `random_block6` 每个有效行人的缺失集合恰为 0–5；
- [ ] 所有新条件的帧 6、7 都为可见；
- [ ] 每个 condition 的有效行人数、样本数和源索引集合与完整源数据一致；
- [ ] 同一 source sample 在不同 condition 中的未来 12 帧逐位一致；
- [ ] 实际缺失率与标称值一致：37.5%、50%、75%。

### 7.2 特征和接口检查

- [ ] 缺失帧不参与局部坐标锚点、差分、速度、速度变化和角度计算；
- [ ] `x_centers` 始终来自最后一个有效历史位置；
- [ ] `random_block6` 的 focal 至少有帧 6、7 两个有效位置；
- [ ] 三个 ETH/UCY 新连续性条件仍保留邻居 actor token；
- [ ] SDD 新条件的 actor 维度仍为 1；
- [ ] Dataset → collate → `ModelForecast(use_map=False)` 的输出没有 `NaN` 或 `Inf`；
- [ ] 未来 12 帧、`target` 和 `target_mask` 没有被掩码污染。

### 7.3 Manifest

`manifest.json` 至少记录：

- `version: missing_history_v2_high`；
- 所有 condition 名称及其缺失帧数、缺失率和连续性规则；
- `mask_seed=42`、SDD 的 `split_seed=2024`；
- 源数据版本和源根目录；
- 每个 condition、fold、split 的样本数、有效行人数、实际缺失帧数和实际缺失率；
- source file、source index 范围和输出文件校验值；
- 审计脚本版本或提交号。

## 8. 实现边界与后续顺序

本次只写文档，后续按以下顺序执行：

1. 扩展缺失掩码生成器，使其支持 `random_fixed3`、`random_fixed4`、`random_block3`、`random_block4` 和 `random_block6`；
2. 为 ETH/UCY 和 SDD 新建独立 v2 输出根目录，不覆盖 v1；
3. 扩展审计脚本的 condition 解析和连续块检查；
4. 完成新数据集的结构、掩码、未来一致性和模型接口审计；
5. 审计通过后再按第 6 节执行训练和零样本评估。

在第 1–4 步完成前，不得把新 condition 写入训练命令或把其结果写入第一轮结果表。当前 v1 的数据和结果仍是已完成实验的唯一结果来源。

## 9. 与 v1 文档的关系

- [ETH/UCY 与 SDD 第一轮缺失历史数据集制作说明](ETHUCY与SDD第一轮缺失历史数据集制作说明.md)：v1 数据格式、轻度缺失条件和已完成的验收口径；
- [缺失历史第一阶段实验汇总](../results/缺失历史第一阶段汇总.md)：v1 训练期适应结果；
- [缺失历史数据集 v1 审计](../audits/缺失历史数据集v1审计.md)：v1 数据审计记录。

本方案只扩展未来实验条件，不修改上述 v1 文档所记录的历史事实。
