# ETH/UCY Benchmark 数据重建、DeMo 重训练与评估执行方案

> **面向其他 AI Agent 的执行文档：** 本文是可直接执行的工程方案。执行 agent 需要先阅读现有仓库和工作区改动，再按本文完成数据重建、代码修正、训练、评估和结果汇总。
>
> **执行边界：** 本方案只验证 DeMo 从自动驾驶运动预测迁移到 ETH/UCY 行人轨迹预测后的基础能力。不加入历史缺失建模、运动模式增强、MoFlow、地图分支、额外生成器或新的科研模块。

**目标：** 保留原始 ETH/UCY 文本数据，按照 SocialGAN 风格的 ETH/UCY benchmark 协议生成新的 `.pt` 数据，修正现有 ETH/UCY 训练和评估流程，并完成严格的五折 Leave-One-Scene-Out 重训练与测试。

**架构：** 原始 `.txt` 文件只读；先依据官方 train/val/test 文件建立 manifest，再按 20 帧滑动窗口生成严格完整轨迹样本。DeMo 使用现有 actor-only、无地图配置，训练阶段只使用对应 fold 的 train/val，checkpoint 由 val 指标选择，最终只在独立 test split 上评估。

**技术栈：** Python、PyTorch、PyTorch Lightning、Hydra、TorchMetrics、现有 Mamba/DeMo 实现、pytest、CSV/JSON/Markdown 结果记录。

---

## 1. 执行原则

### 1.1 必须遵守

- 不修改任何原始 `.txt` 文件。
- 不覆盖当前 `data/ETHUCY_processed`，新数据必须写入新目录。
- 不使用 `raw/all_data` 直接作为正式训练或测试数据源。
- 不把缺失轨迹用 0 填充后当作真实监督目标。
- 不使用测试 split 选择 checkpoint、调参或决定训练轮数。
- 不修改 DeMo 的核心 Mode/State 解耦结构。
- 不加入历史缺失、轨迹插补、MoFlow、地图、图像或新生成器模块。
- 不回退工作区中已有的用户改动；执行前先检查 `git status`。
- 每个阶段都要先运行验证命令，再进入下一阶段。

### 1.2 允许的改动

- 新增 benchmark 专用预处理器、manifest、dataset/datamodule 配置和 runner。
- 修正 ETH/UCY 数据字段、mask、坐标变换、指标和 checkpoint 选择逻辑。
- 为 ETH/UCY 使用显式的 actor-only 配置。
- （历史记录，仓库已收缩为行人专用）当时为避免环境依赖阻塞，将上游 AV2 submission import 改为延迟导入。
- 新增单元测试、集成测试和结果汇总脚本。

### 1.3 不允许的实验解释

本阶段只能回答：

> DeMo 在不使用地图、只使用行人轨迹和社会上下文的条件下，迁移到 ETH/UCY 后的基础预测能力如何？

不能把结果解释为：

- DeMo 在行人预测上达到了新的 state of the art。
- 模型已经解决了历史轨迹缺失问题。
- 6 个 mode 的结果可以直接与所有 `minADE20/minFDE20` 论文结果比较。
- 当前结果能证明某个新增模块有效。

---

## 2. 当前仓库基线与已知风险

### 2.1 当前已有适配

现有仓库已经包含以下迁移基础：

- [`src/datamodule/ethucy_utils.py`](/home/lbh/DeMo/src/datamodule/ethucy_utils.py)
- [`src/datamodule/ethucy_extractor.py`](/home/lbh/DeMo/src/datamodule/ethucy_extractor.py)
- [`src/datamodule/ethucy_dataset.py`](/home/lbh/DeMo/src/datamodule/ethucy_dataset.py)
- [`src/datamodule/ethucy_datamodule.py`](/home/lbh/DeMo/src/datamodule/ethucy_datamodule.py)
- [`conf/config_ethucy.yaml`](/home/lbh/DeMo/conf/config_ethucy.yaml)
- [`conf/datamodule/ethucy.yaml`](/home/lbh/DeMo/conf/datamodule/ethucy.yaml)
- [`conf/model/ethucy_model_forecast.yaml`](/home/lbh/DeMo/conf/model/ethucy_model_forecast.yaml)
- [`src/utils/submission_ethucy.py`](/home/lbh/DeMo/src/utils/submission_ethucy.py)
- [`tests/test_ethucy_dataset.py`](/home/lbh/DeMo/tests/test_ethucy_dataset.py)
- [`tests/test_ethucy_model.py`](/home/lbh/DeMo/tests/test_ethucy_model.py)
- [`tests/test_ethucy_trainer.py`](/home/lbh/DeMo/tests/test_ethucy_trainer.py)
- [`scripts/训练与评估/run_5fold_loo.py`](/home/lbh/DeMo/scripts/训练与评估/run_5fold_loo.py)

这些文件可以复用，但不能假设当前实现符合 benchmark。

### 2.2 必须修正的风险

1. 当前预处理从 `raw/all_data` 递归收集文件，正式 train/val/test 边界没有保留。
2. 当前滑窗逻辑只要求 focal 历史完整，没有要求 focal 未来 12 帧完整。
3. 当前 Dataset 将 focal 的未来 `target_mask` 强制设为全 `True`，可能把零填充坐标当成标签。
4. 当前邻居可能只在部分历史帧出现，但仍使用固定末帧位置构造 `x_centers` 和 `x_angles`。
5. 当前五折脚本把同一场景目录同时用于验证和测试，没有使用独立的官方 val/test 文件。
6. （历史记录）当时 `Trainer` 无条件 import 上游 AV2 submission，会阻塞 ETH/UCY 测试；现已只保留 SubmissionEthUcy。
7. 当前环境可能存在 `mamba_ssm` 二进制与系统 glibc/CUDA 不兼容，必须先完成环境 smoke test。
8. 当前模型输出 6 个 mode，不能直接声称与使用 20 个样本的 `minADE20/minFDE20` 结果等价。

---

## 3. Benchmark 定义

### 3.1 固定协议

正式结果必须使用以下固定协议：

| 项目 | 固定值 |
|---|---|
| 数据集 | ETH、HOTEL、UNIV、ZARA1、ZARA2 |
| 观测长度 | 8 个时间步 |
| 预测长度 | 12 个时间步 |
| 总窗口 | 20 个时间步 |
| 采样间隔 | 0.4 秒 |
| 采样频率 | 2.5 Hz |
| 滑窗步长 | 1 个时间步 |
| 地图 | 不使用 |
| 输入 | 行人历史轨迹、位移、速度差、方向和社会上下文 |
| 模型 | `ModelForecast`（actor-only，仓库收缩后已无 use_map 参数） |
| mode 数量 | 6 |
| 主指标 | `minADE1`、`minFDE1`、`minADE6`、`minFDE6`、`MR` |
| MR 阈值 | 2.0 米 |
| fold 数量 | 5 |
| checkpoint 选择 | 只依据对应 fold 的 val `minFDE6` |
| 最终测试 | 只在对应 fold 的 test split 运行一次 |

SocialGAN 风格的核心是固定长度序列和官方场景划分；Trajectron++ 等实现也沿用了类似的 ETH/UCY train/validation/test 组织。正式报告中必须明确记录 `K=6`，不能将 `minADE6/minFDE6` 与 `minADE20/minFDE20` 直接混为一谈。

### 3.2 官方 split 使用规则

如果本机存在以下目录，应优先使用它们：

```text
/home/lbh/MoFlow/data/eth_ucy/original/
    eth/
        train/
        val/
        test/
    hotel/
        train/
        val/
        test/
    univ/
        train/
        val/
        test/
    zara1/
        train/
        val/
        test/
    zara2/
        train/
        val/
        test/
```

每个目录名表示一个 held-out fold：

```text
fold=ETH:
    train = eth/train/*
    val   = eth/val/*
    test  = eth/test/*

fold=HOTEL:
    train = hotel/train/*
    val   = hotel/val/*
    test  = hotel/test/*

fold=UNIV:
    train = univ/train/*
    val   = univ/val/*
    test  = univ/test/*

fold=ZARA1:
    train = zara1/train/*
    val   = zara1/val/*
    test  = zara1/test/*

fold=ZARA2:
    train = zara2/train/*
    val   = zara2/val/*
    test  = zara2/test/*
```

不能通过 `scene_names=["ETH", ...]` 从一个混合目录推断 split。Dataset 必须读取具体 manifest 或具体 fold/split 目录。

### 3.3 原始文件到标准场景的映射

建立显式映射表，不允许通过模糊字符串匹配静默归类：

| 原始文件 | 标准场景 |
|---|---|
| `biwi_eth.txt` | `ETH` |
| `biwi_hotel.txt` | `HOTEL` |
| `students001.txt` | `UNIV` |
| `students003.txt` | `UNIV` |
| `uni_examples.txt` | `UNIV` |
| `crowds_zara01.txt` | `ZARA1` |
| `crowds_zara02.txt` | `ZARA2` |
| `crowds_zara03.txt` | `ZARA2` |

带有 `_train`、`_val` 后缀的文件必须根据其所在 split 目录处理，不能仅根据文件名推断最终评估 split。

如果某个文件不在映射表中，预处理必须报错并停止，而不是自动归入某个场景。

---

## 4. 目标输出目录与文件

执行 agent 需要创建以下新目录和文件。文件名可以小幅调整，但职责不能改变。

### 4.1 数据输出

```text
data/ETHUCY_benchmark_v1/
    manifest.json
    fold_ETH/
        train/
            ETH/
            HOTEL/
            UNIV/
            ZARA1/
            ZARA2/
        val/
            ETH/
            HOTEL/
            UNIV/
            ZARA1/
            ZARA2/
        test/
            ETH/
            HOTEL/
            UNIV/
            ZARA1/
            ZARA2/
    fold_HOTEL/
        ...
    fold_UNIV/
        ...
    fold_ZARA1/
        ...
    fold_ZARA2/
        ...
```

空场景目录可以不创建。每个 `.pt` 文件必须保存足够的审计字段：

```python
{
    "scene_id": "ETH",
    "fold": "ETH",
    "split": "test",
    "source_file": "/absolute/path/to/biwi_eth.txt",
    "sample_index": 0,
    "focal_id": 12,
    "agent_ids": Tensor[N],
    "positions": Tensor[N, 20, 2],
    "valid_mask": BoolTensor[N, 20],
    "frame_ids": Tensor[20],
}
```

`source_file` 可以保存绝对路径或相对于 raw root 的路径，但必须可追溯，不能只保存 `scene_id`。

### 4.2 Manifest

`manifest.json` 至少包含：

```json
{
  "version": "ethucy_benchmark_v1",
  "obs_len": 8,
  "pred_len": 12,
  "sequence_length": 20,
  "dt": 0.4,
  "frame_stride": 1,
  "scenes": ["ETH", "HOTEL", "UNIV", "ZARA1", "ZARA2"],
  "folds": {
    "ETH": {
      "train": [],
      "val": [],
      "test": []
    }
  }
}
```

每个 manifest item 至少包含：

```json
{
  "fold": "ETH",
  "split": "train",
  "scene_id": "HOTEL",
  "source_file": "eth/train/biwi_hotel_train.txt",
  "num_source_rows": 0,
  "num_source_frames": 0,
  "num_samples": 0
}
```

manifest 生成后必须保存原始文件哈希，推荐使用 SHA-256，便于确认训练结果对应的数据版本。

---

## 5. 数据重建任务

### 任务 1：环境和工作区检查

**文件：**

- 读取：`README.md`
- 读取：`requirements.txt`
- 读取：`git status`
- 读取：`src/model/layers/mamba/vim_mamba.py`
- 读取：（已删，上游 AV2 submission 见 DeMo_Origin 存档）

**步骤：**

- [ ] 确认当前工作目录为 `/home/lbh/DeMo`。
- [ ] 保存执行前的 `git status --short` 到本次运行日志。
- [ ] 确认 Python、PyTorch、CUDA、Mamba、PyTorch Lightning 版本。
- [ ] 检查 `mamba_ssm` 是否能成功 import 和完成一次 CPU/GPU 前向。
- [x] （已完成并超出：仓库收缩时已删除 AV2 submission，Trainer 只保留 ethucy 分支）
- [ ] 确认 raw root 存在且包含五个 fold 的 `train/val/test` 目录。

**验证命令：**

```bash
pwd
git status --short
python -V
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
PY
```

**通过标准：**

- Python 和 PyTorch 可运行。
- 至少可以 import `torch`、`pytorch_lightning`、`hydra`、`torchmetrics`。
- 若 Mamba import 失败，必须先修复环境或提供兼容 fallback；不得直接跳过模型验证。
- raw root 五个 fold 的 split 文件可枚举。

### 任务 2：建立显式 source manifest

**文件：**

- 创建：`scripts/数据集构建/build_ethucy_manifest.py`
- 创建：`data/ETHUCY_benchmark_v1/source_manifest.json`
- 测试：`tests/test_ethucy_manifest.py`

**实现要求：**

- 参数：

```text
--raw-root
--output
--expected-folds ETH HOTEL UNIV ZARA1 ZARA2
```

- 只扫描：

```text
<raw_root>/<fold>/{train,val,test}/*.txt
```

- 不扫描 `raw/all_data` 作为正式 manifest。
- 检查每个 fold 都有 train、val、test。
- 检查同一 fold 的 source file 不同时出现在多个 split。
- 检查文件名映射明确。
- 记录文件大小、行数、frame 数、SHA-256。
- 对缺失文件、未知文件、重复文件、空文件直接报错。

**测试必须覆盖：**

- 一个完整 fold manifest 可以生成。
- 同一文件出现在 train 和 val 时测试失败。
- 未知场景文件测试失败。
- 缺少 `test/` 目录测试失败。

**验证命令：**

```bash
PYTHONPATH=. pytest -q tests/test_ethucy_manifest.py
python scripts/数据集构建/build_ethucy_manifest.py \
    --raw-root /home/lbh/MoFlow/data/eth_ucy/original \
    --output data/ETHUCY_benchmark_v1/source_manifest.json
```

### 任务 3：实现 SocialGAN 风格固定窗口提取

**文件：**

- 创建或修改：`src/datamodule/ethucy_benchmark_preprocess.py`
- 修改：`scripts/数据集构建/preprocess_ethucy.py`，只在需要时增加 benchmark 子命令
- 测试：`tests/test_ethucy_benchmark_preprocess.py`

**时间处理要求：**

- 当前原始文本的 frame ID 可能以原视频 frame 编号保存，例如间隔 10。
- 不要再次通过 `frame // 10` 进行隐式分箱。
- 将排序后的 source frame ID 映射成连续 benchmark timestep `0, 1, ...`。
- 保留原始 frame ID 到 `frame_ids_raw`，保留连续索引到 `frame_ids`。
- 如果同一个 source file 中不同 raw frame 映射到同一 timestep，直接报错。
- 默认 `dt=0.4`、`frame_stride=1` 表示“处理后的每一行 unique frame 是一个 benchmark timestep”。

**固定窗口规则：**

- `obs_len=8`。
- `pred_len=12`。
- `sequence_length=20`。
- `skip=1`。
- 每个滑动窗口遍历所有有效 focal 行人。
- focal 的历史 8 帧和未来 12 帧必须全部存在。
- 按 SocialGAN 风格，正式 benchmark 版本只保留窗口内覆盖完整 20 帧的 actor 作为场景 actor。
- 不对缺失坐标做线性插值。
- 不把 NaN/缺失位置填 0 后继续作为有效目标。
- 可将原始稀疏矩阵中的无效位置暂时置零用于保存，但 `valid_mask` 必须保留；正式 Dataset 不得把这些位置作为有效标签。
- `target_mask[0]` 必须由实际未来有效性决定，不能无条件设为 `True`。

**样本保留规则：**

```python
window_valid = valid_mask[:, start:start + 20]
complete_actor = window_valid.all(axis=1)
complete_focal = complete_actor
```

如果某个窗口没有完整 focal，跳过窗口。

如果采用严格 SocialGAN 场景形式，窗口内 actor 集合使用 `complete_actor`。如果实现 agent 选择允许单 actor 窗口或保留部分可见邻居，必须：

- 将该策略写入 manifest；
- 在结果中记录 `context_policy`；
- 不把该结果称为严格 SocialGAN reproduction；
- 至少额外提供 strict-complete 版本作为主结果。

**预处理输出必须包含：**

- `positions`
- `valid_mask`
- `agent_ids`
- `focal_id`
- `frame_ids`
- `frame_ids_raw`
- `scene_id`
- `fold`
- `split`
- `source_file`
- `sample_index`

**验证命令：**

```bash
PYTHONPATH=. pytest -q tests/test_ethucy_benchmark_preprocess.py
python scripts/数据集构建/preprocess_ethucy.py benchmark \
    --raw-root /home/lbh/MoFlow/data/eth_ucy/original \
    --output-root data/ETHUCY_benchmark_v1 \
    --obs-len 8 \
    --pred-len 12 \
    --dt 0.4
```

如果不采用子命令，使用一个明确的新脚本：

```bash
python scripts/数据集构建/preprocess_ethucy_benchmark.py \
    --raw-root /home/lbh/MoFlow/data/eth_ucy/original \
    --output-root data/ETHUCY_benchmark_v1
```

### 任务 4：数据完整性审计

**文件：**

- 创建：`scripts/审计与校验/audit_ethucy_benchmark.py`
- 创建：`docs/audits/ETHUCY基准数据审计.md`
- 测试：`tests/test_ethucy_benchmark_audit.py`

**必须输出：**

| 统计项 | 说明 |
|---|---|
| source 文件数 | 每个 fold/split/scene |
| source 行数 | 原始文本有效行数 |
| source frame 数 | unique raw frame 数 |
| 生成样本数 | 每个 fold/split/scene |
| actor 数分布 | min/median/max |
| focal 数量 | 每个 split |
| 无效 focal 历史数 | 必须为 0 |
| 无效 focal 未来数 | 必须为 0 |
| sample split overlap | 必须为 0 |
| source file overlap | 必须符合官方 fold 组织，不能出现同 fold 跨 split 重复 |

**强制断言：**

```python
assert focal_valid[:, :8].all()
assert focal_valid[:, 8:20].all()
assert target_mask[0].all()
assert torch.isfinite(positions).all()
assert frame_ids.shape == (20,)
```

如果使用 strict-complete actor 策略：

```python
assert valid_mask.all()
```

**验证命令：**

```bash
python scripts/审计与校验/audit_ethucy_benchmark.py \
    --data-root data/ETHUCY_benchmark_v1 \
    --manifest data/ETHUCY_benchmark_v1/manifest.json \
    --output docs/audits/ETHUCY基准数据审计.md
```

在数据审计未通过前，不得开始训练。

---

## 6. DeMo Dataset 和模型适配任务

### 任务 5：实现 benchmark Dataset/DataModule

**文件：**

- 创建或修改：`src/datamodule/ethucy_benchmark_dataset.py`
- 创建或修改：`src/datamodule/ethucy_benchmark_datamodule.py`
- 创建：`conf/datamodule/ethucy_benchmark.yaml`
- 测试：`tests/test_ethucy_benchmark_dataset.py`

**Dataset 行为：**

1. 读取一个 `.pt` 样本。
2. 将 focal 放到 actor index 0。
3. 使用 focal 观测末帧作为局部坐标原点。
4. 使用 focal 最后一个有效位移确定旋转角。
5. 如果最后位移过小，向前寻找最近的非零位移；若整个历史没有足够位移，使用 `theta=0` 并记录 `degenerate_heading=True`。
6. 生成 DeMo 需要的 actor-only 字段：

```text
target
target_diff
target_vel_diff
target_mask
x_positions_diff
x_positions
x_attr
x_centers
x_angles
x_velocity
x_velocity_diff
x_valid_mask
x_key_valid_mask
origin
theta
scene_id
track_id
timestamp
```

7. `x_attr[..., 2]` 全部为 pedestrian type `0`。
8. 不生成 lane 字段。
9. `timestamp` 使用 `obs_len * 0.4`，并在代码注释中明确单位。
10. 保证 `x_key_valid_mask[:, 0]` 对 focal 为 `True`。

**必须有坐标往返测试：**

```python
global_position
    -> local_transform
    -> inverse_transform
    -> recovered_global_position
```

误差阈值建议为 `1e-5`。测试至少覆盖：

- focal 向右运动；
- focal 向上运动；
- focal 末两帧位移接近零；
- 多 actor；
- batch 内 actor 数不同；
- `N=1`；
- `N>1`。

**DataModule 接口：**

```python
EthUcyBenchmarkDataModule(
    data_root="data/ETHUCY_benchmark_v1",
    fold="ETH",
    obs_len=8,
    pred_len=12,
    train_batch_size=...,
    val_batch_size=...,
    test_batch_size=...,
    num_workers=...,
    test=False,
)
```

`test=False` 时只构建当前 fold 的 train 和 val；`test=True` 时只构建当前 fold 的 test。不能在 `test=True` 时加载 train 或 val。

### 任务 6：确认并最小化模型改动

**文件：**

- 复核：`src/model/model_forecast.py`
- 复核：`src/model/layers/time_decoder.py`
- 复核：`src/model/trainer_forecast.py`
- 创建：`conf/model/ethucy_benchmark_model_forecast.yaml`
- 创建：`conf/config_ethucy_benchmark.yaml`
- 测试：`tests/test_ethucy_benchmark_model.py`

**固定模型配置：**

```yaml
model:
  type: ModelForecast
  embed_dim: 128
  future_steps: 12
  num_heads: 8
  mlp_ratio: 4.0
  qkv_bias: false
  drop_path: 0.2
  num_actor_types: 1
  num_modes: 6
```

**模型边界：**

- 保留现有 DeMo 主干。
- 不新增运动模式编码器。
- 不新增缺失感知门控。
- 不新增图像、地图或外部特征。
- `pretrained_weights` 默认为空，避免上游自动驾驶 checkpoint 输入维度和任务定义污染迁移实验。
- 保证输出形状：

```text
y_hat: [B, 6, 12, 2]
pi:    [B, 6]
scal:  [B, 6, 12, 2]
```

**训练损失边界：**

- focal 轨迹作为主监督目标。
- 其他 actor 的辅助损失只能使用真实有效的 `target_mask`。
- 如果严格完整 actor 策略使所有 actor 都完整，则可以保留现有 `y_hat_others` 辅助损失。
- 不得通过 `target_mask` 强制覆盖无效未来标签。
- 检查 `LaplaceNLLLoss` 的 mode 数量来自 `out_mu.size(1)`，不能写死为 6。

**依赖隔离：**

如果 `Trainer` 为 ETH/UCY 仍然无条件 import AV2：

（历史记录，已过时：仓库收缩为行人专用后，`Trainer` 已只保留 SubmissionEthUcy，上游 AV2 submission 整体移至 DeMo_Origin 存档，下述选项不再适用。）

---

## 7. Smoke Test 任务

### 任务 7：单样本、单 batch、单折快速验证

**文件：**

- 创建：`scripts/审计与校验/smoke_test_ethucy_benchmark.py`
- 创建：`docs/results/ETHUCY基准冒烟测试记录.md`

**阶段 A：数据 smoke test**

运行：

```bash
python scripts/审计与校验/smoke_test_ethucy_benchmark.py \
    --stage data \
    --data-root data/ETHUCY_benchmark_v1 \
    --fold ETH
```

必须验证：

- train、val、test 均可加载；
- 每个 sample 为 20 个时间步；
- focal 历史和未来完整；
- batch collate 成功；
- 不存在 NaN/Inf；
- 不存在 lane 字段依赖。

**阶段 B：模型 smoke test**

运行：

```bash
python train.py \
    --config-name=config_ethucy_benchmark \
    datamodule.target.fold=ETH \
    epochs=1 \
    limit_train_batches=2 \
    limit_val_batches=2 \
    batch_size=2 \
    gpus=1 \
    precision=32 \
    output=smoke_ethucy_benchmark
```

必须验证：

- forward 成功；
- loss 为有限值；
- 反向传播成功；
- 至少生成一个 checkpoint；
- validation 输出 `val_minADE6` 和 `val_minFDE6`；
- 没有 CUDA OOM、shape mismatch、Mamba import error。

**阶段 C：test smoke test**

使用阶段 B checkpoint：

```bash
python eval.py \
    --config-name=config_ethucy_benchmark \
    datamodule.target.fold=ETH \
    test=true \
    checkpoint=/absolute/path/to/smoke_checkpoint.ckpt \
    precision=32 \
    gpus=1 \
    output=smoke_ethucy_benchmark_eval
```

必须验证：

- 只加载 ETH fold 的 test split；
- 输出预测文件；
- 预测形状为 `[B, 6, 12, 2]`；
- 坐标反变换后数值有限；
- test 指标可以计算。

未通过 smoke test，不得启动完整五折训练。

---

## 8. 完整训练和评估任务

### 任务 8：创建严格五折 runner

**文件：**

- 创建或重写：`scripts/训练与评估/run_ethucy_benchmark.py`
- 创建：`scripts/parse_ethucy_results.py`
- 创建：`docs/results/ETHUCY基准训练日志.md`
- 测试：`tests/test_ethucy_benchmark_runner.py`

不要直接复用当前 [`scripts/训练与评估/run_5fold_loo.py`](/home/lbh/DeMo/scripts/训练与评估/run_5fold_loo.py)，除非修正其 split 和 checkpoint 逻辑后经过测试。

**五折定义：**

```python
FOLDS = ["ETH", "HOTEL", "UNIV", "ZARA1", "ZARA2"]
```

每个 fold：

```text
train = fold/<heldout>/train
val   = fold/<heldout>/val
test  = fold/<heldout>/test
```

**训练阶段：**

- 固定随机种子 `2024`。
- 默认最大训练轮数 `100`。
- warmup `5` epochs。
- 使用 `AdamW`。
- 使用当前 ETH/UCY 配置的学习率和 weight decay 作为初始配置。
- 默认 `batch_size=64` 或根据显存自动调整，但完整五折必须记录实际 batch size。
- 默认单 GPU 顺序运行，避免共享输出目录和显存冲突。
- 如果并行运行，每个 fold 必须拥有独立 `CUDA_VISIBLE_DEVICES` 和输出目录。
- checkpoint 监控 `val_minFDE6`，保存 best 和 last。
- 可加入 EarlyStopping，但 patience、monitor 和 mode 必须记录。
- 不使用 test 指标参与训练控制。

推荐完整训练命令：

```bash
python scripts/训练与评估/run_ethucy_benchmark.py \
    --config-name config_ethucy_benchmark \
    --data-root data/ETHUCY_benchmark_v1 \
    --output-root outputs/ethucy_benchmark_v1 \
    --epochs 100 \
    --batch-size 64 \
    --gpus 1 \
    --seed 2024
```

### 任务 9：checkpoint 选择和最终测试

对每个 fold：

1. 从该 fold 的 `metrics.csv` 中找最小 `val_minFDE6`。
2. 读取对应 `epoch=N.ckpt`。
3. 只使用该 checkpoint 在该 fold 的 test split 上运行一次。
4. 保存 test 指标，不回头修改 checkpoint。
5. 保存 checkpoint epoch、val 指标、test 指标、训练时长和 git commit。

严禁以下做法：

- 从 test 指标选择 checkpoint。
- 看到 test 结果不好后重新选择 epoch。
- 将 test split 合并回 train。
- 使用全场景 `.pt` 目录让 Dataset 自行推断 split。
- 使用已有旧实验 checkpoint 作为主结果。

### 任务 10：结果汇总

**输出文件：**

```text
outputs/ethucy_benchmark_v1/
    fold_ETH/
        train/
        eval/
        best.ckpt
        metrics.csv
    fold_HOTEL/
        ...
    fold_UNIV/
        ...
    fold_ZARA1/
        ...
    fold_ZARA2/
        ...
    results.csv
    results.json
    SUMMARY.md
```

`results.csv` 至少包含：

```text
fold
seed
checkpoint_epoch
train_samples
val_samples
test_samples
minADE1
minFDE1
minADE6
minFDE6
MR
b-minFDE6
train_seconds
eval_seconds
git_commit
data_manifest_sha256
```

`SUMMARY.md` 必须包含：

- 每个 fold 的样本数；
- 每个 fold 的最佳 checkpoint epoch；
- 每个 fold 的 val 指标；
- 每个 fold 的 test 指标；
- 五折均值；
- 五折标准差；
- 最好和最差场景；
- 训练失败或重试记录；
- 当前结果的可比性限制。

均值必须只对成功完成且指标完整的 fold 计算。如果某 fold 失败，不得用 0 填充后求平均。

---

## 9. 最终验收清单

### 9.1 数据验收

- [ ] 原始 `.txt` 文件的修改时间和内容未改变。
- [ ] 正式数据不来自 `raw/all_data`。
- [ ] 五个 fold 的 train/val/test manifest 均存在。
- [ ] 同一 fold 内 source file 没有跨 split 重复。
- [ ] 所有正式样本窗口长度为 20。
- [ ] focal 历史 8 帧全部有效。
- [ ] focal 未来 12 帧全部有效。
- [ ] 训练标签中没有 NaN/Inf。
- [ ] 未使用插值产生监督目标。
- [ ] 数据审计报告已生成。

### 9.2 Dataset/模型验收

- [ ] actor-only forward 成功。
- [x] （已完成并超出：模型已收缩为纯 actor-only，无 lane 分支）
- [ ] `y_hat` 形状为 `[B, 6, 12, 2]`。
- [ ] 局部坐标与全局坐标往返误差小于 `1e-5`。
- [ ] `target_mask` 反映真实有效性。
- [ ] `N=1` 和不同 `N` batch 均能运行。
- [ ] loss、梯度和输出均为有限值。

### 9.3 训练/评估验收

- [ ] 单折 smoke test 通过。
- [ ] 五个 fold 均有独立输出目录。
- [ ] 每个 fold 的 checkpoint 只由 val `minFDE6` 选择。
- [ ] test 只使用对应 fold 的 test split。
- [ ] 五折结果均有 `minADE1/minFDE1/minADE6/minFDE6/MR`。
- [ ] `results.csv`、`results.json`、`SUMMARY.md` 均生成。
- [ ] 结果中记录 seed、git commit、数据 manifest hash。

### 9.4 结果解释验收

- [ ] 报告明确这是 DeMo 的基础迁移实验。
- [ ] 报告明确模型使用 6 个 mode。
- [ ] 不把 `minADE6/minFDE6` 直接等同于 `minADE20/minFDE20`。
- [ ] 不声称解决历史缺失或实现新的运动模式模块。
- [ ] 说明当前没有使用地图。
- [ ] 说明使用的具体 split、数据版本、训练轮数和 checkpoint 选择规则。

---

## 10. 推荐结果报告模板

执行 agent 完成后，在 `outputs/ethucy_benchmark_v1/SUMMARY.md` 中使用类似结构：

```markdown
# DeMo on ETH/UCY Benchmark

## Protocol

- Dataset:
- Observation / prediction:
- Sampling interval:
- Split:
- Context policy:
- Number of modes:
- Seed:
- Checkpoint selection:

## Results

| Fold | minADE1 | minFDE1 | minADE6 | minFDE6 | MR |
|---|---:|---:|---:|---:|---:|
| ETH |  |  |  |  |  |
| HOTEL |  |  |  |  |  |
| UNIV |  |  |  |  |  |
| ZARA1 |  |  |  |  |  |
| ZARA2 |  |  |  |  |  |
| Mean |  |  |  |  |  |
| Std |  |  |  |  |  |

## Interpretation

本实验只用于评估 DeMo 从自动驾驶运动预测迁移到 ETH/UCY 行人轨迹预测后的基础能力。
结果不用于证明新的缺失轨迹建模、运动模式增强或生成式模型贡献。

## Limitations

- DeMo 输出 6 个 mode。
- 没有使用地图。
- 不与不同 K、不同 split 或不同采样频率的论文结果直接比较。
- 需要根据每个 fold 的样本数量和场景差异解释结果。
```

---

## 11. 执行完成后的交付物

其他 agent 只有在以下内容全部存在后，才能向用户汇报“基线迁移实验完成”：

1. `data/ETHUCY_benchmark_v1/manifest.json`
2. 数据审计报告
3. benchmark Dataset/DataModule 代码
4. benchmark 配置文件
5. 单元测试和 smoke test 输出
6. 五个 fold 的 checkpoint 或明确失败日志
7. `results.csv`
8. `results.json`
9. `SUMMARY.md`
10. 每个 fold 的训练/评估日志

如果任何 fold 失败，agent 必须报告具体失败阶段、错误信息、已生成文件和是否存在可用的部分结果，不能用不完整结果冒充五折完成。

---

## 12. 参考协议

本方案以 SocialGAN 风格的固定长度轨迹窗口和 ETH/UCY 官方场景划分为 benchmark 锚点；Trajectron++、AgentFormer 和近期 ETH/UCY 工作可作为实现交叉核对来源。由于不同论文可能使用不同的采样数量、指标和 split，最终报告必须把这些配置写清楚。

[^1]: https://github.com/agrimgupta92/sgan
[^2]: https://github.com/StanfordASL/Trajectron-plus-plus
[^3]: https://proceedings.neurips.cc/paper_files/paper/2024/file/69f3eb242c7c9df9ea2f2b66ea8b3c0f-Paper-Conference.pdf
