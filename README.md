# DeMo: 面向 ETH/UCY + SDD 的行人轨迹预测实验代码库

基于 [DeMo (NeurIPS 2024)](https://arxiv.org/abs/2410.05982) 的行人轨迹预测（pedestrian trajectory prediction）研究代码库，收缩为 ETH/UCY + SDD 双数据集、DeMo actor-only（无地图/车道）模型、含缺失历史（missing-history）数据协议的实验框架。

- 模型：DeMo 单向（UniMamba）/ 双向（BiMamba）actor-only 骨干，唯一变量为方向性
- 数据：ETH/UCY 5 折留一（LOO）+ SDD，缺失历史 v1（轻度）/ v2（高缺失）协议
- 上游完整版本（含 AV2/自动驾驶管线）存档于 [DeMo_Origin](https://github.com/oAyanameio/DeMo_Origin)

## 环境安装

```bash
conda create -n DeMo python=3.10
conda activate DeMo
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Mamba（VideoMamba fork，支持 bimamba=True）
git clone git@github.com:OpenGVLab/VideoMamba.git
cd VideoMamba
pip install -e causal-conv1d
pip install -e mamba
```

注意：RTX 5880 Ada / CUDA 12.8 下 Triton RMSNorm 内核不兼容，`vim_mamba.py` 已改用 PyTorch 原生回退，详见 `docs/audits/`。

## 数据准备

### ETH/UCY（基准五折）

```bash
# 1. 源 manifest（fold/split/scene 清单）+ benchmark 预处理，一步完成：
python src/datamodule/ethucy_benchmark_preprocess.py \
    --raw-root /path/to/ethucy_raw \
    --output-root data/ETHUCY_benchmark_v1
# 或分两步（manifest 也可由 scripts/数据集构建/build_ethucy_manifest.py 单独生成）

# 2. 原始 ETH/UCY txt 预处理（非 benchmark 管线时）
python scripts/数据集构建/preprocess_ethucy.py \
    --data_root /path/to/ethucy \
    --output_root data/ETHUCY_processed \
    --frame_stride 10 --obs_len 8 --pred_len 12
```

### SDD（Stanford Drone Dataset）

仓库内无 SDD 预处理脚本。需自行准备 **MoFlow 格式源数据**（`data/sdd/`，即 MoFlow 协议的 pkl/划分格式）；缺失历史数据集直接在该源上构建。

### 缺失历史数据集（v1/v2，同一构建脚本，--version 区分）

```bash
# v1 轻度：complete / random_single / random_block2
python scripts/数据集构建/build_missing_history_dataset.py --dataset ethucy \
    --source-root data/ETHUCY_benchmark_v1 --output-root data/ETHUCY_missing_v1 \
    --conditions complete random_single random_block2
python scripts/数据集构建/build_missing_history_dataset.py --dataset sdd \
    --source-root data/sdd --output-root data/SDD_missing_v1 \
    --conditions complete random_single random_block2

# v2 高缺失：fixed3/fixed4、block3/block4/block6（37.5%~75%）
python scripts/数据集构建/build_missing_history_dataset.py --dataset ethucy \
    --source-root data/ETHUCY_benchmark_v1 --output-root data/ETHUCY_missing_v2_high \
    --version missing_history_v2_high \
    --conditions random_fixed3 random_fixed4 random_block3 random_block4 random_block6
python scripts/数据集构建/build_missing_history_dataset.py --dataset sdd \
    --source-root data/sdd --output-root data/SDD_missing_v2_high \
    --version missing_history_v2_high \
    --conditions random_fixed3 random_fixed4 random_block3 random_block4 random_block6

# 验收审计
python scripts/审计与校验/audit_missing_history_dataset.py --dataset ethucy \
    --data-root data/ETHUCY_missing_v1 --source-root data/ETHUCY_benchmark_v1
```

## 训练与评估

```bash
# ETH/UCY 基准（默认入口：config.yaml → ethucy_benchmark + ETH fold）
python train.py
python train.py fold=UNIV                 # 换折
python eval.py checkpoint='/path/to/ckpt' test=true

# 缺失历史训练适应臂（单向）
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/训练与评估/run_ethucy_benchmark.py \
    --config-name config_ethucy_benchmark \
    --data-root data/ETHUCY_missing_v1/random_block2 \
    --output-root outputs/ethucy_missing_v1_random_block2 --gpu 0

# 双向（bimamba）臂
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python scripts/训练与评估/run_ethucy_benchmark.py \
    --config-name config_ethucy_benchmark_bimamba \
    --data-root data/ETHUCY_missing_v1/random_block2 \
    --output-root outputs/ethucy_missing_v1_bimamba_random_block2 --gpu 1

# SDD（缺失历史，含 bimamba）
CUDA_VISIBLE_DEVICES=2 python scripts/训练与评估/run_sdd_missing.py random_block2
CUDA_VISIBLE_DEVICES=2 python scripts/训练与评估/run_sdd_missing_bimamba.py complete 100

# 零样本（complete 模型直测掩码测试集）
bash scripts/训练与评估/run_zeroshot_v2.sh   # SDD 补跑见 rerun_zeroshot_sdd.sh
```

## 目录结构

```
conf/                 Hydra 配置（ETH/UCY、SDD、缺失历史、uni/bi）
src/datamodule/       ETH/UCY、SDD、MoFlow 协议、missing 数据管线
src/model/            DeMo actor-only（ModelForecast + TimeDecoder）
src/metrics/          minADE/minFDE/MR/brierFDE
scripts/              数据集构建 / 训练与评估 / 审计与校验 / 结果分析
docs/                 研究文档、实验结果、审计报告、数据集说明
```

## 实验结果索引

- 总汇总：`docs/results/缺失历史实验总汇总.md`（唯一权威，随实验追加）
- 逐折数字：`outputs/*/results.json`、`outputs/sdd_*_summary.txt`
- 审计报告：`docs/audits/`
- MoFlow vs DeMo 留出对照：`docs/results/MoFlow与DeMo留出测试对比.md`

## 上游出处

```bibtex
@inproceedings{zhang2024demo,
 title={DeMo: Decoupling Motion Forecasting into Directional Intentions and Dynamic States},
 author={Zhang, Bozhou and Song, Nan and Zhang, Li},
 booktitle={NeurIPS},
 year={2024},
}
```

致谢：[VideoMamba](https://github.com/OpenGVLab/VideoMamba)、[Forecast-MAE](https://github.com/jchengai/forecast-mae)、[StreamPETR](https://github.com/exiawsh/StreamPETR)
