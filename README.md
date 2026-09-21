# DeMo: 面向 ETH/UCY + SDD 的行人轨迹预测实验代码库

基于 [DeMo (NeurIPS 2024)](https://arxiv.org/abs/2410.05982) 的行人轨迹预测（pedestrian trajectory prediction）研究代码库，收缩为 ETH/UCY + SDD 双数据集、DeMo actor-only（无地图/车道）模型，以及 TrajImpute 官方缺失历史数据协议。

- 模型：DeMo 单向（UniMamba）/ 双向（BiMamba）actor-only 骨干，唯一变量为方向性
- 数据：ETH/UCY 5 折留一（LOO）+ SDD 原始协议；缺失历史主线使用 TrajImpute 官方 Easy/Hard release
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

### ETH/UCY（完整数据）

预处理产物已在 `data/ETHUCY_processed/`（5 场景 .pt，经典 LOO 划分）；
原始 txt 预处理脚本已随 benchmark_v1 作废删除，正常使用无须重建。

### SDD（Stanford Drone Dataset）

仓库内无 SDD 预处理脚本。需自行准备 **标准 pkl/划分格式源数据**（`data/sdd/`）。

### TrajImpute 官方缺失历史数据

正式缺失历史数据位于 `/home/lbh/TrajImpute/dataset/TrajImpute`，包含
ETH-M、HOTEL-M、UNIV-M、ZARA1-M、ZARA2-M 的 Easy/Hard train/val/test pkl。
使用 `config_missing_aware_trajimpute` 和 `run_trajimpute_experiments.py`，不再生成仓库内自定义缺失数据。

## 训练与评估

```bash
# ETH/UCY 完整数据（原始协议）
python train.py
python train.py --config-name=config_ethucy fold=UNIV
python eval.py checkpoint='/path/to/ckpt' test=true

# TrajImpute Easy/Hard direct 缺失历史训练
PYTHONNOUSERSITE=1 PYTHONPATH=. python scripts/训练与评估/run_trajimpute_experiments.py \
    --protocol easy-direct --variant M0-current \
    --scenes ETH-M HOTEL-M UNIV-M ZARA1-M ZARA2-M --gpu 0
```

## 目录结构

```
conf/                 Hydra 配置（ETH/UCY、SDD、TrajImpute）
src/datamodule/       ETH/UCY、SDD 与 TrajImpute 数据管线
src/model/            DeMo actor-only（ModelForecast + TimeDecoder）
src/metrics/          minADE/minFDE/MR/brierFDE
scripts/              数据集构建 / 训练与评估 / 审计与校验 / 结果分析
docs/                 研究文档、实验结果、审计报告、数据集说明
```

## 实验结果索引

- 结果总索引：`docs/results/README.md`（按实验性质分类）
- 总汇总：`docs/results/主线实验记录.md`
- 逐折数字：`outputs/*/results.json`
- 审计报告：`docs/audits/`
- 完整数据基线与公开 SOTA 参考：`docs/results/完整数据基线对比.md`

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
