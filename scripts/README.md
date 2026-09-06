# scripts/ 目录结构

所有脚本统一约定：**从仓库根目录（/home/lbh/DeMo）运行**，入口为仓库根的 `train.py` / `eval.py`。

| 子目录 | 内容 |
|---|---|
| `数据集构建/` | 数据预处理与数据集制作：`preprocess_ethucy.py`（原始 ETH/UCY → .pt）、`build_ethucy_manifest.py`（benchmark v1 manifest）、`build_missing_history_dataset.py`（缺失历史 v1 / v2_high 数据集，ETH/UCY + SDD） |
| `训练与评估/` | 各轮实验的批量运行入口：`run_ethucy_benchmark.py`（benchmark 五折训练+评估）、`run_missing_aware_ethucy.py`（Missing-Aware M0/M1/M2 ETH/UCY 五折正式实验，variant 固定映射模型开关，val_minFDE6 选点，experiment_meta.json 追溯）、`run_missing_aware_sdd.py`（同协议 SDD 版，condition/data_root 走 datamodule.target.*，val_minFDE20 选点）、`run_ma_screening_round1.sh`（第一组筛选链：M0→M1→M2→analyze rc 门控）、`run_missing_round1.sh`（第一轮三臂并行总控）、`run_missing_round2.sh`（第二轮 v2_high 训练与编排）、`run_zeroshot_v2.sh`（第二轮零样本评估）、`run_sdd.py` / `run_sdd_missing.py`（SDD 训练评估）、`run_moflow_protocol{,_bimamba}.py`、`run_heldout_test.py`、`run_5fold_*.py`（旧版五折，已被 benchmark 版取代）、`rerun_eval_v1_round1.sh`（第一轮 eval 补跑） |
| `审计与校验/` | 数据与配置正确性检查：`audit_ethucy_benchmark.py`、`audit_missing_history_dataset.py`（v1 / v2_high）、`smoke_test_ethucy_benchmark.py`、`smoke_missing_history_interface.py`（v1/v2/v3 三版本断言集）、`smoke_missing_aware_m1m2.py`（M1_obs/M2_history 新字段 + 模型前向真实数据 smoke）、`smoke_trainer_missing_aware.py`（M1+M2 Trainer 一步训练 + hydra 默认配置回归）、`check_config_parity.py`（uni/bi 对比 + Missing-Aware 配置一致性检查） |
| `结果分析/` | 指标提取与汇总：`analyze_missing_aware.py`（Missing-Aware M0/M1/M2 配对分析：完整性校验、M1-M0/M2-M1/M2-M0 配对差、bootstrap CI、summary.json/csv/md 三格式输出）、`extract_val_metrics.py`（从 metrics.csv 抽 val 曲线）、`parse_eval_rerun.py`（第一轮 eval 补跑日志 → 三臂五折汇总表）、`parse_eval_bimamba.py`（双向对照结果解析） |

注意事项：

- 第二轮 v2_high 数据已生成并通过全量审计：`data/ETHUCY_missing_v2_high/` 和 `data/SDD_missing_v2_high/` 支持 `random_fixed3`、`random_fixed4`、`random_block3`、`random_block4`、`random_block6`；`random_fixed5` 仍是未生成的可选条件。第二轮训练与零样本实验已于 2026-09-01 启动，结果尚未汇总。

- checkpoint 文件名含 `=`（如 `epoch=73.ckpt`）时，Hydra override 必须整体加引号：`"checkpoint='outputs/.../epoch=73.ckpt'"`。Missing-Aware runner 内部已用 symlink（`best_for_eval.ckpt`）规避此问题。
- eval 输出目录已存在 `model.log` 会 FileExistsError，重跑前先删整个 eval 目录。
- 带 `=` 的路径也可用 `run_heldout_test.py` 的 symlink 方案规避。
- Missing-Aware 正式实验三件套用法：
  ```bash
  # ETH/UCY 五折（variant 固定映射开关；输出 outputs/missing_aware/ethucy/train_adapt/<variant>/<condition>/seed_<seed>/）
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/训练与评估/run_missing_aware_ethucy.py \
      --variant M1_obs --condition random_fixed4_ng \
      --data-root data/ETHUCY_missing_v3_noguard/random_fixed4_ng \
      --output-root outputs/missing_aware/ethucy/train_adapt --seed 2024 --gpu 0
  # SDD（condition/data_root 必须走 datamodule.target.*，runner 已处理）
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/训练与评估/run_missing_aware_sdd.py \
      --variant M2_history --condition random_fixed4_ng \
      --data-root data/SDD_missing_v3_noguard \
      --output-root outputs/missing_aware/sdd/train_adapt --seed 2024 --gpu 0
  # 配对分析（拒绝不完整/不一致组；输出 summary.json/csv/md）
  python scripts/结果分析/analyze_missing_aware.py --input-root outputs/missing_aware \
      --dataset ethucy --protocol train_adapt --conditions random_fixed4_ng \
      --variants M0_base M1_obs M2_history --seeds 2024
  ```
