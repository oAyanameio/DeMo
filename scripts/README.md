# scripts/ 目录结构

所有脚本统一约定：**从仓库根目录（/home/lbh/DeMo）运行**，入口为仓库根的 `train.py` / `eval.py`。

| 子目录 | 内容 |
|---|---|
| `训练与评估/` | TrajImpute 官方 Easy/Hard direct、原始 ETH/UCY 和原始 SDD 的训练评估脚本 |
| `审计与校验/` | 当前模型与配置的静态检查 |
| `结果分析/` | TrajImpute direct 评估与汇总：`evaluate_trajimpute_direct.py`（分组直接预测评估）、`summarize_trajimpute.py`（重训结果汇总） |

注意事项：

- 历史自定义缺失数据已清理；正式缺失实验统一使用 TrajImpute 官方 release，SDD 扩展仅使用原始 SDD 数据。

- checkpoint 文件名含 `=`（如 `epoch=73.ckpt`）时，Hydra override 必须整体加引号：`"checkpoint='outputs/.../epoch=73.ckpt'"`。Missing-Aware runner 内部已用 symlink（`best_for_eval.ckpt`）规避此问题。
- eval 输出目录已存在 `model.log` 会 FileExistsError，重跑前先删整个 eval 目录。
- 带 `=` 的路径也可用 `run_heldout_test.py` 的 symlink 方案规避。
