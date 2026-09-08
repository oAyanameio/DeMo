"""修复 M0 results.json：从五折 fold_result.json 重汇总（fold_result 全部完好）。

也作为收尾通用工具：--skip-train 的 runner 会话只汇总本次跑的折，
全量汇总需从磁盘 fold_result.json 回读。M1/M2 收尾时会用到同一逻辑。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
# 直接复用 runner 的 summarize 逻辑但以磁盘 rows 为准
import importlib.util

spec = importlib.util.spec_from_file_location(
    "ma_eth", Path(__file__).resolve().parents[2] / "scripts" / "训练与评估" / "run_missing_aware_ethucy.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

exp_dir = Path(sys.argv[1])
variant = sys.argv[2]
rows = []
for f in mod.FOLDS:
    fr = exp_dir / f"fold_{f}" / "fold_result.json"
    if fr.exists():
        rows.append(json.loads(fr.read_text()))
    else:
        print(f"WARN: 缺 {fr}")

class A:
    pass
args = A()
args.variant = variant
args.condition = rows[0]["condition"] if rows else ""
args.seed = rows[0]["seed"] if rows else None
args.folds = [r["fold"] for r in rows]
summary = mod.summarize(exp_dir, rows, args)
print(f"variant={variant} status={summary['status']} folds_ok={summary['folds_ok']}")
for k in ("mean_test_minFDE6", "mean_test_minADE6", "mean_test_MR", "mean_test_b-minFDE6"):
    if k in summary:
        print(f"  {k} = {summary[k]}")
