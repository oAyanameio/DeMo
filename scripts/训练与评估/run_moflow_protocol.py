"""MoFlow 协议 5 子集（eth/hotel/univ/zara1/zara2）批量训练+评测。

用法:
    CUDA_VISIBLE_DEVICES=2 python scripts/训练与评估/run_moflow_protocol.py [epochs]

每个子集：标准 leave-one-out（该子集作 test，其余 4 场景作 train/val，
SocialGAN 官方划分，与 MoFlow 论文一致），
指标 minADE20 / minFDE20 / b-minFDE20（米制）。
结果汇总写入 outputs/moflow_protocol_summary.txt。
"""
import subprocess
import sys
from pathlib import Path

SUBSETS = ["eth", "hotel", "univ", "zara1", "zara2"]


def find_best_metrics(run_dir: Path):
    """从 metrics.csv 中找 val_new_minFDE20 最小的 epoch 行。"""
    import csv

    csvs = sorted(run_dir.glob("logs/version_*/metrics.csv"))
    best = None
    for c in csvs:
        with open(c) as f:
            rows = [r for r in csv.DictReader(f) if r.get("val_new_minFDE20")]
        for r in rows:
            v = float(r["val_new_minFDE20"])
            if best is None or v < best[0]:
                best = (v, r)
    return best


def main():
    epochs = sys.argv[1] if len(sys.argv) > 1 else "100"
    root = Path(__file__).parent
    results = {}
    for subset in SUBSETS:
        print(f"===== {subset} =====", flush=True)
        env = {
            **__import__("os").environ,
            "CUDA_VISIBLE_DEVICES": env_gpu,
        }
        proc = subprocess.Popen(
            [
                "conda", "run", "-n", "DeMo", "python", "-u", "train.py",
                "--config-name", "config_moflow_ethucy",
                f"datamodule.target.subset={subset}",
                f"epochs={epochs}",
            ],
            cwd=root,
            env=env,
        )
        proc.wait()
        out_root = root / "outputs" / f"moflow_{subset}"
        runs = sorted(out_root.glob("*/"), key=lambda p: p.stat().st_mtime)
        best = find_best_metrics(runs[-1]) if runs else None
        if best:
            _, row = best
            results[subset] = {k: row.get(k, "") for k in [
                "val_new_minADE20", "val_new_minFDE20", "val_new_b-minFDE20"]}
            print(subset, results[subset], flush=True)

    summary = root / "outputs" / "moflow_protocol_summary.txt"
    with open(summary, "w") as f:
        f.write(f"subset  minADE20  minFDE20  b-minFDE20 (best val)\n")
        for s in SUBSETS:
            r = results.get(s, {})
            f.write(f"{s:>7s}  {r.get('val_new_minADE20','')}  "
                    f"{r.get('val_new_minFDE20','')}  {r.get('val_new_b-minFDE20','')}\n")
    print(f"Summary -> {summary}")


if __name__ == "__main__":
    env_gpu = __import__("os").environ.get("CUDA_VISIBLE_DEVICES", "2")
    main()
