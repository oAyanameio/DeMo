"""BiMamba 对照组：MoFlow 协议 5 子集批量训练+评测。

用法：
    CUDA_VISIBLE_DEVICES=2 python scripts/训练与评估/run_moflow_protocol_bimamba.py [epochs]

与 scripts/训练与评估/run_moflow_protocol.py 完全相同，但使用 config_moflow_ethucy_bimamba.yaml
（bimamba=True），输出目录 moflow_bimamba_*。
"""
import subprocess
import sys
from pathlib import Path

SUBSETS = ["eth", "hotel", "univ", "zara1", "zara2"]


def find_best_metrics(run_dir: Path):
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
        print(f"===== BiMamba {subset} =====", flush=True)
        env = {
            **__import__("os").environ,
            "CUDA_VISIBLE_DEVICES": env_gpu,
        }
        proc = subprocess.Popen(
            [
                "conda", "run", "-n", "DeMo", "python", "-u", "train.py",
                "--config-name", "config_moflow_ethucy_bimamba",
                f"datamodule.target.subset={subset}",
                f"epochs={epochs}",
            ],
            cwd=root,
            env=env,
        )
        proc.wait()
        out_root = root / "outputs" / f"moflow_bimamba_{subset}"
        runs = sorted(out_root.glob("*/"), key=lambda p: p.stat().st_mtime)
        best = find_best_metrics(runs[-1]) if runs else None
        if best:
            _, row = best
            results[subset] = {k: row.get(k, "") for k in [
                "val_new_minADE20", "val_new_minFDE20", "val_new_b-minFDE20"]}
            print(subset, results[subset], flush=True)

    summary = root / "outputs" / "moflow_bimamba_protocol_summary.txt"
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