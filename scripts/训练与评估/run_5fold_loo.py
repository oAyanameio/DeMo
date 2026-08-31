#!/usr/bin/env python3
"""5-fold Leave-One-Scene-Out CV for ETH/UCY — STANDARD PROTOCOL.

Train on 4 scenes, val+test on the held-out scene (val used only for
checkpoint selection, which is the common ETH/UCY reproduction protocol).
Runs folds SEQUENTIALLY on one GPU.
"""
import csv
import os
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 30
BATCH = int(sys.argv[2]) if len(sys.argv) > 2 else 128

ALL = ["ETH", "HOTEL", "UNIV", "ZARA1", "ZARA2"]

OUTPUT_BASE = Path("outputs")


def find_best_ckpt(output_name):
    """Find best checkpoint by val_minFDE6 from metrics.csv; symlink-safe."""
    dirs = sorted(OUTPUT_BASE.glob(f"{output_name}/*/"), reverse=True)
    for d in dirs:
        ckpt_dir = d / "checkpoints"
        if not ckpt_dir.exists():
            continue
        metrics_csv = d / "logs" / "version_0" / "metrics.csv"
        if metrics_csv.exists():
            try:
                best_epoch, best_val = None, float("inf")
                with open(metrics_csv) as f:
                    for row in csv.DictReader(f):
                        try:
                            e = int(row["epoch"])
                            v = float(row["val_minFDE6"])
                            if v < best_val:
                                best_val, best_epoch = v, e
                        except (ValueError, KeyError):
                            pass
                if best_epoch is not None:
                    ckpt = ckpt_dir / f"epoch={best_epoch}.ckpt"
                    if ckpt.exists():
                        return str(ckpt), best_epoch
            except Exception:
                pass
        ckpts = [c for c in sorted(ckpt_dir.glob("epoch=*.ckpt"))]
        if ckpts:
            return str(ckpts[-1]), None
        last = ckpt_dir / "last.ckpt"
        if last.exists():
            return str(last), None
    return None, None


def parse_metrics(stdout):
    metrics = {}
    for line in stdout.split("\n"):
        if "│" not in line:
            continue
        for key in ["val_minADE6", "val_minFDE6", "val_MR"]:
            if key in line:
                for p in line.split("│"):
                    try:
                        metrics[key] = float(p.strip())
                        break
                    except ValueError:
                        continue
    return metrics


def main():
    print(f"=== 5-Fold LOO (standard) at {datetime.now()} ===", flush=True)
    print(f"GPU={os.environ.get('CUDA_VISIBLE_DEVICES')} epochs={EPOCHS} batch={BATCH}", flush=True)
    print("Protocol: train=4 scenes, val=test scene (ckpt selection only)", flush=True)

    results = {}
    for fold_num, test in enumerate(ALL, 1):
        train = ",".join(s for s in ALL if s != test)
        output_name = f"loo_test-{test}"

        cmd = [
            sys.executable, "-u", "train.py",
            "--config-name=config_ethucy",
            "gpus=1",
            f"epochs={EPOCHS}",
            f"batch_size={BATCH}",
            f"datamodule.target.train_scenes=[{train}]",
            f"datamodule.target.val_scenes=[{test}]",
            f"datamodule.target.test_scenes=[{test}]",
            f"output={output_name}",
        ]
        t0 = time.time()
        print(f"\n[Fold {fold_num}/5] test={test} train=[{train}] TRAIN starting...", flush=True)
        ret = subprocess.run(cmd).returncode
        print(f"[Fold {fold_num}] TRAIN ret={ret} elapsed={time.time()-t0:.0f}s", flush=True)

        ckpt, best_epoch = find_best_ckpt(output_name)
        if not ckpt:
            print(f"[Fold {fold_num}] ERROR: no checkpoint", flush=True)
            continue
        print(f"[Fold {fold_num}] best ckpt: {ckpt} (epoch={best_epoch})", flush=True)

        # symlink to avoid '=' in Hydra CLI
        ckpt_path = Path(ckpt)
        safe = ckpt_path.parent / "best_for_eval.ckpt"
        if safe.exists() or safe.is_symlink():
            safe.unlink()
        safe.symlink_to(ckpt_path.name)

        eval_cmd = [
            sys.executable, "-u", "eval.py",
            "--config-name=config_ethucy",
            "gpus=1", "test=false",
            f"checkpoint={safe}",
            f"datamodule.target.val_scenes=[{test}]",
            f"output={output_name}",
        ]
        t0 = time.time()
        r = subprocess.run(eval_cmd, capture_output=True, text=True)
        print(f"[Fold {fold_num}] EVAL ret={r.returncode} elapsed={time.time()-t0:.0f}s", flush=True)
        if r.returncode != 0:
            print(f"[Fold {fold_num}] Eval stderr: {r.stderr[-500:]}", flush=True)
            continue

        metrics = parse_metrics(r.stdout)
        results[test] = metrics
        print(f"[Fold {fold_num}] {test}: {metrics}", flush=True)

        # append partial results after each fold
        with open("5fold_loo_results.txt", "w") as f:
            f.write(f"5-Fold LOO (standard) — {datetime.now()}\n")
            f.write(f"epochs={EPOCHS} batch={BATCH}\n\n")
            ade = [results[t].get("val_minADE6", 0) for t in ALL if t in results]
            fde = [results[t].get("val_minFDE6", 0) for t in ALL if t in results]
            mr = [results[t].get("val_MR", 0) for t in ALL if t in results]
            for t in ALL:
                if t in results:
                    m = results[t]
                    f.write(f"{t}: ADE6={m.get('val_minADE6',0):.4f}  "
                            f"FDE6={m.get('val_minFDE6',0):.4f}  MR={m.get('val_MR',0):.4f}\n")
            if results:
                n = len(results)
                f.write(f"\nAVG: ADE6={sum(ade)/n:.4f}  FDE6={sum(fde)/n:.4f}  MR={sum(mr)/n:.4f}\n")

    print(f"\n{'='*50}\nFinal Summary at {datetime.now()}\n{'='*50}", flush=True)
    if results:
        for t in ALL:
            if t in results:
                m = results[t]
                print(f"{t:>8s}  {m.get('val_minADE6',0):>8.4f}  "
                      f"{m.get('val_minFDE6',0):>8.4f}  {m.get('val_MR',0):>8.4f}", flush=True)
        n = len(results)
        print("Saved to 5fold_loo_results.txt", flush=True)
    else:
        print("No results!", flush=True)


if __name__ == "__main__":
    main()
