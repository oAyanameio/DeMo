#!/usr/bin/env python3
"""5-fold Leave-One-Out CV for ETH/UCY — SIMPLE VERSION.

Uses subprocess.Popen + wait() to avoid all the issues with run/os.system.
"""
import os
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 80
BATCH = int(sys.argv[2]) if len(sys.argv) > 2 else 128

FOLDS = [
    ("HOTEL,UNIV,ZARA2", "ZARA1", "ETH"),
    ("ETH,UNIV,ZARA2", "ZARA1", "HOTEL"),
    ("ETH,HOTEL,ZARA2", "ZARA1", "UNIV"),
    ("ETH,HOTEL,UNIV", "ZARA2", "ZARA1"),
    ("ETH,HOTEL,UNIV", "ZARA1", "ZARA2"),
]

OUTPUT_BASE = Path("outputs")


def find_best_ckpt(output_name):
    dirs = sorted(OUTPUT_BASE.glob(f"{output_name}/*/checkpoints/"), reverse=True)
    for d in dirs:
        ckpts = sorted(d.glob("epoch=*.ckpt"))
        ckpts = [c for c in ckpts if "last.ckpt" not in str(c)]
        if ckpts:
            return str(ckpts[-1])
        last = d / "last.ckpt"
        if last.exists():
            return str(last)
    return None


def run_training(train, val, test, output_name):
    """Run training subprocess, wait for completion, return exit code."""
    cmd = [
        sys.executable, "-u", "train.py",
        "--config-name=config_ethucy",
        f"gpus=1",
        f"epochs={EPOCHS}",
        f"batch_size={BATCH}",
        f"datamodule.target.train_scenes=[{train}]",
        f"datamodule.target.val_scenes=[{val}]",
        f"datamodule.target.test_scenes=[{test}]",
        f"output={output_name}",
    ]
    t0 = time.time()
    print(f"[Train] Starting: test={test}", flush=True)

    # Use Popen + wait — most reliable subprocess approach
    proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
    ret = proc.wait()
    elapsed = time.time() - t0

    print(f"[Train] Done: test={test}  ret={ret}  elapsed={elapsed:.0f}s", flush=True)
    return ret


def run_eval(test, ckpt, output_name):
    """Run evaluation, capture output for parsing."""
    cmd = [
        sys.executable, "-u", "eval.py",
        "--config-name=config_ethucy",
        "gpus=1",
        "test=false",
        f"checkpoint={ckpt}",
        f"datamodule.target.val_scenes=[{test}]",
        f"output={output_name}",
    ]
    t0 = time.time()
    print(f"[Eval] Starting: test={test}", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0
    print(f"[Eval] Done: test={test}  ret={result.returncode}  elapsed={elapsed:.0f}s", flush=True)
    return result.returncode, result.stdout


def parse_metrics(stdout):
    metrics = {}
    for line in stdout.split("\n"):
        for key in ["val_minADE6", "val_minFDE6", "val_MR"]:
            if key in line and "│" in line:
                try:
                    parts = line.split("│")
                    val_str = parts[-2].strip()
                    metrics[key] = float(val_str)
                except (ValueError, IndexError):
                    pass
    return metrics


def main():
    print(f"=== 5-Fold CV at {datetime.now()} ===", flush=True)
    print(f"GPU={os.environ.get('CUDA_VISIBLE_DEVICES')} epochs={EPOCHS} batch={BATCH}", flush=True)
    print(flush=True)

    results = {}

    for fold_num, (train, val, test) in enumerate(FOLDS, 1):
        print(f"{'='*50}", flush=True)
        print(f"  Fold {fold_num}/5: test={test}  val={val}", flush=True)
        print(f"{'='*50}", flush=True)

        output_name = f"ethucy_ethucy_ethucy_test-{test}"

        # Train
        ret = run_training(train, val, test, output_name)
        if ret != 0:
            print(f"[Fold {fold_num}] Train FAILED (ret={ret})", flush=True)
            ckpt = find_best_ckpt(output_name)
            if not ckpt:
                continue
            print(f"[Fold {fold_num}] Using partial ckpt: {ckpt}", flush=True)
        else:
            ckpt = find_best_ckpt(output_name)
            if not ckpt:
                print(f"[Fold {fold_num}] No ckpt found", flush=True)
                continue
            print(f"[Fold {fold_num}] Best ckpt: {ckpt}", flush=True)

        # Eval
        ret, stdout = run_eval(test, ckpt, output_name)
        if ret != 0:
            print(f"[Fold {fold_num}] Eval FAILED", flush=True)
            continue

        metrics = parse_metrics(stdout)
        results[test] = metrics
        print(f"[Fold {fold_num}] Done: {metrics}", flush=True)
        print(flush=True)

    # Summary
    print(f"\n{'='*50}", flush=True)
    print(f"Summary at {datetime.now()}", flush=True)
    print(f"{'='*50}", flush=True)
    if not results:
        print("No results!", flush=True)
        return

    print(f"{'Test':>8s}  {'minADE6':>10s}  {'minFDE6':>10s}  {'MR':>8s}", flush=True)
    print("-" * 45, flush=True)
    ade_sum = fde_sum = mr_sum = 0.0
    for test in ["ETH", "HOTEL", "UNIV", "ZARA1", "ZARA2"]:
        if test in results:
            m = results[test]
            ade_sum += m.get("val_minADE6", 0)
            fde_sum += m.get("val_minFDE6", 0)
            mr_sum += m.get("val_MR", 0)
            print(f"{test:>8s}  {m.get('val_minADE6',0):>10.4f}  "
                  f"{m.get('val_minFDE6',0):>10.4f}  {m.get('val_MR',0):>8.4f}", flush=True)
    n = len(results)
    print("-" * 45, flush=True)
    print(f"{'AVG':>8s}  {ade_sum/n:>10.4f}  {fde_sum/n:>10.4f}  {mr_sum/n:>8.4f}", flush=True)

    # Save
    with open("5fold_cv_results.txt", "w") as f:
        f.write(f"5-Fold CV Results — {datetime.now()}\n")
        f.write(f"GPU={os.environ.get('CUDA_VISIBLE_DEVICES')} epochs={EPOCHS} batch={BATCH}\n\n")
        for test in ["ETH", "HOTEL", "UNIV", "ZARA1", "ZARA2"]:
            if test in results:
                m = results[test]
                f.write(f"{test}: ADE6={m.get('val_minADE6',0):.4f}  "
                        f"FDE6={m.get('val_minFDE6',0):.4f}  MR={m.get('val_MR',0):.4f}\n")
        f.write(f"\nAVG: ADE6={ade_sum/n:.4f}  FDE6={fde_sum/n:.4f}  MR={mr_sum/n:.4f}\n")
    print("\nSaved to 5fold_cv_results.txt", flush=True)


if __name__ == "__main__":
    main()