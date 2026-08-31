#!/usr/bin/env python3
"""5-fold CV — PARALLEL version. Each fold on a different GPU."""

import os
import csv
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 80
BATCH = int(sys.argv[2]) if len(sys.argv) > 2 else 128

# 5 folds: (train, val, test)
FOLDS = [
    ("HOTEL,UNIV,ZARA2", "ZARA1", "ETH"),
    ("ETH,UNIV,ZARA2", "ZARA1", "HOTEL"),
    ("ETH,HOTEL,ZARA2", "ZARA1", "UNIV"),
    ("ETH,HOTEL,UNIV", "ZARA2", "ZARA1"),
    ("ETH,HOTEL,UNIV", "ZARA1", "ZARA2"),
]

# Available GPUs (use 0,1,2,3 — skip any that are too busy)
GPUS = [0, 1, 2, 3]

OUTPUT_BASE = Path("outputs")


def find_best_ckpt(output_name):
    """Find the best checkpoint by val_minFDE6 from metrics.csv.

    Falls back to largest epoch number if metrics.csv is unavailable.
    Returns: (ckpt_path, best_epoch) or (None, None).
    """
    dirs = sorted(OUTPUT_BASE.glob(f"{output_name}/*/"), reverse=True)
    for d in dirs:
        ckpt_dir = d / "checkpoints"
        if not ckpt_dir.exists():
            continue

        # Try to find best epoch from metrics.csv
        metrics_csv = d / "logs" / "version_0" / "metrics.csv"
        if metrics_csv.exists():
            try:
                best_epoch = None
                best_val = float('inf')
                with open(metrics_csv) as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        try:
                            epoch = int(row['epoch'])
                            val = float(row.get('val_minFDE6', 'inf'))
                            if val < best_val:
                                best_val = val
                                best_epoch = epoch
                        except (ValueError, KeyError):
                            pass
                if best_epoch is not None:
                    ckpt = ckpt_dir / f"epoch={best_epoch}.ckpt"
                    if ckpt.exists():
                        return str(ckpt), best_epoch
            except Exception:
                pass

        # Fallback: largest epoch number
        ckpts = sorted(ckpt_dir.glob("epoch=*.ckpt"))
        ckpts = [c for c in ckpts if "last.ckpt" not in str(c)]
        if ckpts:
            return str(ckpts[-1]), None

        last = ckpt_dir / "last.ckpt"
        if last.exists():
            return str(last), None
    return None, None


def run_fold(fold_num, train, val, test, gpu):
    """Run one fold (train + eval) on a specific GPU."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    output_name = f"ethucy_ethucy_ethucy_test-{test}"

    # --- Train ---
    train_cmd = [
        sys.executable, "-u", "train.py",
        "--config-name=config_ethucy",
        "gpus=1",
        f"epochs={EPOCHS}",
        f"batch_size={BATCH}",
        f"datamodule.target.train_scenes=[{train}]",
        f"datamodule.target.val_scenes=[{val}]",
        f"datamodule.target.test_scenes=[{test}]",
        f"output={output_name}",
    ]
    t0 = time.time()
    print(f"[Fold{fold_num}] GPU={gpu} test={test} TRAIN starting...", flush=True)
    ret = subprocess.run(train_cmd, env=env).returncode
    elapsed = time.time() - t0
    print(f"[Fold{fold_num}] GPU={gpu} test={test} TRAIN ret={ret} elapsed={elapsed:.0f}s", flush=True)

    if ret != 0:
        ckpt, best_epoch = find_best_ckpt(output_name)
        if not ckpt:
            return test, None, f"Train FAILED (ret={ret})"
        print(f"[Fold{fold_num}] GPU={gpu} Using partial ckpt: {ckpt} (best_epoch={best_epoch})", flush=True)
    else:
        ckpt, best_epoch = find_best_ckpt(output_name)
        if not ckpt:
            return test, None, "No checkpoint found"
        print(f"[Fold{fold_num}] GPU={gpu} Best ckpt: {ckpt} (best_epoch={best_epoch})", flush=True)

    # --- Eval ---
    # Use a symlink without '=' to avoid Hydra parsing issues
    ckpt_path = Path(ckpt)
    safe_ckpt = ckpt_path.parent / "best_for_eval.ckpt"
    if safe_ckpt.exists():
        safe_ckpt.unlink()
    safe_ckpt.symlink_to(ckpt_path.name)

    eval_cmd = [
        sys.executable, "-u", "eval.py",
        "--config-name=config_ethucy",
        "gpus=1",
        "test=false",
        f"checkpoint={safe_ckpt}",
        f"datamodule.target.val_scenes=[{test}]",
        f"output={output_name}",
    ]
    t0 = time.time()
    print(f"[Fold{fold_num}] GPU={gpu} test={test} EVAL starting...", flush=True)
    result = subprocess.run(eval_cmd, capture_output=True, text=True, env=env)
    elapsed = time.time() - t0
    print(f"[Fold{fold_num}] GPU={gpu} test={test} EVAL ret={result.returncode} elapsed={elapsed:.0f}s", flush=True)

    # Clean up symlink
    if safe_ckpt.exists():
        safe_ckpt.unlink()

    if result.returncode != 0:
        print(f"[Fold{fold_num}] Eval stderr: {result.stderr[-500:]}", flush=True)
        return test, None, f"Eval FAILED (ret={result.returncode})"

    # Parse metrics
    metrics = {}
    for line in result.stdout.split("\n"):
        for key in ["val_minADE6", "val_minFDE6", "val_MR"]:
            if key in line and "│" in line:
                try:
                    parts = line.split("│")
                    val_str = parts[-2].strip()
                    metrics[key] = float(val_str)
                except (ValueError, IndexError):
                    pass

    return test, metrics, None


def main():
    print(f"=== 5-Fold CV PARALLEL at {datetime.now()} ===", flush=True)
    print(f"GPUs={GPUS} epochs={EPOCHS} batch={BATCH}", flush=True)
    print(flush=True)

    results = {}
    errors = []

    with ThreadPoolExecutor(max_workers=len(GPUS)) as executor:
        futures = {}
        for i, (train, val, test) in enumerate(FOLDS):
            gpu = GPUS[i % len(GPUS)]
            fut = executor.submit(run_fold, i + 1, train, val, test, gpu)
            futures[fut] = (i + 1, test, gpu)

        for fut in as_completed(futures):
            fold_num, test, gpu = futures[fut]
            test_name, metrics, error = fut.result()
            if error:
                errors.append(f"Fold{fold_num} {test_name}: {error}")
                print(f"[Fold{fold_num}] GPU={gpu} test={test_name} ERROR: {error}", flush=True)
            else:
                results[test_name] = metrics
                print(f"[Fold{fold_num}] GPU={gpu} test={test_name} DONE: {metrics}", flush=True)

    # --- Summary ---
    print(f"\n{'='*50}", flush=True)
    print(f"Summary at {datetime.now()}", flush=True)
    print(f"{'='*50}", flush=True)

    for err in errors:
        print(f"  ERROR: {err}", flush=True)

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

    with open("5fold_cv_results.txt", "w") as f:
        f.write(f"5-Fold CV Results — {datetime.now()}\n")
        f.write(f"GPUs={GPUS} epochs={EPOCHS} batch={BATCH}\n\n")
        for test in ["ETH", "HOTEL", "UNIV", "ZARA1", "ZARA2"]:
            if test in results:
                m = results[test]
                f.write(f"{test}: ADE6={m.get('val_minADE6',0):.4f}  "
                        f"FDE6={m.get('val_minFDE6',0):.4f}  MR={m.get('val_MR',0):.4f}\n")
        f.write(f"\nAVG: ADE6={ade_sum/n:.4f}  FDE6={fde_sum/n:.4f}  MR={mr_sum/n:.4f}\n")
    print("\nSaved to 5fold_cv_results.txt", flush=True)


if __name__ == "__main__":
    main()
