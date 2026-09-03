"""ETH/UCY benchmark 严格五折 runner。

每折：train/val 来自 fold_<F>/train|val，checkpoint 由 val_minFDE6 选择，
随后只用该折 test split 评估一次。使用 subprocess.Popen + wait()（可靠等待）。
"""
import argparse
import csv
import glob
import json
import os
import subprocess
import sys
import time

FOLDS = ["ETH", "HOTEL", "UNIV", "ZARA1", "ZARA2"]


def sh(cmd, log_path):
    print("+", cmd, flush=True)
    with open(log_path, "a") as f:
        f.write(f"\n$ {cmd}\n")
        p = subprocess.Popen(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT)
        rc = p.wait()
    return rc


PY = "/home/lbh/.conda/envs/DeMo/bin/python"


def latest_run_dir(base):
    runs = sorted(glob.glob(os.path.join(base, "*")))
    assert runs, f"no run dirs under {base}"
    return runs[-1]


def read_best_val(metrics_csv):
    """从 CSVLogger 的 metrics.csv 找最小 val_minFDE6 的 epoch。"""
    import csv as _csv
    best = None
    with open(metrics_csv) as f:
        for row in _csv.DictReader(f):
            v = row.get("val_minFDE6")
            e = row.get("epoch")
            if v is None or e is None or v == "":
                continue
            v, e = float(v), int(float(e))
            if best is None or v < best[1]:
                best = (e, v)
    return best  # (epoch, val_minFDE6)


def run_fold(fold, args):
    out_dir = os.path.abspath(os.path.join(args.output_root, f"fold_{fold}"))
    os.makedirs(out_dir, exist_ok=True)
    train_log = os.path.join(out_dir, "train.log")
    eval_log = os.path.join(out_dir, "eval.log")

    if not args.skip_train:
        t0 = time.time()
        rc = sh(
            f"cd {args.repo} && "
            f"CUDA_VISIBLE_DEVICES={args.gpu} PYTHONPATH=. {PY} train.py "
            f"--config-name={args.config_name} fold={fold} "
            f"data_root={args.data_root} "
            f"epochs={args.epochs} batch_size={args.batch_size} "
            f"seed={args.seed} precision={args.precision} "
            f"num_workers={args.num_workers} "
            f"output=runs hydra.run.dir={out_dir}/train > /dev/null 2>&1",
            train_log,
        )
        train_seconds = time.time() - t0
        if rc != 0:
            return {"fold": fold, "status": f"train_failed_rc{rc}"}
    else:
        train_seconds = 0.0

    run_dir = os.path.join(out_dir, "train")
    # 合并所有 version_*/metrics.csv（Lightning 续训会开新 version），best 取全集最小 val
    all_metrics = sorted(glob.glob(os.path.join(run_dir, "logs", "version_*", "metrics.csv")))
    assert all_metrics, f"metrics.csv not found under {run_dir}"
    best = None
    for metrics_csv in all_metrics:
        b = read_best_val(metrics_csv)
        if b is not None and (best is None or b[1] < best[1]):
            best = b
    assert best is not None, "no val_minFDE6 recorded"
    best_epoch, best_val = best
    ckpt = os.path.join(run_dir, "checkpoints", f"epoch={best_epoch}.ckpt")
    if not os.path.exists(ckpt):
        # save_top_k=1 -> best checkpoint kept as top file；续训后旧 best 已被替换，
        # 用 mtime 最新的非 last checkpoint 兜底
        cands = [c for c in glob.glob(os.path.join(run_dir, "checkpoints", "*.ckpt"))
                 if "last" not in c]
        assert cands, f"no epoch checkpoint for best epoch {best_epoch}"
        ckpt = sorted(cands, key=os.path.getmtime)[-1]
        print(f"WARN: epoch={best_epoch}.ckpt missing, using {ckpt}")

    t0 = time.time()
    rc = sh(
        f"cd {args.repo} && "
        f"CUDA_VISIBLE_DEVICES={args.gpu} PYTHONPATH=. {PY} eval.py "
        f"--config-name={args.config_name} fold={fold} test=true "
        f"data_root={args.data_root} "
        f"precision={args.precision} checkpoint=\"'{ckpt}'\" "
        f"hydra.run.dir={out_dir}/eval",
        eval_log,
    )
    eval_seconds = time.time() - t0
    if rc != 0:
        return {"fold": fold, "status": f"eval_failed_rc{rc}"}

    # 解析 eval 日志里的 test 指标（on_test_end 打印 TEST METRICS）
    import re
    text = open(eval_log).read()
    m = re.search(r"TEST METRICS: (\{.*\})", text)
    assert m, "TEST METRICS not found in eval log"
    # 值形如 tensor(0.0109, device='cuda:0') 或裸浮点，逐 key 提取数字（json.loads 解析不了 tensor(...)）
    metrics = {
        km.group(1): float(km.group(2))
        for km in re.finditer(r"'(test_[\w\-]+)':\s*(?:tensor\()?\s*([-+0-9.eE]+)", m.group(1))
    }
    missing = [k for k in ("test_minADE1", "test_minFDE1", "test_minADE6", "test_minFDE6",
                           "test_MR", "test_b-minFDE6") if k not in metrics]
    assert not missing, f"metrics not parsed: {missing}"
    row = {
        "fold": fold,
        "seed": args.seed,
        "checkpoint_epoch": best_epoch,
        "val_minFDE6": best_val,
        "train_seconds": round(train_seconds, 1),
        "eval_seconds": round(eval_seconds, 1),
        "status": "ok",
    }
    for k in ("test_minADE1", "test_minFDE1", "test_minADE6", "test_minFDE6",
              "test_MR", "test_b-minFDE6"):
        row[k.replace("test_", "")] = round(float(metrics[k]), 4)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/home/lbh/DeMo")
    ap.add_argument("--config-name", default="config_ethucy_benchmark")
    ap.add_argument("--data-root", default="data/ETHUCY_benchmark_v1")
    ap.add_argument("--output-root", default="outputs/ethucy_benchmark_v1")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--gpu", type=int, default=2)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--precision", default="bf16")
    ap.add_argument("--num-workers", type=int, default=16)
    ap.add_argument("--folds", nargs="+", default=FOLDS)
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.output_root, exist_ok=True)
    rows = []
    for fold in args.folds:
        print(f"===== fold {fold} =====", flush=True)
        row = run_fold(fold, args)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        rows.append(row)
        with open(os.path.join(args.output_root, "results_partial.json"), "w") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)

    # 汇总
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    summary = {"rows": rows}
    if ok_rows:
        for k in ("minADE1", "minFDE1", "minADE6", "minFDE6", "MR"):
            vals = [r[k] for r in ok_rows]
            mean = sum(vals) / len(vals)
            std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
            summary[f"mean_{k}"] = round(mean, 4)
            summary[f"std_{k}"] = round(std, 4)
    with open(os.path.join(args.output_root, "results.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    fields = ["fold", "seed", "checkpoint_epoch", "val_minFDE6", "minADE1", "minFDE1",
              "minADE6", "minFDE6", "MR", "b-minFDE6", "train_seconds", "eval_seconds", "status"]
    with open(os.path.join(args.output_root, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print("DONE. folds ok:", len(ok_rows), "/", len(rows))


if __name__ == "__main__":
    main()
