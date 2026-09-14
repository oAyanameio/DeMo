#!/usr/bin/env python3
"""M2-social Clean 臂：原始 ETH/UCY LOO 五折（与 8/28 M0-uni Clean 0.232/0.389 同协议）。

协议：train=4 场景聚合，val/test=留出场景；8帧→12帧，K=20，米制。
选点：val_minFDE20 最优 epoch（config_moflow_ethucy monitor 口径）。
评测：eval.py test=true → test_minADE20 / test_minFDE20（focal 分支）。

唯一改动：model.target.model.use_social_evidence=true（社会证据补偿分支，
零初始化残差，初始严格等价 M0-uni）。bimamba 默认 false = 单向，与 M0-uni 一致。
"""
import csv
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path("/home/lbh/DeMo")
PY = os.environ.get("DEPY", f"{Path.home()}/.conda/envs/DeMo/bin/python")
FOLDS = ["eth", "hotel", "univ", "zara1", "zara2"]
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 100
SOCIAL = True

MODEL_OV = ["model.target.model.use_social_evidence=true"] if SOCIAL else []


def find_best_ckpt(out_name):
    """按 val_minFDE20 找最优 epoch 的 checkpoint（focal 分支，与 config monitor 一致）。"""
    base = ROOT / "outputs" / out_name
    if not base.exists():
        return None, None
    dirs = sorted(base.glob("*/"), key=lambda p: p.stat().st_mtime, reverse=True)
    for d in dirs:
        metrics_csv = d / "logs" / "version_0" / "metrics.csv"
        ckpt_dir = d / "checkpoints"
        if not metrics_csv.exists() or not ckpt_dir.exists():
            continue
        best_epoch, best_val = None, float("inf")
        with open(metrics_csv) as f:
            for row in csv.DictReader(f):
                try:
                    v = float(row.get("val_minFDE20") or "nan")
                    e = int(row["epoch"])
                    if v < best_val:
                        best_val, best_epoch = v, e
                except (ValueError, KeyError, TypeError):
                    continue
        if best_epoch is not None:
            ckpt = ckpt_dir / f"epoch={best_epoch}.ckpt"
            if ckpt.exists():
                return ckpt, best_epoch, best_val
    return None, None, None


def main():
    print(f"=== M2 Clean LOO at {datetime.now()} ===", flush=True)
    print(f"GPU={os.environ.get('CUDA_VISIBLE_DEVICES')} epochs={EPOCHS} social={SOCIAL}", flush=True)

    results = {}
    for fold in FOLDS:
        out_name = f"moflow_social_{fold}"
        print(f"\n[Fold] test={fold} TRAIN ...", flush=True)
        cmd = [PY, "-u", "train.py",
               "--config-name=config_moflow_ethucy",
               "gpus=1",
               f"epochs={EPOCHS}",
               f"datamodule.target.subset={fold}",
               f"output={out_name}",
               ] + MODEL_OV
        ret = subprocess.run(cmd, cwd=ROOT).returncode
        print(f"[Fold] TRAIN ret={ret}", flush=True)
        if ret != 0:
            print(f"[Fold] {fold} 训练失败，跳过", flush=True)
            continue

        ckpt, ep, val = find_best_ckpt(out_name)
        if not ckpt:
            print(f"[Fold] {fold} 无 checkpoint", flush=True)
            continue
        print(f"[Fold] {fold} best ckpt epoch={ep} val_minFDE20={val:.4f}", flush=True)

        safe = ckpt.parent / "best_for_eval.ckpt"
        safe.unlink(missing_ok=True)
        safe.symlink_to(ckpt.name)

        eval_cmd = [PY, "-u", "eval.py",
                    "--config-name=config_moflow_ethucy",
                    "gpus=1", "test=true",
                    f"datamodule.target.subset={fold}",
                    f"checkpoint={safe}",
                    ] + MODEL_OV
        r = subprocess.run(eval_cmd, cwd=ROOT, capture_output=True, text=True)
        safe.unlink(missing_ok=True)
        ade = re.search(r"test_minADE20[^0-9]*([0-9.]+)", r.stdout)
        fde = re.search(r"test_minFDE20[^0-9]*([0-9.]+)", r.stdout)
        if ade and fde:
            results[fold] = (float(ade.group(1)), float(fde.group(1)))
            print(f"[Fold] {fold}: minADE20={ade.group(1)} minFDE20={fde.group(1)}", flush=True)
        else:
            print(f"[Fold] {fold}: 未解析到 test 指标\n{r.stdout[-800:]}", flush=True)

        # 逐折落盘
        with open(ROOT / "outputs" / "m2_clean_results.txt", "w") as f:
            f.write(f"M2-social Clean LOO — {datetime.now()}\nepochs={EPOCHS}\n\n")
            for s in FOLDS:
                if s in results:
                    a, d = results[s]
                    f.write(f"{s}: minADE20={a:.4f} minFDE20={d:.4f}\n")
            if results:
                na = sum(v[0] for v in results.values()) / len(results)
                nd = sum(v[1] for v in results.values()) / len(results)
                f.write(f"\nAVG: minADE20={na:.4f} minFDE20={nd:.4f}\n")

    if results:
        na = sum(v[0] for v in results.values()) / len(results)
        nd = sum(v[1] for v in results.values()) / len(results)
        print(f"\n===== M2 Clean 五折平均: minADE20={na:.4f} minFDE20={nd:.4f} =====", flush=True)
        print("(对照 M0-uni Clean 0.232/0.389)", flush=True)


if __name__ == "__main__":
    main()
