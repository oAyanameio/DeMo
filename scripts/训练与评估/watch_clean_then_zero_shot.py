"""等 5 个 Clean fold 训练全部完成后，自动对每场景 Easy/Hard test 跑零样本评估。

用法: python -u scripts/训练与评估/watch_clean_then_zero_shot.py
逻辑: 每 5 分钟检查一次各 fold 是否到 epoch 100；全完成后按 val_minFDE20 选点，
      逐场景逐难度调用 evaluate_trajimpute_direct.py，结果写入各 fold 的 eval_zero_shot/。
"""
import csv
import glob
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PY = "/home/lbh/.conda/envs/DeMo/bin/python"
EVAL = REPO / "scripts" / "结果分析" / "evaluate_trajimpute_direct.py"
DATA_ROOT = "/home/lbh/TrajImpute/dataset/TrajImpute"
FOLDS = ["ETH", "HOTEL", "UNIV", "ZARA1", "ZARA2"]
MONITOR = "val_minFDE20"
POLL_SECONDS = 300


def fold_dir(fold: str) -> Path:
    return REPO / "outputs" / "clean_retrain" / f"fold_{fold}"


def current_epoch(fold: str):
    """返回该 fold 所有 version 中的最大 epoch；无数据返回 0。"""
    best_ep = 0
    for f in glob.glob(str(fold_dir(fold) / "logs" / "version_*" / "metrics.csv")):
        with open(f) as handle:
            for row in csv.DictReader(handle):
                if row.get("epoch") and row.get(MONITOR):
                    best_ep = max(best_ep, int(float(row["epoch"])))
    return best_ep


def select_best_ckpt(fold: str):
    """按 monitor 最小选点，返回 (epoch, val, ckpt_path)。禁止用最后 epoch。"""
    best = None
    for f in glob.glob(str(fold_dir(fold) / "logs" / "version_*" / "metrics.csv")):
        with open(f) as handle:
            for row in csv.DictReader(handle):
                value, epoch = row.get(MONITOR), row.get("epoch")
                if value in (None, "") or epoch in (None, ""):
                    continue
                cand = (int(float(epoch)), float(value))
                if best is None or cand[1] < best[1]:
                    best = cand
    if best is None:
        raise RuntimeError(f"{fold}: 无 {MONITOR} 指标")
    epoch, val = best
    ckpt = fold_dir(fold) / "checkpoints" / f"epoch={epoch}.ckpt"
    if not ckpt.exists():
        raise RuntimeError(f"{fold}: best epoch={epoch} 的 ckpt 不存在: {ckpt}")
    return epoch, val, ckpt


def all_done() -> bool:
    return all(current_epoch(f) >= 100 for f in FOLDS)


def run_eval(scene: str, difficulty: str, ckpt: Path, out_root: Path):
    cmd = [
        PY, "-u", str(EVAL),
        "--data-root", DATA_ROOT,
        "--scene", scene,
        "--difficulty", difficulty,
        "--split", "test",
        "--variant", "M0",
        "--K", "20",
        "--seed", "2024",
        "--checkpoint", str(ckpt),
        "--output-root", str(out_root),
        "--no-bimamba",
    ]
    print(f"[eval] {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd).returncode


def main():
    print(f"[watch] 开始等待 5 个 Clean fold 训练完成: {FOLDS}", flush=True)
    while not all_done():
        status = " ".join(f"{f}={current_epoch(f)}" for f in FOLDS)
        print(f"[watch] {time.strftime('%H:%M:%S')} 等待中 | {status}", flush=True)
        time.sleep(POLL_SECONDS)
    print(f"[watch] 全部 fold 训练完成，开始零样本评估", flush=True)

    rc_all = 0
    for fold in FOLDS:
        scene = f"{fold}-M"
        epoch, val, ckpt = select_best_ckpt(fold)
        print(f"[watch] {scene}: best_epoch={epoch} val={val:.4f} ckpt={ckpt}", flush=True)
        out_root = fold_dir(fold) / "eval_zero_shot"
        for difficulty in ("Easy", "Hard"):
            rc = run_eval(scene, difficulty, ckpt, out_root)
            print(f"[watch] {scene} {difficulty} rc={rc}", flush=True)
            rc_all |= rc
    print(f"[watch] 全部完成，总 rc={rc_all}", flush=True)
    return rc_all


if __name__ == "__main__":
    sys.exit(main())
