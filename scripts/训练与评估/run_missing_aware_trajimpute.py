"""TrajImpute 缺失数据重训 smoke runner（正式模型链路的最小验证）。

只做单场景、小 batch 的最小运行验证（任务书 §十三 10）：
  1) 用 config_missing_aware_trajimpute 跑几个 train step（limit_train_batches）
  2) 从该 checkpoint（或指定 checkpoint）跑 Easy/Hard test 的分组评估

用法：
  # 冒烟：ETH-M/Easy，M0-current，2 个 batch 训练 + test 评估
  CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. python scripts/训练与评估/run_missing_aware_trajimpute.py \
      --variant M0-current --scene ETH-M --difficulty Easy --gpu 3 \
      --train-batches 2 --eval-max-batches 2

只做单场景、小 batch 验证；不覆盖历史 outputs；正式训练使用
`run_trajimpute_experiments.py`。
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PY = sys.executable
CONFIG_NAME = "config_missing_aware_trajimpute"

VARIANTS = {
    "M0-current": {"use_observation_features": False, "use_missing_summary": False},
    "M0_base": {"use_observation_features": False, "use_missing_summary": False},
    "M1_obs": {"use_observation_features": True, "use_missing_summary": False},
    "M2_history": {"use_observation_features": True, "use_missing_summary": True},
}


def get_git_revision():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="M0-current", choices=list(VARIANTS))
    ap.add_argument("--data-root", default="/home/lbh/TrajImpute/dataset/TrajImpute")
    ap.add_argument("--scene", default="ETH-M")
    ap.add_argument("--difficulty", default="Easy", choices=["Easy", "Hard"])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--train-batches", type=int, default=2,
                    help="limit_train_batches：smoke 只跑几个 step")
    ap.add_argument("--eval-max-batches", type=int, default=2,
                    help="test 评估的最大 batch 数（None=全部）")
    ap.add_argument("--checkpoint", default=None, help="评估用的已有 checkpoint")
    ap.add_argument("--output-root", default="outputs/trajimpute_retrain_smoke")
    ap.add_argument("--K", type=int, default=20, choices=[20])
    args = ap.parse_args()

    if args.checkpoint is not None:
        raise SystemExit("该 runner 只执行从头训练；请勿传入 checkpoint")

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.output_root) / f"{args.variant}_{args.scene}_{args.difficulty}_seed{args.seed}_{ts}"
    if run_dir.exists():
        raise SystemExit(f"refusing to overwrite: {run_dir}")
    run_dir.mkdir(parents=True)
    log = run_dir / "runner.log"

    meta = {
        "runner": "run_missing_aware_trajimpute.py (smoke)",
        "variant": args.variant,
        "switches": VARIANTS[args.variant],
        "data_root": args.data_root,
        "scene": args.scene,
        "difficulty": args.difficulty,
        "seed": args.seed,
        "K": args.K,
        "training_mode": "retrain_missing_data",
        "zero_shot": False,
        "batch_size": args.batch_size,
        "train_batches": args.train_batches,
        "eval_max_batches": args.eval_max_batches,
        "git_revision": get_git_revision(),
        "started": ts,
        "release_source": {
            "type": "official_trajimpute_release",
            "difficulty": args.difficulty,
            "direct_prediction": True,
            "zero_missing_rows_are_diagnostic_only": True,
        },
    }
    with open(run_dir / "experiment_meta.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    def sh(cmd):
        pretty = " ".join(str(c) for c in cmd)
        print("+", pretty, flush=True)
        with open(log, "a") as f:
            f.write(f"\n$ {pretty}\n")
            rc = subprocess.Popen([str(c) for c in cmd], stdout=f,
                                  stderr=subprocess.STDOUT, cwd=REPO).wait()
        return rc

    assert args.train_batches > 0
    overrides = [
        f"scene={args.scene}", f"difficulty={args.difficulty}",
        f"data_root={args.data_root}", f"seed={args.seed}",
        f"batch_size={args.batch_size}", f"num_workers={args.num_workers}",
        f"epochs={args.epochs}", f"limit_train_batches={args.train_batches}",
        f"limit_val_batches={args.train_batches}",
        f"monitor=val_minFDE{args.K}",
        f"model.target.model.num_modes={args.K}",
        f"model_version={'0' if args.variant in ('M0_base', 'M0-current') else ('1' if args.variant=='M1_obs' else '2')}",
        f"model.target.model.use_observation_features={str(VARIANTS[args.variant]['use_observation_features']).lower()}",
        f"model.target.model.use_missing_summary={str(VARIANTS[args.variant]['use_missing_summary']).lower()}",
        f"hydra.run.dir={run_dir / 'train'}",
    ]
    rc = sh([PY, "-u", "train.py", f"--config-name={CONFIG_NAME}"] + overrides)
    if rc != 0:
        raise SystemExit(f"smoke train failed rc={rc}; see {log}")
    ckpts = sorted((run_dir / "train" / "checkpoints").glob("*.ckpt"))
    if not ckpts:
        raise SystemExit(f"no checkpoint produced; see {log}")
    ckpt = ckpts[-1]
    print(f"[ckpt] {ckpt}")

    # 分组评估
    eval_cmd = [
        PY, "-u", str(REPO / "scripts/结果分析/evaluate_trajimpute_direct.py"),
        "--data-root", args.data_root, "--scene", args.scene,
        "--difficulty", args.difficulty, "--split", "test",
        "--variant", args.variant, "--K", str(args.K), "--seed", str(args.seed),
        "--checkpoint", str(ckpt),
        "--output-root", str(run_dir / "eval"),
        "--device", f"cuda:0",
    ]
    if args.eval_max_batches:
        eval_cmd += ["--max-batches", str(args.eval_max_batches)]
    rc = sh(eval_cmd)
    if rc != 0:
        raise SystemExit(f"eval failed rc={rc}; see {log}")

    meta["finished"] = datetime.now().strftime("%Y%m%d-%H%M%S")
    meta["status"] = "complete"
    with open(run_dir / "experiment_meta.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"\n[done] {run_dir}")


if __name__ == "__main__":
    main()
