"""TrajImpute 缺失数据重训入口（Easy-direct / Hard-direct）。

协议（任务书 2026-09-07 + 方案 §1.4）：
- 默认从 `M0-current + Easy-direct` 开始；
- `Easy-direct` / `Hard-direct` 均使用 TrajImpute 官方缺失 train/val/test；
- 每个场景从头训练，不使用零样本 checkpoint；
- 训练输出、验证选点和测试评估统一使用 K=20；
- 每个实验写入独立 manifest，拒绝把不同数据源混为同一协议；
- variant -> 选题模型开关固定映射；
- 失败时保留完整 traceback 到 runner.log。

用法：
  PYTHONNOUSERSITE=1 PYTHONPATH=. python scripts/训练与评估/run_trajimpute_experiments.py \
      --protocol easy-direct --variant M0-current \
      --scenes ETH-M HOTEL-M UNIV-M ZARA1-M ZARA2-M \
      --seed 2024 --gpu 3 --output-root outputs/clean_ethucy
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PY = sys.executable
CONFIG_NAME = "config_missing_aware_trajimpute"
EVAl_SCRIPT = REPO / "scripts/结果分析/evaluate_trajimpute_direct.py"

SCENES = ["ETH-M", "HOTEL-M", "UNIV-M", "ZARA1-M", "ZARA2-M"]
FOLDS = ["ETH", "HOTEL", "UNIV", "ZARA1", "ZARA2"]
DIRECT_NUM_MODES = 20
DEFAULT_PROTOCOL = "easy-direct"
DEFAULT_VARIANT = "M0-current"

# variant -> 模型开关（唯一事实来源）
VARIANTS = {
    "M0-current": {  # = B0：缺失感知全关
        "use_observation_features": False, "use_missing_summary": False,
        "use_motion_features": False,
    },
    "B0": {"use_observation_features": False, "use_missing_summary": False,
           "use_motion_features": False},
    "B1": {"use_observation_features": False, "use_missing_summary": False,
           "use_motion_features": True},
    "M1-evidence": {"use_observation_features": False, "use_missing_summary": False,
                    "use_motion_features": True, "use_evidence_clock": True},
    "M1_obs": {"use_observation_features": True, "use_missing_summary": False,
               "use_motion_features": False},
    "M2_history": {"use_observation_features": True, "use_missing_summary": True,
                   "use_motion_features": False},
}

PROTOCOLS = {
    "easy-direct": {
        "dataset": "trajimpute",
        "difficulty": "Easy",
        "zero_missing_only": False,
        "suffix": "",
    },
    "hard-direct": {
        "dataset": "trajimpute",
        "difficulty": "Hard",
        "zero_missing_only": False,
        "suffix": "",
    },
}


def get_git_revision():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       cwd=REPO, text=True).strip()
    except Exception:
        return "unknown"


def env_info():
    import torch
    return {
        "python": sys.executable,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def select_best_checkpoint(train_dir, monitor):
    """按验证集 monitor 选择 checkpoint，禁止用最后 epoch 代替选点。"""
    train_dir = Path(train_dir)
    metrics_files = sorted((train_dir / "logs").glob("version_*/metrics.csv"))
    if not metrics_files:
        raise RuntimeError(f"未找到 metrics.csv: {train_dir / 'logs'}")

    best = None
    for metrics_file in metrics_files:
        with metrics_file.open() as handle:
            for row in csv.DictReader(handle):
                value = row.get(monitor)
                epoch = row.get("epoch")
                if value in (None, "") or epoch in (None, ""):
                    continue
                candidate = (int(float(epoch)), float(value))
                if best is None or candidate[1] < best[1]:
                    best = candidate
    if best is None:
        raise RuntimeError(f"metrics.csv 没有指标 {monitor}: {train_dir}")

    best_epoch, best_value = best
    exact = train_dir / "checkpoints" / f"epoch={best_epoch}.ckpt"
    candidates = [exact] if exact.exists() else sorted(
        (train_dir / "checkpoints").glob(f"epoch={best_epoch}*.ckpt")
    )
    if not candidates:
        raise RuntimeError(
            f"最佳 epoch={best_epoch} 的 checkpoint 不存在: "
            f"{train_dir / 'checkpoints'}"
        )
    return best_epoch, best_value, str(candidates[0])


def build_manifest(args, protocol_cfg):
    data_root = Path(args.data_root)
    return {
        "protocol": args.protocol,
        "dataset": "TrajImpute",
        "data_source": {
            "type": "official_release",
            "root": str(data_root),
            "difficulty": protocol_cfg["difficulty"],
            "direct_prediction": True,
            "is_trajimpute_clean_split": False,
        },
        "data_root": str(data_root),
        "scenes": args.scenes,
        "files": {s: [f"{s}/{protocol_cfg['difficulty']}/data_{sp}.pkl"
                      for sp in ("train", "val", "test")] for s in args.scenes},
        "git_revision": get_git_revision(),
        "python": sys.executable,
        "env": env_info(),
        "model_version": args.variant,
        "backbone": {"bimamba": args.bimamba},
        "seed": args.seed,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "monitor": f"val_minFDE{args.K}",
        "K": args.K,
        "training_mode": "retrain_missing_data",
        "zero_missing_only": False,
        "gpu": args.gpu,
        "started": datetime.now().isoformat(timespec="seconds"),
        "experiment_class": "smoke" if args.smoke else (
            "screening" if args.screening else "confirmatory"),
        "output_root": str(args.output_root),
        "command": " ".join(sys.argv),
    }



def sh(cmd, log_path, env=None):
    pretty = " ".join(str(c) for c in cmd)
    print("+", pretty, flush=True)
    with open(log_path, "a") as f:
        f.write(f"\n$ {pretty}\n")
        p = subprocess.Popen([str(c) for c in cmd], stdout=f,
                             stderr=subprocess.STDOUT, cwd=REPO, env=env)
        rc = p.wait()
    return rc


def run_one_scene(args, scene, protocol_cfg, manifest, gpu_env):
    if protocol_cfg["dataset"] != "trajimpute":
        raise ValueError("run_one_scene 只接受 TrajImpute protocol")
    sw = VARIANTS[args.variant]
    tag = f"{args.variant}_{scene}_{args.protocol}_seed{args.seed}"
    if args.bimamba is False:
        tag += "_uni"
    run_dir = Path(args.output_root) / tag
    if run_dir.exists():
        # 断点续训放行：已有 last.ckpt 的半成品目录允许继续（2026-09-13）
        if (run_dir / "train" / "checkpoints" / "last.ckpt").exists():
            print(f"[resume-ok] 半成品目录续训: {run_dir}", flush=True)
        else:
            raise RuntimeError(
                f"正式重训输出目录已存在，拒绝跳过或复用: {run_dir}；"
                "请更换 --output-root 或显式删除该实验目录"
            )
    run_dir.mkdir(parents=True, exist_ok=True)
    log = run_dir / "runner.log"

    _manifest = run_dir / "manifest.json"
    if not _manifest.exists():
        m = dict(manifest)
        m["scenes"] = [scene]
        m["files"] = {scene: manifest["files"][scene]}
        with open(_manifest, "w") as f:
            json.dump(m, f, indent=2, ensure_ascii=False)

    # 训练
    train_overrides = [
        f"scene={scene}",
        f"difficulty={protocol_cfg['difficulty']}",
        f"zero_missing_only={str(protocol_cfg['zero_missing_only']).lower()}",
        f"data_root={args.data_root}",
        f"seed={args.seed}",
        f"batch_size={args.batch_size}",
        f"num_workers={args.num_workers}",
        f"epochs={args.epochs}",
        f"monitor=val_minFDE{args.K}",
        f"model.target.model.num_modes={args.K}",
        f"lr={args.lr}",
        f"weight_decay={args.weight_decay}",
        f"bimamba={str(args.bimamba).lower()}",
        f"model_version={args.model_version_num}",
        f"clean_suffix={protocol_cfg['suffix']}",
        f"model.target.model.use_observation_features={str(sw['use_observation_features']).lower()}",
        f"model.target.model.use_missing_summary={str(sw['use_missing_summary']).lower()}",
        f"model.target.model.use_motion_features={str(sw.get('use_motion_features', False)).lower()}",
        f"model.target.model.use_evidence_clock={str(sw.get('use_evidence_clock', False)).lower()}",
    ]
    if args.smoke:
        train_overrides += [f"limit_train_batches={args.limit_batches}",
                            f"limit_val_batches={args.limit_batches}"]
    train_cmd = [PY, "-u", "train.py", f"--config-name={CONFIG_NAME}"] + train_overrides + [
        f"hydra.run.dir={run_dir / 'train'}"]

    # 2026-09-13：断点续训 + 被杀自动重拉（外源 SIGTERM 防御）。
    # train.py 原生支持 ckpt_path=last.ckpt；训练失败时只要 last.ckpt 存在就续训，
    # 最多 args.max_train_retries 次；目录守卫对"已有 last.ckpt 的半成品"放行续训。
    last_ckpt = run_dir / "train" / "checkpoints" / "last.ckpt"
    rc = 1
    for attempt in range(1, getattr(args, "max_train_retries", 6) + 1):
        cmd = list(train_cmd)
        if last_ckpt.exists():
            cmd.append(f"checkpoint={last_ckpt}")
            print(f"[resume] attempt={attempt} from {last_ckpt}", flush=True)
        rc = sh(cmd, log, env=gpu_env)
        if rc == 0:
            break
        print(f"[train-failed] attempt={attempt} rc={rc}", flush=True)
        if not last_ckpt.exists():
            break  # 无断点可续，放弃
    if rc != 0:
        return {"scene": scene, "status": "train_failed", "dir": str(run_dir), "rc": rc}

    # checkpoint：严格按 val_minFDE20 选点，不使用最后 epoch。
    try:
        best_epoch, best_val, ckpt = select_best_checkpoint(
            run_dir / "train", f"val_minFDE{args.K}"
        )
    except RuntimeError as error:
        return {
            "scene": scene,
            "status": "checkpoint_selection_failed",
            "dir": str(run_dir),
            "error": str(error),
        }

    # 评估（独立进程，direct evaluator）
    eval_cmd = [
        PY, "-u", str(EVAl_SCRIPT),
        "--data-root", args.data_root,
        "--scene", scene,
        "--difficulty", protocol_cfg["difficulty"],
        "--split", "test",
        "--variant", args.variant,
        "--K", str(args.K),
        "--seed", str(args.seed),
        "--checkpoint", str(ckpt),
        "--output-root", str(run_dir / "eval"),
        "--bimamba" if args.bimamba else "--no-bimamba",
    ]
    if protocol_cfg["zero_missing_only"]:
        eval_cmd += ["--zero-missing-only"]
    if args.smoke:
        eval_cmd += ["--max-batches", str(args.limit_batches)]
    rc = sh(eval_cmd, log, env=gpu_env)
    status = "complete" if rc == 0 else "eval_failed"
    return {
        "scene": scene,
        "status": status,
        "dir": str(run_dir),
        "ckpt": str(ckpt),
        "best_epoch": best_epoch,
        "best_val": best_val,
        "monitor": f"val_minFDE{args.K}",
        "K": args.K,
        "training_mode": "retrain_missing_data",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default=DEFAULT_PROTOCOL, choices=list(PROTOCOLS))
    ap.add_argument("--variant", default=DEFAULT_VARIANT, choices=list(VARIANTS))
    ap.add_argument("--scenes", nargs="+", default=SCENES)
    ap.add_argument("--folds", nargs="+", default=FOLDS, choices=FOLDS)
    ap.add_argument("--data-root", default="/home/lbh/TrajImpute/dataset/TrajImpute")
    ap.add_argument("--precision", default="bf16")
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    epochs_default = 100
    ap.add_argument("--epochs", type=int, default=epochs_default)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--K", type=int, default=DIRECT_NUM_MODES, choices=[DIRECT_NUM_MODES])
    ap.add_argument("--bimamba", action=argparse.BooleanOptionalAction, default=False,
                    help="主链固定单向(2026-09-12裁定)；旗标仅作用于encoder")
    ap.add_argument("--output-root", default="outputs/trajimpute_retrain")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-train-retries", type=int, default=6,
                    help="训练被杀后自动断点续训的最大重试次数")
    ap.add_argument("--screening", action="store_true",
                    help="标记为筛选实验（正式非确认性）")
    ap.add_argument("--limit-batches", type=int, default=2, help="smoke 用")
    args = ap.parse_args()

    if args.K != DIRECT_NUM_MODES:
        raise SystemExit("TrajImpute 正式重训协议固定 K=20")

    # model_version 标签（仅 output 命名）：M0-current/B0->0, B1->1
    args.model_version_num = {"M0-current": "0", "B0": "0", "B1": "1",
                              "M1-evidence": "1",
                              "M1_obs": "1", "M2_history": "2"}[args.variant]

    protocol_cfg = PROTOCOLS[args.protocol]
    if protocol_cfg["dataset"] == "trajimpute" and protocol_cfg["zero_missing_only"]:
        raise SystemExit("正式 TrajImpute 重训禁止 zero_missing_only")
    args.data_root = str(Path(args.data_root).expanduser().resolve())

    gpu_env = dict(os.environ)
    gpu_env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    gpu_env["PYTHONNOUSERSITE"] = "1"
    gpu_env["PYTHONPATH"] = "."

    manifest = build_manifest(args, protocol_cfg)
    results = []
    for scene in args.scenes:
        try:
            r = run_one_scene(args, scene, protocol_cfg, manifest, gpu_env)
        except Exception:
            err = traceback.format_exc()
            print(err, file=sys.stderr)
            r = {"scene": scene, "status": "exception", "traceback": err}
        results.append(r)
        print(f"[{scene}] {r['status']}", flush=True)

    with open(Path(args.output_root) / f"runs_{args.variant}_{args.protocol}_seed{args.seed}.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
