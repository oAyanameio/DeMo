"""Missing-Aware SDD 正式实验 Runner（M0/M1/M2，训练适应协议）。

与 ETH/UCY runner 同协议，差异：
- condition/data_root 必须以 datamodule.target.condition / datamodule.target.data_root
  传递（SDD DataModule 的实际读取键），不能只改输出目录或顶层 condition；
- monitor=val_minFDE20，指标 test_minADE20/test_minFDE20 等；
- 固定 90/10 train/val + held-out test（无 fold 维度）。

用法示例：
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/训练与评估/run_missing_aware_sdd.py \
      --variant M2_history --condition random_fixed4_ng \
      --data-root data/SDD_missing_v3_noguard \
      --output-root outputs/missing_aware/sdd/train_adapt \
      --seed 2024 --gpu 0
"""
import argparse
import csv
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PY = sys.executable
CONFIG_NAME = "config_missing_aware_sdd"
MONITOR = "val_minFDE20"
TEST_KEYS = ["test_minADE1", "test_minFDE1", "test_minADE20", "test_minFDE20",
             "test_MR", "test_b-minFDE20"]

VARIANTS = {
    "M0_base": {"use_observation_features": False, "use_missing_summary": False},
    "M1_obs": {"use_observation_features": True, "use_missing_summary": False},
    "M2_history": {"use_observation_features": True, "use_missing_summary": True},
}
EXPECTED = {
    "M0_base": (4, False),
    "M1_obs": (8, False),
    "M2_history": (8, True),
}


# ---------------------------------------------------------------- 命令构造（纯函数）
def build_train_overrides(variant, condition, data_root, seed, epochs,
                          batch_size, num_workers, precision):
    return [
        f"seed={seed}",
        f"epochs={epochs}",
        f"batch_size={batch_size}",
        f"num_workers={num_workers}",
        f"precision={precision}",
        f"datamodule.target.condition={condition}",
        f"datamodule.target.data_root={data_root}",
        f"model.target.model.use_observation_features={str(VARIANTS[variant]['use_observation_features']).lower()}",
        f"model.target.model.use_missing_summary={str(VARIANTS[variant]['use_missing_summary']).lower()}",
        "model.target.model.use_gap_condition=false",
    ]


def build_train_command(variant, condition, data_root, seed, epochs, batch_size,
                        num_workers, precision, run_dir, resume_ckpt=None):
    cmd = [PY, "-u", "train.py", f"--config-name={CONFIG_NAME}"]
    cmd += build_train_overrides(variant, condition, data_root, seed, epochs,
                                 batch_size, num_workers, precision)
    if resume_ckpt is not None:
        cmd.append(f"checkpoint={resume_ckpt}")
    cmd.append(f"hydra.run.dir={run_dir}")
    return cmd


def build_eval_command(condition, data_root, seed, precision, ckpt_link, eval_dir):
    return [
        PY, "-u", "eval.py", f"--config-name={CONFIG_NAME}",
        f"seed={seed}", f"precision={precision}", "test=true",
        f"datamodule.target.condition={condition}",
        f"datamodule.target.data_root={data_root}",
        f"checkpoint={ckpt_link}",
        f"hydra.run.dir={eval_dir}",
    ]


# ---------------------------------------------------------------- 子进程与解析（与 ETH/UCY 同构）
def sh(cmd, log_path):
    pretty = " ".join(str(c) for c in cmd)
    print("+", pretty, flush=True)
    with open(log_path, "a") as f:
        f.write(f"\n$ {pretty}\n")
        p = subprocess.Popen([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT,
                             cwd=REPO)
        rc = p.wait()
    return rc


def read_best_val(metrics_csv, monitor=MONITOR):
    best = None
    with open(metrics_csv) as f:
        for row in csv.DictReader(f):
            v, e = row.get(monitor), row.get("epoch")
            if v is None or e is None or v == "":
                continue
            v, e = float(v), int(float(e))
            if best is None or v < best[1]:
                best = (e, v)
    return best


def collect_best_checkpoint(run_dir, monitor=MONITOR):
    all_metrics = sorted(glob.glob(os.path.join(run_dir, "logs", "version_*", "metrics.csv")))
    assert all_metrics, f"metrics.csv not found under {run_dir}"
    best = None
    for metrics_csv in all_metrics:
        b = read_best_val(metrics_csv, monitor)
        if b is not None and (best is None or b[1] < best[1]):
            best = b
    assert best is not None, f"no {monitor} recorded under {run_dir}"
    best_epoch, best_val = best
    ckpt = os.path.join(run_dir, "checkpoints", f"epoch={best_epoch}.ckpt")
    if not os.path.exists(ckpt):
        cands = [c for c in glob.glob(os.path.join(run_dir, "checkpoints", "*.ckpt"))
                 if "last" not in c]
        assert cands, f"no epoch checkpoint for best epoch {best_epoch} under {run_dir}"
        ckpt = sorted(cands, key=os.path.getmtime)[-1]
        print(f"WARN: epoch={best_epoch}.ckpt missing, using {ckpt}")
    return best_epoch, best_val, ckpt


def parse_test_metrics(eval_log, keys=TEST_KEYS):
    text = open(eval_log).read()
    m = re.search(r"TEST METRICS: (\{.*\})", text)
    assert m, "TEST METRICS not found in eval log"
    metrics = {
        km.group(1): float(km.group(2))
        for km in re.finditer(r"'(test_[\w\-]+)':\s*(?:tensor\()?(\s*[-+0-9.eE]+)", m.group(1))
    }
    missing = [k for k in keys if k not in metrics]
    assert not missing, f"metrics not parsed: {missing}"
    return metrics


# ---------------------------------------------------------------- 元数据与安全检查
def exp_dir_for(output_root, variant, condition, seed):
    return Path(output_root) / variant / condition / f"seed_{seed}"


def git_state(repo=REPO):
    try:
        rev = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(repo), "status", "--porcelain"],
                                    capture_output=True, text=True).stdout.strip())
    except Exception:
        rev, dirty = "unknown", None
    return rev, dirty


def write_meta(exp_dir, args, status, extra=None):
    rev, dirty = git_state()
    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "git_revision": rev,
        "git_dirty": dirty,
        "dataset": "SDD",
        "protocol": "train_adapt",
        "variant": args.variant,
        "condition": args.condition,
        "data_root": str(args.data_root),
        "seed": args.seed,
        "folds": [],
        "config_name": CONFIG_NAME,
        "model_flags": {
            "use_observation_features": VARIANTS[args.variant]["use_observation_features"],
            "use_missing_summary": VARIANTS[args.variant]["use_missing_summary"],
            "use_gap_condition": False,
        },
        "model_parameters": getattr(args, "_model_parameters", None),
        "hist_input_dim": EXPECTED[args.variant][0],
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "precision": args.precision,
        "checkpoint_monitor": MONITOR,
        "command": " ".join(sys.argv),
        "status": status,
    }
    meta.update(extra or {})
    with open(Path(exp_dir) / "experiment_meta.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return meta


def check_overwrite(exp_dir, args):
    exp_dir = Path(exp_dir)
    if not exp_dir.exists():
        return
    complete = False
    rj = exp_dir / "results.json"
    if rj.exists():
        try:
            complete = json.loads(rj.read_text()).get("status") == "complete"
        except Exception:
            complete = False
    if complete and not args.overwrite:
        sys.exit(f"REFUSE: {exp_dir} 已有完整结果（results.json status=complete）。"
                 f"如确认覆盖请加 --overwrite（仅删除该 variant/condition/seed 目录）。")
    if args.overwrite:
        target = exp_dir.resolve()
        root = Path(args.output_root).resolve()
        assert str(target).startswith(str(root) + os.sep), "overwrite 目标越界"
        print(f"--overwrite: 删除 {target}")
        shutil.rmtree(target)


def compose_cfg(overrides):
    import hydra
    rel = os.path.relpath(REPO / "conf", Path(__file__).resolve().parent)
    with hydra.initialize(config_path=rel, version_base=None, job_name="ma_runner"):
        return hydra.compose(config_name=CONFIG_NAME, overrides=overrides)


def verify_variant_config(args):
    ov = build_train_overrides(args.variant, args.condition, str(args.data_root),
                               args.seed, args.epochs, args.batch_size,
                               args.num_workers, args.precision)
    cfg = compose_cfg(ov)
    m = cfg.model.target.model
    flags_ok = (bool(m.use_observation_features) == VARIANTS[args.variant]["use_observation_features"]
                and bool(m.use_missing_summary) == VARIANTS[args.variant]["use_missing_summary"])
    for forbidden in ("condition_state_query", "condition_mode_query", "condition_hybrid"):
        assert forbidden not in m, f"配置中出现未实现参数 {forbidden}"
    if not flags_ok:
        sys.exit(f"CONFIG MISMATCH: variant={args.variant} 期望 {VARIANTS[args.variant]}, "
                 f"resolved: uof={m.use_observation_features}, ums={m.use_missing_summary}")
    from hydra.utils import instantiate
    trainer = instantiate(cfg.model.target)
    net = trainer.net
    hist_in = net.hist_embed_mlp[0].in_features
    has_summary = hasattr(net, "missing_summary_embed")
    exp_in, exp_summary = EXPECTED[args.variant]
    if hist_in != exp_in or has_summary != exp_summary:
        sys.exit(f"MODEL MISMATCH: variant={args.variant} 期望 hist_in={exp_in}, "
                 f"summary={exp_summary}; 实际 hist_in={hist_in}, summary={has_summary}")
    args._model_parameters = sum(p.numel() for p in net.parameters())
    print(f"[config-check] variant={args.variant} hist_in={hist_in} "
          f"missing_summary_embed={has_summary} params={args._model_parameters}")
    return hist_in, has_summary


def load_ckpt_model_flags(ckpt_path):
    import torch
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    h = ckpt.get("hyper_parameters", {})
    model = h.get("model", {}) if isinstance(h, dict) else {}
    return {
        "use_observation_features": model.get("use_observation_features", False),
        "use_missing_summary": model.get("use_missing_summary", False),
    }


def verify_resume(ckpt_path, args):
    ckpt_path = Path(ckpt_path).resolve()
    if not ckpt_path.exists():
        sys.exit(f"RESUME REFUSED: checkpoint 不存在: {ckpt_path}")
    flags = load_ckpt_model_flags(ckpt_path)
    want = VARIANTS[args.variant]
    if (bool(flags["use_observation_features"]) != want["use_observation_features"]
            or bool(flags["use_missing_summary"]) != want["use_missing_summary"]):
        sys.exit(f"RESUME REFUSED: checkpoint 模型开关 {flags} 与 variant={args.variant} "
                 f"期望 {want} 不一致（禁止跨 variant 恢复，含 M0->M1/M2）")
    exp_root = exp_dir_for(args.output_root, args.variant, args.condition, args.seed).resolve()
    if not str(ckpt_path).startswith(str(exp_root) + os.sep):
        sys.exit(f"RESUME REFUSED: checkpoint 不在本实验目录内 ({exp_root})，"
                 f"无法证明同 condition/seed")
    ovr_file = ckpt_path
    while ovr_file.parent != ovr_file:
        cand = ovr_file.parent / ".hydra" / "overrides.yaml"
        if cand.exists():
            text = cand.read_text()
            need = [f"datamodule.target.data_root={args.data_root}",
                    f"seed={args.seed}"]
            missing = [t for t in need if t not in text]
            if missing:
                sys.exit(f"RESUME REFUSED: run overrides 与当前实验不一致，缺 {missing}")
            return str(ckpt_path)
        ovr_file = ovr_file.parent
    sys.exit("RESUME REFUSED: 未找到 checkpoint 所属 run 的 .hydra/overrides.yaml")


# ---------------------------------------------------------------- 主流程
def run_experiment(args):
    exp_dir = exp_dir_for(args.output_root, args.variant, args.condition, args.seed)
    exp_dir.mkdir(parents=True, exist_ok=True)
    train_log = str(exp_dir / "train" / "runner.log")
    eval_log = str(exp_dir / "eval" / "runner.log")
    Path(train_log).parent.mkdir(exist_ok=True)
    Path(eval_log).parent.mkdir(exist_ok=True)

    print(json.dumps({
        "variant": args.variant, **VARIANTS[args.variant],
        "hist_embed_mlp.in_features": EXPECTED[args.variant][0],
        "condition": args.condition, "data_root": str(args.data_root),
        "seed": args.seed, "config_name": CONFIG_NAME,
    }, ensure_ascii=False), flush=True)

    resume_ckpt = None
    if args.resume_checkpoint:
        resume_ckpt = verify_resume(args.resume_checkpoint, args)

    train_seconds = 0.0
    if not args.skip_train:
        t0 = time.time()
        rc = sh(build_train_command(args.variant, args.condition, str(args.data_root),
                                    args.seed, args.epochs, args.batch_size,
                                    args.num_workers, args.precision,
                                    str(exp_dir / "train" / "run"),
                                    resume_ckpt=resume_ckpt),
                train_log)
        train_seconds = time.time() - t0
        if rc != 0:
            return {"status": f"train_failed_rc{rc}"}, train_seconds
    else:
        print("[skip-train] 跳过训练，使用已有 run 目录", flush=True)

    run_dir = str(exp_dir / "train" / "run")
    try:
        best_epoch, best_val, ckpt = collect_best_checkpoint(run_dir)
    except AssertionError as e:
        return {"status": f"checkpoint_missing: {e}"}, train_seconds

    link = exp_dir / "train" / "best_for_eval.ckpt"
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(ckpt)

    t0 = time.time()
    rc = sh(build_eval_command(args.condition, str(args.data_root), args.seed,
                               args.precision, str(link), str(exp_dir / "eval")),
            eval_log)
    eval_seconds = time.time() - t0
    if rc != 0:
        return {"status": f"eval_failed_rc{rc}"}, train_seconds + eval_seconds

    try:
        metrics = parse_test_metrics(eval_log)
    except AssertionError as e:
        return {"status": f"metrics_parse_failed: {e}"}, train_seconds + eval_seconds

    row = {
        "dataset": "SDD",
        "protocol": "train_adapt",
        "variant": args.variant,
        "condition": args.condition,
        "seed": args.seed,
        "config_name": CONFIG_NAME,
        "use_observation_features": VARIANTS[args.variant]["use_observation_features"],
        "use_missing_summary": VARIANTS[args.variant]["use_missing_summary"],
        "checkpoint_epoch": best_epoch,
        "checkpoint_path": ckpt,
        "val_minFDE20": best_val,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "precision": args.precision,
        "train_seconds": round(train_seconds, 1),
        "eval_seconds": round(eval_seconds, 1),
        "status": "ok",
    }
    row.update({k: round(float(metrics[k]), 4) for k in TEST_KEYS})
    with open(exp_dir / "result.json", "w") as f:
        json.dump(row, f, indent=2, ensure_ascii=False)
    return row, train_seconds + eval_seconds


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    ap.add_argument("--condition", required=True)
    ap.add_argument("--data-root", required=True,
                    help="SDD missing 根目录，如 data/SDD_missing_v3_noguard")
    ap.add_argument("--output-root", default="outputs/missing_aware/sdd/train_adapt")
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=16)
    ap.add_argument("--precision", default="bf16")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--resume-checkpoint", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    args.data_root = Path(args.data_root)
    if not args.data_root.is_absolute():
        args.data_root = (REPO / args.data_root).resolve()
    assert args.data_root.exists(), f"data_root 不存在: {args.data_root}"

    exp_dir = exp_dir_for(args.output_root, args.variant, args.condition, args.seed)
    check_overwrite(exp_dir, args)
    exp_dir.mkdir(parents=True, exist_ok=True)

    verify_variant_config(args)
    write_meta(exp_dir, args, "running")

    try:
        row, _ = run_experiment(args)
        status = "complete" if row.get("status") == "ok" else "failed"
        write_meta(exp_dir, args, status,
                   {"failure_stage": row.get("status"), "return_code": 1}
                   if status == "failed" else None)
        print(f"DONE. status={row.get('status')}")
    except Exception as e:
        write_meta(exp_dir, args, "failed",
                   {"failure_stage": "runner", "return_code": 1, "error": str(e)})
        raise


if __name__ == "__main__":
    main()
