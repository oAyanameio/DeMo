"""SDD missing-history（v1 / v2_high）：训练 + held-out test 评测（单 condition）。

用法:
    CUDA_VISIBLE_DEVICES=<gpu> python scripts/训练与评估/run_sdd_missing.py <condition> [epochs] [data_root]

condition ∈ {complete, random_single, random_block2,
             random_fixed3/4/5, random_block3/4/6}
data_root 缺省 data/SDD_missing_v1；v2 高缺失条件传 data/SDD_missing_v2_high。
流程与 scripts/训练与评估/run_sdd.py 相同：train（monitor=val_minFDE20, save_top_k=3）→
best epoch → eval test=true 重算 test_minADE20/test_minFDE20。
结果追加到 outputs/sdd_missing_<condition>_summary.txt。
"""
import csv
import os
import re
import subprocess
import sys
from pathlib import Path

D = Path("/home/lbh/DeMo")
PY = f"{Path.home()}/.conda/envs/DeMo/bin/python"
CONDITIONS = ["complete", "random_single", "random_block2",
              "random_fixed3", "random_fixed4", "random_fixed5",
              "random_block3", "random_block4", "random_block6"]
CFG = "config_sdd_missing"


def find_best_epoch(run_dir: Path):
    csvs = sorted(run_dir.glob("logs/version_*/metrics.csv"))
    best = None
    for c in csvs:
        with open(c) as fh:
            for r in csv.DictReader(fh):
                if not r.get("val_minFDE20"):
                    continue
                v = float(r["val_minFDE20"])
                if best is None or v < best[0]:
                    best = (v, int(float(r["epoch"])))
    return best


def main():
    cond = sys.argv[1]
    epochs = sys.argv[2] if len(sys.argv) > 2 else "100"
    data_root = sys.argv[3] if len(sys.argv) > 3 else str(D / "data" / "SDD_missing_v1")
    assert cond in CONDITIONS, f"bad condition: {cond}"
    out_name = f"sdd_missing_{cond}"
    env = dict(os.environ)

    # 1. train
    print(f"===== TRAIN {CFG} condition={cond} data_root={data_root} (epochs={epochs}) =====", flush=True)
    p = subprocess.run(
        [PY, "-u", "train.py", f"--config-name={CFG}",
         f"datamodule.target.condition={cond}", f"datamodule.target.data_root={data_root}",
         f"epochs={epochs}", f"output={out_name}"],
        cwd=D,
        env=env,
    )
    if p.returncode != 0:
        print(f"TRAIN FAILED rc={p.returncode}", flush=True)
        sys.exit(p.returncode)

    # 2. best checkpoint (monitor=val_minFDE20)
    runs = sorted((D / "outputs" / out_name).glob("*/"),
                  key=lambda x: x.stat().st_mtime)
    assert runs, "no output run dir found"
    run_dir = runs[-1]
    best = find_best_epoch(run_dir)
    assert best, "no val_minFDE20 found in metrics.csv"
    best_val, best_epoch = best
    ck = run_dir / "checkpoints" / f"epoch={best_epoch}.ckpt"
    assert ck.exists(), f"checkpoint missing: {ck}"

    # 3. eval held-out test
    link = run_dir / "checkpoints" / "best_for_eval.ckpt"
    link.unlink(missing_ok=True)
    link.symlink_to(ck)
    out = subprocess.run(
        [PY, "-u", "eval.py", f"--config-name={CFG}",
         f"datamodule.target.condition={cond}", f"datamodule.target.data_root={data_root}",
         "gpus=1", "test=true",
         f"checkpoint={link}"],
        cwd=D,
        env=env,
        capture_output=True,
        text=True,
    ).stdout
    link.unlink(missing_ok=True)

    n = re.search(r"Total predictions:\s+(\d+)", out)
    m = re.search(r"test_minADE20[^0-9]*([0-9.]+)", out)
    f = re.search(r"test_minFDE20[^0-9]*([0-9.]+)", out)

    line = (
        f"sdd_missing_{cond:13s} best_epoch={best_epoch:>3d} val_minFDE20={best_val:.4f}  "
        f"N={n.group(1) if n else '?':>5s}  "
        f"test_minADE20={m.group(1) if m else '?':>8s}  "
        f"test_minFDE20={f.group(1) if f else '?':>8s}"
    )
    print("===== RESULT =====", flush=True)
    print(line, flush=True)
    with open(D / "outputs" / f"{out_name}_summary.txt", "a") as fh:
        fh.write(line + "\n")


if __name__ == "__main__":
    main()
