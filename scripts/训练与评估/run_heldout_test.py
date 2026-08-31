#!/usr/bin/env python
"""在 held-out test 场景上重算 DeMo 两个模型真正的最小测试指标(可与论文比)."""
import os, re, subprocess, sys, shutil
from pathlib import Path

GPU = sys.argv[1] if len(sys.argv) > 1 else "2"
D = Path("/home/lbh/DeMo")
PY = f"{Path.home()}/.conda/envs/DeMo/bin/python"

JOBS = [
    ("config_moflow_ethucy",          "eth",  "moflow_eth",          "20260826-183752", 82),
    ("config_moflow_ethucy",          "hotel","moflow_hotel",        "20260826-215931", 61),
    ("config_moflow_ethucy",          "univ", "moflow_univ",         "20260827-020739", 92),
    ("config_moflow_ethucy",          "zara1","moflow_zara1",        "20260827-035014", 69),
    ("config_moflow_ethucy",          "zara2","moflow_zara2",        "20260827-080537", 89),
    ("config_moflow_ethucy_bimamba",  "eth",  "moflow_bimamba_eth",  "20260826-212037", 98),
    ("config_moflow_ethucy_bimamba",  "hotel","moflow_bimamba_hotel","20260827-020642", 99),
    ("config_moflow_ethucy_bimamba",  "univ", "moflow_bimamba_univ", "20260827-073801", 99),
    ("config_moflow_ethucy_bimamba",  "zara1","moflow_bimamba_zara1","20260827-085216", 67),
    ("config_moflow_ethucy_bimamba",  "zara2","moflow_bimamba_zara2","20260827-130643", 96),
]

def run(cfg, sub, outdir, ts, ep):
    ck = D / "outputs" / outdir / ts / "checkpoints" / f"epoch={ep}.ckpt"
    if not ck.exists():
        print(f"{sub}({cfg}) : MISSING {ck}", flush=True); return
    link = ck.parent / "best_for_eval.ckpt"
    link.unlink(missing_ok=True)
    link.symlink_to(ck)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU)
    cmd = [PY, "-u", "eval.py", f"--config-name={cfg}", "gpus=1", "test=true",
           f"datamodule.target.subset={sub}", f"checkpoint={link}"]
    out = subprocess.run(cmd, cwd=D, env=env, capture_output=True, text=True).stdout
    link.unlink(missing_ok=True)
    n = re.search(r"Total predictions:\s+(\d+)", out)
    m = re.search(r"test_minADE20[^0-9]*([0-9.]+)", out)
    f = re.search(r"test_minFDE20[^0-9]*([0-9.]+)", out)
    ok = "OK" if "Loaded model weights" in out else "LOAD_WARN"
    print(f"{sub:6s}({cfg.split('_')[-1]:6s}): N={n.group(1) if n else '?':>5s} "
          f"minADE20={m.group(1) if m else '?':>8s} minFDE20={f.group(1) if f else '?':>8s} [{ok}]", flush=True)

for j in JOBS:
    run(*j)