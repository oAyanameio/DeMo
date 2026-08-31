#!/usr/bin/env python
"""提取 DeMo 单向/双向 MoFlow 协议每个 fold 的 val_new 指标、last-epoch 指标,并 diff overrides.yaml。"""
import csv, glob, json, os, sys
from pathlib import Path

ROOT = Path("/home/lbh/DeMo")

def best_metrics(run_dir):
    """返回 (best_eval, last_eval); each: dict(epoch,ade20,fde20,bminfde20) 基于 val_new_*"""
    csvs = sorted(Path(run_dir).glob("logs/version_*/metrics.csv"))
    rows = []
    for c in csvs:
        with open(c) as f:
            for r in csv.DictReader(f):
                if r.get("epoch") is None: continue
                try:
                    e=int(r["epoch"]); a=float(r["val_new_minADE20"]); fv=float(r["val_new_minFDE20"]); b=r.get("val_new_b-minFDE20"); b=float(b) if b else None
                    rows.append(dict(epoch=e,ade20=a,fde20=fv,bminfde20=b))
                except (ValueError, TypeError): pass
    if not rows: return None, None
    rows.sort(key=lambda r: r["epoch"])
    # best 按 val_new_minFDE20 最小
    best = min(rows, key=lambda r: r["fde20"])
    # 也单独按 val_new_minADE20 最小
    best_ade = min(rows, key=lambda r: r["ade20"])
    last = rows[-1]
    return {"best_fde20":best, "best_ade20":best_ade, "last":last}

arms = {
    "uni": [f"moflow_{s}" for s in ["eth","hotel","univ","zara1","zara2"]],
    "bi":  [f"moflow_bimamba_{s}" for s in ["eth","hotel","univ","zara1","zara2"]],
}
SUBS = ["eth","hotel","univ","zara1","zara2"]
print(f"{'arm':3s} {'sub':6s} {'run':20s} "
      f"{'bestFDE_ep':>11s} {'bFDE_ade20':>10s} {'bFDE_fde20':>10s} {'bFDE_bmin':>10s} "
      f"{'bestADE_ep':>11s} {'bADE_ade20':>10s} "
      f"{'last_ep':>8s} {'last_ade20':>10s} {'last_fde20':>10s}")
for arm, outnames in arms.items():
    for s,o in zip(SUBS,outnames):
        base = ROOT/"outputs"/o
        if not base.exists():
            print(f"{arm:3s} {s:6s} NO_DIR"); continue
        runs = sorted([p for p in base.glob("*/") if (p/"logs").exists()], key=lambda p: p.stat().st_mtime)
        if not runs:
            print(f"{arm:3s} {s:6s} NO_RUNS"); continue
        rd = runs[-1]
        best_ade, last0 = None, None
        b = best_metrics(rd)
        bF, bA, last = b["best_fde20"], b["best_ade20"], b["last"]
        print(f"{arm:3s} {s:6s} {Path(rd).name:20s} "
              f"{bF['epoch']:>11d} {bF['ade20']:>10.4f} {bF['fde20']:>10.4f} {bF['bminfde20'] if bF['bminfde20'] is not None else -1:>10.4f} "
              f"{bA['epoch']:>11d} {bA['ade20']:>10.4f} "
              f"{last['epoch']:>8d} {last['ade20']:>10.4f} {last['fde20']:>10.4f}")

print("\n===== overrides.yaml diff (uni vs bi, 各场景) =====")
for s in SUBS:
    un = ROOT/"outputs"/f"moflow_{s}"
    bi = ROOT/"outputs"/f"moflow_bimamba_{s}"
    def ov(base):
        runs = sorted([p for p in base.glob("*/")], key=lambda p: p.stat().st_mtime) if base.exists() else []
        for rd in reversed(runs):
            f = rd/".hydra/overrides.yaml"
            if f.exists(): return Path(rd).name, f.read_text().strip()
        return None, None
    nu,ou = ov(un); nb,ob = ov(bi)
    print(f"--- {s} ---")
    print(f"  uni run={nu} overrides={ou or 'N/A'}")
    print(f"  bi  run={nb} overrides={ob or 'N/A'}")
    print(f"  overrides相同: {ou==ob}")