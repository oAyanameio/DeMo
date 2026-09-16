"""TrajImpute P0/P0.5 结果汇总。

扫描 outputs/trajimpute_direct/<variant>_<scene>_<protocol>_seed<seed>/eval/*/results.json，
按 scene/seed/variant 聚合 minFDE20 等指标，输出：
  summary.json / summary.csv / comparison.md（含配对差值）

不把 Clean-direct 与 Easy-direct、不同 seed、非配对版本混合平均。
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

METRICS = ["minADE_K", "minFDE_K", "ADE@1", "FDE@1", "MR"]
SCENES = ["ETH-M", "HOTEL-M", "UNIV-M", "ZARA1-M", "ZARA2-M"]
FORMAL_K = 20


def protocol_label(manifest, meta):
    """遗留 Easy 零缺失子集不能作为 Clean-direct 汇总。"""
    clean_source = manifest.get("clean_source", {})
    if isinstance(clean_source, dict) and \
            clean_source.get("type") == "easy_zero_missing_subset":
        return "easy-zero-missing-diagnostic"
    if meta.get("zero_missing_only"):
        return "easy-zero-missing-diagnostic"
    return manifest.get("protocol", "?")


def collect(root: Path):
    rows = []
    for res_file in sorted(root.glob("*/eval/*/results.json")):
        payload = json.loads(res_file.read_text())
        meta = payload["meta"]
        ov = payload["results"]["overall"]
        run_dir = res_file.parents[2]
        manifest_path = run_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        row = {
            "variant": meta["variant"],
            "scene": meta["scene"],
            "protocol": protocol_label(manifest, meta),
            "seed": meta["seed"],
            "K": meta["K"],
            "backbone": manifest.get("backbone", {}).get("bimamba", True),
            "n": ov.get("n"),
            "ckpt": meta.get("checkpoint", {}).get("path") if isinstance(meta.get("checkpoint"), dict) else None,
            "run_dir": str(run_dir),
        }
        for m in METRICS:
            row[m] = ov.get(m)
        rows.append(row)
    return rows


def agg(rows, key_fields):
    """按 key_fields 聚合：均值/标准差/最差/最好。"""
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in key_fields)].append(r)
    out = []
    for key, grp in sorted(groups.items()):
        entry = dict(zip(key_fields, key))
        entry["n_runs"] = len(grp)
        entry["scenes"] = ",".join(g["scene"] for g in grp)
        for m in METRICS:
            vals = [g[m] for g in grp if g[m] is not None]
            if vals:
                entry[f"{m}_mean"] = sum(vals) / len(vals)
                entry[f"{m}_min"] = min(vals)
                entry[f"{m}_max"] = max(vals)
                if len(vals) > 1:
                    mu = entry[f"{m}_mean"]
                    entry[f"{m}_std"] = (sum((v - mu) ** 2 for v in vals)
                                         / (len(vals) - 1)) ** 0.5
        out.append(entry)
    return out


def paired_diff(rows, base_variant, comp_variant, protocol, seed, backbone=True):
    """同 scene/protocol/seed 配对：comp - base（minFDE 越低越好，负值=comp 更好）。"""
    def index(variant):
        return {r["scene"]: r for r in rows
                if r["variant"] == variant and r["protocol"] == protocol
                and r["seed"] == seed and r["backbone"] == backbone}
    b, c = index(base_variant), index(comp_variant)
    diffs = []
    for scene in SCENES:
        if scene in b and scene in c:
            for m in METRICS:
                if b[scene][m] is None or c[scene][m] is None:
                    continue
            d = {m: c[scene][m] - b[scene][m] for m in METRICS}
            d["scene"] = scene
            d["base_minFDE"] = b[scene]["minFDE_K"]
            d["comp_minFDE"] = c[scene]["minFDE_K"]
            d["rel_minFDE_pct"] = (d["minFDE_K"] / b[scene]["minFDE_K"] * 100
                                   if b[scene]["minFDE_K"] else None)
            diffs.append(d)
    return diffs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/trajimpute_retrain")
    args = ap.parse_args()
    root = Path(args.root)
    all_rows = collect(root)
    rows = [row for row in all_rows if row["K"] == FORMAL_K]
    excluded = [
        {
            "run_dir": row["run_dir"],
            "K": row["K"],
            "protocol": row["protocol"],
            "reason": f"formal TrajImpute summary requires K={FORMAL_K}",
        }
        for row in all_rows
        if row["K"] != FORMAL_K
    ]
    if not rows:
        if excluded:
            print(f"no K={FORMAL_K} results found; excluded {len(excluded)} legacy rows")
        else:
            print("no results found")
        return

    # 明细
    detail_path = root / "summary.json"
    by_variant = agg(rows, ["protocol", "variant", "seed", "backbone", "K"])
    payload = {
        "formal_K": FORMAL_K,
        "rows": rows,
        "excluded_legacy_rows": excluded,
        "by_variant": by_variant,
    }
    detail_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str))

    # CSV
    csv_path = root / "summary.csv"
    fields = ["variant", "scene", "protocol", "seed", "backbone", "K", "n"] + METRICS
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # comparison.md：配对比较 + 聚合表
    md = ["# TrajImpute 选题模型重训结果汇总\n"]
    md.append("数据来源：`" + str(root) + "`；训练与评估均为 direct 重训，K=20；")
    md.append("MR=全部 K 模态终点误差>2.0。Easy 零缺失子集仅作诊断，不是 Clean-direct。\n")

    md.append("## 各版本×场景×seed 明细\n")
    md.append("| variant | scene | protocol | seed | n | minADE20 | minFDE20 | ADE@1 | FDE@1 | MR |")
    md.append("|---|---|---|---|---:|---:|---:|---:|---:|---:|")
    for r in sorted(rows, key=lambda x: (x["protocol"], x["variant"], x["seed"], SCENES.index(x["scene"]))):
        md.append(
            f"| {r['variant']} | {r['scene']} | {r['protocol']} | {r['seed']} | {r['n']} "
            f"| {r['minADE_K']:.4f} | {r['minFDE_K']:.4f} | {r['ADE@1']:.4f} "
            f"| {r['FDE@1']:.4f} | {r['MR']:.4f} |")

    # 配对差（E1 vs M0；同 protocol/seed/backbone）
    seeds = sorted({r["seed"] for r in rows})
    for seed in seeds:
        for proto in sorted({r["protocol"] for r in rows}):
            for comp in ("E1-gap-scaling",):
                diffs = paired_diff(rows, "M0", comp, proto, seed)
                if not diffs:
                    continue
                md.append(f"\n## 配对差 {comp} − M0（{proto}，seed={seed}，负值={comp} 更好）\n")
                md.append("| scene | ΔminADE | ΔminFDE | ΔminFDE% | ΔMR |")
                md.append("|---|---:|---:|---:|---:|")
                for d in diffs:
                    md.append(f"| {d['scene']} | {d['minADE_K']:+.4f} | {d['minFDE_K']:+.4f} "
                              f"| {d['rel_minFDE_pct']:+.1f}% | {d['MR']:+.4f} |")
                n_better = sum(1 for d in diffs if d["minFDE_K"] < 0)
                md.append(f"\nminFDE 改善场景数：{n_better}/{len(diffs)}")

    md.append("\n## 聚合（protocol × variant × seed × K）\n")
    md.append("| protocol | variant | seed | K | runs | minFDE20均值 | minFDE20最好 | minFDE20最差 | minFDE20标准差 |")
    md.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for e in by_variant:
        md.append(f"| {e['protocol']} | {e['variant']} | {e['seed']} | {e['K']} | {e['n_runs']} "
                  f"| {e.get('minFDE_K_mean', float('nan')):.4f} "
                  f"| {e.get('minFDE_K_min', float('nan')):.4f} "
                  f"| {e.get('minFDE_K_max', float('nan')):.4f} "
                  f"| {e.get('minFDE_K_std', 0):.4f} |")

    (root / "comparison.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[summary] {detail_path}")
    print(f"[summary] {csv_path}")
    print(f"[summary] {root / 'comparison.md'}")


if __name__ == "__main__":
    main()
