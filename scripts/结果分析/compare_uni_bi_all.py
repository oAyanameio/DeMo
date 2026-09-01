#!/usr/bin/env python3
"""跨轮次 bi vs uni 对照汇总（第一轮/零样本/v2-SDD）。

数据源:
  A. docs/results/raw/missing_v1_round1_ethucy_15folds.json   (v1 训练期适应, uni)
  B. outputs/missing_v1_bimamba_logs/eval_rerun.log           (v1 训练期适应, bi)
  C. outputs/missing_v2_zeroshot_logs/zeroshot_summary.txt    (零样本, uni+bi)
  D. outputs/sdd_missing_*_summary.txt                        (v2 SDD 适应臂, uni+bi)
"""
import json, re, sys
from pathlib import Path

D = Path("/home/lbh/DeMo")

def mean(xs): return sum(xs) / len(xs)

# ---------- A. v1 uni 适应轮 ----------
def load_v1_uni():
    d = json.load(open(D / "docs/results/raw/missing_v1_round1_ethucy_15folds.json"))
    out = {}
    for r in d["rows"]:
        out[(r["cond"], r["fold"])] = {k: r[k] for k in ("FDE6", "ADE6", "MR")}
    return out

# ---------- B. v1 bi 适应轮 (从 eval_rerun.log 解析) ----------
def load_v1_bi():
    text = open(D / "outputs/missing_v1_bimamba_logs/eval_rerun.log").read()
    out = {}
    # 每段: ===== cond / fold (ckpt) ===== ... TEST METRICS: {...}
    for m in re.finditer(r"===== (\w+) / (\w+) \([^)]+\) =====.*?TEST METRICS: (\{.*?\})\n",
                         text, re.S):
        cond, fold, met = m.group(1), m.group(2), m.group(3)
        vals = dict(re.findall(r"'(test_\w+)': tensor\(([0-9.eE+-]+)", met))
        out[(cond, fold)] = {
            "FDE6": float(vals["test_minFDE6"]),
            "ADE6": float(vals["test_minADE6"]),
            "MR": float(vals["test_MR"]),
        }
    return out

# ---------- C. 零样本轮 ----------
def load_zeroshot():
    out = {}
    for line in open(D / "outputs/missing_v2_zeroshot_logs/zeroshot_summary.txt"):
        m = re.match(r"(ETHUCY|SDD) (uni|bi) (\w+) (\w+)? ?rc=(\d+) TEST METRICS: (\{.*\})", line.strip())
        if not m:
            m2 = re.match(r"(ETHUCY|SDD) (uni|bi) (\w+) rc=(\d+) TEST METRICS: (\{.*\})", line.strip())
            if not m2: continue
            ds, dirn, cond, rc, met = m2.groups()
            fold = "-"
        else:
            ds, dirn, cond, fold, rc, met = m.groups()
            fold = (fold or "-").strip() or "-"
        vals = dict(re.findall(r"'(test_\w+)': tensor\(([0-9.eE+-]+)", met))
        fde_key = "test_minFDE6" if ds == "ETHUCY" else "test_minFDE20"
        ade_key = "test_minADE6" if ds == "ETHUCY" else "test_minADE20"
        out[(ds, dirn, cond, fold)] = {
            "FDE": float(vals[fde_key]), "ADE": float(vals[ade_key]),
            "MR": float(vals.get("test_MR", "nan")),
        }
    return out

# ---------- D. v2 SDD 适应臂 ----------
def load_v2_sdd():
    out = {}
    for suffix, dirn in (("", "uni"), ("_bimamba", "bi")):
        for cond in ("random_fixed3", "random_fixed4", "random_block3",
                     "random_block4", "random_block6"):
            f = D / f"outputs/sdd_missing{suffix}_{cond}_summary.txt"
            if not f.exists(): continue
            for line in open(f):
                m = re.search(r"best_epoch=\s*(\d+).*test_minAde=\s*([0-9.]+| \?).*test_minFDE20=\s*([0-9.]+)", line)
                m2 = re.search(r"test_minADE20=\s*([0-9.]+)\s+test_minFDE20=\s*([0-9.]+)", line)
                use = m2 or m
                if not use: continue
                if m2:
                    ade, fde = float(m2.group(1)), float(m2.group(2))
                else:
                    ade, fde = (float(use.group(2)) if use.group(2).strip() != "?" else None,
                                float(use.group(3)))
                be = re.search(r"best_epoch=\s*(\d+)", line)
                out[(dirn, cond)] = {"FDE": fde, "ADE": ade,
                                     "best_ep": int(be.group(1)) if be else None}
    return out

def main():
    uniA = load_v1_uni(); biB = load_v1_bi()
    zs = load_zeroshot(); sdd = load_v2_sdd()

    print("=" * 78)
    print("【一】v1 训练期适应 (ETH/UCY 五折宏平均, 米制, FDE6/minADE6/MR)")
    print("=" * 78)
    folds = ("ETH", "HOTEL", "UNIV", "ZARA1", "ZARA2")
    conds_v1 = ("complete", "random_single", "random_block2")
    hdr = f"{'条件':<16}{'uni FDE6':>10}{'bi FDE6':>10}{'ΔFDE%':>8}{'uni MR':>9}{'bi MR':>9}{'n(uni/bi)':>10}"
    print(hdr)
    for c in conds_v1:
        uf = [uniA[(c, f)]["FDE6"] for f in folds if (c, f) in uniA]
        bf = [biB[(c, f)]["FDE6"] for f in folds if (c, f) in biB]
        um = [uniA[(c, f)]["MR"] for f in folds if (c, f) in uniA]
        bm = [biB[(c, f)]["MR"] for f in folds if (c, f) in biB]
        if uf and bf:
            d = (mean(bf) / mean(uf) - 1) * 100
            print(f"{c:<16}{mean(uf):>10.4f}{mean(bf):>10.4f}{d:>+7.1f}%{mean(um):>9.4f}{mean(bm):>9.4f}{len(uf):>6}/{len(bf)}")

    print()
    print("=" * 78)
    print("【二】零样本高缺失 (complete ckpt 直测 v2 test; ETH/UCY 五折宏平均)")
    print("=" * 78)
    conds_v2 = ("random_fixed3", "random_fixed4", "random_block3",
                "random_block4", "random_block6")
    print(f"{'条件':<16}{'uni FDE6':>10}{'bi FDE6':>10}{'ΔFDE%':>8}{'uni MR':>9}{'bi MR':>9}")
    for c in conds_v2:
        uf = [zs[("ETHUCY", "uni", c, f)]["FDE"] for f in folds if ("ETHUCY", "uni", c, f) in zs]
        bf = [zs[("ETHUCY", "bi", c, f)]["FDE"] for f in folds if ("ETHUCY", "bi", c, f) in zs]
        um = [zs[("ETHUCY", "uni", c, f)]["MR"] for f in folds if ("ETHUCY", "uni", c, f) in zs]
        bm = [zs[("ETHUCY", "bi", c, f)]["MR"] for f in folds if ("ETHUCY", "bi", c, f) in zs]
        if uf and bf:
            d = (mean(bf) / mean(uf) - 1) * 100
            print(f"{c:<16}{mean(uf):>10.4f}{mean(bf):>10.4f}{d:>+7.1f}%{mean(um):>9.4f}{mean(bm):>9.4f}")
    print("  SDD 侧 (像素制, 单点):")
    for c in conds_v2:
        u = zs.get(("SDD", "uni", c, "-")); b = zs.get(("SDD", "bi", c, "-"))
        if u and b:
            d = (b["FDE"] / u["FDE"] - 1) * 100
            print(f"  {c:<14}{u['FDE']:>10.2f}{b['FDE']:>10.2f}{d:>+7.1f}%{u['MR']:>9.3f}{b['MR']:>9.3f}")

    print()
    print("=" * 78)
    print("【三】v2 SDD 训练期适应 (像素制 minFDE20; complete: uni 14.55 / bi 15.42)")
    print("=" * 78)
    print(f"{'条件':<16}{'uni FDE20':>11}{'bi FDE20':>11}{'ΔFDE%':>8}{'uni best':>9}{'bi best':>9}")
    for c in conds_v2:
        u = sdd.get(("uni", c)); b = sdd.get(("bi", c))
        if u and b:
            d = (b["FDE"] / u["FDE"] - 1) * 100
            print(f"{c:<16}{u['FDE']:>11.2f}{b['FDE']:>11.2f}{d:>+7.1f}%{u.get('best_ep'):>9}{b.get('best_ep'):>9}")

    print()
    print("=" * 78)
    print("【四】同条件对照总表 (uni vs bi, 全部已有数据; Δ>0 = bi 更差)")
    print("=" * 78)
    # FDE 相对差汇总
    rows = []
    def safe_mean(d, key, c, folds, metric):
        xs = [d[(c, f)][metric] for f in folds if (c, f) in d]
        return mean(xs) if xs else None
    for c in conds_v1:
        uf = safe_mean(uniA, None, c, folds, "FDE6")
        bf = safe_mean(biB, None, c, folds, "FDE6")
        if uf and bf: rows.append(("v1适应 ETH/UCY", c, uf, bf))
    for c in conds_v2:
        uf = safe_mean(zs, None, ("ETHUCY", "uni", c), folds, "FDE") if False else None
        u_l = [zs[("ETHUCY", "uni", c, f)]["FDE"] for f in folds if ("ETHUCY", "uni", c, f) in zs]
        b_l = [zs[("ETHUCY", "bi", c, f)]["FDE"] for f in folds if ("ETHUCY", "bi", c, f) in zs]
        if u_l and b_l: rows.append(("零样本 ETH/UCY", c, mean(u_l), mean(b_l)))
        if ("SDD", "uni", c, "-") in zs and ("SDD", "bi", c, "-") in zs:
            rows.append(("零样本 SDD", c, zs[("SDD","uni",c,"-")]["FDE"], zs[("SDD","bi",c,"-")]["FDE"]))
        if ("uni", c) in sdd and ("bi", c) in sdd:
            rows.append(("v2适应 SDD", c, sdd[("uni",c)]["FDE"], sdd[("bi",c)]["FDE"]))
    print(f"{'实验':<16}{'条件':<16}{'uni':>10}{'bi':>10}{'Δ(bi-uni)%':>12}")
    for tag, c, u, b in rows:
        if u and b:
            print(f"{tag:<16}{c:<16}{u:>10.4f}{b:>10.4f}{(b/u-1)*100:>+11.1f}%")

if __name__ == "__main__":
    main()
