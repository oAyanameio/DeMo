"""Missing-Aware M0/M1/M2 配对结果分析（ETH/UCY 五折 / SDD 单协议）。

输入：run_missing_aware_{ethucy,sdd}.py 产出的 results.json / result.json 树。
比较（相同 condition+seed+fold 配对）：M1-M0、M2-M1、M2-M0。
完整性校验：status!=ok 拒绝、fold 缺失/重复拒绝、variant-开关不一致拒绝、
训练预算不一致拒绝、指标缺失拒绝。
SDD 单 seed 只报绝对差与相对变化，不输出显著性结论。

输出：summary.json / summary.csv / summary.md（--output-dir）。
"""
import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

VARIANTS = {
    "M0_base": {"use_observation_features": False, "use_missing_summary": False},
    "M1_obs": {"use_observation_features": True, "use_missing_summary": False},
    "M2_history": {"use_observation_features": True, "use_missing_summary": True},
}
FOLDS = ["ETH", "HOTEL", "UNIV", "ZARA1", "ZARA2"]
PAIRS = [("M1_obs", "M0_base"), ("M2_history", "M1_obs"), ("M2_history", "M0_base")]
ETHUCY_METRICS = ["test_minADE6", "test_minFDE6", "test_MR", "test_b-minFDE6"]
PRIMARY = "test_minFDE6"
BOOTSTRAP_N = 10000
BOOTSTRAP_SEED = 2024


class IntegrityError(Exception):
    """完整性校验失败（分析拒绝执行）。"""


# ---------------------------------------------------------------- 读取与校验
def load_ethucy_run(exp_root, variant, condition, seed):
    """读取并校验一个 ETH/UCY (variant, condition, seed) 组，返回 rows。"""
    d = Path(exp_root) / variant / condition / f"seed_{seed}"
    rj = d / "results.json"
    if not rj.exists():
        raise IntegrityError(f"缺少 results.json: {d}")
    data = json.loads(rj.read_text())
    rows = [r for r in data.get("rows", []) if isinstance(r, dict)]
    if not rows:
        raise IntegrityError(f"{d}: results.json 无 rows")
    problems = []
    # 1) status
    if data.get("status") != "complete" or any(r.get("status") != "ok" for r in rows):
        bad = [r.get("fold") for r in rows if r.get("status") != "ok"]
        problems.append(f"status 不完整（summary={data.get('status')}, 非 ok 折: {bad}）")
    # 2) fold 完整性与重复
    folds = [r.get("fold") for r in rows]
    if sorted(f for f in folds if f) != sorted(FOLDS):
        problems.append(f"fold 集合 != 五折全集: {sorted(set(folds))}")
    if len(folds) != len(set(folds)):
        problems.append(f"重复 fold: {[f for f in folds if folds.count(f) > 1]}")
    for r in rows:
        # 3) variant-开关一致性
        want = VARIANTS[r.get("variant", variant)]
        if (bool(r.get("use_observation_features")) != want["use_observation_features"]
                or bool(r.get("use_missing_summary")) != want["use_missing_summary"]):
            problems.append(f"fold {r.get('fold')}: 模型开关与 variant 不一致 "
                            f"(uof={r.get('use_observation_features')}, "
                            f"ums={r.get('use_missing_summary')})")
        # 4) condition/seed 一致
        if r.get("condition") != condition or r.get("seed") != seed:
            problems.append(f"fold {r.get('fold')}: condition/seed 不一致 "
                            f"({r.get('condition')}/{r.get('seed')})")
        # 5) 指标字段
        for k in ETHUCY_METRICS + ["val_minFDE6"]:
            if r.get(k) is None:
                problems.append(f"fold {r.get('fold')}: 缺指标 {k}")
    if problems:
        raise IntegrityError(f"{d}: " + "; ".join(problems[:6]))
    return rows


def load_sdd_run(exp_root, variant, condition, seed):
    d = Path(exp_root) / variant / condition / f"seed_{seed}"
    rj = d / "result.json"
    if not rj.exists():
        raise IntegrityError(f"缺少 result.json: {d}")
    r = json.loads(rj.read_text())
    want = VARIANTS[r.get("variant", variant)]
    problems = []
    if r.get("status") != "ok":
        problems.append(f"status={r.get('status')}")
    if (bool(r.get("use_observation_features")) != want["use_observation_features"]
            or bool(r.get("use_missing_summary")) != want["use_missing_summary"]):
        problems.append("模型开关与 variant 不一致")
    if r.get("condition") != condition or r.get("seed") != seed:
        problems.append(f"condition/seed 不一致 ({r.get('condition')}/{r.get('seed')})")
    for k in ETHUCY_METRICS:
        if r.get(k.replace("6", "20") if "6" in k else k) is None:
            problems.append(f"缺指标 {k}")
    if problems:
        raise IntegrityError(f"{d}: " + "; ".join(problems))
    return r


def check_budget_consistency(groups):
    """同一比较组（同 dataset/condition/seed 的各 variant）训练预算必须一致。"""
    problems = []
    for (ds, cond, seed), variants in groups.items():
        budgets = {v: (meta.get("epochs"), meta.get("batch_size"))
                   for v, (rows, meta) in variants.items()}
        vals = set(budgets.values())
        if len(vals) > 1:
            problems.append(f"{ds}/{cond}/seed_{seed}: 训练预算不一致 {budgets}")
    return problems


# ---------------------------------------------------------------- 统计
def paired_t(diffs):
    n = len(diffs)
    if n < 2:
        return None
    mean = sum(diffs) / n
    var = sum((x - mean) ** 2 for x in diffs) / (n - 1)
    if var == 0:
        return float("nan") if mean == 0 else (float("inf") if mean > 0 else float("-inf"))
    t = mean / math.sqrt(var / n)
    return t


def t_two_sided_p(t, df):
    """正态近似的双尾 p 值（n>=5 时近似；正式报告用 bootstrap CI 为主）。"""
    # t 分布 CDF 近似： Abramowitz-Stegun 26.7.10 不够稳，这里用 df 自适应正态近似
    z = t / math.sqrt(1 + t * t / df)  # Johnson-Kotz 变换
    return 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))


def bootstrap_ci(diffs, n_boot=BOOTSTRAP_N, seed=BOOTSTRAP_SEED, alpha=0.05):
    rng = random.Random(seed)
    n = len(diffs)
    means = []
    for _ in range(n_boot):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(n_boot * alpha / 2)]
    hi = means[min(n_boot - 1, int(n_boot * (1 - alpha / 2)))]
    return lo, hi


def analyze_pair(paired, metric):
    """paired: list[(fold, v_a, v_b)]；返回 a-b 统计。"""
    diffs = [a - b for _, a, b in paired]
    n = len(diffs)
    mean = sum(diffs) / n
    std = (sum((x - mean) ** 2 for x in diffs) / n) ** 0.5
    base_mean = sum(b for _, a, b in paired) / n
    rel = (mean / base_mean * 100) if base_mean != 0 else None
    out = {
        "n": n,
        "diff_mean": round(mean, 4),
        "diff_std": round(std, 4),
        "rel_change_pct": round(rel, 1) if rel is not None else None,
        "per_fold": [{"fold": f, "a": round(a, 4), "b": round(b, 4),
                      "diff": round(a - b, 4)} for f, a, b in paired],
    }
    if n >= 2:
        t = paired_t(diffs)
        if t is not None and t == t and abs(t) != float("inf"):  # 过滤 nan/inf
            out["t_stat"] = round(t, 4)
            p = t_two_sided_p(t, n - 1)
            out["p_value_norm_approx"] = round(p, 5) if p == p else None
        lo, hi = bootstrap_ci(diffs)
        out["bootstrap_ci95"] = [round(lo, 4), round(hi, 4)]
    return out


# ---------------------------------------------------------------- 主分析
def analyze_ethucy(args):
    results = {"dataset": "ETHUCY", "protocol": args.protocol, "conditions": {}}
    md_sections = []
    for cond in args.conditions:
        cond_block = {"pairs": {}, "variants": {}}
        for v in args.variants:
            rows = load_ethucy_run(args.input_root, v, cond, args.seeds[0]) \
                if len(args.seeds) == 1 else None
            if rows is None:
                # 多 seed：合并 (seed, fold) 为配对单位
                rows = []
                for s in args.seeds:
                    rows.extend(load_ethucy_run(args.input_root, v, cond, s))
            cond_block["variants"][v] = rows
        # 训练预算一致性
        budgets = set()
        for v, rows in cond_block["variants"].items():
            budgets.update((r.get("epochs"), r.get("batch_size")) for r in rows)
        if len(budgets) > 1:
            raise IntegrityError(f"{cond}: 比较组内训练预算不一致 {budgets}")
        for a, b in PAIRS:
            if a not in cond_block["variants"] or b not in cond_block["variants"]:
                continue
            rows_a = {f"{r['seed']}_{r['fold']}": r for r in cond_block["variants"][a]}
            rows_b = {f"{r['seed']}_{r['fold']}": r for r in cond_block["variants"][b]}
            keys = sorted(set(rows_a) & set(rows_b))
            if not keys:
                raise IntegrityError(f"{cond}: {a} 与 {b} 无共同 (seed,fold) 配对")
            pair_block = {}
            for metric in ETHUCY_METRICS:
                paired = [(k.split("_", 1)[1], rows_a[k][metric], rows_b[k][metric])
                          for k in keys]
                pair_block[metric] = analyze_pair(paired, metric)
            cond_block["pairs"][f"{a} - {b}"] = pair_block
        results["conditions"][cond] = cond_block
    return results


def analyze_sdd(args):
    results = {"dataset": "SDD", "protocol": args.protocol, "conditions": {}}
    for cond in args.conditions:
        cond_block = {"pairs": {}, "variants": {}}
        for v in args.variants:
            rows = [load_sdd_run(args.input_root, v, cond, s) for s in args.seeds]
            cond_block["variants"][v] = rows
        budgets = set()
        for v, rows_list in cond_block["variants"].items():
            budgets.update((r.get("epochs"), r.get("batch_size")) for r in rows_list)
        if len(budgets) > 1:
            raise IntegrityError(f"{cond}: 比较组内训练预算不一致 {budgets}")
        sdd_metrics = [m.replace("6", "20") for m in ETHUCY_METRICS]
        for a, b in PAIRS:
            if a not in cond_block["variants"] or b not in cond_block["variants"]:
                continue
            ra = {r["seed"]: r for r in cond_block["variants"][a]}
            rb = {r["seed"]: r for r in cond_block["variants"][b]}
            keys = sorted(set(ra) & set(rb))
            pair_block = {}
            for metric in sdd_metrics:
                paired = [(f"seed_{s}", ra[s][metric], rb[s][metric]) for s in keys]
                st = analyze_pair(paired, metric)
                if len(keys) < 2:
                    # 单 seed：只保留绝对差与相对变化，删除显著性字段
                    st.pop("t_stat", None)
                    st.pop("p_value_norm_approx", None)
                    st.pop("bootstrap_ci95", None)
                pair_block[metric] = st
            cond_block["pairs"][f"{a} - {b}"] = pair_block
        results["conditions"][cond] = cond_block
    return results


# ---------------------------------------------------------------- 输出
def fmt(x, nd=4):
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def write_md(results, path):
    ds = results["dataset"]
    lines = [
        f"# Missing-Aware 配对分析（{ds}，{results['protocol']}）",
        "",
        f"- 实验协议：{results['protocol']}（train_adapt：同条件训练/验证/测试）",
        f"- 数据集：{ds}",
        f"- 主终点：{PRIMARY if ds == 'ETHUCY' else 'test_minFDE20'}",
        f"- 完整性状态：通过（未通过的组合已在读取阶段被拒绝）",
        "",
    ]
    for cond, cb in results["conditions"].items():
        lines.append(f"## condition = {cond}")
        lines.append("")
        # variant 数值表
        lines.append("### 各 variant 均值±标准差（五折）" if ds == "ETHUCY" else "### 各 variant 数值（seed）")
        lines.append("")
        if ds == "ETHUCY":
            metrics = ETHUCY_METRICS
            header = "| variant | " + " | ".join(metrics) + " |"
            lines += [header, "|" + "---|" * (len(metrics) + 1)]
            for v, rows in cb["variants"].items():
                cells = []
                for m in metrics:
                    vals = [r[m] for r in rows]
                    mean = sum(vals) / len(vals)
                    std = (sum((x - mean) ** 2 for x in vals) / len(vals)) ** 0.5
                    cells.append(f"{mean:.4f}±{std:.4f}")
                lines.append(f"| **{v}** | " + " | ".join(cells) + " |")
        else:
            metrics = [m.replace("6", "20") for m in ETHUCY_METRICS]
            header = "| variant | seed | " + " | ".join(metrics) + " |"
            lines += [header, "|" + "---|" * (len(metrics) + 2)]
            for v, rows_list in cb["variants"].items():
                for r in rows_list:
                    lines.append(f"| {v} | {r['seed']} | " +
                                 " | ".join(fmt(r[m]) for m in metrics) + " |")
        lines.append("")
        # 配对表
        for pair_name, pb in cb["pairs"].items():
            lines.append(f"### 配对：{pair_name}")
            lines.append("")
            prim = PRIMARY if ds == "ETHUCY" else "test_minFDE20"
            for m, st in pb.items():
                marker = "（主终点）" if m == prim else ""
                lines.append(f"- **{m}**{marker}: diff={fmt(st['diff_mean'])}±{fmt(st['diff_std'])}"
                             f"（n={st['n']}），rel={fmt(st.get('rel_change_pct'), 1)}%")
                if "per_fold" in st:
                    per = ", ".join(f"{d['fold']}: {d['diff']:+.4f}" for d in st["per_fold"])
                    lines.append(f"  - 各折差值：{per}")
                if "bootstrap_ci95" in st:
                    lines.append(f"  - bootstrap 95% CI: [{st['bootstrap_ci95'][0]}, "
                                 f"{st['bootstrap_ci95'][1]}]"
                                 f"（t={st.get('t_stat')}, p≈{st.get('p_value_norm_approx')}）")
            lines.append("")
    # 一句话结论
    try:
        prim = PRIMARY if ds == "ETHUCY" else "test_minFDE20"
        first_cond = next(iter(results["conditions"].values()))
        pair21 = first_cond["pairs"].get("M2_history - M0_base", {})
        st = pair21.get(prim)
        if st:
            d, r = st["diff_mean"], st.get("rel_change_pct")
            better = "改善" if d < 0 else ("恶化" if d > 0 else "持平")
            lines.append(f"一句话结论：M2_history 相对 M0_base 主终点 {prim} "
                         f"{d:+.4f}（{r:+.1f}%，{better}；配对单位含全部 fold/seed）。")
    except Exception:
        pass
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def write_csv(results, path):
    rows = []
    for cond, cb in results["conditions"].items():
        for pair_name, pb in cb["pairs"].items():
            for m, st in pb.items():
                rows.append({
                    "dataset": results["dataset"],
                    "protocol": results["protocol"],
                    "condition": cond,
                    "pair": pair_name,
                    "metric": m,
                    "n": st["n"],
                    "diff_mean": st["diff_mean"],
                    "diff_std": st["diff_std"],
                    "rel_change_pct": st.get("rel_change_pct"),
                    "p_value_norm_approx": st.get("p_value_norm_approx"),
                    "ci95_lo": st.get("bootstrap_ci95", [None, None])[0],
                    "ci95_hi": st.get("bootstrap_ci95", [None, None])[1],
                })
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else
                           ["dataset", "protocol", "condition", "pair", "metric"])
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-root", default="outputs/missing_aware")
    ap.add_argument("--dataset", choices=["ethucy", "sdd"], required=True)
    ap.add_argument("--protocol", default="train_adapt")
    ap.add_argument("--conditions", nargs="+", required=True)
    ap.add_argument("--variants", nargs="+", default=["M0_base", "M1_obs", "M2_history"],
                    choices=sorted(VARIANTS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[2024])
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()
    if not Path(args.input_root).is_absolute():
        args.input_root = str((REPO / args.input_root).resolve())
    if args.output_dir is None:
        args.output_dir = str(Path(args.input_root) / args.dataset / f"analysis_{args.protocol}")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.dataset == "ethucy":
        results = analyze_ethucy(args)
    else:
        results = analyze_sdd(args)

    out_json = Path(args.output_dir) / "summary.json"
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(results, Path(args.output_dir) / "summary.csv")
    write_md(results, Path(args.output_dir) / "summary.md")
    print(f"OK: {out_json}")
    print(f"OK: {Path(args.output_dir) / 'summary.csv'}")
    print(f"OK: {Path(args.output_dir) / 'summary.md'}")


if __name__ == "__main__":
    main()
