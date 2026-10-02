from pathlib import Path
import csv
import json

SCENES = ["ETH-M", "HOTEL-M", "UNIV-M", "ZARA1-M", "ZARA2-M"]
METRICS = ["minADE_K", "minFDE_K", "ADE@1", "FDE@1", "MR"]
BASE_ROOTS = {
    "M0": Path("outputs/c1a_m0_par_m0_hard"),
    "C1-A": Path("outputs/c1a_m0_par_c1a_hard"),
    "C3": Path("outputs/c3_hard_screening_seed2024"),
}
SUBGROUP_ROOT = Path("outputs/c1a_subgroup_eval/hard")
OUT = Path("outputs/c3_hard_screening_seed2024/summary")


def one_result(root: Path, scene: str):
    paths = list(root.rglob(f"{scene}*/*/results.json")) + list(root.rglob(f"*{scene}*/**/results.json"))
    paths = sorted(set(paths))
    if len(paths) != 1:
        raise RuntimeError(f"expected one result for {root} {scene}, got {paths}")
    return json.loads(paths[0].read_text())


def subgroup_result(variant: str, scene: str):
    paths = list((SUBGROUP_ROOT / f"{variant}_{scene}").rglob("results.json"))
    if len(paths) != 1:
        raise RuntimeError((variant, scene, paths))
    return json.loads(paths[0].read_text())["results"]


def aggregate(entries, metric, weighted):
    if weighted:
        total = sum(x["overall"]["n"] for x in entries)
        return sum(x["overall"][metric] * x["overall"]["n"] for x in entries) / total
    return sum(x["overall"][metric] for x in entries) / len(entries)


payload = {v: {s: one_result(root, s) for s in SCENES} for v, root in BASE_ROOTS.items()}
for v in payload:
    assert set(payload[v]) == set(SCENES)

summary = {"protocol": "hard-direct", "seed": 2024, "scenes": SCENES, "metrics": METRICS, "overall": {}, "subgroups": {}}
for aggregation, weighted in [("macro", False), ("micro", True)]:
    summary["overall"][aggregation] = {}
    for v in payload:
        values = {m: aggregate([payload[v][s]["results"] for s in SCENES], m, weighted) for m in METRICS}
        summary["overall"][aggregation][v] = values
    for base in ["M0", "C1-A"]:
        summary["overall"][aggregation][f"C3_vs_{base}_delta_pct"] = {
            m: (summary["overall"][aggregation]["C3"][m] / summary["overall"][aggregation][base][m] - 1) * 100
            for m in METRICS
        }

for dim in ["missing_count", "valid_count", "anchor_lag", "forecast_gap", "terminal_missing"]:
    summary["subgroups"][dim] = {}
    values = sorted(
        set().union(*(set(payload["C3"][s]["results"]["by_dimension"][dim]) for s in SCENES)),
        key=lambda x: (x not in {"false", "true"}, int(x) if x.isdigit() else x),
    )
    for value in values:
        entries = []
        for scene in SCENES:
            c3 = payload["C3"][scene]["results"]["by_dimension"][dim].get(value)
            if c3 is None:
                continue
            m0 = subgroup_result("M0", scene)["by_dimension"][dim].get(value)
            c1 = subgroup_result("C1-A", scene)["by_dimension"][dim].get(value)
            if m0 is not None and c1 is not None:
                assert c3["n"] == m0["n"] == c1["n"]
                entries.append((c3, m0, c1))
        if not entries:
            continue
        n = sum(x[0]["n"] for x in entries)
        row = {"n": n, "metrics": {}}
        for metric in METRICS:
            c = sum(x[0][metric] * x[0]["n"] for x in entries) / n
            row["metrics"][metric] = {"C3": c}
            for label, index in [("M0", 1), ("C1-A", 2)]:
                b = sum(x[index][metric] * x[0]["n"] for x in entries) / n
                row["metrics"][metric][label] = b
                row["metrics"][metric][f"C3_vs_{label}_delta_pct"] = (c / b - 1) * 100
        summary["subgroups"][dim][value] = row

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "c3_vs_controls.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
with (OUT / "c3_vs_controls.csv").open("w", newline="") as f:
    fields = ["dimension", "value", "n", "metric", "C3", "M0", "C1-A", "C3_vs_M0_delta_pct", "C3_vs_C1-A_delta_pct"]
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for dim, groups in summary["subgroups"].items():
        for value, row in groups.items():
            for metric in METRICS:
                m = row["metrics"][metric]
                w.writerow({"dimension": dim, "value": value, "n": row["n"], "metric": metric, **m})
print(json.dumps({"scenes": len(SCENES), "results": len(SCENES) * 3, "summary": str(OUT)}, ensure_ascii=False))
