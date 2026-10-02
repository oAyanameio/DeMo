from pathlib import Path
import csv
import json

ROOT = Path("outputs/c1a_subgroup_eval")
METRICS = ["minADE_K", "minFDE_K", "ADE@1", "FDE@1", "MR"]
SCENES = ["ETH-M", "HOTEL-M", "UNIV-M", "ZARA1-M", "ZARA2-M"]


def load(difficulty, variant):
    out = {}
    for scene in SCENES:
        paths = list((ROOT / difficulty / f"{variant}_{scene}").rglob("results.json"))
        if len(paths) != 1:
            raise RuntimeError((difficulty, variant, scene, paths))
        payload = json.loads(paths[0].read_text())
        out[scene] = payload["results"]
    return out


summary = {}
rows = []
for difficulty in ["easy", "hard"]:
    models = {variant: load(difficulty, variant) for variant in ["M0", "C1-A"]}
    # Re-evaluation must preserve each scene's original overall result.
    for variant in models:
        for scene in SCENES:
            if models[variant][scene]["overall"]["n"] <= 0:
                raise RuntimeError("empty result")
    summary[difficulty] = {}
    dimensions = sorted(models["M0"][SCENES[0]]["by_dimension"])
    for dimension in dimensions:
        values = sorted(
            set().union(*[
                set(models[v][s]["by_dimension"][dimension])
                for v in models for s in SCENES
            ]),
            key=lambda x: (x not in {"false", "true"}, int(x) if x.isdigit() else x),
        )
        summary[difficulty][dimension] = {}
        for value in values:
            scene_entries = []
            for scene in SCENES:
                m0 = models["M0"][scene]["by_dimension"][dimension].get(value)
                c1 = models["C1-A"][scene]["by_dimension"][dimension].get(value)
                if m0 is not None and c1 is not None:
                    if m0["n"] != c1["n"]:
                        raise RuntimeError((difficulty, dimension, value, scene, m0["n"], c1["n"]))
                    scene_entries.append((scene, int(m0["n"]), m0, c1))
            total_n = sum(x[1] for x in scene_entries)
            entry = {"n": total_n, "scene_count": len(scene_entries), "metrics": {}}
            for metric in METRICS:
                micro_m0 = sum(n * m0[metric] for _, n, m0, _ in scene_entries) / total_n
                micro_c1 = sum(n * c1[metric] for _, n, _, c1 in scene_entries) / total_n
                macro_m0 = sum(m0[metric] for _, _, m0, _ in scene_entries) / len(scene_entries)
                macro_c1 = sum(c1[metric] for _, _, _, c1 in scene_entries) / len(scene_entries)
                wins = sum(c1[metric] < m0[metric] for _, _, m0, c1 in scene_entries)
                metric_entry = {
                    "micro_m0": micro_m0,
                    "micro_c1a": micro_c1,
                    "micro_delta_pct": (micro_c1 / micro_m0 - 1) * 100,
                    "macro_m0": macro_m0,
                    "macro_c1a": macro_c1,
                    "macro_delta_pct": (macro_c1 / macro_m0 - 1) * 100,
                    "scene_wins": wins,
                }
                entry["metrics"][metric] = metric_entry
                rows.append({
                    "difficulty": difficulty,
                    "dimension": dimension,
                    "value": value,
                    "n": total_n,
                    "metric": metric,
                    **metric_entry,
                })
            entry["per_scene"] = {
                scene: {
                    "n": n,
                    "m0": {m: m0[m] for m in METRICS},
                    "c1a": {m: c1[m] for m in METRICS},
                }
                for scene, n, m0, c1 in scene_entries
            }
            summary[difficulty][dimension][value] = entry

out_dir = ROOT / "summary"
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "subgroup_comparison.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
with (out_dir / "subgroup_comparison.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

for difficulty in ["easy", "hard"]:
    print(f"\n## {difficulty.upper()}")
    for dimension in ["missing_count", "valid_count", "anchor_lag", "forecast_gap", "terminal_missing"]:
        print(f"\n[{dimension}]")
        for value, entry in summary[difficulty][dimension].items():
            fde = entry["metrics"]["minFDE_K"]
            mr = entry["metrics"]["MR"]
            print(
                value,
                "n=", entry["n"],
                f"macro_FDE={fde['macro_delta_pct']:+.2f}% ({fde['scene_wins']}/5 wins)",
                f"micro_FDE={fde['micro_delta_pct']:+.2f}%",
                f"macro_MR={mr['macro_delta_pct']:+.2f}% ({mr['scene_wins']}/5 wins)",
                f"micro_MR={mr['micro_delta_pct']:+.2f}%",
            )
