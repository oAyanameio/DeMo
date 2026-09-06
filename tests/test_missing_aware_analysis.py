"""Missing-Aware 分析脚本测试：合成目录/JSON，不依赖真实实验输出。

覆盖任务书第六节分析测试 9 项要求。
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def load_module():
    spec = importlib.util.spec_from_file_location(
        "ma_analysis", REPO / "scripts" / "结果分析" / "analyze_missing_aware.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


AN = load_module()
FOLDS = ["ETH", "HOTEL", "UNIV", "ZARA1", "ZARA2"]
METRICS = ["test_minADE6", "test_minFDE6", "test_MR", "test_b-minFDE6"]
VARIANT_FLAGS = {
    "M0_base": (False, False),
    "M1_obs": (True, False),
    "M2_history": (True, True),
}


def make_fold_row(variant, condition, seed, fold, uof, ums, epochs=100, batch=64,
                  **overrides):
    row = {
        "dataset": "ETHUCY", "protocol": "train_adapt", "variant": variant,
        "condition": condition, "seed": seed, "fold": fold,
        "config_name": "config_missing_aware_ethucy",
        "use_observation_features": uof, "use_missing_summary": ums,
        "checkpoint_epoch": 42, "checkpoint_path": f".../fold_{fold}/train/...",
        "val_minFDE6": 0.9, "epochs": epochs, "batch_size": batch,
        "precision": "bf16", "status": "ok",
        "test_minADE1": 0.5, "test_minFDE1": 0.9,
    }
    # M0/M1/M2 数值递减（模拟 M2 最优），逐 fold 加噪声由调用方传 overrides
    base = {"test_minADE6": 0.60, "test_minFDE6": 1.20, "test_MR": 0.12,
            "test_b-minFDE6": 2.0}
    offset = {"M0_base": 0.10, "M1_obs": 0.03, "M2_history": 0.0}
    for k, v in base.items():
        row[k] = round(v + offset[variant] + FOLDS.index(fold) * 0.01, 4)
    row.update(overrides)
    return row


def build_ethucy_tree(root, variants=("M0_base", "M1_obs", "M2_history"),
                      condition="random_fixed4_ng", seed=2024, epochs=100,
                      folds=FOLDS, statuses=None, mutate=None):
    """构造合成 ETH/UCY 结果树；mutate(row, variant) 可篡改字段。"""
    for v in variants:
        uof, ums = VARIANT_FLAGS[v]
        exp = root / v / condition / f"seed_{seed}"
        rows = []
        for i, f in enumerate(folds):
            st = (statuses or {}).get(f, "ok")
            r = make_fold_row(v, condition, seed, f, uof, ums, epochs=epochs)
            r["status"] = st
            if mutate:
                mutate(r, v)
            rows.append(r)
        n_ok = sum(1 for r in rows if r["status"] == "ok")
        summary = {
            "dataset": "ETHUCY", "protocol": "train_adapt", "variant": v,
            "condition": condition, "seed": seed, "rows": rows,
            "folds_ok": n_ok, "folds_failed": len(rows) - n_ok,
            "status": "complete" if n_ok == 5 else "incomplete",
        }
        exp.mkdir(parents=True, exist_ok=True)
        (exp / "results.json").write_text(json.dumps(summary, ensure_ascii=False))
    return root


def build_sdd_tree(root, variants=("M0_base", "M1_obs", "M2_history"),
                   condition="random_block6_ng", seeds=(2024,)):
    for v in variants:
        uof, ums = VARIANT_FLAGS[v]
        for seed in seeds:
            exp = root / v / condition / f"seed_{seed}"
            row = {
                "dataset": "SDD", "protocol": "train_adapt", "variant": v,
                "condition": condition, "seed": seed,
                "use_observation_features": uof, "use_missing_summary": ums,
                "checkpoint_epoch": 50, "val_minFDE20": 15.0,
                "epochs": 100, "batch_size": 64, "status": "ok",
                "test_minADE1": 8.0, "test_minFDE1": 15.0,
                "test_minADE20": 9.0 + 0.5 * VARIANT_FLAGS[v][0],
                "test_minFDE20": 16.0 + 0.5 * VARIANT_FLAGS[v][0]
                                 + 0.3 * VARIANT_FLAGS[v][1],
                "test_MR": 0.6, "test_b-minFDE20": 20.0,
            }
            exp.mkdir(parents=True, exist_ok=True)
            (exp / "result.json").write_text(json.dumps(row, ensure_ascii=False))
    return root


def make_an_args(tmp_path, dataset="ethucy", conditions=("random_fixed4_ng",),
                 seeds=(2024,), variants=("M0_base", "M1_obs", "M2_history")):
    import argparse as _ap
    ns = _ap.Namespace(
        input_root=str(tmp_path), dataset=dataset, protocol="train_adapt",
        conditions=list(conditions), variants=list(variants),
        seeds=list(seeds), output_dir=str(tmp_path / "analysis"))
    return ns


class TestNormalAnalysis:
    def test_load_and_pair_ethucy(self, tmp_path):
        build_ethucy_tree(tmp_path)
        args = make_an_args(tmp_path)
        res = AN.analyze_ethucy(args)
        cb = res["conditions"]["random_fixed4_ng"]
        # 三个 variant 都在
        assert set(cb["variants"]) == {"M0_base", "M1_obs", "M2_history"}
        pairs = cb["pairs"]
        assert "M1_obs - M0_base" in pairs and "M2_history - M0_base" in pairs

    def test_paired_diffs_correct(self, tmp_path):
        build_ethucy_tree(tmp_path)
        args = make_an_args(tmp_path)
        res = AN.analyze_ethucy(args)
        pb = res["conditions"]["random_fixed4_ng"]["pairs"]["M1_obs - M0_base"]
        st = pb["test_minFDE6"]
        # 每个 variant 内 offset(M1)-offset(M0) = -0.07，逐 fold 相同
        assert st["n"] == 5
        assert abs(st["diff_mean"] - (-0.07)) < 1e-6
        assert abs(st["diff_std"]) < 1e-6
        assert all(abs(d["diff"] - (-0.07)) < 1e-6 for d in st["per_fold"])
        # 相对变化：base_mean = 1.30 -> -0.07/1.30
        assert abs(st["rel_change_pct"] - (-0.07 / 1.30 * 100)) < 0.1
        assert "bootstrap_ci95" in st

    def test_three_outputs_generated(self, tmp_path):
        build_ethucy_tree(tmp_path)
        args = make_an_args(tmp_path)
        res = AN.analyze_ethucy(args)
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        AN.write_md(res, out / "summary.md")
        AN.write_csv(res, out / "summary.csv")
        (out / "summary.json").write_text(json.dumps(res, ensure_ascii=False))
        assert (out / "summary.json").exists() and (out / "summary.csv").exists()
        md = (out / "summary.md").read_text()
        for token in ("train_adapt", "ETHUCY", "random_fixed4_ng",
                      "M1_obs - M0_base", "一句话结论"):
            assert token in md, token
        csv_text = (out / "summary.csv").read_text()
        assert "test_minFDE6" in csv_text and "M2_history - M0_base" in csv_text


class TestIntegrityRejection:
    def test_reject_missing_fold(self, tmp_path):
        build_ethucy_tree(tmp_path, folds=FOLDS[:4])  # 只 4 折
        args = make_an_args(tmp_path)
        with pytest.raises(AN.IntegrityError, match="fold"):
            AN.analyze_ethucy(args)

    def test_reject_duplicate_fold(self, tmp_path):
        build_ethucy_tree(tmp_path, folds=FOLDS[:-1] + ["ETH"])  # ETH 出现两次
        args = make_an_args(tmp_path)
        with pytest.raises(AN.IntegrityError, match="重复"):
            AN.analyze_ethucy(args)

    def test_reject_bad_status(self, tmp_path):
        build_ethucy_tree(tmp_path, statuses={"ZARA2": "eval_failed_rc1"})
        args = make_an_args(tmp_path)
        with pytest.raises(AN.IntegrityError, match="status"):
            AN.analyze_ethucy(args)

    def test_reject_variant_flag_mismatch(self, tmp_path):
        def mutate(row, variant):
            if variant == "M1_obs":
                row["use_missing_summary"] = True  # M1 不应有 summary
        build_ethucy_tree(tmp_path, mutate=mutate)
        args = make_an_args(tmp_path)
        with pytest.raises(AN.IntegrityError, match="开关"):
            AN.analyze_ethucy(args)

    def test_reject_budget_mismatch(self, tmp_path):
        def mutate(row, variant):
            if variant == "M2_history":
                row["epochs"] = 50  # 训练预算不同
        build_ethucy_tree(tmp_path, mutate=mutate)
        args = make_an_args(tmp_path)
        with pytest.raises(AN.IntegrityError, match="预算"):
            AN.analyze_ethucy(args)

    def test_reject_missing_metric(self, tmp_path):
        def mutate(row, variant):
            if variant == "M0_base":
                row.pop("test_MR")
        build_ethucy_tree(tmp_path, mutate=mutate)
        args = make_an_args(tmp_path)
        with pytest.raises(AN.IntegrityError, match="test_MR"):
            AN.analyze_ethucy(args)


class TestSddAnalysis:
    def test_sdd_single_seed_no_significance(self, tmp_path):
        build_sdd_tree(tmp_path, seeds=(2024,))
        args = make_an_args(tmp_path, dataset="sdd", conditions=("random_block6_ng",))
        res = AN.analyze_sdd(args)
        pb = res["conditions"]["random_block6_ng"]["pairs"]["M2_history - M0_base"]
        st = pb["test_minFDE20"]
        assert st["n"] == 1
        assert "t_stat" not in st and "bootstrap_ci95" not in st \
            and "p_value_norm_approx" not in st
        # 绝对差与相对变化仍在
        assert st["diff_mean"] == pytest.approx(16.0 + 0.5 + 0.3 - 16.0, abs=1e-6)

    def test_sdd_multi_seed_allows_stats(self, tmp_path):
        build_sdd_tree(tmp_path, seeds=(2024, 2025))
        args = make_an_args(tmp_path, dataset="sdd", conditions=("random_block6_ng",),
                            seeds=(2024, 2025))
        res = AN.analyze_sdd(args)
        st = res["conditions"]["random_block6_ng"]["pairs"]["M2_history - M0_base"]["test_minFDE20"]
        assert st["n"] == 2
        assert "bootstrap_ci95" in st

    def test_sdd_rejects_flag_mismatch(self, tmp_path):
        root = build_sdd_tree(tmp_path, seeds=(2024,))
        p = root / "M1_obs" / "random_block6_ng" / "seed_2024" / "result.json"
        r = json.loads(p.read_text())
        r["use_missing_summary"] = True
        p.write_text(json.dumps(r))
        args = make_an_args(tmp_path, dataset="sdd", conditions=("random_block6_ng",))
        with pytest.raises(AN.IntegrityError, match="开关"):
            AN.analyze_sdd(args)


class TestStats:
    def test_paired_t_and_bootstrap(self):
        diffs = [0.1, 0.2, 0.3, 0.4, 0.5]
        t = AN.paired_t(diffs)
        # mean=0.3, sample std=sqrt(0.1/4)=0.15811, se=0.15811/sqrt(5)=0.07071 -> t=4.2426
        assert t == pytest.approx(4.2426, rel=1e-3)
        lo, hi = AN.bootstrap_ci(diffs, n_boot=1000)
        assert lo <= 0.3 <= hi

    def test_bootstrap_seed_deterministic(self):
        d = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert AN.bootstrap_ci(d) == AN.bootstrap_ci(d)
