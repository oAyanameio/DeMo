"""Missing-Aware Runner 单元测试（不运行真实训练）。

通过 importlib 加载中文路径下的 runner 模块；子进程与文件系统均用
mock / tmp_path。覆盖任务书第六节 Runner 测试 10 项要求。
"""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]


def load_module(rel_path, name):
    spec = importlib.util.spec_from_file_location(name, REPO / rel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


RUNNER_ETH = load_module("scripts/训练与评估/run_missing_aware_ethucy.py", "ma_ethucy")
RUNNER_SDD = load_module("scripts/训练与评估/run_missing_aware_sdd.py", "ma_sdd")


class TestVariantMapping:
    def test_variant_flags_mapping(self):
        for mod in (RUNNER_ETH, RUNNER_SDD):
            assert mod.VARIANTS["M0_base"] == {"use_observation_features": False,
                                               "use_missing_summary": False}
            assert mod.VARIANTS["M1_obs"] == {"use_observation_features": True,
                                              "use_missing_summary": False}
            assert mod.VARIANTS["M2_history"] == {"use_observation_features": True,
                                                  "use_missing_summary": True}
            assert mod.EXPECTED["M0_base"] == (4, False)
            assert mod.EXPECTED["M1_obs"] == (8, False)
            assert mod.EXPECTED["M2_history"] == (8, True)

    def test_overrides_carry_variant_flags(self):
        for mod, extra in ((RUNNER_ETH, []), (RUNNER_SDD, [])):
            if mod is RUNNER_ETH:
                ov = mod.build_train_overrides("M2_history", "c", "/d", 2024,
                                               "ETH", 10, 32, 4, "bf16")
            else:
                ov = mod.build_train_overrides("M2_history", "c", "/d", 2024,
                                               10, 32, 4, "bf16")
            joined = " ".join(ov)
            assert "use_observation_features=true" in joined
            assert "use_missing_summary=true" in joined
            assert "use_gap_condition=false" in joined


class TestCommandConstruction:
    def test_ethucy_command_carries_data_root_and_fold(self):
        cmd = RUNNER_ETH.build_train_command(
            "M1_obs", "random_fixed4_ng", "data/ETHUCY_missing_v3_noguard/random_fixed4_ng",
            2024, "ETH", 100, 64, 16, "bf16", "/tmp/run")
        s = " ".join(cmd)
        assert "data_root=data/ETHUCY_missing_v3_noguard/random_fixed4_ng" in s
        assert "fold=ETH" in s
        assert "seed=2024" in s
        assert "model.target.model.use_observation_features=true" in s
        assert "model.target.model.use_missing_summary=false" in s
        assert "config_missing_aware_ethucy" in s
        assert "checkpoint=" not in s  # 默认从零训练

    def test_ethucy_eval_command(self):
        cmd = RUNNER_ETH.build_eval_command("M2_history", "c", "/d", 2024, "ZARA1", "bf16",
                                            "/tmp/best.ckpt", "/tmp/eval")
        s = " ".join(cmd)
        assert "fold=ZARA1" in s and "test=true" in s and "checkpoint=/tmp/best.ckpt" in s
        # eval 必须携带与 train 相同的模型开关，否则 checkpoint shape 不匹配
        assert "model.target.model.use_observation_features=true" in s
        assert "model.target.model.use_missing_summary=true" in s

    def test_sdd_command_carries_datamodule_target(self):
        cmd = RUNNER_SDD.build_train_command(
            "M2_history", "random_fixed4_ng", "data/SDD_missing_v3_noguard",
            2024, 100, 64, 16, "bf16", "/tmp/run")
        s = " ".join(cmd)
        assert "datamodule.target.condition=random_fixed4_ng" in s
        assert "datamodule.target.data_root=data/SDD_missing_v3_noguard" in s
        assert "config_missing_aware_sdd" in s

    def test_sdd_eval_command_carries_datamodule_target(self):
        cmd = RUNNER_SDD.build_eval_command("c", "/d", 2024, "bf16", "/tmp/b.ckpt", "/tmp/e")
        s = " ".join(cmd)
        assert "datamodule.target.condition=c" in s
        assert "datamodule.target.data_root=/d" in s
        assert "test=true" in s

    def test_resume_appends_checkpoint(self):
        cmd = RUNNER_ETH.build_train_command("M0_base", "c", "/d", 2024, "ETH", 10, 32, 4,
                                             "bf16", "/tmp/run", resume_ckpt="/tmp/x.ckpt")
        assert "checkpoint=/tmp/x.ckpt" in " ".join(cmd)


class TestOutputDir:
    def test_exp_dir_contains_variant_condition_seed(self):
        for mod in (RUNNER_ETH, RUNNER_SDD):
            d = mod.exp_dir_for("outputs/missing_aware/x/train_adapt",
                                "M1_obs", "random_block6_ng", 2024)
            parts = Path(d).parts
            assert "M1_obs" in parts and "random_block6_ng" in parts and "seed_2024" in parts


def make_args(mod, tmp_path, variant="M1_obs", condition="random_block6_ng",
              seed=2024, **kw):
    ns = SimpleNamespace(
        variant=variant, condition=condition, data_root=Path("/tmp/data"),
        output_root=str(tmp_path / "out"), seed=seed,
        folds=["ETH", "HOTEL", "UNIV", "ZARA1", "ZARA2"],
        epochs=100, batch_size=64, num_workers=16, precision="bf16",
        skip_train=False, resume_checkpoint=None, overwrite=False,
        gpu=0, _model_parameters=123)
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


class TestOverwrite:
    def test_complete_results_refused_without_overwrite(self, tmp_path, capsys):
        exp = RUNNER_ETH.exp_dir_for(tmp_path / "out", "M0_base", "c", 2024)
        exp.mkdir(parents=True)
        (exp / "results.json").write_text(json.dumps({"status": "complete"}))
        args = make_args(RUNNER_ETH, tmp_path, variant="M0_base", condition="c")
        with pytest.raises(SystemExit, match="REFUSE"):
            RUNNER_ETH.check_overwrite(exp, args)

    def test_overwrite_deletes_only_target_experiment(self, tmp_path):
        root = tmp_path / "out"
        target = RUNNER_ETH.exp_dir_for(root, "M0_base", "c", 2024)
        other_v = RUNNER_ETH.exp_dir_for(root, "M1_obs", "c", 2024)
        other_c = RUNNER_ETH.exp_dir_for(root, "M0_base", "other_cond", 2024)
        for d in (target, other_v, other_c):
            d.mkdir(parents=True)
            (d / "results.json").write_text(json.dumps({"status": "complete"}))
        args = make_args(RUNNER_ETH, tmp_path, variant="M0_base", condition="c", overwrite=True)
        RUNNER_ETH.check_overwrite(target, args)
        assert not target.exists()
        assert other_v.exists() and other_c.exists()

    def test_incomplete_results_allowed_without_overwrite(self, tmp_path):
        exp = RUNNER_ETH.exp_dir_for(tmp_path / "out", "M0_base", "c", 2024)
        exp.mkdir(parents=True)
        (exp / "results.json").write_text(json.dumps({"status": "incomplete"}))
        args = make_args(RUNNER_ETH, tmp_path, variant="M0_base", condition="c")
        RUNNER_ETH.check_overwrite(exp, args)  # 不抛异常
        assert exp.exists()


class TestResumeSafety:
    def test_m1_refuses_m0_checkpoint(self, tmp_path, monkeypatch):
        args = make_args(RUNNER_ETH, tmp_path, variant="M1_obs", condition="c")
        ckpt = tmp_path / "m0.ckpt"
        ckpt.write_bytes(b"x")
        exp = RUNNER_ETH.exp_dir_for(args.output_root, "M1_obs", "c", 2024)
        exp.mkdir(parents=True)
        inside = exp / "fold_ETH" / "train" / "checkpoints" / "epoch=1.ckpt"
        inside.parent.mkdir(parents=True)
        inside.write_bytes(b"x")
        monkeypatch.setattr(RUNNER_ETH, "load_ckpt_model_flags",
                            lambda p: {"use_observation_features": False,
                                       "use_missing_summary": False})
        with pytest.raises(SystemExit, match="RESUME REFUSED.*不一致"):
            RUNNER_ETH.verify_resume(inside, args, "ETH", exp / "fold_ETH")

    def test_missing_checkpoint_clear_error(self, tmp_path):
        args = make_args(RUNNER_ETH, tmp_path)
        with pytest.raises(SystemExit, match="不存在"):
            RUNNER_ETH.verify_resume(tmp_path / "nope.ckpt", args, "ETH", tmp_path)

    def test_checkpoint_outside_exp_root_refused(self, tmp_path, monkeypatch):
        args = make_args(RUNNER_ETH, tmp_path, variant="M2_history", condition="c")
        outside = tmp_path / "elsewhere" / "epoch=1.ckpt"
        outside.parent.mkdir(parents=True)
        outside.write_bytes(b"x")
        monkeypatch.setattr(RUNNER_ETH, "load_ckpt_model_flags",
                            lambda p: {"use_observation_features": True,
                                       "use_missing_summary": True})
        with pytest.raises(SystemExit, match="不在本实验目录内"):
            RUNNER_ETH.verify_resume(outside, args, "ETH", tmp_path)


class TestFailureHandling:
    def test_failed_meta_written(self, tmp_path):
        args = make_args(RUNNER_ETH, tmp_path, variant="M0_base", condition="c")
        exp = RUNNER_ETH.exp_dir_for(args.output_root, "M0_base", "c", 2024)
        exp.mkdir(parents=True)
        meta = RUNNER_ETH.write_meta(exp, args, "failed",
                                     {"failure_stage": "train", "return_code": 1})
        assert meta["status"] == "failed"
        assert meta["failure_stage"] == "train"
        on_disk = json.loads((exp / "experiment_meta.json").read_text())
        assert on_disk["status"] == "failed"
        assert on_disk["variant"] == "M0_base"
        assert on_disk["model_flags"] == {"use_observation_features": False,
                                          "use_missing_summary": False,
                                          "use_gap_condition": False}

    def test_train_failure_returns_failed_row(self, tmp_path, monkeypatch):
        args = make_args(RUNNER_ETH, tmp_path)
        monkeypatch.setattr(RUNNER_ETH, "sh", lambda cmd, log: 3)
        row = RUNNER_ETH.run_fold("ETH", args)
        assert row["status"] == "train_failed_rc3"

    def test_missing_checkpoint_reports_clearly(self, tmp_path, monkeypatch):
        args = make_args(RUNNER_ETH, tmp_path, skip_train=True)
        # 不创建 train run 目录 -> collect_best_checkpoint AssertionError
        row = RUNNER_ETH.run_fold("ETH", args)
        assert row["status"].startswith("checkpoint_missing")

    def test_incomplete_folds_not_complete(self, tmp_path):
        args = make_args(RUNNER_ETH, tmp_path)
        exp = RUNNER_ETH.exp_dir_for(args.output_root, "M0_base", "c", 2024)
        exp.mkdir(parents=True)
        rows = [{"fold": f, "status": "ok", "test_minFDE6": 1.0}
                for f in ["ETH", "HOTEL", "UNIV", "ZARA1"]]  # 缺 ZARA2
        summary = RUNNER_ETH.summarize(exp, rows, args)
        assert summary["status"] != "complete"
        assert summary["folds_ok"] == 4
        on_disk = json.loads((exp / "results.json").read_text())
        assert on_disk["status"] == "incomplete"

    def test_full_five_ok_folds_is_complete(self, tmp_path):
        args = make_args(RUNNER_ETH, tmp_path)
        exp = RUNNER_ETH.exp_dir_for(args.output_root, "M0_base", "c", 2024)
        exp.mkdir(parents=True)
        rows = [{"fold": f, "status": "ok", "test_minFDE6": 1.0} for f in
                RUNNER_ETH.FOLDS]
        summary = RUNNER_ETH.summarize(exp, rows, args)
        assert summary["status"] == "complete"

    def test_run_fold_prints_protocol_banner(self, tmp_path, monkeypatch, capsys):
        """启动前打印 variant/开关/维度/condition/data_root/seed/fold/config。"""
        args = make_args(RUNNER_ETH, tmp_path)
        monkeypatch.setattr(RUNNER_ETH, "sh", lambda cmd, log: 3)
        RUNNER_ETH.run_fold("ETH", args)
        out = capsys.readouterr().out
        for token in ("M1_obs", "use_observation_features", "hist_embed_mlp.in_features",
                      "random_block6_ng", "seed", "fold", "config_missing_aware_ethucy"):
            assert token in out, token


class TestSddRunnerSpecifics:
    def test_sdd_failure_meta(self, tmp_path):
        args = make_args(RUNNER_SDD, tmp_path, variant="M2_history", condition="c")
        exp = RUNNER_SDD.exp_dir_for(args.output_root, "M2_history", "c", 2024)
        exp.mkdir(parents=True)
        meta = RUNNER_SDD.write_meta(exp, args, "failed",
                                     {"failure_stage": "eval", "return_code": 1})
        assert meta["status"] == "failed"
        assert meta["dataset"] == "SDD"
        assert meta["checkpoint_monitor"] == "val_minFDE20"

    def test_sdd_train_failure_row(self, tmp_path, monkeypatch):
        args = make_args(RUNNER_SDD, tmp_path)
        monkeypatch.setattr(RUNNER_SDD, "sh", lambda cmd, log: 2)
        row, secs = RUNNER_SDD.run_experiment(args)
        assert row["status"] == "train_failed_rc2"
