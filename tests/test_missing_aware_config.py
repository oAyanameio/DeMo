"""任务四/五：Missing-Aware Hydra 配置组合与 M0/M1/M2 实例化测试。

只做配置组合与 Trainer instantiate，不启动训练。
模型实例化需要 DeMo conda 环境（CUDA Mamba），在 DeMo 环境下运行。
"""
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import hydra  # noqa: E402
from hydra.utils import instantiate  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

CONF_DIR = "../conf"  # 相对本测试文件（hydra.initialize 要求相对路径）


def compose(name, overrides=None):
    with hydra.initialize(config_path=CONF_DIR, version_base=None, job_name="test"):
        return hydra.compose(config_name=name, overrides=overrides or [])


def make_trainer(overrides, config="config_missing_aware_ethucy"):
    """compose 顶层配置并按 hydra 真实路径 instantiate Trainer。"""
    cfg = compose(config, overrides)
    return instantiate(cfg.model.target)


M0 = ["model.target.model.use_observation_features=false",
      "model.target.model.use_missing_summary=false"]
M1 = ["model.target.model.use_observation_features=true",
      "model.target.model.use_missing_summary=false"]
M2 = ["model.target.model.use_observation_features=true",
      "model.target.model.use_missing_summary=true"]


class TestTopLevelCompose:
    def test_ethucy_compose(self):
        cfg = compose("config_missing_aware_ethucy")
        assert cfg.model.target.model.num_modes == 6
        assert cfg.monitor == "val_minFDE6"
        assert cfg.datamodule.target._target_.endswith("EthUcyBenchmarkDataModule")

    def test_sdd_compose(self):
        cfg = compose("config_missing_aware_sdd")
        assert cfg.model.target.model.num_modes == 20
        assert cfg.monitor == "val_minFDE20"
        assert cfg.datamodule.target._target_.endswith("SddMissingDataModule")

    def test_shared_model_params(self):
        e = compose("config_missing_aware_ethucy").model.target.model
        s = compose("config_missing_aware_sdd").model.target.model
        for f in ("embed_dim", "future_steps", "num_heads", "mlp_ratio", "qkv_bias",
                  "drop_path", "num_actor_types", "dt", "obs_len", "bimamba",
                  "use_observation_features", "use_missing_summary", "use_gap_condition"):
            assert getattr(e, f) == getattr(s, f), f
        assert e.num_modes == 6 and s.num_modes == 20  # 唯一允许的差异


class TestModelInstantiation:
    def test_m0(self):
        t = make_trainer(M0)
        assert t.net.hist_embed_mlp[0].in_features == 4
        assert not hasattr(t.net, "missing_summary_embed")

    def test_m1(self):
        t = make_trainer(M1)
        assert t.net.hist_embed_mlp[0].in_features == 8
        assert not hasattr(t.net, "missing_summary_embed")

    def test_m2(self):
        t = make_trainer(M2)
        assert t.net.hist_embed_mlp[0].in_features == 8
        assert hasattr(t.net, "missing_summary_embed")

    def test_m2_sdd(self):
        t = make_trainer(M2, config="config_missing_aware_sdd")
        assert t.net.hist_embed_mlp[0].in_features == 8
        assert hasattr(t.net, "missing_summary_embed")
        assert t.net.num_modes == 20 if hasattr(t.net, "num_modes") else True

    def test_default_is_m0(self):
        t = make_trainer([])
        assert t.net.hist_embed_mlp[0].in_features == 4
        assert not hasattr(t.net, "missing_summary_embed")

    def test_param_counts(self):
        counts = {}
        for tag, ov in (("M0", M0), ("M1", M1), ("M2", M2)):
            t = make_trainer(ov)
            counts[tag] = sum(p.numel() for p in t.net.parameters())
        print(f"param counts: {counts}")
        assert counts["M1"] > counts["M0"]
        assert counts["M2"] > counts["M1"]

    def test_gap_condition_independent_override(self):
        t = make_trainer(M2 + ["model.target.model.use_gap_condition=true"])
        assert hasattr(t.net, "gap_embed")
        t2 = make_trainer(["model.target.model.use_gap_condition=true"])
        assert hasattr(t2.net, "gap_embed")
        assert t2.net.hist_embed_mlp[0].in_features == 4  # gap_condition 不改输入维度


class TestForbiddenAndContract:
    def test_no_m3_m4_params_in_config(self):
        for name in ("config_missing_aware_ethucy", "config_missing_aware_sdd"):
            raw = (REPO / "conf" / f"{name}.yaml").read_text()
            for k in ("condition_state_query", "condition_mode_query", "condition_hybrid"):
                assert k not in raw, (name, k)
            m = (REPO / "conf" / "model" /
                 ("missing_aware_ethucy_model_forecast.yaml" if "ethucy" in name
                  else "missing_aware_sdd_model_forecast.yaml")).read_text()
            assert k not in m, (name, k)

    def test_model_contract_values(self):
        for name, modes in (("config_missing_aware_ethucy", 6),
                            ("config_missing_aware_sdd", 20)):
            m = compose(name).model.target.model
            assert m.future_steps == 12
            assert m.dt == 0.4
            assert m.obs_len == 8
            assert m.bimamba is True
            assert m.num_modes == modes
            assert m.use_observation_features is False
            assert m.use_missing_summary is False

    def test_override_changes_resolved_config(self):
        """override 真正生效（resolved config 中值改变），非仅命令行。"""
        cfg = compose("config_missing_aware_ethucy", M1)
        assert cfg.model.target.model.use_observation_features is True
        assert cfg.model.target.model.use_missing_summary is False
