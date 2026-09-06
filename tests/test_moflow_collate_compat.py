"""任务一：moflow_ethucy_collate_fn 可选扩展字段兼容性测试。

背景：该 collate 被 MoFlow 原始路径（MoFlowEthUcyDataset/MoFlowSddDataset，
不产出扩展字段）与 SDD missing 路径（SddMissingDataset，产出全部扩展字段）
共享。v3/missing-aware 阶段把扩展字段改为无条件读取后，MoFlow 原始路径
第一个 batch 即 KeyError。

要求：
1. 原始 MoFlow 风格样本（无扩展字段）collate 成功；
2. SDD missing 风格样本（全部扩展字段）collate 成功且 shape 正确；
3. 部分样本有/无扩展字段 -> 清晰 ValueError（不得静默生成不一致 batch）；
4. padding actor 的 x_valid_mask=False；
5. x_key_valid_mask 由 x_valid_mask.any(-1) 生成。
"""
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.datamodule.missing_features import build_missing_features  # noqa: E402
from src.datamodule.moflow_ethucy_dataset import moflow_ethucy_collate_fn  # noqa: E402

OPTIONAL_EXT_KEYS = (
    "x_last_valid_angle", "x_last_valid_idx",
    "x_anchor_lag_steps", "x_forecast_gap_steps",
    "x_gap_steps", "x_prev_valid_gap", "x_motion_valid",
    "x_motion_run", "x_missing_summary",
)


def make_moflow_item(N=2, T_hist=8, T_fut=12, seed=0, with_ext=False):
    """合成一个 dataset item：MoFlow 风格（with_ext=False）或 SDD missing 风格。"""
    g = torch.Generator().manual_seed(seed)
    pos = torch.randn(N, T_hist, 2, generator=g)
    fut = torch.randn(N, T_fut, 2, generator=g)
    item = {
        "x_positions_diff": pos[:, 1:] - pos[:, :-1],
        "x_attr": torch.zeros(N, 3, dtype=torch.uint8),
        "x_positions": pos,
        "x_centers": pos[:, -1],
        "x_angles": torch.randn(N, T_hist, generator=g),
        "x_velocity": torch.rand(N, T_hist, generator=g),
        "x_velocity_diff": torch.randn(N, T_hist, generator=g),
        "target": fut,
        "target_diff": fut[:, 1:] - fut[:, :-1],
        "target_vel_diff": torch.randn(N, T_fut, generator=g),
        "target_mask": torch.ones(N, T_fut, dtype=torch.bool),
        "x_valid_mask": torch.ones(N, T_hist, dtype=torch.bool),
        "origin": torch.zeros(1, 2),
        "theta": torch.zeros(1),
        "scene_id": "ETH",
        "track_id": 0,
        "timestamp": torch.zeros(1),
    }
    if with_ext:
        miss = build_missing_features(torch.ones(N, T_hist, dtype=torch.bool))
        item["x_last_valid_angle"] = torch.zeros(N)
        item["x_last_valid_idx"] = torch.full((N,), T_hist - 1, dtype=torch.long)
        item["x_anchor_lag_steps"] = torch.zeros(N, dtype=torch.long)
        item["x_forecast_gap_steps"] = torch.ones(N, dtype=torch.long)
        for src, dst in [("gap_steps", "x_gap_steps"), ("prev_valid_gap", "x_prev_valid_gap"),
                         ("motion_valid", "x_motion_valid"), ("motion_run", "x_motion_run")]:
            item[dst] = miss[src].clone()
        item["x_missing_summary"] = miss["missing_summary"].clone()
    return item


class TestBasicCollateUnchanged:
    def test_moflow_style_without_ext_fields_succeeds(self):
        """原始 MoFlow 风格：无任何扩展字段，collate 必须恢复可用。"""
        batch = moflow_ethucy_collate_fn([make_moflow_item(N=2), make_moflow_item(N=4, seed=1)])
        assert batch["x_positions"].shape == (2, 4, 8, 2)
        assert batch["target"].shape == (2, 4, 12, 2)
        for k in OPTIONAL_EXT_KEYS:
            assert k not in batch, f"{k} should be absent for MoFlow-style batch"

    def test_base_fields_present_for_both_styles(self):
        for with_ext in (False, True):
            batch = moflow_ethucy_collate_fn([make_moflow_item(N=2, with_ext=with_ext)])
            for k in ("x_positions_diff", "x_attr", "x_positions", "x_centers", "x_angles",
                      "x_velocity", "x_velocity_diff", "target", "target_diff",
                      "target_vel_diff", "target_mask", "x_valid_mask"):
                assert k in batch, (with_ext, k)


class TestExtFieldsCollate:
    def test_sdd_missing_style_all_ext_fields_succeeds(self):
        """SDD missing 风格：全部扩展字段在场，collate 后 shape 正确。"""
        batch = moflow_ethucy_collate_fn(
            [make_moflow_item(N=1, with_ext=True), make_moflow_item(N=1, with_ext=True, seed=1)])
        for k in ("x_gap_steps", "x_prev_valid_gap", "x_motion_valid", "x_motion_run"):
            assert batch[k].shape == (2, 1, 8), k
        assert batch["x_missing_summary"].shape == (2, 1, 6)
        assert batch["x_last_valid_angle"].shape == (2, 1)
        assert batch["x_last_valid_idx"].shape == (2, 1)
        assert batch["x_anchor_lag_steps"].shape == (2, 1)
        assert batch["x_forecast_gap_steps"].shape == (2, 1)

    def test_partial_ext_fields_raises_value_error(self):
        """部分样本有扩展字段 -> ValueError（不允许静默不一致 batch）。
        报错在第一个不一致的可选键（x_last_valid_angle）触发，信息含字段名。"""
        mixed = [make_moflow_item(N=2, with_ext=False),
                 make_moflow_item(N=2, with_ext=True, seed=1)]
        with pytest.raises(ValueError, match="x_last_valid_angle"):
            moflow_ethucy_collate_fn(mixed)
        # 只混合单个扩展字段同样报错
        a = make_moflow_item(N=2)
        b = make_moflow_item(N=2, seed=1)
        b["x_gap_steps"] = torch.zeros(2, 8)
        with pytest.raises(ValueError, match="x_gap_steps"):
            moflow_ethucy_collate_fn([a, b])

    def test_ext_fields_padded_across_actor_dim(self):
        """不同 N 的样本 padding 后扩展字段 shape 一致；padding actor 全 0。"""
        batch = moflow_ethucy_collate_fn(
            [make_moflow_item(N=1, with_ext=True), make_moflow_item(N=3, with_ext=True, seed=1)])
        assert batch["x_gap_steps"].shape == (2, 3, 8)
        assert torch.equal(batch["x_gap_steps"][0, 1:], torch.zeros(2, 8))
        assert torch.equal(batch["x_missing_summary"][0, 1:], torch.zeros(2, 6))


class TestMaskSemantics:
    def test_padding_actor_x_valid_mask_false(self):
        """较短样本 padding 出的 actor 其 x_valid_mask 必须为 False。"""
        batch = moflow_ethucy_collate_fn(
            [make_moflow_item(N=1), make_moflow_item(N=3, seed=1)])
        assert not batch["x_valid_mask"][0, 1:].any()
        assert batch["x_valid_mask"][1].all()

    def test_key_valid_mask_from_any(self):
        """x_key_valid_mask == x_valid_mask.any(-1)。"""
        batch = moflow_ethucy_collate_fn(
            [make_moflow_item(N=1), make_moflow_item(N=3, seed=1)])
        assert torch.equal(batch["x_key_valid_mask"], batch["x_valid_mask"].any(-1))


@pytest.mark.skipif(not Path("/home/lbh/MoFlow/data/eth_ucy/original/eth").exists(),
                    reason="MoFlow ETH/UCY raw pkl not available")
class TestRealMoFlowData:
    def test_real_moflow_ethucy_batch(self):
        from src.datamodule.moflow_ethucy_dataset import MoFlowEthUcyDataset
        ds = MoFlowEthUcyDataset(data_root="/home/lbh/MoFlow/data/eth_ucy/original",
                                 subset="eth", split="test")
        batch = moflow_ethucy_collate_fn([ds[0], ds[1]])
        assert batch["x_positions"].shape[0] == 2
        for k in OPTIONAL_EXT_KEYS:
            assert k not in batch

    def test_real_moflow_sdd_batch(self):
        from src.datamodule.moflow_sdd_dataset import MoFlowSddDataset
        ds = MoFlowSddDataset(data_root="data/sdd", split="test")
        batch = moflow_ethucy_collate_fn([ds[0], ds[1]])
        assert batch["x_positions"].shape[0] == 2
        for k in OPTIONAL_EXT_KEYS:
            assert k not in batch


@pytest.mark.skipif(not Path("/home/lbh/DeMo/data/SDD_missing_v2_high/random_block6").exists(),
                    reason="SDD missing v2 data not available")
class TestRealSddMissingData:
    def test_real_sdd_missing_batch_keeps_all_ext_fields(self):
        from src.datamodule.sdd_missing_dataset import SddMissingDataset
        ds = SddMissingDataset("data/SDD_missing_v2_high", "random_block6", "test")
        batch = moflow_ethucy_collate_fn([ds[0], ds[1]])
        for k in OPTIONAL_EXT_KEYS:
            assert k in batch, k
        assert batch["x_gap_steps"].shape[1] == 1  # SDD 单 actor
        assert bool(torch.isfinite(batch["x_missing_summary"]).all())
