"""任务1：mask-only 缺失特征纯函数单元测试（方案 §2.3/§2.4/任务1.2）。

覆盖：
- complete 历史的中性约定
- block6（v2 协议）逐步特征与摘要
- random_fixed3 的 missing_rate
- 前缀缺失 / 中间缺口 / 交替可见
- v1/v2 帧恒可见条件 x_gap_steps[..., 6:8] == 0
- 非法输入（非 2D、非 bool、T<2）抛 ValueError
"""
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.datamodule.missing_features import build_missing_features  # noqa: E402


class TestNeutralCompleteHistory:
    def test_complete_history_features_are_neutral(self):
        mask = torch.ones(1, 8, dtype=torch.bool)
        out = build_missing_features(mask)
        assert torch.equal(out["gap_steps"], torch.zeros(1, 8))
        assert torch.equal(out["prev_valid_gap"], torch.tensor([[0., 1., 1., 1., 1., 1., 1., 1.]]))
        assert torch.equal(out["motion_valid"], torch.tensor([[0., 1., 1., 1., 1., 1., 1., 1.]]))
        assert torch.equal(out["motion_run"], torch.tensor([[0., 1., 2., 3., 4., 5., 6., 7.]]))
        assert torch.allclose(
            out["missing_summary"],
            torch.tensor([[0., 0., 0., 1., 1., 0.]]),
        )


class TestV2ProtocolConditions:
    def test_block6_matches_v2_protocol(self):
        mask = torch.tensor([[False, False, False, False, False, False, True, True]])
        out = build_missing_features(mask)
        assert torch.equal(out["gap_steps"], torch.tensor([[1., 2., 3., 4., 5., 6., 0., 0.]]))
        assert torch.equal(out["prev_valid_gap"], torch.tensor([[0., 0., 0., 0., 0., 0., 0., 1.]]))
        assert torch.equal(out["motion_valid"], torch.tensor([[0., 0., 0., 0., 0., 0., 0., 1.]]))
        assert torch.equal(out["motion_run"], torch.tensor([[0., 0., 0., 0., 0., 0., 0., 1.]]))
        assert torch.allclose(
            out["missing_summary"],
            torch.tensor([[0.75, 0.75, 0.75, 1 / 7, 1 / 7, 21 / 64]]),
        )

    def test_random_fixed3_missing_rate(self):
        # fixed3：前 6 帧中挖 3 帧（帧 6/7 受保护），missing_rate = 3/8
        mask = torch.tensor([[False, True, False, True, False, True, True, True]])
        out = build_missing_features(mask)
        assert torch.allclose(out["missing_summary"][0, 0], torch.tensor(3 / 8))
        # v1/v2 协议：帧 6/7 恒可见 → gap 为 0
        assert torch.equal(out["gap_steps"][0, 6:8], torch.zeros(2))

    def test_middle_gap(self):
        # 帧 2-4 缺，前后可见
        mask = torch.tensor([[True, True, False, False, False, True, True, True]])
        out = build_missing_features(mask)
        assert torch.equal(
            out["gap_steps"],
            torch.tensor([[0., 0., 1., 2., 3., 0., 0., 0.]]),
        )
        assert torch.equal(
            out["prev_valid_gap"],
            torch.tensor([[0., 1., 0., 0., 0., 4., 1., 1.]]),
        )
        assert torch.equal(
            out["motion_run"],
            torch.tensor([[0., 1., 0., 0., 0., 0., 1., 2.]]),
        )
        # longest_gap = 3, prefix = 0, valid_motion_rate = 3/7 (t=1,6,7 有效位移)
        assert torch.allclose(
            out["missing_summary"],
            torch.tensor([[3 / 8, 3 / 8, 0., 3 / 7, 2 / 7, 6 / 64]]),
        )

    def test_prefix_missing_with_later_valid(self):
        mask = torch.tensor([[False, False, True, True, True, True, True, True]])
        out = build_missing_features(mask)
        assert torch.equal(out["gap_steps"], torch.tensor([[1., 2., 0., 0., 0., 0., 0., 0.]]))
        assert torch.allclose(out["missing_summary"][0, 2], torch.tensor(2 / 8))  # prefix
        # 首 2 帧缺失：帧 3 是第一个有效位移步（run 从 1 起累计）
        assert torch.equal(out["motion_run"], torch.tensor([[0., 0., 0., 1., 2., 3., 4., 5.]]))

    def test_single_visible_frame(self):
        # 仅 1 帧可见（v3 uniform_hard m=7 极端情形）
        mask = torch.tensor([[False] * 7 + [True]])
        out = build_missing_features(mask)
        assert torch.equal(out["gap_steps"], torch.tensor([[1., 2., 3., 4., 5., 6., 7., 0.]]))
        assert torch.equal(out["motion_valid"].abs().sum(), torch.tensor(0.0))
        assert torch.equal(out["motion_run"].abs().sum(), torch.tensor(0.0))
        assert torch.allclose(
            out["missing_summary"],
            torch.tensor([[7 / 8, 7 / 8, 7 / 8, 0., 0., 28 / 64]]),
        )

    def test_all_missing_row_is_finite(self):
        # 全缺 actor（key_valid=False，被 attention 排除）：输出必须有限
        mask = torch.zeros(1, 8, dtype=torch.bool)
        out = build_missing_features(mask)
        for v in out.values():
            assert torch.isfinite(v).all()

    def test_alternating_mask(self):
        mask = torch.tensor([[True, False, True, False, True, False, True, True]])
        out = build_missing_features(mask)
        assert torch.equal(
            out["gap_steps"],
            torch.tensor([[0., 1., 0., 1., 0., 1., 0., 0.]]),
        )
        assert torch.equal(
            out["motion_valid"],
            torch.tensor([[0., 0., 0., 0., 0., 0., 0., 1.]]),
        )


class TestBatchAndMultiActor:
    def test_multi_actor_batch(self):
        masks = torch.stack(
            [
                torch.ones(8, dtype=torch.bool),
                torch.tensor([False] * 6 + [True, True]),
                torch.zeros(8, dtype=torch.bool),
            ]
        )  # [3, 8]
        out = build_missing_features(masks)
        assert out["gap_steps"].shape == (3, 8)
        assert out["missing_summary"].shape == (3, 6)
        # actor 0 中性
        assert torch.allclose(out["missing_summary"][0], torch.tensor([0., 0., 0., 1., 1., 0.]))
        # actor 1 == block6
        assert torch.allclose(
            out["missing_summary"][1],
            torch.tensor([0.75, 0.75, 0.75, 1 / 7, 1 / 7, 21 / 64]),
        )
        assert torch.isfinite(out["missing_summary"]).all()

    def test_dtype_is_float(self):
        out = build_missing_features(torch.ones(2, 8, dtype=torch.bool))
        for k in ("gap_steps", "prev_valid_gap", "motion_valid", "motion_run", "missing_summary"):
            assert out[k].dtype == torch.float32, k


class TestInvalidInputs:
    def test_rejects_1d_mask(self):
        with pytest.raises(ValueError):
            build_missing_features(torch.ones(8, dtype=torch.bool))

    def test_rejects_3d_mask(self):
        with pytest.raises(ValueError):
            build_missing_features(torch.ones(2, 8, 8, dtype=torch.bool))

    def test_rejects_non_bool(self):
        with pytest.raises(ValueError):
            build_missing_features(torch.ones(2, 8, dtype=torch.float32))

    def test_rejects_short_window(self):
        with pytest.raises(ValueError):
            build_missing_features(torch.ones(2, 1, dtype=torch.bool))
