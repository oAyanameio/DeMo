"""build_missing_features（gap_steps + per-actor missing_summary）单元测试。"""

import torch

from src.datamodule.missing_features import build_missing_features


def _b(mask):
    return build_missing_features(torch.tensor(mask, dtype=torch.bool))["gap_steps"]

def _s(mask):
    return build_missing_features(torch.tensor(mask, dtype=torch.bool))["missing_summary"]


class TestGapSteps:
    def test_complete_history_is_all_zero(self):
        out = _b([[1, 1, 1, 1, 1, 1, 1, 1]])
        assert torch.equal(out, torch.zeros(1, 8))

    def test_middle_gap_counts_steps(self):
        # t=3,4,5 缺失：gap 依次 1,2,3
        out = _b([[1, 1, 1, 0, 0, 0, 1, 1]])
        assert torch.equal(out, torch.tensor([[0., 0., 0., 1., 2., 3., 0., 0.]]))

    def test_prefix_missing_counts_from_start(self):
        # 窗口起点起全缺失：t=0 gap=1, t=1 gap=2（t+1 约定）
        out = _b([[0, 0, 1, 1, 1, 1, 1, 1]])
        assert torch.equal(out, torch.tensor([[1., 2., 0., 0., 0., 0., 0., 0.]]))

    def test_alternating_mask(self):
        out = _b([[1, 0, 1, 0, 1, 0, 1, 0]])
        assert torch.equal(out, torch.tensor([[0., 1., 0., 1., 0., 1., 0., 1.]]))

    def test_all_missing_row_is_finite(self):
        out = _b([[0, 0, 0, 0, 0, 0, 0, 0]])
        assert torch.isfinite(out).all()
        assert torch.equal(out, torch.arange(1, 9).float().unsqueeze(0))

    def test_multi_actor_independent(self):
        out = _b([
            [1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 1, 0, 1, 0, 1],
        ])
        assert out.shape == (2, 8)
        assert torch.equal(out[0], torch.zeros(8))
        assert torch.equal(out[1], torch.tensor([1., 2., 3., 0., 1., 0., 1., 0.]))


class TestValidation:
    def test_rejects_non_tensor(self):
        try:
            build_missing_features([[1, 1]])
        except ValueError:
            pass
        else:
            raise AssertionError("应拒绝非 Tensor 输入")

    def test_rejects_non_bool(self):
        try:
            build_missing_features(torch.ones(1, 8))
        except ValueError:
            pass
        else:
            raise AssertionError("应拒绝非 bool 输入")

    def test_rejects_short_window(self):
        try:
            build_missing_features(torch.ones(1, 1, dtype=torch.bool))
        except ValueError:
            pass
        else:
            raise AssertionError("应拒绝 T<2")


class TestSummary:
    """模块一 per-actor 摘要：6 维、[0,1] 有界、mask-only 语义正确。"""

    def test_shape_and_bounded(self):
        s = _s([[1, 1, 0, 0, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1, 1, 1]])
        assert s.shape == (2, 6)
        assert torch.isfinite(s).all()
        assert bool((s >= 0).all()) and bool((s <= 1).all())

    def test_complete_history_all_zero_except_valid_rate(self):
        # 完整历史：missing_rate/longest/prefix/tail/gap_area 全 0，valid_rate=1
        s = _s([[1] * 8])
        assert torch.allclose(s, torch.tensor([[0., 0., 0., 0., 0., 1.]]))

    def test_hand_computed_values(self):
        # mask 1 1 0 0 1 1 1 0：缺 3 帧，最长连缺 2，前缀 0，
        # 末帧缺失 tail=7-6=1 -> 1/7，gap_steps=[0,0,1,2,0,0,0,1] sum=4 -> 4/64
        s = _s([[1, 1, 0, 0, 1, 1, 1, 0]])[0]
        assert torch.isclose(s[0], torch.tensor(3 / 8))   # missing_rate
        assert torch.isclose(s[1], torch.tensor(2 / 8))   # longest_gap
        assert torch.isclose(s[2], torch.tensor(0.0))     # prefix
        assert torch.isclose(s[3], torch.tensor(1 / 7))   # tail_gap
        assert torch.isclose(s[4], torch.tensor(4 / 64))  # gap_area
        assert torch.isclose(s[5], torch.tensor(5 / 8))   # valid_rate

    def test_prefix_missing(self):
        # 前 3 帧缺失：prefix=3/8；最后有效帧=7 -> tail=0
        s = _s([[0, 0, 0, 1, 1, 1, 1, 1]])[0]
        assert torch.isclose(s[2], torch.tensor(3 / 8))
        assert torch.isclose(s[3], torch.tensor(0.0))

    def test_gsm_dim_constant(self):
        from src.datamodule.missing_features import GSM_SUMMARY_DIM
        assert GSM_SUMMARY_DIM == 6
        assert _s([[1, 1]]).shape[1] == GSM_SUMMARY_DIM

    def test_all_missing_row_summary_finite_and_saturated(self):
        """验收1-全缺失行：真实数据保证至少一个有效帧，但纯函数须对
        全缺失输入（padding 极端情形）给出有限饱和值：prefix=1,
        missing_rate=1, valid_rate=0, longest=8/8, tail=7/7。"""
        s = _s([[0] * 8])[0]
        assert torch.isfinite(s).all()
        assert torch.isclose(s[0], torch.tensor(1.0))   # missing_rate
        assert torch.isclose(s[1], torch.tensor(1.0))   # longest_gap = 8/8
        assert torch.isclose(s[2], torch.tensor(1.0))   # prefix
        assert torch.isclose(s[3], torch.tensor(1.0))   # tail_gap = 7/7
        assert torch.isclose(s[5], torch.tensor(0.0))   # valid_rate
