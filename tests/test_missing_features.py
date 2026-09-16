"""build_missing_features（gap_steps）单元测试。"""

import torch

from src.datamodule.missing_features import build_missing_features


def _b(mask):
    return build_missing_features(torch.tensor(mask, dtype=torch.bool))["gap_steps"]


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
