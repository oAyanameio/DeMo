"""ETH/UCY 模型单元测试。

测试 ModelForecast 在 use_map=False 模式下的前向传播。
"""

import pytest
import torch

from src.model.model_forecast import ModelForecast


def _make_dummy_batch_actor_only(
    B=2, N=10, obs_len=8, pred_len=12, device="cuda" if __import__("torch").cuda.is_available() else "cpu"
):
    """构造 actor-only 模式的 dummy batch。"""
    data = {
        "x_positions_diff": torch.randn(B, N, obs_len, 2, device=device),
        "x_positions": torch.randn(B, N, obs_len, 2, device=device),
        "x_attr": torch.zeros(B, N, 3, dtype=torch.uint8, device=device),
        "x_centers": torch.randn(B, N, 2, device=device),
        "x_angles": torch.randn(B, N, obs_len, device=device),
        "x_velocity": torch.randn(B, N, obs_len, device=device),
        "x_velocity_diff": torch.randn(B, N, obs_len, device=device),
        "x_valid_mask": torch.ones(B, N, obs_len, dtype=torch.bool, device=device),
        "x_key_valid_mask": torch.ones(B, N, dtype=torch.bool, device=device),
        "target": torch.randn(B, N, pred_len, 2, device=device),
        "target_mask": torch.ones(B, N, pred_len, dtype=torch.bool, device=device),
        "origin": torch.zeros(B, 1, 2, device=device),
        "theta": torch.zeros(B, 1, device=device),
        "timestamp": torch.zeros(B, 1, device=device),
    }
    return data


class TestModelForecastActorOnly:
    """测试 use_map=False 模式。"""

    def test_model_init(self):
        model = ModelForecast(
            embed_dim=128,
            future_steps=12,
            use_map=False,
            num_actor_types=1,
        ).to(__import__("torch").device("cuda" if __import__("torch").cuda.is_available() else "cpu"))
        assert model.use_map is False
        assert model.future_steps == 12
        assert model.lane_embed is None
        assert model.lane_type_embed is None

    def test_forward_shape(self):
        model = ModelForecast(
            embed_dim=128,
            future_steps=12,
            use_map=False,
            num_actor_types=1,
        ).to(__import__("torch").device("cuda" if __import__("torch").cuda.is_available() else "cpu"))
        model.eval()
        data = _make_dummy_batch_actor_only(B=2, N=10, obs_len=8, pred_len=12)

        with torch.no_grad():
            out = model(data)

        # y_hat: [B, M, 12, 2]
        assert out["y_hat"].shape == (2, 6, 12, 2)
        # pi: [B, M]
        assert out["pi"].shape == (2, 6)
        # scal: [B, M, 12, 2]
        assert out["scal"].shape == (2, 6, 12, 2)

    def test_forward_no_lane_access(self):
        """确认 use_map=False 时模型不访问 lane 字段。"""
        model = ModelForecast(
            embed_dim=128,
            future_steps=12,
            use_map=False,
            num_actor_types=1,
        ).to(__import__("torch").device("cuda" if __import__("torch").cuda.is_available() else "cpu"))
        model.eval()
        data = _make_dummy_batch_actor_only(B=2, N=10, obs_len=8, pred_len=12)

        # 不应该抛出 KeyError
        with torch.no_grad():
            out = model(data)

        assert out is not None

    def test_av2_mode_unchanged(self):
        """确认 use_map=True (默认) 行为不变。"""
        model = ModelForecast(
            embed_dim=128,
            future_steps=60,
            use_map=True,
        ).to(__import__("torch").device("cuda" if __import__("torch").cuda.is_available() else "cpu"))
        assert model.use_map is True
        assert model.lane_embed is not None
        assert model.lane_type_embed is not None
        assert model.future_steps == 60