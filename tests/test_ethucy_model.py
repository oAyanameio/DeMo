"""ETH/UCY 模型单元测试。

测试 actor-only ModelForecast 的初始化与前向传播。
"""

import pytest
import torch

from src.model.model_forecast import ModelForecast

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _make_dummy_batch_actor_only(
    B=2, N=10, obs_len=8, pred_len=12, device=DEVICE
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


def _make_model():
    return ModelForecast(
        embed_dim=128,
        future_steps=12,
        num_actor_types=1,
    ).to(DEVICE)


class TestModelForecastActorOnly:
    """测试 actor-only 模式。"""

    def test_model_init(self):
        model = _make_model()
        assert model.future_steps == 12
        assert model.num_actor_types == 1
        assert model.dt == 0.4
        # 模型不含任何地图/车道属性
        assert not hasattr(model, "use_map")
        assert not hasattr(model, "lane_embed")
        assert not hasattr(model, "lane_type_embed")

    def test_forward_shape(self):
        model = _make_model()
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
        # refine 输出
        assert out["new_y_hat"].shape == (2, 6, 12, 2)

    def test_forward_no_lane_access(self):
        """确认模型前向不访问 lane 字段。"""
        model = _make_model()
        model.eval()
        data = _make_dummy_batch_actor_only(B=2, N=10, obs_len=8, pred_len=12)

        # 不应该抛出 KeyError（batch 中无任何 lane_* 键）
        with torch.no_grad():
            out = model(data)

        assert out is not None

    def test_forward_single_agent(self):
        """N=1（无邻居）时前向仍可用。"""
        model = _make_model()
        model.eval()
        data = _make_dummy_batch_actor_only(B=2, N=1, obs_len=8, pred_len=12)

        with torch.no_grad():
            out = model(data)

        assert out["y_hat"].shape == (2, 6, 12, 2)
        assert out["y_hat_others"].shape[1] == 0
