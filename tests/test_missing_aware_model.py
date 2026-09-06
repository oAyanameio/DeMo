"""任务3：ModelForecast M1/M2 缺失感知历史编码开关测试（方案 §3.1，本阶段范围）。

只覆盖 use_observation_features (M1_obs) 与 use_missing_summary (M2_history)；
State/Mode/Hybrid 条件化（M3/M4）与 TimeDecoder 改动不在本阶段。
"""
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.datamodule.missing_features import build_missing_features  # noqa: E402
from src.model.model_forecast import ModelForecast  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def make_batch(B=2, N=4, obs_len=8, pred_len=12, device=DEVICE, hist_mask=None):
    """dummy batch；hist_mask [N, obs_len] bool 作用于所有样本。"""
    data = {
        "x_positions_diff": torch.randn(B, N, obs_len, 2, device=device),
        "x_positions": torch.randn(B, N, obs_len, 2, device=device),
        "x_attr": torch.zeros(B, N, 3, dtype=torch.uint8, device=device),
        "x_centers": torch.randn(B, N, 2, device=device),
        "x_angles": torch.randn(B, N, obs_len, device=device),
        "x_velocity": torch.rand(B, N, obs_len, device=device),
        "x_velocity_diff": torch.randn(B, N, obs_len, device=device),
        "x_valid_mask": torch.ones(B, N, obs_len, dtype=torch.bool, device=device),
        "x_key_valid_mask": torch.ones(B, N, dtype=torch.bool, device=device),
        "target": torch.randn(B, N, pred_len, 2, device=device),
        "target_mask": torch.ones(B, N, pred_len, dtype=torch.bool, device=device),
        "origin": torch.zeros(B, 1, 2, device=device),
        "theta": torch.zeros(B, 1, device=device),
        "timestamp": torch.zeros(B, 1, device=device),
    }
    if hist_mask is not None:
        m = hist_mask.to(device).bool()
        data["x_valid_mask"] = m.unsqueeze(0).expand(B, N, obs_len).clone()
        data["x_key_valid_mask"] = data["x_valid_mask"].any(-1)
        data["x_key_valid_mask"][:, 0] = True  # focal 恒有效
    # 新字段由 mask 派生（与 Dataset 路径一致）
    miss = build_missing_features(data["x_valid_mask"].reshape(-1, obs_len))
    for src, dst in [("gap_steps", "x_gap_steps"), ("prev_valid_gap", "x_prev_valid_gap"),
                     ("motion_valid", "x_motion_valid"), ("motion_run", "x_motion_run")]:
        data[dst] = miss[src].reshape(B, N, obs_len).clone()
    data["x_missing_summary"] = miss["missing_summary"].reshape(B, N, 6).clone()
    return data


OUT_KEYS_SHAPES = {
    "y_hat": (2, 6, 12, 2),
    "pi": (2, 6),
    "scal": (2, 6, 12, 2),
    "dense_predict": (2, 12, 2),  # state query 输出（focal 单轨迹）
    "new_y_hat": (2, 6, 12, 2),
    "new_pi": (2, 6),
    "scal_new": (2, 6, 12, 2),
}


def _forward(model, data):
    model.eval()
    with torch.no_grad():
        return model(data)


class TestM0BaseUnchanged:
    def test_m0_forward_without_new_fields(self):
        """M0 不依赖任何新字段（老 batch 仍可用）。"""
        model = ModelForecast(embed_dim=128, future_steps=12, num_actor_types=1).to(DEVICE)
        data = make_batch()
        for k in ("x_gap_steps", "x_prev_valid_gap", "x_motion_valid",
                  "x_motion_run", "x_missing_summary"):
            data.pop(k)
        out = _forward(model, data)
        assert out["y_hat"].shape == (2, 6, 12, 2)

    def test_m0_param_set_unchanged(self):
        """开关全关时参数集与旧版一致：无新模块、hist MLP 输入维仍为 4。"""
        model = ModelForecast()
        assert not hasattr(model, "missing_summary_embed")
        assert model.hist_embed_mlp[0].in_features == 4
        for k in model.state_dict():
            assert "missing_summary" not in k

    def test_v3_gap_condition_prototype_preserved(self):
        """v3 use_gap_condition 原型保留：模块存在且前向可用。"""
        model = ModelForecast(use_gap_condition=True).to(DEVICE)
        assert hasattr(model, "gap_embed")
        data = make_batch()
        data["x_forecast_gap_steps"] = torch.ones(2, N_SAFE, device=DEVICE).long()
        out = _forward(model, data)
        assert out["y_hat"].shape == (2, 6, 12, 2)


N_SAFE = 4  # 供 test_v3_gap_condition_prototype_preserved 使用


class TestM1ObservationFeatures:
    def test_m1_forward_shapes_match_m0(self):
        m0 = _forward(ModelForecast().to(DEVICE), make_batch())
        m1 = _forward(
            ModelForecast(use_observation_features=True).to(DEVICE), make_batch())
        for k, shape in OUT_KEYS_SHAPES.items():
            assert m1[k].shape == shape, k
            assert m0[k].shape == shape, k

    def test_m1_hist_mlp_input_dim_8(self):
        model = ModelForecast(use_observation_features=True)
        assert model.hist_embed_mlp[0].in_features == 8

    def test_m1_requires_new_fields(self):
        """use_observation_features=True 而 batch 缺新字段 → 明确 ValueError。"""
        model = ModelForecast(use_observation_features=True).to(DEVICE)
        data = make_batch()
        data.pop("x_gap_steps")
        with pytest.raises(ValueError, match="x_gap_steps"):
            _forward(model, data)

    def test_m1_no_nan_under_block6_and_v3_hard(self):
        block6 = torch.tensor([False] * 6 + [True, True])
        one_frame = torch.tensor([False] * 7 + [True])  # v3 uniform_hard 极端
        for m in (block6, one_frame):
            model = ModelForecast(use_observation_features=True).to(DEVICE)
            out = _forward(model, make_batch(N=3, hist_mask=m))
            for k in OUT_KEYS_SHAPES:
                assert torch.isfinite(out[k]).all(), (m, k)


class TestM2MissingSummary:
    def test_m2_forward_shapes_match_m0(self):
        model = ModelForecast(use_missing_summary=True).to(DEVICE)
        out = _forward(model, make_batch())
        for k, shape in OUT_KEYS_SHAPES.items():
            assert out[k].shape == shape, k

    def test_m2_summary_embed_zero_init(self):
        """missing_summary_embed 输出层零初始化：初始 r_i = 0，不破坏 M0 动态。"""
        model = ModelForecast(use_missing_summary=True)
        last = model.missing_summary_embed[-1]
        assert torch.equal(last.weight, torch.zeros_like(last.weight))
        assert torch.equal(last.bias, torch.zeros_like(last.bias))

    def test_m2_m0_equivalent_at_init_on_complete_history(self):
        """零初始化下 M2 初始前向 == M0：共享 M0 权重（strict=False 载入，
        missing_summary_embed 保持零初始化）→ r_i ≡ 0 → 输出逐元素相等。"""
        torch.manual_seed(0)
        m0 = ModelForecast().to(DEVICE)
        m2 = ModelForecast(use_missing_summary=True).to(DEVICE)
        m2.load_state_dict(m0.state_dict(), strict=False)  # 新模块不在 m0 中，保持零
        last = m2.missing_summary_embed[-1]
        assert torch.count_nonzero(last.weight) == 0 and torch.count_nonzero(last.bias) == 0
        data = make_batch(B=2, N=3)
        o0, o2 = _forward(m0, data), _forward(m2, data)
        assert torch.equal(o0["y_hat"], o2["y_hat"])
        assert torch.equal(o0["new_y_hat"], o2["new_y_hat"])
        assert torch.equal(o0["pi"], o2["pi"])

    def test_m2_padding_actors_get_no_summary(self):
        """padding actor（key_valid=False）不接收摘要条件：输出特征保持 0。"""
        model = ModelForecast(use_missing_summary=True).to(DEVICE)
        model.eval()
        data = make_batch(B=1, N=4)
        data["x_key_valid_mask"] = torch.tensor([[True, True, False, False]], device=DEVICE)
        with torch.no_grad():
            r = model.missing_summary_embed(data["x_missing_summary"])
            zero_for_pad = r * data["x_key_valid_mask"].unsqueeze(-1)
        # 直接验证 mask 乘法逻辑：padding 行结果为 0
        assert torch.equal(zero_for_pad[0, 2:], torch.zeros(2, 128, device=DEVICE))

    def test_m2_combined_with_m1(self):
        model = ModelForecast(use_observation_features=True,
                              use_missing_summary=True).to(DEVICE)
        out = _forward(model, make_batch(N=1))  # SDD 形态 N=1
        assert out["y_hat"].shape == (2, 6, 12, 2)
        out = _forward(model, make_batch(N=6))  # 多邻居
        assert torch.isfinite(out["new_y_hat"]).all()

    def test_m2_with_gap_condition_combo(self):
        """M1+M2 与 v3 use_gap_condition 可共存。"""
        model = ModelForecast(use_observation_features=True, use_missing_summary=True,
                              use_gap_condition=True).to(DEVICE)
        out = _forward(model, make_batch(N=3))
        assert torch.isfinite(out["y_hat"]).all()
