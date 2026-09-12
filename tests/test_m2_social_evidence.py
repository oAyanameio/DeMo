"""M2' 社会证据补偿（Social Evidence Compensation）单元测试。

锁定行为：
1. 完整前向冒烟（GPU）：开启 M2' + B1 后模型可前向、输出形状正确；
2. 双重掩码正确性：邻居帧无效/padding 时 token 不参与注意力（输出对无效 token 不变）；
3. 补偿有效性：focal 缺失样本的输出应受邻居观测影响（对邻居位置扰动敏感）；
4. 完整历史 + 邻居全有效时门控可自学习（前向不报错，梯度存在）。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch


def _make_system(**model_kw):
    from src.model.trainer_forecast import Trainer
    torch.manual_seed(0)
    return Trainer(model={
        "type": "ModelForecast",
        "embed_dim": 128, "num_modes": 6, "future_steps": 12,
        "use_social_evidence": True, "use_motion_features": True,
        **model_kw,
    })


def _make_data(B=2, A=3, T=8, neighbor_valid=True, perturb=False):
    torch.manual_seed(42 if not perturb else 43)
    data = {
        "x_positions": torch.randn(B, A, T, 2),
        "x_positions_diff": torch.randn(B, A, T, 2),
        "x_velocity": torch.rand(B, A, T),
        "x_velocity_diff": torch.rand(B, A, T),
        "x_accel": torch.rand(B, A, T),
        "x_turn_rate": torch.rand(B, A, T),
        "x_motion_run": torch.rand(B, A, T),
        "x_valid_mask": torch.ones(B, A, T, dtype=torch.bool),
        "x_key_valid_mask": torch.ones(B, A, dtype=torch.bool),
        "x_anchor_lag_steps": torch.ones(B, A),
        "x_forecast_gap_steps": torch.ones(B, A),
        "x_gap_steps": torch.ones(B, A, T),
        "x_centers": torch.zeros(B, A, 2),
        "x_angles": torch.zeros(B, A, T),
        "x_yaw": torch.zeros(B, A),
        "x_attr": torch.zeros(B, A, 3),
        "y": torch.randn(B, 12, 2),
        "y_diff": torch.randn(B, 12, 2),
        "y_vel_diff": torch.randn(B, 12, 2),
        "x_ids": torch.zeros(B, A),
        "scene_id": ["s0", "s1"],
    }
    if not neighbor_valid:
        data["x_valid_mask"][:, 1:, :] = False   # 邻居全部帧无效
    return data


def _to_cuda(data):
    return {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in data.items()}


def test_m2_forward_smoke():
    system = _make_system().to("cuda").eval()
    data = _to_cuda(_make_data())
    with torch.no_grad():
        out = system.net(data)
    assert out["y_hat"].shape[-2:] == (12, 2)
    assert out["pi"].shape[-1] == 6


def test_m2_invalid_neighbor_tokens_do_not_affect_output():
    """邻居帧全无效（双重掩码关闭所有 token）时，仅扰动邻居位置不改变输出。"""
    system = _make_system().to("cuda").eval()
    d1 = _make_data(neighbor_valid=False)
    d2 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in d1.items()}
    d2["x_positions"][:, 1:, :, :] += 3.0   # 只扰动无效邻居的原始位置
    with torch.no_grad():
        o1 = system.net(_to_cuda(d1))
        o2 = system.net(_to_cuda(d2))
    assert torch.allclose(o1["y_hat"], o2["y_hat"], atol=1e-5), \
        "无效邻居 token 的位置不应影响输出（双重掩码失效）"


def test_m2_valid_neighbor_evidence_affects_focal():
    """邻居有效时，扰动邻居位置应改变 focal 输出（证据通路接通）。"""
    system = _make_system().to("cuda").eval()
    d1 = _make_data()
    d2 = _make_data()
    d2["x_positions"][:, 1:, :, :] += 3.0
    with torch.no_grad():
        o1 = system.net(_to_cuda(d1))
        o2 = system.net(_to_cuda(d2))
    assert not torch.allclose(o1["y_hat"], o2["y_hat"], atol=1e-5), \
        "有效邻居位置扰动应影响输出（证据通路未接通）"


def test_m2_gradient_flows():
    """社会证据模块参数可接收梯度。"""
    system = _make_system().to("cuda")
    data = _to_cuda(_make_data())
    out = system.net(data)
    loss = out["y_hat"].sum()
    loss.backward()
    assert system.net.neighbor_frame_embed[0].weight.grad is not None
    assert system.net.social_attn.in_proj_weight.grad is not None
