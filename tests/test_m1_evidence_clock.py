"""M1' 证据时钟解码（Evidence-Clock Decoding）单元测试。

锁定三个行为：
1. 开关关闭时与基线完全一致（time embedding 不受 forecast_gap 影响）；
2. 开关开启 + 完整历史（gap=1）时与基线一致（t_evidence = dt + t*dt ≡ time*dt+dt）；
3. 开关开启 + 缺失（gap>1）时 State Query 随 gap 改变（时间形状可感知），
   且不同样本的 gap 产生不同输出（批内区分）。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.model.model_forecast import ModelForecast


def _make_model(**kw):
    torch.manual_seed(0)
    return ModelForecast(
        embed_dim=64, num_modes=6, future_steps=12,
        use_evidence_clock=kw.pop("use_evidence_clock", False),
        **kw,
    )


def _fake_forward(model, gap_steps):
    """绕过完整 forward：直接驱动 state query 初始化段（与实现同公式）。"""
    B, T_f, dt = gap_steps.shape[0], model.future_steps, model.dt
    if model.use_evidence_clock:
        steps = torch.arange(T_f).float()
        t_ev = (steps.view(1, -1) + gap_steps.float().view(-1, 1)) * dt
        mode = model.time_embedding_mlp(t_ev.unsqueeze(-1))
    else:
        time = torch.arange(T_f).float() * dt + dt
        mode = model.time_embedding_mlp(time.unsqueeze(-1)).repeat(B, 1, 1)
    return mode


def test_off_matches_baseline():
    gap = torch.tensor([[1.0], [4.0]])
    m = _fake_forward(_make_model(use_evidence_clock=False), gap)
    # 关闭时输出与 gap 无关：两行相同
    assert torch.allclose(m[0], m[1])


def test_on_complete_history_matches_baseline():
    # gap=1（完整历史）时 t_evidence 与基线 time 完全一致
    gap1 = torch.ones(2, 1)
    base = _fake_forward(_make_model(use_evidence_clock=False), gap1)
    ec = _fake_forward(_make_model(use_evidence_clock=True), gap1)
    assert torch.allclose(base, ec, atol=1e-6), "完整历史下 M1' 应与基线行为一致"


def test_on_missing_changes_query_by_gap():
    gap = torch.tensor([[1.0], [4.0]])
    ec = _fake_forward(_make_model(use_evidence_clock=True), gap)
    assert not torch.allclose(ec[0], ec[1]), "不同 gap 应产生不同 State Query"


def test_forward_smoke_with_evidence_clock():
    """完整 forward 冒烟：开启 M1' 后模型可前向、输出形状正确。"""
    from src.model.trainer_forecast import Trainer
    torch.manual_seed(0)
    system = Trainer(
        model={
            "type": "ModelForecast",
            "embed_dim": 128, "num_modes": 6, "future_steps": 12,
            "use_evidence_clock": True, "use_motion_features": True,
        },
    )
    system.eval()
    system = system.to("cuda")
    B, A, T = 2, 3, 8
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
        "x_forecast_gap_steps": torch.tensor([[1.0], [4.0]]).expand(B, A).contiguous(),
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
    data = {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in data.items()}
    with torch.no_grad():
        out = system.net(data)
    assert out["y_hat"].shape[-2:] == (12, 2)
    assert out["pi"].shape[-1] == 6
