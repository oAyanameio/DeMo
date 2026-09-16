"""E1 gap-conditioned temporal scaling 单元测试。

锁定行为：
1. 零初始化等价性：use_gap_scaling=True 加载 M0 权重后前向与 M0 完全一致；
2. gap 单调性输入接通：alpha 通路可接收 x_gap_steps 且影响输出；
3. 缺 batch 字段时报错。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch


def _make_model(**kw):
    from src.model.trainer_forecast import Trainer
    torch.manual_seed(0)
    return Trainer(model={
        "type": "ModelForecast",
        "embed_dim": 128, "num_modes": 6, "future_steps": 12,
        "use_gap_scaling": True,
        **kw,
    })


def _make_data(B=2, A=3, T=8):
    torch.manual_seed(42)
    return {
        "x_positions": torch.randn(B, A, T, 2),
        "x_positions_diff": torch.randn(B, A, T, 2),
        "x_velocity_diff": torch.rand(B, A, T),
        "x_valid_mask": torch.ones(B, A, T, dtype=torch.bool),
        "x_key_valid_mask": torch.ones(B, A, dtype=torch.bool),
        "x_gap_steps": torch.zeros(B, A, T),
        "x_centers": torch.zeros(B, A, 2),
        "x_angles": torch.zeros(B, A, T),
        "x_attr": torch.zeros(B, A, 3, dtype=torch.uint8),
        "y": torch.randn(B, 12, 2),
        "y_diff": torch.randn(B, 12, 2),
        "y_vel_diff": torch.randn(B, 12, 2),
    }


def _to_cuda(data):
    return {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in data.items()}


def test_e1_zero_init_equivalent_to_m0():
    """gap_scale_mlp 零初始化 ⇒ alpha≡1 ⇒ 与 M0 前向完全一致。"""
    m0 = _make_model(use_gap_scaling=False).to("cuda").eval()
    e1 = _make_model(use_gap_scaling=True).to("cuda").eval()
    # 复制共享权重（跳过 gap_scale_mlp——零初始化本身即等价条件）
    missing, unexpected = e1.net.load_state_dict(
        {k: v for k, v in m0.net.state_dict().items()}, strict=False)
    assert all("gap_scale_mlp" in k for k in missing), missing
    assert unexpected == []
    data = _to_cuda(_make_data())
    with torch.no_grad():
        o0 = m0.net(data)
        o1 = e1.net(data)
    assert torch.equal(o0["y_hat"], o1["y_hat"]), "零初始化下 E1 应与 M0 bitwise 一致"


def test_e1_gap_affects_output():
    """gap_scale_mlp 非零化后，gap 变化应影响输出（通路接通）。"""
    e1 = _make_model().to("cuda").eval()
    # 破坏零初始化，让 alpha 不再恒等于 1
    with torch.no_grad():
        e1.net.gap_scale_mlp[-1].weight.fill_(0.5)
        e1.net.gap_scale_mlp[-1].bias.fill_(0.1)
    d1 = _make_data()
    d2 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in d1.items()}
    d2["x_gap_steps"][:] = 3.0   # 全部时刻缺失距离 3 帧
    with torch.no_grad():
        o1 = e1.net(_to_cuda(d1))
        o2 = e1.net(_to_cuda(d2))
    assert not torch.allclose(o1["y_hat"], o2["y_hat"], atol=1e-6), \
        "gap 变化应改变输出（E1 通路未接通）"


def test_e1_requires_gap_field():
    """缺 x_gap_steps 字段时应报 ValueError。"""
    e1 = _make_model().to("cuda").eval()
    data = _to_cuda(_make_data())
    del data["x_gap_steps"]
    try:
        with torch.no_grad():
            e1.net(data)
    except ValueError as e:
        assert "x_gap_steps" in str(e)
    else:
        raise AssertionError("应抛出 ValueError")
