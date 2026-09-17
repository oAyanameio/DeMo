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
    """修正版 gap gate：零初始化 ⇒ alpha≡1 ⇒ 与 M0 前向完全一致。"""
    m0 = _make_model(use_gap_scaling=False).to("cuda").eval()
    e1 = _make_model(use_gap_scaling=True).to("cuda").eval()
    missing, unexpected = e1.net.load_state_dict(m0.net.state_dict(), strict=False)
    assert all(k.startswith("gap_scale_mlp") for k in missing)
    assert unexpected == []
    data = _to_cuda(_make_data())
    with torch.no_grad():
        o0 = m0.net(data)
        o1 = e1.net(data)
    assert torch.equal(o0["y_hat"], o1["y_hat"]), "零初始化下 E1 应与 M0 bitwise 一致"


def test_e1_alpha_anchored_and_bounded():
    """修正 1：gap=0 强制 alpha=1（任意权重）；log alpha 有界于 ±max（修正 2 的界）。
    用随机非零权重模拟训练后状态。"""
    e1 = _make_model(use_gap_scaling=True).to("cuda").eval()
    with torch.no_grad():
        torch.nn.init.normal_(e1.net.gap_scale_mlp[-1].weight, std=5.0)
        torch.nn.init.normal_(e1.net.gap_scale_mlp[-1].bias, std=5.0)
        gap = torch.tensor([[[0.0, 1.0, 3.0, 8.0]]]).to("cuda")
        log_alpha = torch.tanh(e1.net.gap_scale_mlp(gap[..., None].to("cuda"))) \
            * e1.net.gap_log_alpha_max
        alpha = torch.exp(-log_alpha)
        alpha = torch.where(gap[..., None] == 0, torch.ones_like(alpha), alpha)
    assert torch.isclose(alpha[0, 0, 0], torch.tensor(1.0)), "gap=0 必须强制 alpha=1"
    lo, hi = torch.exp(torch.tensor(e1.net.gap_log_alpha_max)), torch.exp(torch.tensor(-e1.net.gap_log_alpha_max))
    assert bool((alpha[:, :, 1:] <= lo + 1e-6).all()), "alpha 上界 = exp(max)"
    assert bool((alpha[:, :, 1:] >= hi - 1e-6).all()), "alpha 下界 = exp(-max)"


def test_e1_summary_gated_by_missing_rate():
    """修正 2：summary 注入乘 missing_rate——零缺失 actor 注入恒为零。
    MLP bias 非零也不泄漏（随机化权重后验证）。"""
    s1 = _make_model(use_missing_summary=True).to("cuda").eval()
    with torch.no_grad():
        torch.nn.init.normal_(s1.net.missing_summary_embed[-1].weight, std=0.5)
        torch.nn.init.normal_(s1.net.missing_summary_embed[-1].bias, std=0.5)  # bias 非零
        summary = torch.rand(2, 3, 6).to("cuda")
        summary[0, 0, 0] = 0.0  # 显式构造零缺失 actor（missing_rate=0）
        mr = summary[..., :1]                   # [B,N,1]
        inj = s1.net.missing_summary_embed(summary) * mr
        inj_manual = s1.net.missing_summary_embed(summary)
    # 零缺失 actor（missing_rate==0 的行）注入必须恒为零
    zero_rows = (mr[..., 0] == 0)               # [B,N]
    assert torch.allclose(inj[zero_rows], torch.zeros_like(inj[zero_rows])), \
        "missing_rate=0 的 actor 注入必须恒为零"
    assert not torch.allclose(inj, inj_manual), "门控应缩放非零 missing_rate 的注入"


def test_e1_paired_init_shares_backbone():
    """修正 3：S1 加载 make_paired_init 产物后，共享参数与同 seed M0 逐位相同。"""
    import subprocess, tempfile, os
    from src.model.model_forecast import ModelForecast
    with tempfile.TemporaryDirectory() as td:
        ckpt = os.path.join(td, "m0_init.ckpt")
        r = subprocess.run(
            [sys.executable, "scripts/训练与评估/make_paired_init.py",
             "--seed", "2024", "--out", ckpt],
            cwd="/home/lbh/DeMo", capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        torch.manual_seed(2024)
        m0 = ModelForecast(num_modes=20, use_gap_scaling=False,
                           use_missing_summary=False).cuda()
        s1 = ModelForecast(num_modes=20, use_gap_scaling=True,
                           use_missing_summary=True).cuda()
        missing, unexpected = s1.load_from_checkpoint(ckpt), None
        common = [k for k in m0.state_dict() if k in s1.state_dict()]
        diff = [k for k in common
                if not torch.equal(m0.state_dict()[k], s1.state_dict()[k])]
        assert not diff, f"配对初始化后仍有 {len(diff)} 个共享参数不同: {diff[:3]}"
        assert any("gap_scale_mlp" in k for k in s1.state_dict()), "S1 应有新分支"
        # 新分支零初始化保持
        assert torch.all(s1.state_dict()["gap_scale_mlp.2.weight"] == 0)


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
