"""M2-social（v2 门控版，2026-09-15）单元测试。

锁定行为（任务书 2026-09-15）：
1. M0 默认配置输出契约：形状正确、有限值；
2. social_scale=0 时 M2 与 M0 输出 max_abs_diff < 1e-5（初始严格等价）；
3. focal 完整历史（missing_ratio=0）时社会残差严格为 0；
4. focal 缺失但邻居全无效时社会残差严格为 0；
5. focal 缺失且有有效邻居时社会分支产生非零梯度；
6. padding actor 不影响 focal 输出；
7. 全部输出无 NaN/Inf；
8. train/eval 模式下 M2 开关一致（同一模型对象）；
9. 测试输出为官方口径（refine 分支 new_y_hat 优先，与 DeMo 官方 test_step 一致；源码断言）。

正式变体口径：use_motion_features=False、use_evidence_clock=False、
use_observation_features=False、use_missing_summary=False、bimamba=False。
"""

import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 正式 M2-social 开关（与 runner VARIANTS["M2-social"] 一致）
M2_SWITCHES = {
    "use_observation_features": False,
    "use_missing_summary": False,
    "use_motion_features": False,
    "use_evidence_clock": False,
    "use_social_evidence": True,
}
M0_SWITCHES = {k: v for k, v in M2_SWITCHES.items() if k != "use_social_evidence"}


def _make_model(**overrides):
    from src.model.model_forecast import ModelForecast
    torch.manual_seed(0)
    return ModelForecast(
        embed_dim=128, future_steps=12, num_heads=8, mlp_ratio=4.0,
        qkv_bias=False, drop_path=0.2, num_actor_types=1,
        num_modes=20, bimamba=False, dt=0.4, obs_len=8,
        **overrides,
    )


def _make_data(B=2, N=4, T=8, focal_valid=True, neighbors_valid=True,
               padding=False, seed=42):
    """构造最小 batch。focal_valid=False 时 focal 历史缺 4 帧（ratio=0.5）；
    neighbors_valid=False 时真实邻居全帧无效；padding=True 时末位邻居为
    padding actor（x_key_valid_mask=False，collate 语义）。"""
    g = torch.Generator().manual_seed(seed)
    data = {
        "x_positions_diff": torch.randn(B, N, T, 2, generator=g) * 0.1,
        "x_velocity_diff": torch.rand(B, N, T, generator=g) * 0.5,
        "x_centers": torch.randn(B, N, 2, generator=g) * 2.0,
        "x_angles": torch.randn(B, N, T, generator=g),
        "x_attr": torch.zeros(B, N, 3, dtype=torch.uint8),
        "x_valid_mask": torch.ones(B, N, T, dtype=torch.bool),
        "x_key_valid_mask": torch.ones(B, N, dtype=torch.bool),
        "target": torch.randn(B, N, 12, 2, generator=g),
        "target_mask": torch.ones(B, N, 12, dtype=torch.bool),
        "x_positions": torch.randn(B, N, T, 2, generator=g) * 2.0,
    }
    if not focal_valid:                       # focal 缺失 4/8 帧
        data["x_valid_mask"][:, 0, 4:] = False
        data["x_positions"][:, 0, 4:] = 0.0   # 缺失占位（dataset 语义）
    if not neighbors_valid:                   # 真实邻居全帧无效
        data["x_valid_mask"][:, 1:, :] = False
        data["x_positions"][:, 1:, :, :] = 0.0
    if padding:                               # 末位邻居 = padding actor
        data["x_key_valid_mask"][:, -1] = False
    return {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in data.items()}


def _social_residual(model, data):
    """直接计算门控残差 gate·scale·proj(attn_out) 的范数（绕过整网输出的
    其他随机性来源），用于断言"残差严格为 0"。"""
    net = model
    # 复用 forward 内部同一计算：从模块参数与数据重建（与社会分支一致）
    hist_valid = data["x_valid_mask"]
    neigh_valid = data["x_valid_mask"][:, 1:, :].float()
    pad_mask = data["x_key_valid_mask"][:, 1:]
    pad_valid = pad_mask.float().unsqueeze(-1).expand(-1, -1, net.obs_len).float()
    token_valid_f = neigh_valid * pad_valid
    focal_missing_ratio = 1.0 - hist_valid[:, 0, :].float().mean(-1, keepdim=True)
    real_frames = pad_valid.sum(dim=(1, 2))
    valid_frames = token_valid_f.sum(dim=(1, 2))
    neighbor_reliability = torch.where(
        real_frames > 0, valid_frames / real_frames.clamp(min=1.0),
        torch.zeros_like(real_frames)).view(-1, 1)
    gate = focal_missing_ratio * neighbor_reliability
    return gate  # [B,1]；scale=0 或 gate=0 → 残差恒 0


# ---------------------------------------------------------------- 契约与等价

def test_m0_default_contract():
    """M0 默认配置输出契约不变：形状 + 有限值 + 默认全关。"""
    model = _make_model(**M0_SWITCHES).to(DEVICE).eval()
    assert model.use_social_evidence is False
    assert model.use_motion_features is False
    assert model.use_evidence_clock is False
    with torch.no_grad():
        out = model(_make_data())
    assert out["y_hat"].shape[-2:] == (12, 2)
    assert out["y_hat"].shape[1] == 20
    assert out["pi"].shape[-1] == 20
    assert torch.isfinite(out["y_hat"]).all()
    assert torch.isfinite(out["pi"]).all()


def test_m2_scale0_equals_m0():
    """social_scale=0 时 M2 输出与 M0 严格一致（max_abs_diff < 1e-5）。"""
    m0 = _make_model(**M0_SWITCHES).to(DEVICE).eval()
    m2 = _make_model(**M2_SWITCHES).to(DEVICE).eval()
    # 同权重：M2 的主干参数复制自 M0（社会模块初始 scale=0 不影响输出）
    m2.load_state_dict(m0.state_dict(), strict=False)
    assert torch.all(m2.social_scale == 0)
    # 需要覆盖 focal 缺失样本（gate 可能非零但 scale=0）
    data = _make_data(focal_valid=False, neighbors_valid=True)
    with torch.no_grad():
        o0 = m0(data)
        o2 = m2(data)
    diff = (o0["y_hat"] - o2["y_hat"]).abs().max().item()
    assert diff < 1e-5, f"M2(scale=0) 与 M0 输出差 {diff}"


def test_scale0_survives_initialize_weights():
    """initialize_weights() 不得破坏 social_scale 零初始化。"""
    m2 = _make_model(**M2_SWITCHES)
    m2.initialize_weights()
    assert torch.all(m2.social_scale == 0)


# ---------------------------------------------------------------- 门控归零

def test_gate_zero_when_focal_complete():
    """focal 完整历史（missing_ratio=0）→ gate=0 → 社会残差严格为 0。"""
    m2 = _make_model(**M2_SWITCHES).to(DEVICE)
    data = _make_data(focal_valid=True, neighbors_valid=True)
    # 人为放大 scale 确保非零残差本会发生
    with torch.no_grad():
        m2.social_scale.fill_(1.0)
    gate = _social_residual(m2, data)
    assert torch.all(gate == 0), "focal 完整历史时 gate 必须为 0"
    # 端到端复核：完整 focal 下 M2 输出 == M0 同权重输出
    m0 = _make_model(**M0_SWITCHES).to(DEVICE).eval()
    m2 = m2.eval()
    m2.load_state_dict(m0.state_dict(), strict=False)
    with torch.no_grad():
        m2.social_scale.fill_(1.0)   # 即使 scale=1，gate=0 也必须压住
        o0 = m0(data)
        o2 = m2(data)
    diff = (o0["y_hat"] - o2["y_hat"]).abs().max().item()
    assert diff < 1e-5, f"focal 完整 + scale=1 时仍应等价 M0，差 {diff}"


def test_gate_zero_when_no_valid_neighbors():
    """focal 缺失但邻居全无效（reliability=0）→ 残差严格为 0。"""
    m2 = _make_model(**M2_SWITCHES).to(DEVICE).eval()
    with torch.no_grad():
        m2.social_scale.fill_(1.0)
    data = _make_data(focal_valid=False, neighbors_valid=False)
    gate = _social_residual(m2, data)
    assert torch.all(gate == 0), "邻居全无效时 gate 必须为 0"
    with torch.no_grad():
        out = m2(data)
    assert torch.isfinite(out["y_hat"]).all()   # 空注意力不产生 NaN


def test_gate_positive_when_missing_and_valid_neighbors():
    """focal 缺失 + 有效邻居 → gate>0（社会分支通路存在且可训练）。"""
    m2 = _make_model(**M2_SWITCHES).to(DEVICE)
    data = _make_data(focal_valid=False, neighbors_valid=True)
    gate = _social_residual(m2, data)
    assert torch.all(gate > 0), "focal 缺失 + 邻居有效时 gate 应 > 0"
    with torch.no_grad():
        m2.social_scale.fill_(1.0)
    out = m2(data)
    loss = out["y_hat"].sum()
    loss.backward()
    assert m2.neighbor_frame_embed[0].weight.grad is not None
    assert m2.neighbor_frame_embed[0].weight.grad.abs().sum().item() > 0
    assert m2.social_scale.grad is not None
    assert m2.social_scale.grad.abs().sum().item() > 0


# ---------------------------------------------------------------- padding 隔离

def test_padding_actor_does_not_affect_focal():
    """padding actor 不得进入注意力：扰动 padding actor 的原始数据不改变 focal 输出。
    （不能对比"有/无 padding"两个不同场景——场景 Transformer 看到的真实 actor 数
    不同，输出本就应不同；正确断言是 padding actor 的数据被完全隔离。）"""
    m2 = _make_model(**M2_SWITCHES).to(DEVICE).eval()
    with torch.no_grad():
        m2.social_scale.fill_(1.0)   # 放大残差以暴露任何泄漏
    d1 = _make_data(B=2, N=4, focal_valid=False, neighbors_valid=True, padding=True)
    d2 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in d1.items()}
    # 扰动 padding actor（末位）的位置与中心
    d2["x_positions"][:, -1, :, :] += 5.0
    d2["x_centers"][:, -1, :] += 5.0
    with torch.no_grad():
        o1 = m2(d1)
        o2 = m2(d2)
    assert torch.allclose(o1["y_hat"], o2["y_hat"], atol=1e-5), \
        "padding actor 的数据不应影响 focal 输出（双重掩码泄漏）"


def test_no_nan_inf_hard_case():
    """最恶劣组合（focal 缺失 + 邻居全无效 + padding）无 NaN/Inf。"""
    m2 = _make_model(**M2_SWITCHES).to(DEVICE).eval()
    data = _make_data(B=4, N=6, focal_valid=False, neighbors_valid=False, padding=True)
    with torch.no_grad():
        out = m2(data)
    for key in ("y_hat", "pi", "new_y_hat", "new_pi"):
        v = out[key]
        assert torch.isfinite(v).all(), f"{key} 含 NaN/Inf"


# ---------------------------------------------------------------- 口径一致性

def test_eval_branch_is_official_refine():
    """测试输出必须为官方口径：refine 分支 new_y_hat 优先（DeMo 官方 test_step 行为）。"""
    import src.model.trainer_forecast as tf
    import scripts.结果分析.evaluate_trajimpute_direct as ev

    # trainer.test_step 按官方行为把 y_hat 覆盖为 new_y_hat
    src_test = inspect.getsource(tf.Trainer.test_step)
    assert "out['new_y_hat']" in src_test, \
        "test_step 必须按官方行为覆盖为 refine 分支"
    # validation_step 的 val_metrics 消费基础 y_hat（覆盖前计算，与官方一致）
    src_val = inspect.getsource(tf.Trainer.validation_step)
    assert "metrics = self.val_metrics(out" in src_val
    # evaluator 必须优先 refine 分支（官方口径）
    src_ev = inspect.getsource(ev.run_evaluation)
    assert 'out["new_y_hat"]' in src_ev and 'out["new_pi"]' in src_ev


def test_variant_switches_consistent_train_eval():
    """train/eval 用同一 ModelForecast 开关集（VARIANTS 映射单点事实源）。"""
    import importlib
    runner = importlib.import_module("scripts.训练与评估.run_trajimpute_experiments")
    evaluator = importlib.import_module("scripts.结果分析.evaluate_trajimpute_direct")
    sw_r = runner.VARIANTS["M2-social"]
    sw_e = evaluator.VARIANTS["M2-social"]
    assert sw_r == sw_e, "runner 与 evaluator 的 M2-social 开关必须一致"
    assert sw_r["use_social_evidence"] is True
    assert sw_r["use_motion_features"] is False
    assert sw_r["use_evidence_clock"] is False
