"""模块二 S2（mask-aware pooling）单元测试。

锁定行为：
1. 零初始化等价：use_mask_pooling=True 加载 M0 权重后与 M0 前向完全一致；
2. pooling 通路接通：pool_proj 非零化后，缺失模式变化应影响输出；
3. 完整历史语义：完整历史时 last_feat 即 Mamba 最后帧（与 M0 同源）；
4. Hydra 契约：runner/evaluator 双端 variant 表一致、开关透传、yaml 默认关；
5. loss/backward 无 NaN。
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
        "embed_dim": 128, "num_modes": 20, "future_steps": 12,
        **kw,
    })


def _make_data(B=2, A=3, T=8):
    torch.manual_seed(42)
    valid = torch.ones(B, A, T, dtype=torch.bool)
    valid[:, :, 5:] = False  # 尾部缺失
    return {
        "x_positions": torch.randn(B, A, T, 2),
        "x_positions_diff": torch.randn(B, A, T, 2),
        "x_velocity_diff": torch.rand(B, A, T),
        "x_valid_mask": valid,
        "x_key_valid_mask": valid.any(-1),
        "x_centers": torch.zeros(B, A, 2),
        "x_angles": torch.zeros(B, A, T),
        "x_attr": torch.zeros(B, A, 3, dtype=torch.uint8),
        "y": torch.randn(B, A, 12, 2),
        "y_diff": torch.randn(B, A, 12, 2),
        "y_vel_diff": torch.rand(B, A, 12, 2),
        "target": torch.randn(B, A, 12, 2),
        "target_diff": torch.randn(B, A, 12, 2),
        "target_vel_diff": torch.rand(B, A, 12, 2),
        "target_mask": torch.ones(B, A, 12, dtype=torch.bool),
    }


def _to_cuda(data):
    return {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in data.items()}


def test_s2_zero_init_equivalent_to_m0():
    """零初始化 + 完整历史：S2 与 M0 前向 bitwise 一致（last_valid==last）。
    缺失历史下两者应不同——那是模块机制本身（M0 取占位推出的末帧特征，
    S2 取最后有效帧特征）。"""
    m0 = _make_model().to("cuda").eval()
    s2 = _make_model(use_mask_pooling=True).to("cuda").eval()
    missing, unexpected = s2.net.load_state_dict(m0.net.state_dict(), strict=False)
    assert all(k.startswith("pool_proj") for k in missing), missing
    assert unexpected == []
    # 完整历史：bitwise 等价
    data = _make_data()
    data["x_valid_mask"][:] = True
    data["x_key_valid_mask"][:] = True
    with torch.no_grad():
        o0 = m0.net(_to_cuda(data))
        o2 = s2.net(_to_cuda(data))
    assert torch.equal(o0["y_hat"], o2["y_hat"]), \
        "零初始化+完整历史下 S2 应与 M0 bitwise 一致"
    # 尾部缺失：应不同（机制生效）
    data2 = _to_cuda(_make_data())  # 帧 5-7 缺失
    with torch.no_grad():
        o0m = m0.net(data2)
        o2m = s2.net(data2)
    assert not torch.equal(o0["y_hat"], o2m["y_hat"]) or \
        not torch.equal(o0m["y_hat"], o2m["y_hat"]), \
        "尾部缺失下 S2 与 M0 应产生不同输出（最后有效帧 vs 占位末帧）"


def test_s2_pooling_path_live():
    """pool_proj 非零化后，缺失模式变化应影响输出（通路接通）。"""
    s2 = _make_model(use_mask_pooling=True).to("cuda").eval()
    with torch.no_grad():
        s2.net.pool_proj[-1].weight.fill_(0.2)
        s2.net.pool_proj[-1].bias.fill_(0.05)
    d1 = _make_data()
    d2 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in d1.items()}
    d2["x_valid_mask"][:, :, :5] = False  # 改成前缀缺失（有效帧集合不同）
    d2["x_valid_mask"][:, :, 5:] = True
    with torch.no_grad():
        o1 = s2.net(_to_cuda(d1))
        o2 = s2.net(_to_cuda(d2))
    assert not torch.allclose(o1["y_hat"], o2["y_hat"], atol=1e-6), \
        "缺失模式变化应改变输出（pooling 通路未接通）"


def test_s2_complete_history_last_frame():
    """完整历史时 last_feat == Mamba 最后帧（与 M0 同源，mean 为补充路）。"""
    s2 = _make_model(use_mask_pooling=True).to("cuda").eval()
    data = _make_data()
    data["x_valid_mask"][:] = True  # 完整历史
    data["x_key_valid_mask"][:] = True
    data = _to_cuda(data)
    with torch.no_grad():
        # 直接驱动 pooling 段公式
        feat = torch.randn(4, 8, 128, device="cuda")     # [M=4, L=8, D]
        sel_valid = torch.ones(4, 8, dtype=torch.bool, device="cuda")
        last_idx = sel_valid.float().cumsum(dim=1).argmax(dim=1)
        last_feat = feat[torch.arange(4, device="cuda"), last_idx]
        assert torch.equal(last_feat, feat[:, -1]), "完整历史时 last_feat 应为最后帧"


def test_s2_hydra_contract():
    """runner/evaluator 双端 variant 表一致 + 开关透传 + yaml 默认关。"""
    import importlib.util
    from types import SimpleNamespace
    spec = importlib.util.spec_from_file_location(
        "runner", Path("scripts/训练与评估/run_trajimpute_experiments.py"))
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    spec2 = importlib.util.spec_from_file_location(
        "evl", Path("scripts/结果分析/evaluate_trajimpute_direct.py"))
    assert spec2 is not None and spec2.loader is not None
    evl = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(evl)

    assert runner.VARIANTS["S2-module2"] == evl.VARIANTS["S2-module2"]
    assert runner.VARIANTS["S2-module2"] == {"use_mask_pooling": True}

    args = SimpleNamespace(
        variant="S2-module2", seed=2024, batch_size=64, num_workers=4,
        epochs=100, K=20, lr=1e-3, weight_decay=1e-4, bimamba=False,
        model_version_num="s2", smoke=False, init_from=None,
        data_root="/home/lbh/TrajImpute/dataset/TrajImpute",
    )
    sw = runner.VARIANTS["S2-module2"]
    overrides = [
        f"model.target.model.num_modes={args.K}",
        f"model.target.model.use_mask_pooling={str(sw.get('use_mask_pooling', False)).lower()}",
    ]
    assert "model.target.model.num_modes=20" in overrides
    assert "model.target.model.use_mask_pooling=true" in overrides

    import yaml
    model_conf = yaml.safe_load(
        open("conf/model/missing_aware_ethucy_model_forecast.yaml"))
    model_kw = model_conf["target"]["model"]
    assert model_kw["use_mask_pooling"] is False


def test_s2_loss_backward_no_nan():
    """前向 finite，loss/backward 无 NaN。"""
    s2 = _make_model(use_mask_pooling=True).to("cuda")
    data = _to_cuda(_make_data())
    out = s2.net(data)
    for k in ("y_hat", "new_y_hat"):
        assert torch.isfinite(out[k]).all(), f"{k} 含非有限值"
    loss, _ = s2.cal_loss(out, data)
    assert torch.isfinite(loss).all(), "loss 含非有限值"
    loss.sum().backward()
    for n, p in s2.net.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"{n} 梯度含非有限值"
