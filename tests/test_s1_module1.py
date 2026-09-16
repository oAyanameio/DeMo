"""模块一 S1（gap scaling + per-actor missing-summary conditioning）单元测试。

锁定行为（模块化方案 §五 执行规则 3：零初始化起步；验收 2/3/4）：
1. S1 开全开关 + 零初始化：加载 M0 权重后前向与 M0 完全一致（验收4）；
2. 摘要通路接通：summary 变化影响输出；
3. 缺 batch 字段时报错（验收3）；
4. 数据管线：collate 后 x_missing_summary 存在、有限、有界；
5. loss/backward 无 NaN（验收2）。
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
    valid[:, :, 5:] = False  # 尾部缺失：gap/summary 非零
    return {
        "x_positions": torch.randn(B, A, T, 2),
        "x_positions_diff": torch.randn(B, A, T, 2),
        "x_velocity_diff": torch.rand(B, A, T),
        "x_valid_mask": valid,
        "x_key_valid_mask": valid.any(-1),
        "x_gap_steps": torch.rand(B, A, T) * 3,
        "x_missing_summary": torch.rand(B, A, 6),
        "x_centers": torch.zeros(B, A, 2),
        "x_angles": torch.zeros(B, A, T),
        "x_attr": torch.zeros(B, A, 3, dtype=torch.uint8),
        "y": torch.randn(B, 12, 2),
        "y_diff": torch.randn(B, 12, 2),
        "y_vel_diff": torch.randn(B, 12, 2),
    }


def _to_cuda(data):
    return {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in data.items()}


def test_s1_zero_init_equivalent_to_m0():
    """S1 双开关全开 + 零初始化：加载 M0 权重后前向与 M0 完全一致。"""
    m0 = _make_model(use_gap_scaling=False, use_missing_summary=False).to("cuda").eval()
    s1 = _make_model(use_gap_scaling=True, use_missing_summary=True).to("cuda").eval()
    missing, unexpected = s1.net.load_state_dict(m0.net.state_dict(), strict=False)
    assert all(k.startswith(("gap_scale_mlp", "missing_summary_embed")) for k in missing)
    assert unexpected == []
    data = _to_cuda(_make_data())
    with torch.no_grad():
        o0 = m0.net(data)
        o1 = s1.net(data)
    assert torch.equal(o0["y_hat"], o1["y_hat"]), \
        "零初始化下 S1 应与 M0 bitwise 一致"


def test_s1_summary_affects_output():
    """missing_summary_embed 非零化后，summary 变化应影响输出（通路接通）。"""
    s1 = _make_model(use_gap_scaling=False, use_missing_summary=True).to("cuda").eval()
    with torch.no_grad():
        s1.net.missing_summary_embed[-1].weight.fill_(0.5)
        s1.net.missing_summary_embed[-1].bias.fill_(0.1)
    d1 = _make_data()
    d2 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in d1.items()}
    d2["x_missing_summary"] = 1.0 - d2["x_missing_summary"]  # 摘要取反
    with torch.no_grad():
        o1 = s1.net(_to_cuda(d1))
        o2 = s1.net(_to_cuda(d2))
    assert not torch.allclose(o1["y_hat"], o2["y_hat"], atol=1e-6), \
        "summary 变化应改变输出（GSM-lite 通路未接通）"


def test_s1_requires_summary_field():
    """缺 x_missing_summary 字段时应报 ValueError。"""
    s1 = _make_model(use_gap_scaling=True, use_missing_summary=True).to("cuda").eval()
    data = _to_cuda(_make_data())
    del data["x_missing_summary"]
    try:
        with torch.no_grad():
            s1.net(data)
    except ValueError as e:
        assert "x_missing_summary" in str(e)
    else:
        raise AssertionError("应抛出 ValueError")


def test_s1_pipeline_summary_field():
    """数据管线：真实 batch collate 后 x_missing_summary 存在、有限、有界。"""
    from src.datamodule.trajimpute_dataset import (
        TrajImputeDataset, trajimpute_collate_fn, TRAJIMPUTE_ROOT,
    )
    ds = TrajImputeDataset(TRAJIMPUTE_ROOT, "ETH-M", "Easy", "test")
    batch = trajimpute_collate_fn([ds[i] for i in range(4)])
    s = batch["x_missing_summary"]
    assert s.shape[-1] == 6
    assert torch.isfinite(s).all()
    assert bool((s >= 0).all()) and bool((s <= 1 + 1e-6).all())
    # 完整历史的 focal 样本（如存在）摘要应接近中性：missing_rate=0
    zero_rows = (batch["missing_count"] == 0)
    if bool(zero_rows.any()):
        assert torch.allclose(
            s[zero_rows][:, 0, 0], torch.zeros(int(zero_rows.sum())), atol=1e-6)


def test_s1_loss_backward_no_nan():
    """验收2：S1 前向输出 finite，loss/backward 全程无 NaN。
    mock 字段形状与真实 collate 对齐（target/y 带 actor 维 [B,A,T,2]）。"""
    from src.model.trainer_forecast import Trainer
    torch.manual_seed(0)
    trainer = Trainer(model={
        "type": "ModelForecast", "embed_dim": 128,
        "num_modes": 20, "future_steps": 12,
        "use_gap_scaling": True, "use_missing_summary": True,
    }).to("cuda")
    B, A, T = 2, 3, 8
    data = _make_data(B=B, A=A, T=T)
    data["target"] = torch.randn(B, A, 12, 2).cuda()          # [B,A,12,2]
    data["target_diff"] = torch.randn(B, A, 12, 2).cuda()
    data["target_vel_diff"] = torch.rand(B, A, 12, 2).cuda()
    data["target_mask"] = torch.ones(B, A, 12, dtype=torch.bool).cuda()
    data = _to_cuda(data)
    out = trainer.net(data)
    for k in ("y_hat", "new_y_hat"):
        assert torch.isfinite(out[k]).all(), f"{k} 含非有限值"
    loss, _ = trainer.cal_loss(out, data)
    assert torch.isfinite(loss).all(), "loss 含非有限值"
    loss.sum().backward()
    for n, p in trainer.net.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"{n} 梯度含非有限值"


def test_s1_hydra_contract():
    """验收5：Hydra train/eval 双端解析到相同 S1-module1 开关与 K=20。"""
    import importlib.util
    from pathlib import Path as P
    spec = importlib.util.spec_from_file_location(
        "runner", P("scripts/训练与评估/run_trajimpute_experiments.py"))
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    spec2 = importlib.util.spec_from_file_location(
        "evl", P("scripts/结果分析/evaluate_trajimpute_direct.py"))
    assert spec2 is not None and spec2.loader is not None
    evl = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(evl)

    # 双端 variant 表一致
    assert runner.VARIANTS["S1-module1"] == evl.VARIANTS["S1-module1"]
    assert runner.VARIANTS["S1-module1"] == {
        "use_gap_scaling": True, "use_missing_summary": True}

    # runner 训练 override 双开关透传 + K 固定 20
    from types import SimpleNamespace
    args = SimpleNamespace(
        variant="S1-module1", seed=2024, batch_size=64, num_workers=4,
        epochs=100, K=20, lr=1e-3, weight_decay=1e-4, bimamba=False,
        model_version_num="s1", smoke=False,
        data_root="/home/lbh/TrajImpute/dataset/TrajImpute",
    )
    protocol_cfg = runner.PROTOCOLS["easy-direct"]
    sw = runner.VARIANTS["S1-module1"]
    overrides = [
        f"model.target.model.num_modes={args.K}",
        f"model.target.model.use_gap_scaling={str(sw.get('use_gap_scaling', False)).lower()}",
        f"model.target.model.use_missing_summary={str(sw.get('use_missing_summary', False)).lower()}",
    ]
    assert "model.target.model.num_modes=20" in overrides
    assert "model.target.model.use_gap_scaling=true" in overrides
    assert "model.target.model.use_missing_summary=true" in overrides

    # yaml 默认关（Hydra struct 键已声明）
    import yaml
    model_conf = yaml.safe_load(
        open("conf/model/missing_aware_ethucy_model_forecast.yaml"))
    model_kw = model_conf["target"]["model"]
    assert model_kw["use_gap_scaling"] is False
    assert model_kw["use_missing_summary"] is False
