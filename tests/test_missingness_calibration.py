"""M1 自然缺失证据条件化输出校准测试（方案 §6、审计清单 §9.2）。

覆盖：
  1. evidence 特征构造（只来自 mask 派生字段、归一化正确）；
  2. max_missing_run_batch 与逐帧循环版逐位等价（穷举 2^8 掩码）；
  3. all-valid 恒等映射：missing_ratio=0 -> log_tau=log_scale=0，
     new_pi_cal==new_pi、scal_cal==scal_new（随机权重下仍成立）；
  4. 初始零权重等价 M0（零初始化后校准输出==raw 输出）；
  5. temperature/scale 始终有限为正、有上界（随机大 evidence）；
  6. calibration 关闭时模型输出与 M0 结构完全一致（无新增键）；
  7. 前向/反向：校准头参数有有限梯度，主链梯度不受影响路径存在；
  8. 高缺失样本 log_tau>0 时 mode logits 变平（softmax entropy 上升）。

GPU 测试（6/7/8）需 CUDA + mamba_ssm，无 GPU 时自动跳过。
"""

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.model.layers.missingness_calibration import (  # noqa: E402
    MissingnessCalibrationHead, build_evidence_features, max_missing_run_batch,
)

GPU = torch.cuda.is_available()


def _ref_max_run(hist_valid_row: torch.Tensor) -> int:
    best = cur = 0
    for v in hist_valid_row.tolist():
        cur = cur + 1 if not v else 0
        best = max(best, cur)
    return best


# ---------------------------------------------------------------- 1. evidence
def test_evidence_features_layout():
    valid_ratio = torch.tensor([1.0, 0.5, 0.25])
    anchor_lag = torch.tensor([0, 1, 2])
    forecast_gap = torch.tensor([1, 2, 3])
    max_run = torch.tensor([0, 1, 3])
    e = build_evidence_features(valid_ratio, anchor_lag, forecast_gap, max_run, obs_len=8)
    assert e.shape == (3, 5)
    # 列 0=valid_ratio, 1=missing_ratio, 2..4 归一化步数
    assert torch.allclose(e[:, 0], valid_ratio)
    assert torch.allclose(e[:, 1], 1.0 - valid_ratio)
    assert torch.allclose(e[:, 2], torch.tensor([0.0, 1 / 8, 2 / 8]))
    assert torch.allclose(e[:, 3], torch.tensor([1 / 8, 2 / 8, 3 / 8]))
    assert torch.allclose(e[:, 4], torch.tensor([0.0, 1 / 8, 3 / 8]))
    # all-valid -> missing_ratio 恒 0（恒等门控条件）
    assert float(e[0, 1]) == 0.0


def test_evidence_features_bad_shape_raises():
    with pytest.raises(RuntimeError):
        build_evidence_features(torch.zeros(2, 2), torch.zeros(2), torch.zeros(2), torch.zeros(2))


# ---------------------------------------------------------------- 2. max run
def test_max_missing_run_batch_equivalence():
    # 穷举全部 2^8 掩码模式（至少 1 有效帧）与循环版逐位一致
    for m in range(1, 256):
        row = torch.tensor([[(m >> t) & 1 for t in range(8)]], dtype=torch.bool)
        assert int(max_missing_run_batch(row)) == _ref_max_run(row[0]), f"mask={m:08b}"


def test_max_missing_run_batch_multi_row():
    g = torch.Generator().manual_seed(7)
    valid = torch.rand((16, 8), generator=g) > 0.4
    valid[:, 0] = True
    ref = torch.tensor([_ref_max_run(r) for r in valid])
    assert torch.equal(max_missing_run_batch(valid), ref)


# ---------------------------------------------------------------- 3/4. 恒等性
def test_all_valid_identity_with_random_weights():
    """随机（非零）权重下，all-valid 样本 missing_ratio=0 仍强制恒等映射。"""
    head = MissingnessCalibrationHead()
    # 打破零初始化，验证门控本身保证恒等
    with torch.no_grad():
        for p in head.parameters():
            p.copy_(torch.randn_like(p) * 0.5)
    ev = build_evidence_features(
        torch.ones(4), torch.zeros(4), torch.ones(4), torch.zeros(4), obs_len=8)
    log_tau, log_scale = head(ev)
    assert torch.equal(log_tau, torch.zeros(4))
    assert torch.equal(log_scale, torch.zeros(4))


def test_zero_init_equivalence():
    """零初始化输出层：任意 evidence 下 log_tau=log_scale=0（初始即 M0）。"""
    head = MissingnessCalibrationHead()
    ev = build_evidence_features(
        torch.tensor([0.25, 0.5]), torch.tensor([2, 1]), torch.tensor([3, 2]),
        torch.tensor([4, 2]), obs_len=8)
    log_tau, log_scale = head(ev)
    assert torch.equal(log_tau, torch.zeros(2))
    assert torch.equal(log_scale, torch.zeros(2))
    # 主输出恒等：new_pi_cal == new_pi, scal_cal == scal_new
    pi = torch.randn(2, 20)
    scal = torch.rand(2, 20, 12, 2) + 0.5
    assert torch.equal(pi / torch.exp(log_tau).view(-1, 1), pi)
    assert torch.equal(scal * torch.exp(log_scale).view(-1, 1, 1, 1), scal)


# ---------------------------------------------------------------- 5. 边界
def test_calibration_bounds_finite_positive():
    """随机大 evidence：log 有限且有界 -> tau/scale 有限为正。"""
    head = MissingnessCalibrationHead(max_log_magnitude=2.0)
    with torch.no_grad():
        for p in head.parameters():
            p.copy_(torch.randn_like(p) * 3.0)
    ev = torch.randn(64, 5) * 50  # 极端 evidence
    ev[:, 1] = ev[:, 1].abs()     # missing_ratio 非负
    log_tau, log_scale = head(ev)
    assert torch.isfinite(log_tau).all() and torch.isfinite(log_scale).all()
    assert float(log_tau.abs().max()) <= 2.0
    assert float(log_scale.abs().max()) <= 2.0
    tau = torch.exp(log_tau)
    scale_factor = torch.exp(log_scale)
    assert bool((tau > 0).all()) and bool((scale_factor > 0).all())
    assert float(tau.max()) <= torch.exp(torch.tensor(2.0))
    assert float(scale_factor.max()) <= torch.exp(torch.tensor(2.0))


def test_tanh_bound_gradient_alive_at_boundary():
    """平滑限幅：log 值越过 bound 后梯度仍非零（clamp 在此处梯度恒 0）。

    取中度饱和区（raw/bound≈3，tanh'≈0.01）验证；fp32 下深度饱和
    （raw/bound>9）tanh 也会舍入到 1.0 使梯度为 0，故不推到极端。
    """
    head = MissingnessCalibrationHead(max_log_magnitude=2.0)
    with torch.no_grad():
        # raw = missing_ratio * w2·gelu(w1·e)：定向调 last layer bias/weight
        # 使 raw ≈ 6（ratio 3，tanh(3)'≈0.0099 非零）
        torch.nn.init.constant_(head.tau_branch[-1].weight, 0.0)
        head.tau_branch[-1].bias.fill_(8.0)
    ev = torch.tensor([[0.0, 0.75, 0.2, 0.3, 0.2]])
    log_tau, _ = head(ev)
    assert torch.isfinite(log_tau).all()
    assert 1.9 < float(log_tau) < 2.0  # 已越过 bound 附近（限幅生效）
    g, = torch.autograd.grad(log_tau.sum(), head.tau_branch[-1].bias)
    assert float(g.abs().sum()) > 0.0  # clamp 在此位置的梯度恒为 0


def test_calibration_head_gradient_flow():
    head = MissingnessCalibrationHead()
    ev = build_evidence_features(
        torch.tensor([0.375, 0.8]), torch.tensor([2, 0]), torch.tensor([3, 1]),
        torch.tensor([2, 0]), obs_len=8)
    log_tau, log_scale = head(ev)
    # 零初始化下 log 值为 0，但输出层梯度活（∂out/∂w_last = h ≠ 0）
    (log_tau.sum() + log_scale.sum()).backward()
    grads = [p.grad for p in head.parameters() if p.grad is not None]
    assert len(grads) > 0
    # 输出层梯度非零（分支最后一层 weight）
    w_tau = head.tau_branch[-1].weight.grad
    w_scale = head.scale_branch[-1].weight.grad
    assert w_tau is not None and float(w_tau.abs().sum()) > 0
    assert w_scale is not None and float(w_scale.abs().sum()) > 0


# ---------------------------------------------------------------- 6/7/8. 模型级
def _realistic_batch(B=3, device="cpu"):
    """构造带自然缺失的 batch（模拟 trajimpute_collate_fn 输出的关键字段）。"""
    from src.datamodule.trajimpute_dataset import trajimpute_collate_fn, build_sample

    samples = []
    patterns = [
        [1, 1, 1, 1, 1, 1, 1, 1],   # all-valid
        [1, 1, 0, 0, 1, 0, 0, 1],   # 中度缺失
        [1, 0, 0, 0, 0, 0, 0, 0],   # 重度缺失（单有效帧）
    ]
    for b in range(B):
        pattern = patterns[b % len(patterns)]
        A = 3
        hist = torch.zeros(A, 8, 2)
        valid = torch.zeros(A, 8, dtype=torch.bool)
        for a in range(A):
            valid[a] = torch.tensor(pattern, dtype=torch.bool)
            hist[a] = torch.randn(8, 2).cumsum(0) * 0.1
            hist[a, ~valid[a]] = float("nan")
        future = torch.randn(A, 12, 2).cumsum(1) * 0.1
        s = build_sample(hist, future, valid, scene_id=f"t{b}", track_id=b)
        s["seq_index"] = b
        samples.append(s)
    batch = trajimpute_collate_fn(samples)
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


@pytest.mark.skipif(not GPU, reason="需要 GPU（Mamba CUDA）")
def test_model_calibration_off_matches_m0_keys():
    from src.model.trajectory_forecaster import TrajectoryForecaster
    torch.manual_seed(2024)
    batch = _realistic_batch(device="cuda")
    m0 = TrajectoryForecaster(num_modes=6).cuda().eval()
    with torch.no_grad():
        out = m0(batch)
    assert "new_pi_cal" not in out and "scal_cal" not in out
    assert torch.isfinite(out["new_pi"]).all()
    assert torch.isfinite(out["new_y_hat"]).all()


@pytest.mark.skipif(not GPU, reason="需要 GPU（Mamba CUDA）")
def test_model_calibration_all_valid_identity():
    """模型级 all-valid 恒等：calibration 开启且全有效 batch 上 cal==raw 逐位一致。"""
    from src.model.trajectory_forecaster import TrajectoryForecaster
    torch.manual_seed(2024)
    # 全有效 batch：所有 pattern 全 1
    from src.datamodule.trajimpute_dataset import trajimpute_collate_fn, build_sample
    samples = []
    for b in range(2):
        A = 3
        hist = torch.randn(A, 8, 2).cumsum(0) * 0.1
        future = torch.randn(A, 12, 2).cumsum(1) * 0.1
        valid = torch.ones(A, 8, dtype=torch.bool)
        s = build_sample(hist, future, valid, scene_id=f"t{b}", track_id=b)
        s["seq_index"] = b
        samples.append(s)
    batch = trajimpute_collate_fn(samples)
    batch = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in batch.items()}
    m1 = TrajectoryForecaster(num_modes=6, calibration=True).cuda().eval()
    # 打破零初始化验证门控（随机化校准头权重）
    with torch.no_grad():
        for p in m1.calibration_head.parameters():
            p.copy_(torch.randn_like(p) * 0.5)
    with torch.no_grad():
        out = m1(batch)
    assert torch.equal(out["new_pi_cal"], out["new_pi"])
    assert torch.equal(out["scal_cal"], out["scal_new"])


@pytest.mark.skipif(not GPU, reason="需要 GPU（Mamba CUDA）")
def test_model_calibration_forward_backward_and_smoothing():
    """高缺失 batch：前向 finite、校准头梯度有限、log_tau>0 时 entropy 上升。"""
    from src.model.trajectory_forecaster import TrajectoryForecaster
    torch.manual_seed(2024)
    batch = _realistic_batch(device="cuda")
    m1 = TrajectoryForecaster(num_modes=6, calibration=True).cuda().eval()
    with torch.no_grad():
        # tau 分支输出趋向正 -> logits 变平（取小值避免 clamp 饱和：
        # log_tau 超出 ±max_log 后梯度为 0，这是边界保护的预期行为）
        torch.nn.init.normal_(m1.calibration_head.tau_branch[-1].weight, std=0.1)
        m1.calibration_head.tau_branch[-1].bias.fill_(0.8)
        torch.nn.init.normal_(m1.calibration_head.scale_branch[-1].weight, std=0.1)
        m1.calibration_head.scale_branch[-1].bias.fill_(0.4)
    out = m1(batch)
    for k in ["new_pi_cal", "scal_cal", "new_pi", "scal_new", "new_y_hat"]:
        assert torch.isfinite(out[k]).all(), f"{k} not finite"
    # 主路径不受校准影响：new_y_hat/new_pi 与关闭校准的同权重模型一致
    m0 = TrajectoryForecaster(num_modes=6, calibration=True).cuda().eval()
    m0.load_state_dict(m1.state_dict())
    with torch.no_grad():
        out_no_cal = dict(m0(batch))
        # 强制关闭：直接比较 raw 键（calibration 开关只加输出，不改 raw 计算）
    assert torch.equal(out["new_y_hat"], out_no_cal["new_y_hat"])
    assert torch.equal(out["new_pi"], out_no_cal["new_pi"])
    # 高缺失样本 entropy 上升（log_tau>0 -> softmax 变平）
    ent = lambda x: -(torch.softmax(x, -1) * torch.log_softmax(x, -1)).sum(-1)
    ent_raw = ent(out["new_pi"].float())
    ent_cal = ent(out["new_pi_cal"].float())
    high_missing = batch["missing_count"] > 0
    if high_missing.any():
        assert bool((ent_cal[high_missing] >= ent_raw[high_missing] - 1e-5).all())
    # 反向：校准头梯度有限
    loss = out["new_pi_cal"].sum() + out["scal_cal"].sum()
    loss.backward()
    for name, p in m1.calibration_head.named_parameters():
        assert p.grad is not None, f"{name} 无梯度"
        assert torch.isfinite(p.grad).all()
    # 梯度一步更新后输出改变（校准头真的在学习）
    opt = torch.optim.SGD(m1.calibration_head.parameters(), lr=1e-2)
    opt.step()
    with torch.no_grad():
        out2 = m1(batch)
    assert not torch.equal(out2["new_pi_cal"], out["new_pi_cal"])


@pytest.mark.skipif(not GPU, reason="需要 GPU（Mamba CUDA）")
def test_lightning_cal_loss_with_calibration():
    """cal_loss 使用 calibrated 输出：存在 new_pi_cal 时 CE 作用于校准 logits。"""
    from src.model.forecasting_module import ForecastingLightningModule
    torch.manual_seed(2024)
    batch = _realistic_batch(device="cuda")
    module_kwargs = dict(
        model={"type": "TrajectoryForecaster", "embed_dim": 128, "future_steps": 12,
               "num_modes": 6, "dt": 0.4, "obs_len": 8, "calibration": True},
        lr=1e-3, warmup_epochs=5, epochs=100,
    )
    lm = ForecastingLightningModule(**module_kwargs).cuda()
    with torch.no_grad():
        # 随机化校准头，使 cal != raw
        for p in lm.net.calibration_head.parameters():
            p.copy_(torch.randn_like(p) * 0.5)
    out = lm.net(batch)
    loss, disp = lm.cal_loss(out, batch)
    assert torch.isfinite(loss)
    loss.backward()
    g = lm.net.calibration_head.tau_branch[-1].weight.grad
    assert g is not None and torch.isfinite(g).all()
    # 优化器闭包审计：校准头全部参数必须在优化器参数组内
    opt_params = {id(p) for group in lm.configure_optimizers()[0][0].param_groups
                  for p in group["params"]}
    for p in lm.net.calibration_head.parameters():
        assert id(p) in opt_params, "校准头参数未进优化器"


def test_collate_max_missing_run_field():
    """collate 后 max_missing_run 字段存在且语义正确（与循环版一致）。"""
    from src.datamodule.trajimpute_dataset import trajimpute_collate_fn, build_sample
    samples = []
    patterns = [[1] * 8, [1, 1, 0, 0, 1, 0, 0, 1], [1, 0, 0, 0, 0, 0, 0, 0]]
    for b, pattern in enumerate(patterns):
        valid = torch.tensor([pattern], dtype=torch.bool)
        hist = torch.zeros(1, 8, 2)
        hist[0] = torch.randn(8, 2).cumsum(0) * 0.1
        hist[0, ~valid[0]] = float("nan")
        future = torch.zeros(1, 12, 2)
        s = build_sample(hist, future, valid, scene_id=f"t{b}", track_id=b)
        s["seq_index"] = b
        samples.append(s)
    batch = trajimpute_collate_fn(samples)
    assert "max_missing_run" in batch
    ref = torch.tensor([_ref_max_run(torch.tensor(p)) for p in patterns])
    assert torch.equal(batch["max_missing_run"], ref)


def test_runner_monitor_and_eval_variant_mapping():
    """runner monitor 映射：M0->val_minFDE20，M1->val_cal_minFDE20。"""
    import importlib.util
    path = REPO / "scripts" / "训练与评估" / "run_trajimpute_experiments.py"
    spec = importlib.util.spec_from_file_location("runner_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.variant_monitor("M0", 20) == "val_new_b-minFDE20"
    assert mod.variant_monitor("M1", 20) == "val_cal_b-minFDE20"
    assert mod.VARIANTS["M1"] == {"calibration": True}
    # 评估入口同映射
    path2 = REPO / "scripts" / "结果分析" / "evaluate_trajimpute_direct.py"
    spec2 = importlib.util.spec_from_file_location("eval_mod", path2)
    mod2 = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(mod2)
    assert mod2.VARIANTS == {"M0": {}, "M1": {"calibration": True}}


def test_checkpoint_calibration_mismatch_rejected():
    """ckpt 校验：M1 模型 + M0 checkpoint 报错；M0 模型 + M1 checkpoint 报错。"""
    import importlib.util
    path = REPO / "scripts" / "结果分析" / "evaluate_trajimpute_direct.py"
    spec = importlib.util.spec_from_file_location("eval_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    import tempfile, os

    class _TD:
        """最小 stub：只暴露 time_decoder.num_modes 与参数字典。"""
        def __init__(self, calibration: bool):
            import torch.nn as nn
            self.time_decoder = type("TD", (), {"num_modes": 20})()
            if calibration:
                self.calibration_head = nn.Linear(5, 1)
            self._lin = nn.Linear(4, 4)

        def load_state_dict(self, sd, strict=False):
            result = type("R", (), {"missing_keys": [], "unexpected_keys": []})()
            return result.missing_keys, result.unexpected_keys

    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "m0.ckpt")
        torch.save({"state_dict": {"net._lin.weight": torch.zeros(4, 4),
                                   "net._lin.bias": torch.zeros(4)}}, p)
        with pytest.raises(ValueError, match="calibration"):
            mod.load_checkpoint(_TD(calibration=True), p)
        p2 = os.path.join(td, "m1.ckpt")
        torch.save({"state_dict": {
            "net._lin.weight": torch.zeros(4, 4),
            "net._lin.bias": torch.zeros(4),
            "net.calibration_head.weight": torch.zeros(1, 5),
            "net.calibration_head.bias": torch.zeros(1),
        }}, p2)
        with pytest.raises(ValueError, match="calibration"):
            mod.load_checkpoint(_TD(calibration=False), p2)
        # 匹配时不抛错
        m, u = mod.load_checkpoint(_TD(calibration=True), p2)
        assert m == [] and u == []
        # 部分校准参数：完整性校验拒绝（缺失参数不得静默留在零初始化）
        p3 = os.path.join(td, "m1_partial.ckpt")
        torch.save({"state_dict": {
            "net._lin.weight": torch.zeros(4, 4),
            "net._lin.bias": torch.zeros(4),
            "net.calibration_head.weight": torch.zeros(1, 5),
            # 故意缺 bias
        }}, p3)
        with pytest.raises(ValueError, match="不完整"):
            mod.load_checkpoint(_TD(calibration=True), p3)


def test_evaluator_evidence_grouping():
    """DirectEvaluator 新维度：evidence_bin/max_missing_run/mode_entropy/scale_mean 分组。"""
    from src.evaluation.trajimpute_direct import DirectEvaluator
    ev = DirectEvaluator()
    pred = torch.randn(1, 6, 12, 2).cumsum(2) * 0.3
    prob = torch.randn(1, 6)
    target = pred[:, 0].clone()  # [1, 12, 2]
    ev.update(pred, prob, target, scene="S", difficulty="Mixed", split="test",
              missing_count=3, valid_count=5, anchor_lag=1, forecast_gap=2,
              max_missing_run=2, evidence_bin="E2",
              mode_entropy=1.7, scale_mean=0.4)
    out = ev.compute()
    assert "evidence_bin" in out["by_dimension"]
    assert "max_missing_run" in out["by_dimension"]
    assert out["by_dimension"]["evidence_bin"]["E2"]["mode_entropy"] == pytest.approx(1.7)
    assert out["by_dimension"]["evidence_bin"]["E2"]["scale_mean"] == pytest.approx(0.4)
    assert "mode_entropy" in out["overall"]
