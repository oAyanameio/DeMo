"""ETH/UCY 模型单元测试。

测试 actor-only ModelForecast 的初始化与前向传播。
"""

import pytest
import torch

from src.model.model_forecast import ModelForecast, select_history_readout
from src.model.layers.evidence_state import (
    GapConditionedFeatureControl,
    ObservedWriteGapState,
)

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


def _make_model(readout_mode="last", evidence_state_mode="none"):
    return ModelForecast(
        embed_dim=128,
        future_steps=12,
        num_actor_types=1,
        num_modes=6,  # 显式指定：正式主链默认 20，本测试只测机制不测默认值
        readout_mode=readout_mode,
        evidence_state_mode=evidence_state_mode,
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

    def test_forward_last_valid_readout(self):
        """C1-A forward path supports terminal-missing actor histories."""
        model = _make_model(readout_mode="last_valid")
        model.eval()
        data = _make_dummy_batch_actor_only(B=2, N=3, obs_len=8, pred_len=12)
        data["x_valid_mask"][0, 0, -1] = False
        data["x_valid_mask"][0, 1, -2:] = False
        data["x_valid_mask"][1, 2, 0] = False

        with torch.no_grad():
            out = model(data)

        assert out["y_hat"].shape == (2, 6, 12, 2)
        assert torch.isfinite(out["y_hat"]).all()

    def test_forward_observed_write_gap_state(self):
        """C3 forward path supports gapped histories with finite outputs."""
        model = _make_model(
            readout_mode="last",
            evidence_state_mode="observed_write_gap",
        )
        model.eval()
        data = _make_dummy_batch_actor_only(B=2, N=3, obs_len=8, pred_len=12)
        data["x_gap_steps"] = torch.zeros(2, 3, 8, device=DEVICE)
        data["x_valid_mask"][0, 0, 3:6] = False
        data["x_gap_steps"][0, 0, 3:6] = torch.tensor(
            [1.0, 2.0, 3.0], device=DEVICE
        )
        data["x_positions_diff"][0, 0, 3:6] = 0
        data["x_velocity_diff"][0, 0, 3:6] = 0

        with torch.no_grad():
            out = model(data)

        assert out["y_hat"].shape == (2, 6, 12, 2)
        assert torch.isfinite(out["y_hat"]).all()

    def test_forward_integrated_evidence_conditioning(self):
        """C4 combines propagated state, adaptive anchor, and evidence conditioning."""
        model = _make_model(readout_mode="last", evidence_state_mode="integrated")
        model.eval()
        data = _make_dummy_batch_actor_only(B=2, N=3, obs_len=8, pred_len=12)
        data["x_gap_steps"] = torch.zeros(2, 3, 8, device=DEVICE)
        data["x_valid_mask"][0, 0, 4:6] = False
        data["x_gap_steps"][0, 0, 4:6] = torch.tensor([1.0, 2.0], device=DEVICE)
        with torch.no_grad():
            out = model(data)
        assert out["new_y_hat"].shape == (2, 6, 12, 2)
        assert torch.isfinite(out["new_pi"]).all()


def test_last_valid_readout_selects_each_actor_last_observation():
    """C1-A must gather the last valid timestep independently per actor."""
    sequence = torch.tensor(
        [
            [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]],
            [[5.0, 50.0], [6.0, 60.0], [7.0, 70.0], [8.0, 80.0]],
        ]
    )
    valid_mask = torch.tensor(
        [[True, False, True, False], [False, True, False, False]]
    )

    output = select_history_readout(sequence, valid_mask, mode="last_valid")

    expected = torch.tensor([[3.0, 30.0], [6.0, 60.0]])
    torch.testing.assert_close(output, expected)


def test_last_valid_readout_matches_last_for_all_valid_history():
    """All-valid C1-A must preserve M0's fixed-final-timestep readout."""
    sequence = torch.randn(3, 8, 16)
    valid_mask = torch.ones(3, 8, dtype=torch.bool)

    output_m0 = select_history_readout(sequence, valid_mask, mode="last")
    output_c1 = select_history_readout(sequence, valid_mask, mode="last_valid")

    torch.testing.assert_close(output_c1, output_m0)


def _make_deterministic_evidence_state(dim=4):
    module = ObservedWriteGapState(input_dim=dim, state_dim=dim)
    with torch.no_grad():
        module.observation_proj.weight.copy_(torch.eye(dim))
        module.observation_proj.bias.zero_()
        module.propagation_rate.fill_(1.0)
    return module


def test_observed_write_is_zero_on_invalid_frames():
    """Invalid frames may propagate state but must not write observations."""
    module = _make_deterministic_evidence_state()
    tokens = torch.randn(2, 4, 4)
    valid = torch.tensor([[True, False, True, False], [True, True, False, True]])
    gaps = torch.tensor([[0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]])

    _, trace = module(tokens, valid, gaps, return_trace=True)

    assert torch.equal(
        trace["observation_write"][~valid],
        torch.zeros_like(trace["observation_write"][~valid]),
    )
    assert torch.equal(trace["effective_write_gate"], valid)


def test_gap_length_changes_propagation_without_observation_write():
    """Longer missing gaps must alter propagation while write remains disabled."""
    module = _make_deterministic_evidence_state()
    tokens = torch.tensor([[[1.0, 2.0, 3.0, 4.0], [9.0, 9.0, 9.0, 9.0]]])
    valid = torch.tensor([[True, False]])

    state_short, trace_short = module(
        tokens, valid, torch.tensor([[0.0, 1.0]]), return_trace=True
    )
    state_long, trace_long = module(
        tokens, valid, torch.tensor([[0.0, 4.0]]), return_trace=True
    )

    assert not torch.allclose(state_short, state_long)
    assert not torch.allclose(
        trace_short["propagation_output"][:, 1],
        trace_long["propagation_output"][:, 1],
    )
    assert torch.count_nonzero(trace_long["observation_write"][:, 1]) == 0


def test_gap_propagation_uses_elapsed_time_without_compounding_gap_counts():
    """A three-step gap decays by exp(-3λ), not exp(-(1+2+3)λ)."""
    module = _make_deterministic_evidence_state(dim=1)
    tokens = torch.tensor([[[2.0], [99.0], [99.0], [99.0]]])
    valid = torch.tensor([[True, False, False, False]])
    gaps = torch.tensor([[0.0, 1.0, 2.0, 3.0]])

    _, trace = module(tokens, valid, gaps, return_trace=True)

    rate = torch.nn.functional.softplus(module.propagation_rate)[0]
    expected = torch.tensor(2.0) * torch.exp(-3.0 * rate)
    torch.testing.assert_close(trace["state_sequence"][0, 3, 0], expected)


def test_invalid_token_values_do_not_change_evidence_state():
    """Masked placeholder token values must not influence C3 state."""
    module = _make_deterministic_evidence_state()
    tokens = torch.randn(2, 5, 4)
    valid = torch.tensor(
        [[True, False, True, False, True], [True, True, False, False, True]]
    )
    gaps = torch.tensor(
        [[0.0, 1.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 2.0, 0.0]]
    )
    perturbed = tokens.clone()
    perturbed[~valid] = 1000.0 * torch.randn_like(perturbed[~valid])

    state_a = module(tokens, valid, gaps)
    state_b = module(perturbed, valid, gaps)

    torch.testing.assert_close(state_a, state_b)


def test_c3_preserves_paired_backbone_initialization():
    """Adding C3 parameters must not shift initialization of shared modules."""
    torch.manual_seed(2024)
    baseline = ModelForecast(num_modes=6, evidence_state_mode="none")
    torch.manual_seed(2024)
    c3 = ModelForecast(num_modes=6, evidence_state_mode="observed_write_gap")

    c3_state = c3.state_dict()
    for name, value in baseline.state_dict().items():
        torch.testing.assert_close(value, c3_state[name], rtol=0, atol=0)


def test_c3_construction_preserves_training_rng_stream():
    """Optional C3 parameters must not shift later dropout/data RNG draws."""
    torch.manual_seed(2024)
    ModelForecast(num_modes=6, evidence_state_mode="none")
    baseline_rng = torch.random.get_rng_state()

    torch.manual_seed(2024)
    ModelForecast(num_modes=6, evidence_state_mode="observed_write_gap")
    c3_rng = torch.random.get_rng_state()

    assert torch.equal(baseline_rng, c3_rng)


def test_c3_zero_initialized_residual_matches_m0_output():
    """C3 starts from the paired baseline before learning state residuals."""
    torch.manual_seed(2024)
    baseline = ModelForecast(num_modes=6, evidence_state_mode="none").to(DEVICE).eval()
    torch.manual_seed(2024)
    c3 = ModelForecast(
        num_modes=6,
        readout_mode="last",
        evidence_state_mode="observed_write_gap",
    ).to(DEVICE).eval()
    batch = _make_dummy_batch_actor_only(B=2, N=3, obs_len=8, pred_len=12)
    batch["x_gap_steps"] = torch.zeros(2, 3, 8, device=DEVICE)
    batch["x_valid_mask"][0, 0, 4:6] = False
    batch["x_gap_steps"][0, 0, 4:6] = torch.tensor([1.0, 2.0], device=DEVICE)

    with torch.no_grad():
        baseline_out = baseline(batch)
        c3_out = c3(batch)

    for key in ["y_hat", "pi", "scal", "new_y_hat", "new_pi", "scal_new"]:
        assert torch.equal(baseline_out[key], c3_out[key])


def test_c3_zero_residual_has_live_training_gradient():
    """Identity initialization must still let the residual branch start learning."""
    module = ObservedWriteGapState(input_dim=4, state_dim=4)
    module.zero_initialize_residual()
    tokens = torch.randn(2, 5, 4, requires_grad=True)
    valid = torch.tensor(
        [[True, False, True, False, True], [True, True, False, False, True]]
    )
    gaps = torch.tensor(
        [[0.0, 1.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 2.0, 0.0]]
    )

    residual = module.forward_residual(tokens, valid, gaps)
    residual.sum().backward()

    assert module.residual_proj.weight.grad.abs().sum() > 0


def test_c2_gap_control_has_no_observation_write_and_preserves_all_valid():
    """C2 only reacts to elapsed gaps and is an exact no-op for all-valid history."""
    module = GapConditionedFeatureControl(input_dim=4)
    with torch.no_grad():
        module.residual_proj.weight.copy_(torch.eye(4))
        module.residual_proj.bias.zero_()
        module.propagation_rate.fill_(0.1)

    tokens = torch.ones(1, 4, 4)
    all_valid_gaps = torch.zeros(1, 4)
    gapped = torch.tensor([[0.0, 1.0, 2.0, 0.0]])

    all_valid_residual = module.forward_residual(tokens, all_valid_gaps)
    gapped_residual = module.forward_residual(tokens, gapped)

    torch.testing.assert_close(all_valid_residual, torch.zeros_like(all_valid_residual))
    assert torch.count_nonzero(gapped_residual[:, 1:3]) > 0
    torch.testing.assert_close(gapped_residual[:, 0], torch.zeros_like(gapped_residual[:, 0]))
    torch.testing.assert_close(gapped_residual[:, 3], torch.zeros_like(gapped_residual[:, 3]))
