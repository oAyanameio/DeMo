"""Evidence-state layers for missing-aware temporal encoding."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GapConditionedFeatureControl(nn.Module):
    """Gap-only feature modulation control without observation writes.

    The control applies a learned decay to the current token only when the
    supplied elapsed gap is positive. It does not maintain a state and never
    consumes token validity or placeholder content as an observation write.
    """

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.propagation_rate = nn.Parameter(torch.full((self.input_dim,), 0.1))
        self.residual_proj = nn.Linear(self.input_dim, self.input_dim)

    def forward(self, tokens: torch.Tensor, gap_steps: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"tokens must be [B,T,F], got {tuple(tokens.shape)}")
        if gap_steps.shape != tokens.shape[:2]:
            raise ValueError(
                "gap_steps must be [B,T] matching tokens, "
                f"got {tuple(gap_steps.shape)} for {tuple(tokens.shape)}"
            )
        if tokens.shape[-1] != self.input_dim:
            raise ValueError(
                f"tokens feature dim must be {self.input_dim}, got {tokens.shape[-1]}"
            )
        if not torch.isfinite(gap_steps).all():
            raise ValueError("gap_steps must contain only finite values")
        if (gap_steps < 0).any():
            raise ValueError("gap_steps must be non-negative")

        gaps = gap_steps.to(device=tokens.device, dtype=tokens.dtype)
        rate = F.softplus(self.propagation_rate).to(dtype=tokens.dtype)
        decay_delta = torch.expm1(-gaps.unsqueeze(-1) * rate.unsqueeze(0))
        raw_delta = tokens * decay_delta
        residual = self.residual_proj(raw_delta)
        active_gap = gaps.gt(0).unsqueeze(-1)
        return residual * active_gap.to(dtype=residual.dtype)

    def forward_residual(self, tokens: torch.Tensor, gap_steps: torch.Tensor) -> torch.Tensor:
        """Return the per-token gap-conditioned residual."""
        return self(tokens, gap_steps)

    def zero_initialize_residual(self) -> None:
        """Start as an exact identity on the token path."""
        nn.init.zeros_(self.residual_proj.weight)
        nn.init.zeros_(self.residual_proj.bias)


class ObservedWriteGapState(nn.Module):
    """Single-state propagation with validity-gated observation writes.

    At each history step the state first propagates through a gap-dependent
    decay, then receives an observation write only when that frame is valid.
    Invalid token values are masked before the observation projection so
    placeholder coordinates cannot influence the state.
    """

    def __init__(self, input_dim: int, state_dim: int) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.state_dim = int(state_dim)
        self.observation_proj = nn.Linear(self.input_dim, self.state_dim)
        self.propagation_rate = nn.Parameter(torch.full((self.state_dim,), 0.1))
        self.residual_proj = nn.Linear(self.state_dim, self.input_dim)

    def forward(
        self,
        tokens: torch.Tensor,
        valid_mask: torch.Tensor,
        gap_steps: torch.Tensor,
        *,
        return_trace: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if tokens.ndim != 3:
            raise ValueError(f"tokens must be [B,T,F], got {tuple(tokens.shape)}")
        if valid_mask.shape != tokens.shape[:2]:
            raise ValueError(
                "valid_mask must be [B,T] matching tokens, "
                f"got {tuple(valid_mask.shape)} for {tuple(tokens.shape)}"
            )
        if gap_steps.shape != tokens.shape[:2]:
            raise ValueError(
                "gap_steps must be [B,T] matching tokens, "
                f"got {tuple(gap_steps.shape)} for {tuple(tokens.shape)}"
            )
        if tokens.shape[-1] != self.input_dim:
            raise ValueError(
                f"tokens feature dim must be {self.input_dim}, got {tokens.shape[-1]}"
            )
        if not torch.isfinite(gap_steps).all():
            raise ValueError("gap_steps must contain only finite values")
        if (gap_steps < 0).any():
            raise ValueError("gap_steps must be non-negative")

        valid = valid_mask.to(device=tokens.device, dtype=torch.bool)
        gaps = gap_steps.to(device=tokens.device, dtype=tokens.dtype)
        safe_tokens = torch.where(valid.unsqueeze(-1), tokens, torch.zeros_like(tokens))
        observation = self.observation_proj(safe_tokens)
        observation_write = observation * valid.unsqueeze(-1).to(observation.dtype)

        rate = F.softplus(self.propagation_rate).to(dtype=tokens.dtype)
        state = tokens.new_zeros(tokens.shape[0], self.state_dim)
        last_written_state = state
        outputs = []
        propagation_outputs = []
        effective_write_gate = valid
        for t in range(tokens.shape[1]):
            if t == 0:
                elapsed = gaps[:, t]
            else:
                elapsed = torch.where(
                    valid[:, t],
                    gaps[:, t - 1] + 1.0,
                    gaps[:, t],
                )
            decay = torch.exp(-elapsed.unsqueeze(-1) * rate.unsqueeze(0))
            propagated = last_written_state * decay
            candidate = propagated + observation_write[:, t]
            state = torch.where(valid[:, t].unsqueeze(-1), candidate, propagated)
            last_written_state = torch.where(
                valid[:, t].unsqueeze(-1), state, last_written_state
            )
            propagation_outputs.append(propagated)
            outputs.append(state)

        sequence = torch.stack(outputs, dim=1)
        if not return_trace:
            return sequence[:, -1]

        trace = {
            "state_sequence": sequence,
            "propagation_output": torch.stack(propagation_outputs, dim=1),
            "observation_write": observation_write,
            "effective_write_gate": effective_write_gate,
        }
        return sequence[:, -1], trace

    def forward_sequence(
        self,
        tokens: torch.Tensor,
        valid_mask: torch.Tensor,
        gap_steps: torch.Tensor,
    ) -> torch.Tensor:
        """Return the per-step propagated-and-written state sequence."""
        _, trace = self(tokens, valid_mask, gap_steps, return_trace=True)
        return trace["state_sequence"]

    def forward_residual(
        self,
        tokens: torch.Tensor,
        valid_mask: torch.Tensor,
        gap_steps: torch.Tensor,
    ) -> torch.Tensor:
        """Return a learned residual over the original token sequence."""
        state_sequence = self.forward_sequence(tokens, valid_mask, gap_steps)
        return self.residual_proj(state_sequence)

    def zero_initialize_residual(self) -> None:
        """Start as an exact identity on the token path."""
        nn.init.zeros_(self.residual_proj.weight)
        nn.init.zeros_(self.residual_proj.bias)


class EvidenceConditionedAnchor(nn.Module):
    """Identity-started fusion of final-time and last-valid actor states."""

    def __init__(self, feature_dim: int, evidence_dim: int = 4) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(evidence_dim, 32), nn.GELU(), nn.Linear(32, feature_dim)
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(self, propagated, last_valid, evidence):
        if propagated.shape != last_valid.shape:
            raise ValueError("anchor states must have identical shapes")
        if evidence.ndim != 2 or evidence.shape[0] != propagated.shape[0]:
            raise ValueError("evidence must be [A,F] aligned with states")
        alpha = torch.tanh(self.gate(evidence))
        return propagated + alpha * (last_valid - propagated)
