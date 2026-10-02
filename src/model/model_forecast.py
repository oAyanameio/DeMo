from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.transformer_blocks import Block
from .layers.time_decoder import TimeDecoder
from .layers.evidence_state import (
    GapConditionedFeatureControl,
    ObservedWriteGapState,
    EvidenceConditionedAnchor,
)
from .layers.mamba.vim_mamba import init_weights, create_block
from functools import partial
from timm.models.layers import DropPath, to_2tuple
try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


def select_history_readout(
    sequence: torch.Tensor,
    valid_mask: torch.Tensor,
    mode: str = "last",
) -> torch.Tensor:
    """Select an actor-level history summary from temporal features.

    Args:
        sequence: Temporal features with shape ``[A, T, D]``.
        valid_mask: Per-frame validity mask with shape ``[A, T]``.
        mode: ``"last"`` preserves M0's fixed-final-timestep readout;
            ``"last_valid"`` gathers each actor's last valid timestep.

    Raises:
        ValueError: If the shapes are incompatible, the mode is unknown, or
            an actor has no valid history frame in ``last_valid`` mode.
    """
    if sequence.ndim != 3:
        raise ValueError(f"sequence must be [A,T,D], got {tuple(sequence.shape)}")
    if valid_mask.ndim != 2 or valid_mask.shape != sequence.shape[:2]:
        raise ValueError(
            "valid_mask must be [A,T] matching sequence, "
            f"got {tuple(valid_mask.shape)} for {tuple(sequence.shape)}"
        )
    if mode == "last":
        return sequence[:, -1]
    if mode != "last_valid":
        raise ValueError(f"unknown history readout mode: {mode!r}")

    valid_mask = valid_mask.to(device=sequence.device, dtype=torch.bool)
    has_valid = valid_mask.any(dim=1)
    if not bool(has_valid.all()):
        raise ValueError("last_valid readout requires at least one valid frame per actor")

    time_indices = torch.arange(sequence.size(1), device=sequence.device)
    last_valid = torch.where(valid_mask, time_indices.unsqueeze(0), -1).amax(dim=1)
    actor_indices = torch.arange(sequence.size(0), device=sequence.device)
    return sequence[actor_indices, last_valid]


# only 'DeMo'
class ModelForecast(nn.Module):
    def __init__(
        self,
        embed_dim=128,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop_path=0.2,
        future_steps: int = 12,
        num_actor_types: int = 1,
        num_modes: int = 20,
        bimamba: bool = False,
        dt: float = 0.4,
        obs_len: int = 8,
        readout_mode: str = "last",
        evidence_state_mode: str = "none",
        evidence_conditioning: bool = False,
    ) -> None:
        super().__init__()
        if readout_mode not in {"last", "last_valid"}:
            raise ValueError(
                f"unknown history readout mode: {readout_mode!r}; "
                "expected 'last' or 'last_valid'"
            )
        if evidence_state_mode not in {"none", "gap_control", "observed_write_gap", "integrated"}:
            raise ValueError(
                f"unknown evidence state mode: {evidence_state_mode!r}; "
                "expected 'none', 'gap_control', 'observed_write_gap', or 'integrated'"
            )
        self.future_steps = future_steps
        self.dt = dt
        self.num_actor_types = num_actor_types
        self.obs_len = obs_len
        self.readout_mode = readout_mode
        self.evidence_state_mode = evidence_state_mode
        self.evidence_conditioning = bool(
            evidence_conditioning or evidence_state_mode == "integrated"
        )

        hist_input_dim = 4

        self.hist_embed_mlp = nn.Sequential(
            nn.Linear(hist_input_dim, 64),
            nn.GELU(),
            nn.Linear(64, embed_dim),
        )

        # Agent Encoding Mamba
        self.hist_embed_mamba = nn.ModuleList(  
            [
                create_block(  
                    d_model=embed_dim,
                    layer_idx=i,
                    drop_path=0.2,
                    bimamba=bimamba,
                    rms_norm=True,  
                )
                for i in range(4)
            ]
        )
        self.norm_f = RMSNorm(embed_dim, eps=1e-5)
        self.drop_path = DropPath(drop_path)

        self.pos_embed = nn.Sequential(
            nn.Linear(4, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Scene Context Transformer
        self.blocks = nn.ModuleList(
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=0.2,
            )
            for i in range(5)
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.actor_type_embed = nn.Parameter(torch.Tensor(num_actor_types, embed_dim))

        self.dense_predictor = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.GELU(), nn.Linear(256, future_steps * 2)
        )

        self.time_embedding_mlp = nn.Sequential(
            nn.Linear(1, 64), nn.GELU(), nn.Linear(64, embed_dim)
        )

        self.time_decoder = TimeDecoder(
            future_len=future_steps,
            dim=embed_dim,
            num_modes=num_modes,
            evidence_conditioning=self.evidence_conditioning,
        )

        # Keep C3 out of the baseline initialization RNG stream. Shared
        # modules initialize first; the optional state layer is created and
        # initialized afterward so paired M0/C1-A weights remain bit-identical.
        self.gap_control = None
        self.evidence_state = None
        self.evidence_anchor = None

        self.initialize_weights()
        if evidence_state_mode in {"gap_control", "observed_write_gap", "integrated"}:
            rng_state = torch.random.get_rng_state()
            if evidence_state_mode == "gap_control":
                self.gap_control = GapConditionedFeatureControl(input_dim=embed_dim)
                self.gap_control.apply(self._init_weights)
                self.gap_control.zero_initialize_residual()
            else:
                self.evidence_state = ObservedWriteGapState(
                    input_dim=embed_dim, state_dim=embed_dim
                )
                self.evidence_state.apply(self._init_weights)
                self.evidence_state.zero_initialize_residual()
            if evidence_state_mode == "integrated":
                self.evidence_anchor = EvidenceConditionedAnchor(embed_dim)
                self.time_decoder.build_evidence_conditioning()
            torch.random.set_rng_state(rng_state)

    @staticmethod
    def _max_missing_run(valid: torch.Tensor) -> torch.Tensor:
        current = torch.zeros(valid.shape[0], device=valid.device, dtype=torch.long)
        best = current.clone()
        for t in range(valid.shape[1]):
            current = torch.where(valid[:, t], torch.zeros_like(current), current + 1)
            best = torch.maximum(best, current)
        return best

    def _evidence_features(self, valid: torch.Tensor) -> torch.Tensor:
        valid_count = valid.sum(-1).float() / float(self.obs_len)
        idx = torch.arange(valid.shape[1], device=valid.device)
        last_valid = torch.where(valid, idx.unsqueeze(0), -1).amax(-1)
        anchor_lag = (self.obs_len - 1 - last_valid).clamp(min=0).float()
        forecast_gap = (self.obs_len - last_valid).clamp(min=1).float()
        max_run = self._max_missing_run(valid).float()
        return torch.stack((
            valid_count,
            anchor_lag / float(max(1, self.obs_len - 1)),
            forecast_gap / float(self.obs_len),
            max_run / float(self.obs_len),
        ), dim=-1)

    def initialize_weights(self):
        nn.init.normal_(self.actor_type_embed, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def load_from_checkpoint(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        state_dict = {
            k[len("net.") :]: v for k, v in ckpt.items() if k.startswith("net.")
        }
        return self.load_state_dict(state_dict=state_dict, strict=False)

    def forward(self, data):
        ###### Scene context encoding ###### 
        # agent encoding
        hist_valid_mask = data["x_valid_mask"]
        hist_key_valid_mask = data["x_key_valid_mask"]

        hist_feat = torch.cat([
            data["x_positions_diff"],
            data["x_velocity_diff"][..., None],
            hist_valid_mask[..., None],
        ], dim=-1)

        B, N, L, D = hist_feat.shape
        hist_feat = hist_feat.view(B * N, L, D)
        hist_feat_key_valid = hist_key_valid_mask.view(B * N)

        actor_tokens = self.hist_embed_mlp(hist_feat[hist_feat_key_valid].contiguous())
        actor_valid = hist_valid_mask.view(B * N, L)[hist_feat_key_valid]
        if self.evidence_state_mode == "integrated":
            # Do not let a placeholder write into the unchanged Mamba stack.
            # The learned state residual can still propagate across this gap.
            actor_tokens = actor_tokens * actor_valid.unsqueeze(-1).to(actor_tokens.dtype)
        evidence_real = None
        if self.evidence_state_mode in {"gap_control", "observed_write_gap", "integrated"}:
            if "x_gap_steps" not in data:
                raise KeyError(
                    f"{self.evidence_state_mode} requires data['x_gap_steps']"
                )
            actor_gaps = data["x_gap_steps"].view(B * N, L)[hist_feat_key_valid]
            if self.evidence_state_mode == "gap_control":
                if self.gap_control is None:
                    raise RuntimeError("gap_control mode is missing its control module")
                actor_tokens = actor_tokens + self.gap_control.forward_residual(
                    actor_tokens, actor_gaps
                )
            else:
                if self.evidence_state is None:
                    raise RuntimeError("observed_write_gap mode is missing its evidence_state module")
                # C3: explicit propagation/write semantics before the unchanged
                # Mamba backbone. Invalid observations cannot write into state.
                actor_tokens = actor_tokens + self.evidence_state.forward_residual(
                    actor_tokens, actor_valid, actor_gaps
                )
            if self.evidence_state_mode == "integrated":
                evidence_real = self._evidence_features(actor_valid)

        # The temporal Mamba stack remains shared across M0/C1-A/C3.
        actor_feat = actor_tokens
        residual = None
        for blk_mamba in self.hist_embed_mamba:
            actor_feat, residual = blk_mamba(actor_feat, residual)
        fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
        actor_feat = fused_add_norm_fn(
            self.drop_path(actor_feat),
            self.norm_f.weight,
            self.norm_f.bias,
            eps=self.norm_f.eps,
            residual=residual,
            prenorm=False,
            residual_in_fp32=True
        )
        if self.evidence_state_mode == "integrated":
            last_state = actor_feat[:, -1]
            last_valid_state = select_history_readout(actor_feat, actor_valid, mode="last_valid")
            actor_feat = self.evidence_anchor(last_state, last_valid_state, evidence_real)
        else:
            actor_feat = select_history_readout(
                actor_feat,
                actor_valid,
                mode=self.readout_mode,
            )
        actor_feat_tmp = torch.zeros(
            B * N, actor_feat.shape[-1], dtype=actor_feat.dtype, device=actor_feat.device
        )
        actor_feat_tmp[hist_feat_key_valid] = actor_feat
        actor_feat = actor_feat_tmp.view(B, N, actor_feat.shape[-1])
        decoder_evidence = None
        if self.evidence_conditioning:
            if evidence_real is None:
                raise RuntimeError("integrated evidence features were not constructed")
            evidence_tmp = evidence_real.new_zeros((B * N, 4))
            evidence_tmp[hist_feat_key_valid] = evidence_real
            decoder_evidence = evidence_tmp.view(B, N, 4)[:, 0]

        # type embedding and position embedding
        x_centers = data["x_centers"]
        # 朝向输入：v3_noguard 下帧 7 可能缺失，x_angles[...,-1] 不再可靠；
        # 改用 dataset 提供的每 actor 最近有效运动对朝向 x_last_valid_angle。
        # 兼容：老 batch（无该键，v1/v2 旧 collate）回退 x_angles[..., -1]，
        # 数值与旧版完全一致（v1/v2 下两者相等）。
        if "x_last_valid_angle" in data:
            angles = data["x_last_valid_angle"]
        else:
            angles = data["x_angles"][:, :, -1]
        x_angles = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
        pos_feat = torch.cat([x_centers, x_angles], dim=-1)
        pos_embed = self.pos_embed(pos_feat)

        actor_type_embed = self.actor_type_embed[data["x_attr"][..., 2].long()]
        actor_feat = actor_feat + actor_type_embed


        # scene context features
        x_encoder = actor_feat
        key_valid_mask = data["x_key_valid_mask"]

        x_encoder = x_encoder + pos_embed

        #  intra-interaction learning for scene context features
        for blk in self.blocks:
            x_encoder = blk(x_encoder, key_padding_mask=~key_valid_mask)
        x_encoder = self.norm(x_encoder)

        ###### Trajectory decoding with decoupled queries ###### 
        new_y_hat = None
        new_pi = None
        dense_predict = None
        mode = None

        # outputs of other agents (handle N=1: no other agents)
        x_others = x_encoder[:, 1:N]
        if x_others.size(1) > 0:
            y_hat_others = self.dense_predictor(x_others).view(B, x_others.size(1), -1, 2)
        else:
            y_hat_others = x_encoder.new_zeros((B, 0, self.future_steps, 2))


        # state query initialization
        # dt 参数化：ETH/UCY 与 SDD 均为 0.4s/帧（frame stride=10 @ 2.5Hz）
        time = torch.arange(self.future_steps).long().to(x_encoder.device)
        time = time * self.dt + self.dt
        time = time.unsqueeze(-1)
        mode = self.time_embedding_mlp(time)
        mode = mode.repeat(x_encoder.size(0), 1, 1)

        # decoder module with decoupled queries
        dense_predict, y_hat, pi, x_mode, new_y_hat, new_pi, mode_dense, scal, scal_new = \
        self.time_decoder(
            mode, x_encoder, mask=~key_valid_mask, evidence=decoder_evidence
        )

        ret_dict = {
            "y_hat": y_hat,  # trajectory output from mode query
            "pi": pi,  # probability output from mode query
            "scal": scal,  # output for Laplace loss from mode query

            "dense_predict": dense_predict,  # trajectory output from state query

            "y_hat_others": y_hat_others,  # trajectory of other agents

            "new_y_hat": new_y_hat,  # final trajectory output
            "new_pi": new_pi,  # final probability output     
            "scal_new": scal_new,  # final output for Laplace loss
        }

        return ret_dict
