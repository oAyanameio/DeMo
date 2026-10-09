from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.transformer_blocks import Block
from .layers.structured_future_decoder import FutureDistributionDecoder
from .layers.missingness_calibration import (
    MissingnessCalibrationHead, build_evidence_features, max_missing_run_batch,
)
from .layers.mamba.vim_mamba import init_weights, create_block
from functools import partial
from timm.models.layers import DropPath, to_2tuple
try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


# Actor-centric trajectory forecaster. The legacy class name was ModelForecast.
class TrajectoryForecaster(nn.Module):
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
        calibration: bool = False,
        calibration_hidden_dim: int = 32,
        calibration_max_log: float = 2.0,
    ) -> None:
        super().__init__()
        self.future_steps = future_steps
        self.dt = dt
        self.num_actor_types = num_actor_types
        self.obs_len = obs_len

        hist_input_dim = 4

        # Legacy attribute: hist_embed_mlp.
        self.history_input_projection = nn.Sequential(
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

        # Legacy attribute: pos_embed.
        self.actor_context_projection = nn.Sequential(
            nn.Linear(4, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Scene Context Transformer
        # Legacy attribute: blocks.
        self.scene_interaction_blocks = nn.ModuleList(
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

        # Legacy attribute: dense_predictor.
        self.neighbor_prediction_head = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.GELU(), nn.Linear(256, future_steps * 2)
        )

        # Legacy attribute: time_embedding_mlp.
        self.future_time_embedding = nn.Sequential(
            nn.Linear(1, 64), nn.GELU(), nn.Linear(64, embed_dim)
        )

        # Legacy attribute: time_decoder. Protected Mode Query, State Query,
        # and Hybrid Coupling terminology remains unchanged.
        self.future_decoder = FutureDistributionDecoder(
            future_len=future_steps, dim=embed_dim, num_modes=num_modes
        )

        self.initialize_weights()

        # M1 校准头：在 initialize_weights() 之后创建，避免被 _init_weights
        # 的 xavier 覆写；其自身 reset_calibration_parameters() 用 constant_
        # 零初始化（不消耗 RNG draw），保证同 seed 下主链参数与 M0 逐位一致。
        self.calibration_enabled = bool(calibration)
        if self.calibration_enabled:
            self.calibration_head = MissingnessCalibrationHead(
                hidden_dim=calibration_hidden_dim,
                max_log_magnitude=calibration_max_log,
            )

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

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict,
        missing_keys, unexpected_keys, error_msgs,
    ):
        """Migrate legacy parameter paths during direct or Lightning resume loads."""
        replacements = (
            ("hist_embed_mlp.", "history_input_projection."),
            ("pos_embed.", "actor_context_projection."),
            ("blocks.", "scene_interaction_blocks."),
            ("dense_predictor.", "neighbor_prediction_head."),
            ("time_embedding_mlp.", "future_time_embedding."),
            ("time_decoder.predictor_dense.", "future_decoder.coupled_trajectory_head."),
            ("time_decoder.predictor.", "future_decoder.mode_trajectory_head."),
            ("time_decoder.cross_block_time.", "future_decoder.state_context_layers."),
            ("time_decoder.dense_predict.", "future_decoder.state_forecast_head."),
            ("time_decoder.timequery_norm_f.", "future_decoder.state_sequence_norm."),
            ("time_decoder.timequery_drop_path.", "future_decoder.state_sequence_drop_path."),
            ("time_decoder.timequery_embed_mamba.", "future_decoder.timequery_embed_mamba."),
            ("time_decoder.self_block_mode.", "future_decoder.mode_self_attention_layers."),
            ("time_decoder.cross_block_mode.", "future_decoder.mode_context_layers."),
            ("time_decoder.multi_modal_query_embedding.", "future_decoder.mode_query_embedding."),
            ("time_decoder.self_block_dense.", "future_decoder.coupling_joint_layers."),
            ("time_decoder.cross_block_dense.", "future_decoder.coupling_context_layers."),
            ("time_decoder.self_block_different_mode.", "future_decoder.coupling_mode_interaction_layers."),
            ("time_decoder.dense_embed_mamba.", "future_decoder.coupled_temporal_mamba."),
            ("time_decoder.dense_norm_f.", "future_decoder.coupled_temporal_norm."),
            ("time_decoder.dense_drop_path.", "future_decoder.coupled_temporal_drop_path."),
            ("time_decoder.", "future_decoder."),
        )
        head_replacements = (
            (".gaussian.", ".trajectory_mean_head."),
            (".score.", ".mode_score_head."),
            (".scale.", ".trajectory_scale_head."),
        )
        for key in list(state_dict):
            if not key.startswith(prefix):
                continue
            relative = key[len(prefix):]
            migrated = relative
            for old, new in replacements:
                if migrated.startswith(old):
                    migrated = new + migrated[len(old):]
                    break
            for old, new in head_replacements:
                migrated = migrated.replace(old, new)
            if migrated != relative:
                state_dict[prefix + migrated] = state_dict.pop(key)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )

    @property
    def time_decoder(self):
        """Legacy attribute alias for the future distribution decoder."""
        return self.future_decoder

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

        # unidirectional mamba
        actor_feat = self.history_input_projection(
            hist_feat[hist_feat_key_valid].contiguous()
        )
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

        actor_feat = actor_feat[:, -1]
        actor_feat_tmp = torch.zeros(
            B * N, actor_feat.shape[-1], dtype=actor_feat.dtype, device=actor_feat.device
        )
        actor_feat_tmp[hist_feat_key_valid] = actor_feat
        actor_feat = actor_feat_tmp.view(B, N, actor_feat.shape[-1])

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
        pos_embed = self.actor_context_projection(pos_feat)

        actor_type_embed = self.actor_type_embed[data["x_attr"][..., 2].long()]
        actor_feat = actor_feat + actor_type_embed


        # scene context features
        x_encoder = actor_feat
        key_valid_mask = data["x_key_valid_mask"]

        x_encoder = x_encoder + pos_embed

        #  intra-interaction learning for scene context features
        for blk in self.scene_interaction_blocks:
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
            y_hat_others = self.neighbor_prediction_head(x_others).view(
                B, x_others.size(1), -1, 2
            )
        else:
            y_hat_others = x_encoder.new_zeros((B, 0, self.future_steps, 2))


        # state query initialization
        # dt 参数化：ETH/UCY 与 SDD 均为 0.4s/帧（frame stride=10 @ 2.5Hz）
        time = torch.arange(self.future_steps).long().to(x_encoder.device)
        time = time * self.dt + self.dt
        time = time.unsqueeze(-1)
        mode = self.future_time_embedding(time)
        mode = mode.repeat(x_encoder.size(0), 1, 1)

        # decoder module with decoupled queries
        dense_predict, y_hat, pi, x_mode, new_y_hat, new_pi, mode_dense, scal, scal_new = \
        self.future_decoder(mode, x_encoder, mask=~key_valid_mask)

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

        # M1：自然缺失证据条件化校准（只改 new_pi/scal_new，不动 new_y_hat）
        if self.calibration_enabled:
            ret_dict.update(
                self._calibrate_outputs(new_pi, scal_new, data)
            )

        return ret_dict

    def _calibrate_outputs(self, new_pi, scal_new, data):
        """按方案 §6.3 计算 new_pi_cal / scal_cal 与诊断字段。

        evidence 只来自 focal(=actor 0) 历史 mask 派生字段；
        all-valid 样本 missing_ratio=0 -> 恒等映射（log_tau=log_scale=0）。

        边界（温度校准的数学性质）：logits/τ 不改变 mode 排序，因此
        同一 checkpoint 上 raw/cal 的 ADE@1/FDE@1/MR/min* 逐位相同；
        推理期直接变化的只有 b-minFDE（Brier 项用概率值）、mode entropy
        与 scale。M1 对排序/top-1 的改善只能来自训练期效应（校准后的
        CE/Laplace 梯度改变 logits 与轨迹头的联合学习），跨训练比较
        （M1 ckpt vs M0 ckpt）才有意义。
        """
        focal_valid = data["x_valid_mask"][:, 0]            # [B, T]
        valid_ratio = focal_valid.float().mean(dim=-1)      # [B]
        anchor_lag = data["x_anchor_lag_steps"][:, 0].float()
        forecast_gap = data["x_forecast_gap_steps"][:, 0].float()
        if "max_missing_run" in data:
            max_run = data["max_missing_run"].float()
        else:
            # 老数据路径回退：由 mask 现算（语义与 dataset 字段一致）
            max_run = max_missing_run_batch(focal_valid).float()
        evidence = build_evidence_features(
            valid_ratio, anchor_lag, forecast_gap, max_run, obs_len=self.obs_len
        )
        log_tau, log_scale = self.calibration_head(evidence)  # [B]
        # temperature：new_pi / exp(log_tau) —— 均匀化 mode logits
        new_pi_cal = new_pi / torch.exp(log_tau).view(-1, *([1] * (new_pi.dim() - 1)))
        # scale：[B,1,1,1] 广播到 scal_new [B, M, T, 2]
        scal_cal = scal_new * torch.exp(log_scale).view(-1, 1, 1, 1)
        return {
            "new_pi_cal": new_pi_cal,
            "scal_cal": scal_cal,
            "calibration_log_tau": log_tau,
            "calibration_log_scale": log_scale,
        }


# Legacy class name kept for old configs and external scripts.
ModelForecast = TrajectoryForecaster
