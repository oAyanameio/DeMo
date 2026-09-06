from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.transformer_blocks import Block
from .layers.time_decoder import TimeDecoder
from .layers.mamba.vim_mamba import init_weights, create_block
from functools import partial
from timm.models.layers import DropPath, to_2tuple
try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


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
        num_modes: int = 6,
        bimamba: bool = False,
        dt: float = 0.4,
        obs_len: int = 8,
        use_gap_condition: bool = False,
        use_observation_features: bool = False,
        use_missing_summary: bool = False,
    ) -> None:
        super().__init__()
        # v3 缺失感知条件化通路（默认关闭：参数集与既有 checkpoint 完全一致）。
        # 开启时 focal forecast_gap_steps 经 MLP 融入 time embedding（State Query
        # 初始化路径），并输出 focal_anchor_lag/forecast_gap 供 Mode Query 与
        # Hybrid Coupling 的后续条件化消费（选题 §五.3 r_h 通路）。
        self.use_gap_condition = use_gap_condition
        if use_gap_condition:
            self.gap_embed = nn.Sequential(
                nn.Linear(1, 64), nn.GELU(), nn.Linear(64, embed_dim)
            )

        # M1_obs（方案 §3.1）：历史输入追加 4 个掩码派生时间步特征
        # [x_gap_steps/obs_len, x_prev_valid_gap/obs_len, x_motion_valid,
        #  x_motion_run/(obs_len-1)]，输入维度 4 -> 8。
        self.use_observation_features = use_observation_features
        self.obs_len = obs_len
        hist_input_dim = 4 + (4 if use_observation_features else 0)

        # M2_history（方案 §3.1/§2.6）：x_missing_summary -> embed_dim 条件向量，
        # 加到历史 actor token（TypeEmbedding 之后、场景编码之前）。
        self.use_missing_summary = use_missing_summary

        self.future_steps = future_steps
        self.dt = dt
        self.num_actor_types = num_actor_types

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

        # M2_history：摘要编码器。输出层零初始化 -> 初始 r_i ≡ 0，
        # 开启开关即刻与 M0 数值等价（不影响既有训练动态）。
        if use_missing_summary:
            self.missing_summary_embed = nn.Sequential(
                nn.Linear(6, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim),
            )
            nn.init.zeros_(self.missing_summary_embed[-1].weight)
            nn.init.zeros_(self.missing_summary_embed[-1].bias)

        self.time_decoder = TimeDecoder(future_len=future_steps, dim=embed_dim, num_modes=num_modes)

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.actor_type_embed, std=0.02)

        self.apply(self._init_weights)
        # missing_summary_embed 输出层的零初始化必须在 self.apply 之后重申，
        # 否则会被 _init_weights 的 xavier_uniform 覆盖
        if self.use_missing_summary:
            nn.init.zeros_(self.missing_summary_embed[-1].weight)
            nn.init.zeros_(self.missing_summary_embed[-1].bias)

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

        hist_feat_parts = [
            data["x_positions_diff"],
            data["x_velocity_diff"][..., None],
            hist_valid_mask[..., None],
        ]
        # M1_obs：追加 4 个掩码派生时间步特征（顺序固定，方案 §3.1）
        if self.use_observation_features:
            missing_keys = [k for k in
                            ("x_gap_steps", "x_prev_valid_gap", "x_motion_valid", "x_motion_run")
                            if k not in data]
            if missing_keys:
                raise ValueError(
                    f"use_observation_features=True requires batch fields "
                    f"{missing_keys} (enable via missing-aware datasets)"
                )
            obs_len = self.obs_len
            hist_feat_parts.extend([
                data["x_gap_steps"][..., None] / obs_len,
                data["x_prev_valid_gap"][..., None] / obs_len,
                data["x_motion_valid"][..., None],
                data["x_motion_run"][..., None] / (obs_len - 1),
            ])
        hist_feat = torch.cat(hist_feat_parts, dim=-1)

        B, N, L, D = hist_feat.shape
        hist_feat = hist_feat.view(B * N, L, D)
        hist_feat_key_valid = hist_key_valid_mask.view(B * N)

        # unidirectional mamba
        actor_feat = self.hist_embed_mlp(hist_feat[hist_feat_key_valid].contiguous())
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
        pos_embed = self.pos_embed(pos_feat)

        actor_type_embed = self.actor_type_embed[data["x_attr"][..., 2].long()]
        actor_feat = actor_feat + actor_type_embed

        # M2_history：摘要条件向量加到（真实 actor 的）历史 actor token。
        # padding actor 不参与（key_valid=False 的行乘 0），不污染场景上下文。
        if self.use_missing_summary:
            if "x_missing_summary" not in data:
                raise ValueError(
                    "use_missing_summary=True requires batch field "
                    "'x_missing_summary' (enable via missing-aware datasets)"
                )
            summary_condition = self.missing_summary_embed(data["x_missing_summary"])
            actor_feat = actor_feat + summary_condition * hist_key_valid_mask[..., None]

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

        # v3 缺失感知：focal forecast_gap 条件化 State Query 初始化
        # （anchor_lag/forecast_gap 同时进 ret_dict，供 Mode Query/Hybrid
        #  Coupling 条件化路径消费；v1/v2 下恒为 1/0，无影响）
        focal_anchor_lag = data.get("x_anchor_lag_steps", None)
        focal_forecast_gap = data.get("x_forecast_gap_steps", None)
        if focal_forecast_gap is not None:
            gap_focal = focal_forecast_gap[:, 0].float()  # [B]
            if self.use_gap_condition:
                mode = mode + self.gap_embed(gap_focal.view(-1, 1, 1)).expand_as(mode)

        # decoder module with decoupled queries
        dense_predict, y_hat, pi, x_mode, new_y_hat, new_pi, mode_dense, scal, scal_new = \
        self.time_decoder(mode, x_encoder, mask=~key_valid_mask)

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

        # v3 缺失条件通路：focal 时间间隔暴露给后续条件化模块（不作为轨迹特征输入）
        if focal_anchor_lag is not None:
            ret_dict["focal_anchor_lag"] = focal_anchor_lag[:, 0]
            ret_dict["focal_forecast_gap"] = focal_forecast_gap[:, 0]

        return ret_dict
