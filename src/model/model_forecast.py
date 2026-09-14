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
        use_evidence_clock: bool = False,
        use_social_evidence: bool = False,
        use_observation_features: bool = False,
        use_missing_summary: bool = False,
        use_motion_features: bool = False,
    ) -> None:
        super().__init__()
        # B1（P0.5 运动证据增强）：历史输入追加 4 维有效运动统计
        # [x_velocity, x_accel, x_turn_rate, x_motion_run/(obs_len-1)]，
        # 全部由历史窗口内有效观测派生（跨缺口归一化速度、其变化率、
        # 相邻有效步航向变化率、连续有效运动长度）。默认关闭 = M0-current。
        self.use_motion_features = use_motion_features
        # v3 缺失感知条件化通路（默认关闭：参数集与既有 checkpoint 完全一致）。
        # 开启时 focal forecast_gap_steps 经 MLP 融入 time embedding（State Query
        # 初始化路径），并输出 focal_anchor_lag/forecast_gap 供 Mode Query 与
        # Hybrid Coupling 的后续条件化消费（选题 §五.3 r_h 通路）。
        self.use_gap_condition = use_gap_condition
        self.use_evidence_clock = use_evidence_clock
        # M2'（P2，主贡献）：社会证据补偿——时间对齐的跨行人证据注意力。
        # 邻居帧级 token（相对几何+时间归一+有效位 → 线性投影）作为
        # attention 的 key/value；query 为场景编码后的 focal token。
        # 双重掩码：邻居帧无效或邻居为 padding actor 的 token 不参与注意力。
        # 补偿向量经学习门控残差注入 focal 表征——不输出重建历史轨迹，
        # 与插补路线（MS-TIP/GC-VRNN/BRITS）划清边界。
        self.use_social_evidence = use_social_evidence
        if use_social_evidence:
            self.neighbor_frame_embed = nn.Sequential(
                nn.Linear(4, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim))
            self.social_attn = nn.MultiheadAttention(
                embed_dim, num_heads=4, batch_first=True)
            self.social_proj = nn.Sequential(
                nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim))
            self.social_scale = nn.Parameter(torch.zeros(1))  # P2: 零初始化残差缩放，初始严格等价 M0
        if use_gap_condition:
            self.gap_embed = nn.Sequential(
                nn.Linear(1, 64), nn.GELU(), nn.Linear(64, embed_dim)
            )

        # M1_obs（方案 §3.1）：历史输入追加 4 个掩码派生时间步特征
        # [x_gap_steps/obs_len, x_prev_valid_gap/obs_len, x_motion_valid,
        #  x_motion_run/(obs_len-1)]，输入维度 4 -> 8。
        self.use_observation_features = use_observation_features
        self.obs_len = obs_len
        hist_input_dim = 4 + (4 if use_observation_features else 0) \
            + (4 if use_motion_features else 0)

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

        # B1：追加 4 维有效运动统计（P0.5；trajimpute/missing 数据集提供）
        if self.use_motion_features:
            missing_keys = [k for k in
                            ("x_velocity", "x_accel", "x_turn_rate", "x_motion_run")
                            if k not in data]
            if missing_keys:
                raise ValueError(
                    f"use_motion_features=True requires batch fields "
                    f"{missing_keys} (enable via trajimpute/missing-aware datasets)"
                )
            obs_len_m = self.obs_len
            hist_feat_parts.extend([
                data["x_velocity"][..., None],
                data["x_accel"][..., None],
                data["x_turn_rate"][..., None],
                data["x_motion_run"][..., None] / (obs_len_m - 1),
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

        # M2' 社会证据补偿（2026-09-10）：时间对齐的跨行人证据注意力。
        # focal 的缺失帧证据从邻居同期观测聚合：邻居帧 token =
        # [相对位置(focal最后有效位置系), 相对几何归一, t/obs_len, 邻居帧有效位]，
        # 双重掩码（邻居帧无效 × padding actor）后 attention 聚合到 focal token，
        # 门控残差注入。缺失帧越多可用邻居证据越重要——补偿量天然与 gap 相关。
        if self.use_social_evidence:
            # 相对几何：邻居各帧位置 - focal 最后有效位置（缺失帧被
            # valid_mask 置零占位，不产生虚假相对几何）
            focal_last = data["x_centers"][:, 0:1, :]            # [B,1,2] focal 锚点
            rel_pos = data["x_positions"][:, 1:, :, :] - focal_last.unsqueeze(2)  # [B,N-1,L,2]
            t_norm = torch.arange(self.obs_len, device=x_encoder.device).float().view(1, 1, -1, 1) / self.obs_len
            neigh_valid = data["x_valid_mask"][:, 1:, :].float()                    # [B,N-1,L]
            pad_mask = data["x_key_valid_mask"][:, 1:]                # [B,N-1] 或 [B,N-1,L]
            if pad_mask.dim() == 2:
                pad_mask = pad_mask.unsqueeze(-1).expand(-1, -1, self.obs_len)
            pad_valid = pad_mask.float()                              # [B,N-1,L]
            frame_feat = torch.cat([
                rel_pos / 5.0,                        # 相对位置（米→归一尺度）
                t_norm.expand(B, rel_pos.size(1), self.obs_len, 1),
                (neigh_valid * pad_valid).unsqueeze(-1),
            ], dim=-1)                                # [B,N-1,L,4]
            tokens = self.neighbor_frame_embed(frame_feat)      # [B,N-1,L,D]
            B_, Nn, L_, D_ = tokens.shape
            tokens = tokens.view(B_, Nn * L_, D_)                # [B, (N-1)*L, D]
            token_valid = (neigh_valid * pad_valid).view(B_, Nn * L_).bool()  # 双重掩码
            query = x_encoder[:, 0:1, :]                         # [B,1,D] focal token
            # 边界防护：全 token 无效的样本（邻居恰好全缺失/padding）跳过补偿，
            # 否则 softmax 对空集产生 NaN（Hard 高缺失下真实会出现）
            has_any = token_valid.any(dim=-1, keepdim=True)      # [B,1]
            attn_out = torch.zeros_like(query)
            if has_any.all():
                attn_out, _ = self.social_attn(
                    query, tokens, tokens, key_padding_mask=~token_valid)
            elif has_any.any():
                sub, _ = self.social_attn(
                    query[has_any[:, 0]], tokens[has_any[:, 0]], tokens[has_any[:, 0]],
                    key_padding_mask=~token_valid[has_any[:, 0]])
                attn_out[has_any[:, 0]] = sub
            # has_any 全 False 的样本 attn_out 保持零 → 缩放后无补偿。
            # 残差注入：focal + social_scale · proj(attn_out)，scale=0 初始严格等价 M0
            delta = self.social_proj(attn_out)
            x_encoder = torch.cat([
                query + self.social_scale * delta,
                x_encoder[:, 1:N, :],
            ], dim=1)

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

        # v3 缺失感知：focal forecast_gap 条件化 State Query 初始化
        # （anchor_lag/forecast_gap 同时进 ret_dict，供 Mode Query/Hybrid
        #  Coupling 条件化路径消费；v1/v2 下恒为 1/0，无影响）
        focal_anchor_lag = data.get("x_anchor_lag_steps", None)
        focal_forecast_gap = data.get("x_forecast_gap_steps", None)

        # state query initialization
        # dt 参数化：ETH/UCY 与 SDD 均为 0.4s/帧（frame stride=10 @ 2.5Hz）
        time = torch.arange(self.future_steps).long().to(x_encoder.device)
        time = time * self.dt + self.dt

        # M1' 证据时钟解码（2026-09-10）：State Query 的时间自变量从
        # "距历史窗口末端"改为"距最后有效观测"——t_evidence(t) = gap + t（帧），
        # 秒制 (gap + t) * dt。gap=1（完整历史）时与基线 time=(t+1)*dt 严格一致。
        # 同一未来帧在证据新（gap 小）与证据旧（gap 大）下获得不同时间嵌入，
        # 缺失直接改变输出分布的时间形状。
        if self.use_evidence_clock and focal_forecast_gap is not None:
            steps = torch.arange(self.future_steps).to(x_encoder.device).float()  # [T]
            gap_frames = focal_forecast_gap[:, 0].float().view(-1, 1)             # [B,1]
            t_evidence = (steps.view(1, -1) + gap_frames) * self.dt               # [B,T]
            mode = self.time_embedding_mlp(t_evidence.unsqueeze(-1))              # [B,T,D]
        else:
            time = time.unsqueeze(-1)
            mode = self.time_embedding_mlp(time)
            mode = mode.repeat(x_encoder.size(0), 1, 1)

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
