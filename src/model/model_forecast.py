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
        num_modes: int = 20,
        bimamba: bool = False,
        dt: float = 0.4,
        obs_len: int = 8,
        use_gap_scaling: bool = False,
        use_missing_summary: bool = False,
    ) -> None:
        super().__init__()
        # E1（Sports-Traj 方案 §五 阶段 A）：gap-conditioned temporal scaling。
        # R1 判负后修正版（2026-09-17，见 docs/results/实验总汇总.md §8）：
        #   alpha(g) = exp(-tanh(MLP(g/obs_len)) * GAP_LOG_ALPHA_MAX)
        # 修正点：
        #   1) 锚定：有效帧（gap=0）在 forward 中显式 where 强制 alpha=1，
        #      完整历史对该机制精确无操作（不论训练后 bias 为何值）；
        #   2) 有界：tanh∈[-1,1] 限制 log alpha ∈ [-max, +max]（默认 max=2，
        #      即 alpha ∈ [exp(-2), exp(2)]≈[0.135, 7.39]），杜绝训后 8千~1万倍
        #      全局尺度捷径；"缺失降权"仍是训练学得的方向。
        self.use_gap_scaling = use_gap_scaling
        if use_gap_scaling:
            self.gap_log_alpha_max = 2.0
            self.gap_scale_mlp = nn.Sequential(
                nn.Linear(1, 64), nn.GELU(), nn.Linear(64, 1)
            )
            nn.init.zeros_(self.gap_scale_mlp[-1].weight)
            nn.init.zeros_(self.gap_scale_mlp[-1].bias)

        # 模块一 per-actor missing-summary conditioning（方案 §3.3）：
        # R1 判负后修正版：注入形式 r_i = missing_rate_i · MLP(s_i)，
        # 零缺失 actor（missing_rate=0）注入恒为零——即使 MLP bias 学到
        # 非零值也不泄漏到完整历史 actor；missing_rate 取 summary[0]。
        # 非 Sports-Traj GSM（scene-time ghost token 不在本模块）。
        # 不新增 token、不动 encoding[:,0] 的 focal 语义；padding actor
        # 由 key_valid 掩码屏蔽。末层零初始化 => 初始注入 ≡ 0。
        self.use_missing_summary = use_missing_summary
        if use_missing_summary:
            self.missing_summary_embed = nn.Sequential(
                nn.Linear(6, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim)
            )
            nn.init.zeros_(self.missing_summary_embed[-1].weight)
            nn.init.zeros_(self.missing_summary_embed[-1].bias)

        self.future_steps = future_steps
        self.dt = dt
        self.num_actor_types = num_actor_types
        self.obs_len = obs_len

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

        self.time_decoder = TimeDecoder(future_len=future_steps, dim=embed_dim, num_modes=num_modes)

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.actor_type_embed, std=0.02)

        self.apply(self._init_weights)

        # E1 gap_scale_mlp / missing_summary_embed 同理：零初始化必须在 apply 之后重申
        if self.use_gap_scaling:
            nn.init.zeros_(self.gap_scale_mlp[-1].weight)
            nn.init.zeros_(self.gap_scale_mlp[-1].bias)
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

        hist_feat = torch.cat([
            data["x_positions_diff"],
            data["x_velocity_diff"][..., None],
            hist_valid_mask[..., None],
        ], dim=-1)

        B, N, L, D = hist_feat.shape
        hist_feat = hist_feat.view(B * N, L, D)
        hist_feat_key_valid = hist_key_valid_mask.view(B * N)

        # unidirectional mamba
        actor_feat = self.hist_embed_mlp(hist_feat[hist_feat_key_valid].contiguous())
        # E1：gap-conditioned temporal scaling（修正版）——缺失距离越长，
        # 该时刻特征可靠性越低。alpha = exp(-tanh(MLP(g/obs_len))·max)：
        # g=0 恒 alpha=1（完整历史精确无操作）；log alpha 有界于 ±max，
        # 杜绝全局尺度捷径（R1 判负根因之一）。
        if self.use_gap_scaling:
            if "x_gap_steps" not in data:
                raise ValueError(
                    "use_gap_scaling=True requires batch field 'x_gap_steps' "
                    "(enable via trajimpute/missing-aware datasets)"
                )
            gap_all = data["x_gap_steps"] / self.obs_len          # [B, N, L]
            gap_sel = gap_all.view(-1, L)[hist_feat_key_valid]     # [M, L]
            log_alpha = torch.tanh(self.gap_scale_mlp(gap_sel[..., None])) \
                * self.gap_log_alpha_max                           # [M, L, 1]
            alpha = torch.exp(-log_alpha)                          # [M, L, 1]
            # 显式锚定：有效帧（gap=0）强制 alpha=1——不论 MLP bias 训成
            # 什么值，完整历史对该机制精确无操作（R1 修正要求 1）
            alpha = torch.where(gap_sel[..., None] == 0,
                                torch.ones_like(alpha), alpha)
            actor_feat = actor_feat * alpha
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

        # 模块一：缺失摘要条件向量加到各 actor 自身 token。
        # padding actor 不参与（key_valid=False 的行乘 0），不污染场景上下文。
        if self.use_missing_summary:
            if "x_missing_summary" not in data:
                raise ValueError(
                    "use_missing_summary=True requires batch field "
                    "'x_missing_summary' (enable via trajimpute/missing-aware datasets)"
                )
            summary_input = data["x_missing_summary"].float()
            # 修正版：missing_rate 门控——零缺失 actor 注入恒为零，
            # 消除 R1 判负根因之二（完整历史 token 被重参数化）。
            missing_rate = summary_input[..., :1]                  # [B, N, 1]
            summary_cond = self.missing_summary_embed(summary_input) * missing_rate
            actor_feat = actor_feat + summary_cond * hist_key_valid_mask[..., None]


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

        return ret_dict
