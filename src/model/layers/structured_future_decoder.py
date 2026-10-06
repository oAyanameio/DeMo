import torch
import torch.nn as nn
from .transformer_blocks import Cross_Block, Block
import torch.nn.functional as F
from .mamba.vim_mamba import init_weights, create_block
from functools import partial
from timm.models.layers import DropPath, to_2tuple
try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


class CoupledTrajectoryHead(nn.Module):
    def __init__(self, future_len=60, dim=128):
        super().__init__()
        self.future_horizon = future_len  # Legacy attribute: _future_len.
        # Legacy attribute: gaussian.
        self.trajectory_mean_head = nn.Sequential(
            nn.Linear(dim, 64), 
            nn.GELU(), 
            nn.Linear(64, 2)
        )
        # Legacy attribute: score.
        self.mode_score_head = nn.Sequential(
            nn.Linear(dim, 64), 
            nn.GELU(), 
            nn.Linear(64, 1),
        )
        # Legacy attribute: scale.
        self.trajectory_scale_head = nn.Sequential(
            nn.Linear(dim, 64), 
            nn.GELU(), 
            nn.Linear(64, 2)
        )
    
    def forward(self, input):
        res = self.trajectory_mean_head(input)
        scal = F.elu_(self.trajectory_scale_head(input), alpha=1.0) + 1.0 + 0.0001
        input = input.max(dim=2)[0]  
        score = self.mode_score_head(input).squeeze(-1)

        return res, score, scal


class ModeTrajectoryHead(nn.Module):
    def __init__(self, future_len=60, dim=128):
        super().__init__()
        self.future_horizon = future_len  # Legacy attribute: _future_len.
        # Legacy attribute: gaussian.
        self.trajectory_mean_head = nn.Sequential(
            nn.Linear(dim, 256), 
            nn.GELU(), 
            nn.Linear(256, self.future_horizon*2)
        )
        # Legacy attribute: score.
        self.mode_score_head = nn.Sequential(
            nn.Linear(dim, 64), 
            nn.GELU(), 
            nn.Linear(64, 1),
        )
        # Legacy attribute: scale.
        self.trajectory_scale_head = nn.Sequential(
            nn.Linear(dim, 256), 
            nn.GELU(), 
            nn.Linear(256, self.future_horizon*2)
        )
    
    def forward(self, input):
        B, M, _ = input.shape
        res = self.trajectory_mean_head(input).view(B, M, self.future_horizon, 2) 
        scal = F.elu_(self.trajectory_scale_head(input), alpha=1.0) + 1.0 + 0.0001
        scal = scal.view(B, M, self.future_horizon, 2) 
        score = self.mode_score_head(input).squeeze(-1)

        return res, score, scal
    

class FutureDistributionDecoder(nn.Module):
    # Legacy class name: TimeDecoder. Mode Query, State Query, and Hybrid
    # Coupling terminology intentionally remains unchanged.
    def __init__(self, future_len=60, dim=128, num_modes=20):
        super().__init__()
        self.num_modes = num_modes

        ###### State Consistency Module ######
        # state cross attention
        # Legacy attribute: cross_block_time.
        self.state_context_layers = nn.ModuleList(
            Cross_Block()
            for i in range(2)
        )

        # state bidirectional mamba
        self.timequery_embed_mamba = nn.ModuleList(  
            [
                create_block(  
                    d_model=dim,
                    layer_idx=i,
                    drop_path=0.2,  
                    bimamba=True,  
                    rms_norm=True,  
                )
                for i in range(2)
            ]
        )  
        # Legacy attributes: timequery_norm_f, timequery_drop_path.
        self.state_sequence_norm = RMSNorm(dim, eps=1e-5)
        self.state_sequence_drop_path = DropPath(0.2)

        # MLP for state query
        self.state_forecast_head = nn.Sequential(
            nn.Linear(dim, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, dim),
            nn.GELU(),
            nn.Linear(dim, 64),
            nn.GELU(),
            nn.Linear(64, 2),
        )

        ###### Mode Localization Module ######
        # mode self attention
        # Legacy attribute: self_block_mode.
        self.mode_self_attention_layers = nn.ModuleList(
            Block()
            for i in range(3)
        )

        # mode cross attention
        # Legacy attribute: cross_block_mode.
        self.mode_context_layers = nn.ModuleList(
            Cross_Block()
            for i in range(3)
        )

        # mode query initialization
        # Legacy attribute: multi_modal_query_embedding.
        self.mode_query_embedding = nn.Embedding(num_modes, dim)
        self.register_buffer('modal', torch.arange(num_modes).long())

        # MLP for mode query
        # Legacy attribute: predictor.
        self.mode_trajectory_head = ModeTrajectoryHead(future_len)

        ###### Hybrid Coupling Module ######
        # hybrid self attention
        # Legacy attribute: self_block_dense.
        self.coupling_joint_layers = nn.ModuleList(
            Block()
            for i in range(3)
        )

        # hybrid cross attention
        # Legacy attribute: cross_block_dense.
        self.coupling_context_layers = nn.ModuleList(
            Cross_Block()
            for i in range(3)
        )

        # mode self attention for hybrid spatiotemporal queries
        # Legacy attribute: self_block_different_mode.
        self.coupling_mode_interaction_layers = nn.ModuleList(
            Block()
            for i in range(3)
        )

        # state bidirectional mamba for hybrid spatiotemporal queries 
        self.coupled_temporal_mamba = nn.ModuleList(  
            [
                create_block(  
                    d_model=dim,
                    layer_idx=i,
                    drop_path=0.2,  
                    bimamba=True,  
                    rms_norm=True,  
                )
                for i in range(2)
            ]
        )
        # Legacy attributes: dense_norm_f, dense_drop_path.
        self.coupled_temporal_norm = RMSNorm(dim, eps=1e-5)
        self.coupled_temporal_drop_path = DropPath(0.2)

        # MLP for final output
        # Legacy attribute: predictor_dense.
        self.coupled_trajectory_head = CoupledTrajectoryHead(future_len)

    def forward(self, mode, encoding, mask=None):
        # Dynamic state consistency
        for blk in self.state_context_layers:
            mode = blk(mode, encoding, key_padding_mask=mask)
        
        residual = None
        for blk_mamba in self.timequery_embed_mamba:
            mode, residual = blk_mamba(mode, residual)
        fused_add_norm_fn = rms_norm_fn if isinstance(self.state_sequence_norm, RMSNorm) else layer_norm_fn
        mode = fused_add_norm_fn(
            self.state_sequence_drop_path(mode),
            self.state_sequence_norm.weight,
            self.state_sequence_norm.bias,
            eps=self.state_sequence_norm.eps,
            residual=residual,
            prenorm=False,
            residual_in_fp32=True  
        )
        
        dense_pred = self.state_forecast_head(mode)

        mode_tmp = mode
        
        # Directional intention localization
        multi_modal_query = self.mode_query_embedding(self.modal)
        mode_query = encoding[:, 0]
        mode = mode_query[:, None] + multi_modal_query

        for blk in self.mode_context_layers:
            mode = blk(mode, encoding, key_padding_mask=mask)
        for blk in self.mode_self_attention_layers:
            mode = blk(mode)

        y_hat, pi, scal = self.mode_trajectory_head(mode)

        # Hybrid query coupling
        mode_dense = mode[:, :, None] + mode_tmp[:, None, :]
        B, M, T, C = mode_dense.shape
        
        mode_dense = mode_dense.reshape(B, -1, C)
        for blk in self.coupling_context_layers:
            mode_dense = blk(mode_dense, encoding, key_padding_mask=mask)
        for blk in self.coupling_joint_layers:
            mode_dense = blk(mode_dense)
        mode_dense = mode_dense.reshape(B, M, T, C)
        
        mode_dense = mode_dense.transpose(1, 2).reshape(-1, M, C)
        for blk in self.coupling_mode_interaction_layers:
            mode_dense = blk(mode_dense)
        mode_dense = mode_dense.reshape(B, -1, M, C).transpose(1, 2)

        mode_dense = mode_dense.reshape(-1, T, C)
        residual = None
        for blk_mamba in self.coupled_temporal_mamba:
            mode_dense, residual = blk_mamba(mode_dense, residual)
        fused_add_norm_fn = rms_norm_fn if isinstance(self.coupled_temporal_norm, RMSNorm) else layer_norm_fn
        mode_dense = fused_add_norm_fn(
            self.coupled_temporal_drop_path(mode_dense),
            self.coupled_temporal_norm.weight,
            self.coupled_temporal_norm.bias,
            eps=self.coupled_temporal_norm.eps,
            residual=residual,
            prenorm=False,
            residual_in_fp32=True  
        )
        mode_dense = mode_dense.reshape(B, M, T, C)

        y_hat_new, pi_new, scal_new = self.coupled_trajectory_head(mode_dense)

        return dense_pred, y_hat, pi, mode, y_hat_new, pi_new, mode_dense, scal, scal_new


# Legacy names kept for old imports.
GMMPredictor_dense = CoupledTrajectoryHead
GMMPredictor = ModeTrajectoryHead
TimeDecoder = FutureDistributionDecoder
