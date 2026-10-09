"""M1：自然缺失证据条件化的输出校准头（missingness-aware output calibration）。

方案依据：docs/research/缺失历史预测实验方案.md §6（2026-10-08 版）。

机制：
    log_tau   = missing_ratio · f_tau(e)
    log_scale = missing_ratio · f_scale(e)
    new_pi_cal = new_pi / exp(log_tau)
    scal_cal   = scal_new · exp(log_scale)

约束（方案 §6.3）：
    - evidence 只来自 focal 历史 mask 派生字段（valid_ratio / missing_ratio /
      anchor_lag / forecast_gap / max_missing_run，全部除以 obs_len 归一化）；
    - all-valid 样本 missing_ratio=0 -> log_tau=log_scale=0，恒等映射；
    - f_* 最后一层零初始化，训练初始与 M0 等价；零初始化用 constant_
      覆写（不消耗 RNG draw，保持配对初始化的 RNG 流不变）；
    - log 量经平滑限幅 log = bound * tanh(raw)：保证 tau、scale 有限、
      为正且有上界，且边界附近梯度饱和但不为 0（clamp 在边界梯度恒 0，
      会让校准头进入边界后失去学习信号）；
    - 只校准 new_pi / scal_new，不改变候选轨迹均值 new_y_hat；
    - 候选轨迹均值不参与校准；校准头参数进入 checkpoint。
"""

import torch
import torch.nn as nn


def max_missing_run_batch(hist_valid: torch.Tensor) -> torch.Tensor:
    """向量化最长连续缺失段：hist_valid [B, T] bool -> [B] long。

    纯 mask 派生（无位置/未来参与）。与逐帧循环版 max_missing_run 等价。
    """
    if hist_valid.dim() != 2:
        raise ValueError(f"hist_valid must be [B, T], got {tuple(hist_valid.shape)}")
    missing = (~hist_valid).long()
    c = torch.cumsum(missing, dim=1)
    breaks = torch.where(hist_valid, c, torch.full_like(c, -1))
    last_break = torch.cummax(breaks, dim=1).values
    run_len = c - last_break.clamp(min=0)
    return run_len.max(dim=1).values


class MissingnessCalibrationHead(nn.Module):
    """样本级标量校准：temperature（mode logits）+ scale（Laplace 尺度）。

    输入 evidence 特征 [B, 5]（构造见 build_evidence_features），
    输出 [B] 的 log_tau / log_scale（已乘 missing_ratio 门控并限幅）。
    """

    EVIDENCE_DIM = 5

    def __init__(self, hidden_dim: int = 32, max_log_magnitude: float = 2.0):
        super().__init__()
        self.max_log_magnitude = float(max_log_magnitude)
        # 两个独立分支共享同一输入特征；输出层零初始化保证初始等价 M0
        self.tau_branch = nn.Sequential(
            nn.Linear(self.EVIDENCE_DIM, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1),
        )
        self.scale_branch = nn.Sequential(
            nn.Linear(self.EVIDENCE_DIM, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1),
        )
        self.reset_calibration_parameters()

    def reset_calibration_parameters(self):
        """最后一层权重/偏置零初始化（constant_，不消耗 RNG）。

        首层保持默认初始化：训练开始后梯度即可流动（输出层为零时，
        ∂out/∂last_w = h ≠ 0，梯度活）；恒等起点只由输出层零化保证。
        """
        for branch in (self.tau_branch, self.scale_branch):
            last = branch[-1]
            nn.init.constant_(last.weight, 0.0)
            nn.init.constant_(last.bias, 0.0)

    def forward(self, evidence: torch.Tensor) -> tuple:
        """evidence [B, 5] -> (log_tau [B], log_scale [B])。

        evidence[..., 1] 必须是 missing_ratio（门控项）。
        """
        raw_tau = self.tau_branch(evidence).squeeze(-1)      # [B]
        raw_scale = self.scale_branch(evidence).squeeze(-1)  # [B]
        missing_ratio = evidence[..., 1]
        log_tau = missing_ratio * raw_tau
        log_scale = missing_ratio * raw_scale
        if self.max_log_magnitude is not None and self.max_log_magnitude > 0:
            # 平滑限幅：边界附近梯度饱和但不为 0（clamp 边界梯度恒 0）
            log_tau = self.max_log_magnitude * torch.tanh(log_tau / self.max_log_magnitude)
            log_scale = self.max_log_magnitude * torch.tanh(log_scale / self.max_log_magnitude)
        return log_tau, log_scale


def build_evidence_features(
    valid_ratio: torch.Tensor,
    anchor_lag: torch.Tensor,
    forecast_gap: torch.Tensor,
    max_missing_run: torch.Tensor,
    obs_len: int = 8,
) -> torch.Tensor:
    """按方案 §6.2 组装 evidence 向量 e = [valid_ratio, missing_ratio,
    anchor_lag/obs_len, forecast_gap/obs_len, max_missing_run/obs_len]。

    Args:
        valid_ratio: [B]（或任意前置维度）focal 历史有效帧比例，取值 [0,1]
        anchor_lag:  [B] 最后有效观测距观测窗口末端间隔（步）
        forecast_gap:[B] 最后有效观测到未来第一步的时间间隔（步）
        max_missing_run: [B] focal 历史最长连续缺失长度（步）
        obs_len: 观测窗口长度（归一化分母）
    Returns:
        [B, 5] float 张量（与输入同 device/dtype 转 float）
    """
    valid_ratio = valid_ratio.float()
    missing_ratio = 1.0 - valid_ratio
    evidence = torch.stack([
        valid_ratio,
        missing_ratio,
        anchor_lag.float() / obs_len,
        forecast_gap.float() / obs_len,
        max_missing_run.float() / obs_len,
    ], dim=-1)
    return evidence
