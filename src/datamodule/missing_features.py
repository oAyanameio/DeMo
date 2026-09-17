"""mask-only 缺失特征构造。

纯函数：只由历史掩码 [A, T] 派生缺失距离特征，不读取位置、未来轨迹、
condition 名、scene 或 fold —— 无未来泄漏。

字段语义（完整历史时的中性约定）：
    gap_steps  距最近有效观测的离散步数；当前帧有效为 0；
               从窗口起点到当前都无有效帧时为 t + 1
"""

import torch


def build_missing_features(history_mask: torch.Tensor) -> dict:
    """Build mask-only history features.

    Args:
        history_mask: Boolean tensor shaped [A, T], True = 该帧可见。

    Returns:
        dict，含 gap_steps [A, T]（float）。
    """
    if not torch.is_tensor(history_mask):
        raise ValueError(f"history_mask must be torch.Tensor, got {type(history_mask)}")
    if history_mask.dim() != 2:
        raise ValueError(
            f"history_mask must be 2D [A, T], got shape {tuple(history_mask.shape)}"
        )
    if history_mask.dtype != torch.bool:
        raise ValueError(f"history_mask must be bool dtype, got {history_mask.dtype}")
    A, T = history_mask.shape
    if T < 2:
        raise ValueError(f"history window T must be >= 2, got {T}")

    device = history_mask.device
    j = torch.arange(T, device=device).unsqueeze(0).expand(A, T)  # [A, T]

    # --- gap_steps：距最近有效观测的步数（含"从未见过有效帧"的 t+1 约定）---
    # lv[t] = t 及之前最后一个有效帧下标（无则 -1）；缺失帧 gap = t - lv，
    # 从未有效时 = t - (-1) = t + 1，自动满足前缀约定。
    cand = torch.where(history_mask, j, torch.full_like(j, -1))
    lv = torch.cummax(cand, dim=1).values  # [A, T]
    gap_steps = torch.where(history_mask, torch.zeros(A, T, device=device), (j - lv).float())

    return {"gap_steps": gap_steps}
