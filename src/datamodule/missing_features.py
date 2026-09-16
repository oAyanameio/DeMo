"""mask-only 缺失特征构造。

纯函数：只由历史掩码 [A, T] 派生缺失距离特征与逐 actor 缺失摘要，
不读取位置、未来轨迹、condition 名、scene 或 fold —— 无未来泄漏。

字段语义（完整历史时的中性约定）：
    gap_steps  距最近有效观测的离散步数；当前帧有效为 0；
               从窗口起点到当前都无有效帧时为 t + 1
    missing_summary [A, 6]（per-actor 窗口级：可见性 + 缺失时长，全部归一化；
               非 Sports-Traj GSM——后者是 scene-time 级 ghost token）
        0 missing_rate    缺失帧比例 mean(1-mask)
        1 longest_gap     最长连续缺失 max(gap_steps)/T
        2 missing_prefix  首个有效帧前的缺失帧数/T（全缺失为 1）
        3 tail_gap        末帧距最后有效帧的步数/(T-1)（=anchor lag；末帧有效为 0）
        4 gap_area        缺失时长质量 sum(gap_steps)/(T*T)
        5 valid_rate      有效帧比例 sum(mask)/T（=1-missing_rate 的冗余显式形式，
                          保留以让"可见性"独立于"缺失分布"成通道）
"""

import torch

GSM_SUMMARY_DIM = 6


def build_missing_features(history_mask: torch.Tensor) -> dict:
    """Build mask-only history features.

    Args:
        history_mask: Boolean tensor shaped [A, T], True = 该帧可见。

    Returns:
        dict，含 gap_steps [A, T]（float）与 missing_summary [A, 6]。
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

    # --- per-actor missing_summary [A, 6]（模块一：逐 actor 缺失摘要条件化）---
    missing = (~history_mask).float()
    missing_rate = missing.mean(dim=1)                                   # [A]
    longest_gap = gap_steps.max(dim=1).values / T                        # [A]
    idx_or_T = torch.where(history_mask, j, torch.full_like(j, T))
    first_valid = idx_or_T.min(dim=1).values                             # 无有效帧=T
    missing_prefix = first_valid.float() / T                             # 全缺失=1
    any_valid = history_mask.any(dim=1)
    last_valid = torch.where(any_valid, lv[:, -1], torch.full_like(lv[:, -1], -1))
    # 全缺失行（真实数据不出现，padding 极端）：tail_gap 饱和为 1 而非 8/7
    tail_gap = torch.where(
        any_valid,
        (T - 1 - last_valid).clamp(min=0).float() / (T - 1),
        torch.ones(A, device=device),
    )                                                              # [A]
    gap_area = gap_steps.sum(dim=1) / float(T * T)                       # [A]
    valid_rate = history_mask.float().mean(dim=1)                        # [A]

    missing_summary = torch.stack(
        [missing_rate, longest_gap, missing_prefix, tail_gap, gap_area, valid_rate],
        dim=1,
    )  # [A, 6]

    return {"gap_steps": gap_steps, "missing_summary": missing_summary}
