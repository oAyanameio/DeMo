"""mask-only 缺失特征构造（方案《缺失感知模型构建与实验验证方案》§2.3/§2.4，任务 1）。

纯函数：只由历史掩码 [A, T] 派生时间步观测特征与历史级缺失摘要，
不读取位置、未来轨迹、condition 名、scene 或 fold —— 无未来泄漏。

字段语义（完整历史时的中性约定）：
    gap_steps       距最近有效观测的离散步数；当前帧有效为 0；
                    从窗口起点到当前都无有效帧时为 t + 1
    prev_valid_gap  当前有效帧距前一有效帧的步数；当前帧无效/首个有效帧为 0
    motion_valid    当前位移步是否由相邻有效帧构成；t=0 恒 0
    motion_run      以当前时刻结尾的连续有效位移步长度；无效步为 0，从 1 起计
    missing_summary [A, 6] 顺序固定：
        [missing_rate, longest_gap_norm, missing_prefix,
         valid_motion_rate, last_motion_run, gap_area_norm]
        missing_rate      = mean(1 - mask)
        longest_gap_norm  = 最长连续缺失帧数 / T
        missing_prefix    = 首个有效帧之前的缺失帧数 / T（全缺失为 1）
        valid_motion_rate = mean(motion_valid[1:])
        last_motion_run   = motion_run[-1] / (T - 1)
        gap_area_norm     = sum(gap_steps) / (T * T)
"""

import torch

TIME_STEP_FEATURE_KEYS = (
    "gap_steps",
    "prev_valid_gap",
    "motion_valid",
    "motion_run",
)

SUMMARY_DIMS = 6


def build_missing_features(history_mask: torch.Tensor) -> dict:
    """Build mask-only history features.

    Args:
        history_mask: Boolean tensor shaped [A, T], True = 该帧可见。

    Returns:
        dict，含 gap_steps/prev_valid_gap/motion_valid/motion_run [A, T]
        与 missing_summary [A, 6]，全部 float。
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

    # --- prev_valid_gap：仅当当前帧有效且存在更早有效帧时 = t - 前一有效帧下标 ---
    lv_excl = torch.cat(
        [torch.full((A, 1), -1, dtype=torch.long, device=device), lv[:, :-1]], dim=1
    )
    has_earlier = lv_excl >= 0
    prev_valid_gap = torch.where(
        history_mask & has_earlier,
        (j - lv_excl).clamp(min=0).float(),
        torch.zeros(A, T, device=device),
    )

    # --- motion_valid / motion_run ---
    motion_valid = torch.zeros(A, T, device=device)
    motion_valid[:, 1:] = (history_mask[:, :-1] & history_mask[:, 1:]).float()
    # 连续 run：以最近一次 motion_valid==0 的位置为断点，run[t] = t - last_break
    breaks = torch.where(motion_valid > 0, torch.full_like(j, -1), j)
    breaks[:, 0] = 0  # t=0 恒为断点
    last_break = torch.cummax(breaks, dim=1).values
    motion_run = torch.where(
        motion_valid > 0, (j - last_break).float(), torch.zeros(A, T, device=device)
    )

    # --- missing_summary [A, 6] ---
    missing = (~history_mask).float()
    missing_rate = missing.mean(dim=1)
    # gap_steps 在连续缺失段内随 t 递增，max 即最长连续缺失帧数（含未闭合段）
    longest_gap_norm = gap_steps.max(dim=1).values / T
    idx_or_T = torch.where(history_mask, j, torch.full_like(j, T))
    first_valid = idx_or_T.min(dim=1).values
    missing_prefix = first_valid.float() / T  # 无有效帧时 = T/T = 1
    valid_motion_rate = motion_valid[:, 1:].mean(dim=1)
    last_motion_run = motion_run[:, -1] / (T - 1)
    gap_area_norm = gap_steps.sum(dim=1) / float(T * T)

    missing_summary = torch.stack(
        [
            missing_rate,
            longest_gap_norm,
            missing_prefix,
            valid_motion_rate,
            last_motion_run,
            gap_area_norm,
        ],
        dim=1,
    )

    return {
        "gap_steps": gap_steps,
        "prev_valid_gap": prev_valid_gap,
        "motion_valid": motion_valid,
        "motion_run": motion_run,
        "missing_summary": missing_summary,
    }
