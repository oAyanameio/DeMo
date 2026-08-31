"""ETH/UCY 行人轨迹数据解析工具。

支持 ETH、HOTEL、UNIV、ZARA1、ZARA2 五个场景的原始 .txt 格式。
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch


# ETH/UCY 原始帧率约 25 Hz，下采样 stride=10 得到 2.5 Hz
DEFAULT_FRAME_STRIDE = 10
DEFAULT_OBS_LEN = 8
DEFAULT_PRED_LEN = 12


def load_ethucy_file(path: Path) -> pd.DataFrame:
    """加载单个 ETH/UCY 场景的 .txt 文件。

    支持空格、Tab、逗号分隔，自动跳过空行和表头行。
    表头行检测：首列非数字的行视为表头。

    返回列：frame, ped_id, x, y
    """
    rows: List[Dict] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 尝试多种分隔符
            parts = None
            for sep in ["\t", " ", ","]:
                parts = line.split(sep)
                parts = [p.strip() for p in parts if p.strip()]
                if len(parts) >= 4:
                    break
            if parts is None or len(parts) < 4:
                continue
            # 跳过表头行（首列非数字）
            try:
                float(parts[0])
            except ValueError:
                continue
            # 取前 4 列
            frame_id = int(float(parts[0]))
            ped_id = int(float(parts[1]))
            x = float(parts[2])
            y = float(parts[3])
            rows.append({"frame": frame_id, "ped_id": ped_id, "x": x, "y": y})

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No valid data found in {path}")
    df = df.sort_values(["frame", "ped_id"]).reset_index(drop=True)
    return df


def normalize_scene_name(path: Path) -> str:
    """从文件路径推断标准化场景名（ETH, HOTEL, UNIV, ZARA1, ZARA2）。"""
    name = path.stem.upper()
    # 通用映射：将常见变体映射到标准名称
    mapping = {
        "ETH": "ETH",
        "HOTEL": "HOTEL",
        "UNIV": "UNIV",
        "ZARA1": "ZARA1",
        "ZARA2": "ZARA2",
        "ZARA01": "ZARA1",
        "ZARA02": "ZARA2",
        "ZARA03": "ZARA2",  # ZARA3 不是标准 5 场景，忽略或归入 ZARA2
        "STUDENTS003": "UNIV",
        "STUDENTS001": "UNIV",
        "UNI_EXAMPLES": "UNIV",
        "UNI": "UNIV",
        "BIWI": "ETH",
    }
    for key, val in mapping.items():
        if key in name:
            return val
    return name


def resample_scene(
    df: pd.DataFrame,
    frame_stride: int = DEFAULT_FRAME_STRIDE,
) -> pd.DataFrame:
    """按固定帧间隔下采样。

    保留 frame % frame_stride == 0 的帧，并重新编号 frame 为 0, 1, 2, ...
    """
    if frame_stride <= 1:
        return df
    df = df.copy()
    # 先将原始 frame 对齐到 stride 的倍数
    min_frame = df["frame"].min()
    df["_aligned"] = (df["frame"] - min_frame) // frame_stride
    df["frame"] = df["_aligned"]
    df = df.drop(columns=["_aligned"])
    # 对每个 (frame, ped_id) 去重，保留第一条
    df = df.drop_duplicates(subset=["frame", "ped_id"])
    df = df.sort_values(["frame", "ped_id"]).reset_index(drop=True)
    return df


def sliding_window_samples(
    df: pd.DataFrame,
    obs_len: int = DEFAULT_OBS_LEN,
    pred_len: int = DEFAULT_PRED_LEN,
) -> List[Dict]:
    """对单个场景执行滑动窗口，生成样本列表。

    每个样本包含一个 focal agent 和周围所有行人。
    对每一个 (end_obs, focal_ped_id) 组合生成一个样本。

    返回列表，每个元素为 dict：
        scene_id, focal_id, agent_ids, positions, valid_mask, frame_ids
    """
    total_len = obs_len + pred_len
    ped_ids = sorted(df["ped_id"].unique())
    frames = sorted(df["frame"].unique())
    num_frames = len(frames)

    if num_frames < total_len:
        return []

    # 构建 actor_id -> index 映射
    ped_to_idx = {pid: i for i, pid in enumerate(ped_ids)}
    num_peds = len(ped_ids)

    # 构建稀疏张量：positions[num_peds, num_frames, 2], valid[num_peds, num_frames]
    positions = np.full((num_peds, num_frames, 2), np.nan, dtype=np.float32)
    for _, row in df.iterrows():
        pi = ped_to_idx[row["ped_id"]]
        fi = frames.index(row["frame"])
        positions[pi, fi, 0] = row["x"]
        positions[pi, fi, 1] = row["y"]

    valid_mask_np = ~np.isnan(positions[:, :, 0])

    samples = []
    for end_obs in range(obs_len - 1, num_frames - pred_len):
        start = end_obs - obs_len + 1
        end = end_obs + pred_len
        window_frames = frames[start : end + 1]

        # 找出窗口内存在完整 8 帧历史的行人作为候选 focal
        window_valid = valid_mask_np[:, start : end + 1]
        obs_valid = window_valid[:, :obs_len]
        has_full_obs = obs_valid.all(axis=1)

        if not has_full_obs.any():
            continue

        # 对每个候选 focal agent 生成一个样本
        focal_indices = np.where(has_full_obs)[0]
        for focal_idx in focal_indices:
            focal_ped_id = ped_ids[focal_idx]

            # 收集所有在窗口内至少有一帧有效数据的行人
            any_valid = window_valid.any(axis=1)
            included = np.where(any_valid)[0]

            agent_ids = [ped_ids[i] for i in included]
            pos = positions[included][:, start : end + 1, :]
            valid = window_valid[included]

            # 用 0 填充 NaN
            pos = np.nan_to_num(pos, nan=0.0)

            sample = {
                "scene_id": "unknown",  # 由调用者设置
                "focal_id": int(focal_ped_id),
                "agent_ids": torch.tensor(agent_ids, dtype=torch.long),
                "positions": torch.from_numpy(pos).float(),
                "valid_mask": torch.from_numpy(valid).bool(),
                "frame_ids": torch.tensor(window_frames, dtype=torch.long),
            }
            samples.append(sample)

    return samples


def compute_focal_rotation(
    pos_last: torch.Tensor, pos_prev: torch.Tensor
) -> torch.Tensor:
    """根据 focal 最后两帧位移计算旋转角。

    如果位移过小（< 1e-4），返回 0。
    """
    delta = pos_last - pos_prev
    if torch.norm(delta) < 1e-4:
        return torch.tensor(0.0)
    return torch.atan2(delta[1], delta[0])


def transform_to_local(
    positions: torch.Tensor,
    valid_mask: torch.Tensor,
    origin: torch.Tensor,
    theta: torch.Tensor,
) -> torch.Tensor:
    """将全局坐标转换到 focal 局部坐标系。

    positions: [N, T, 2]
    origin: [2] — focal 最后一帧坐标
    theta: [] — 旋转角
    """
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    rot_mat = torch.tensor(
        [[cos_t, -sin_t], [sin_t, cos_t]],
        dtype=positions.dtype,
        device=positions.device,
    )
    local = positions - origin
    local[valid_mask] = torch.matmul(
        local[valid_mask].double(), rot_mat.double()
    ).float()
    return local