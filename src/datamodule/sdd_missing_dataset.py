"""SDD missing-history v1 Dataset（读取 data/SDD_missing_v1/<condition>）。

每个 .pt 是 missing_history_v1 canonical 样本：
    positions [1,20,2]（原始像素坐标，缺失历史位置为 0.0）
    valid_mask [1,20] / history_mask [1,8]
    condition/source_file/source_index/sample_index 等元数据

process() 复用 MoFlowSddDataset 的掩码感知逻辑（velocity/positions_diff/
velocity_diff/x_angles 均跳过缺失步），focal=唯一行人（index 0），
局部坐标旋转用恒可见的帧 7、6。
"""

import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .ethucy_utils import compute_focal_rotation, transform_to_local


class SddMissingDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        condition: str,
        split: str = "train",
        obs_len: int = 8,
        pred_len: int = 12,
    ) -> None:
        super().__init__()
        # v1 + v2_high 全部条件（数据格式相同，仅掩码不同；v2 高缺失数据
        # 位于 data/SDD_missing_v2_high/<condition>/，data_root 指过去即可）
        _KNOWN = ("complete", "random_single", "random_block2",
                  "random_fixed3", "random_fixed4", "random_fixed5",
                  "random_block3", "random_block4", "random_block6")
        assert condition in _KNOWN, condition
        assert split in ("train", "val", "test"), split
        self.data_root = Path(data_root)
        self.condition = condition
        self.split = split
        self.obs_len = obs_len
        self.pred_len = pred_len

        split_dir = self.data_root / condition / split
        assert split_dir.exists(), f"missing split dir: {split_dir}"
        self.file_list = sorted(split_dir.glob("*.pt"))
        assert self.file_list, f"no .pt files under {split_dir}"
        print(f"SddMissingDataset cond={condition} split={split}: {len(self.file_list)} samples")

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, index: int):
        d = torch.load(self.file_list[index], weights_only=False)
        positions = d["positions"].float()  # [1, 20, 2]
        valid_mask = d["valid_mask"].bool()  # [1, 20]
        item = self.process(positions, valid_mask)
        # 调试/审计字段（不进模型 forward）
        item["history_mask"] = d["history_mask"]
        item["condition"] = d["condition"]
        item["source_index"] = d["source_index"]
        item["split"] = d["split"]
        return item

    def process(self, positions: torch.Tensor, valid_mask: torch.Tensor) -> dict:
        """掩码感知的 DeMo 字段构造（与 MoFlowSddDataset.process 同构）。"""
        obs_len = self.obs_len
        pred_len = self.pred_len
        N = positions.size(0)

        # focal 局部坐标系：帧 7、6 在 v1 下恒可见，锚定稳定
        origin = positions[0, obs_len - 1].clone()
        theta = compute_focal_rotation(
            positions[0, obs_len - 1], positions[0, obs_len - 2]
        )
        positions_local = transform_to_local(positions, valid_mask, origin, theta)

        hist_pos = positions_local[:, :obs_len]
        hist_valid = valid_mask[:, :obs_len]
        future_pos = positions_local[:, obs_len:]
        future_valid = valid_mask[:, obs_len:]

        # 速度：涉及缺失帧的相邻步置零
        velocity = torch.zeros(N, obs_len)
        for t in range(1, obs_len):
            velocity[:, t] = torch.norm(hist_pos[:, t] - hist_pos[:, t - 1], dim=-1)
        diff_mask = hist_valid[:, :-1] & hist_valid[:, 1:]
        velocity[:, 1:] = velocity[:, 1:] * diff_mask.float()

        x_positions_diff = torch.zeros_like(hist_pos)
        x_positions_diff[:, 1:] = hist_pos[:, 1:] - hist_pos[:, :-1]
        x_positions_diff[:, 1:][~diff_mask] = 0.0

        # velocity_diff[t] 需要 t 与 t-1 两步的位移都有效（velocity[t-1] 对应步 t-2→t-1）
        x_velocity_diff = torch.zeros_like(velocity)
        x_velocity_diff[:, 1:] = velocity[:, 1:] - velocity[:, :-1]
        both_steps = diff_mask[:, 1:] & diff_mask[:, :-1]  # [N, obs_len-2]
        x_velocity_diff[:, 2:] = x_velocity_diff[:, 2:] * both_steps.float()

        # v1：帧 6/7 恒可见，x_centers=帧 7 位置（= 最后有效帧）
        x_centers = hist_pos[:, obs_len - 1].clone()

        # x_angles：涉及缺失帧的方向置零；模型只用 x_angles[..., -1]（恒有效）
        x_angles = torch.zeros(N, obs_len)
        for t in range(1, obs_len):
            d = hist_pos[:, t] - hist_pos[:, t - 1]
            ang = torch.atan2(d[:, 1], d[:, 0])
            x_angles[:, t] = torch.where(diff_mask[:, t - 1], ang, torch.zeros_like(ang))
        x_angles[:, 0] = x_angles[:, 1]

        x_attr = torch.zeros(N, 3, dtype=torch.uint8)

        target = future_pos.clone()
        target_padded = torch.cat([hist_pos[:, -1:], target], dim=1)
        target_diff = target_padded[:, 1:] - target_padded[:, :-1]
        target_valid_padded = torch.cat([hist_valid[:, -1:], future_valid], dim=1)
        target_diff_mask = target_valid_padded[:, :-1] & target_valid_padded[:, 1:]
        target_diff[~target_diff_mask] = 0.0

        vel_future = torch.zeros(N, pred_len)
        for t in range(pred_len):
            d = future_pos[:, t] - (
                hist_pos[:, -1] if t == 0 else future_pos[:, t - 1]
            )
            vel_future[:, t] = torch.norm(d, dim=-1)
        target_vel_diff = torch.zeros_like(vel_future)
        vel_padded = torch.cat([velocity[:, -1:], vel_future], dim=1)
        target_vel_diff = vel_padded[:, 1:] - vel_padded[:, :-1]
        target_vel_diff[~target_diff_mask] = 0.0

        target_mask = future_valid.clone()
        target_mask[0] = True

        return {
            "target": target,
            "target_diff": target_diff,
            "target_vel_diff": target_vel_diff,
            "target_mask": target_mask,
            "x_positions_diff": x_positions_diff,
            "x_positions": hist_pos,
            "x_attr": x_attr,
            "x_centers": x_centers,
            "x_angles": x_angles,
            "x_velocity": velocity,
            "x_velocity_diff": x_velocity_diff,
            "x_valid_mask": hist_valid,
            "x_key_valid_mask": hist_valid.any(-1),
            "origin": origin.view(1, 2),
            "theta": theta.view(1),
            "scene_id": "SDD",
            "track_id": 0,
            "timestamp": torch.tensor([obs_len * 0.4]),
        }
