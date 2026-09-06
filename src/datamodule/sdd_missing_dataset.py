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
from .missing_features import build_missing_features


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
        # v1 + v2_high + v3_noguard 全部条件（数据格式相同，仅掩码不同；
        # v2/v3 数据位于 data/SDD_missing_v2_high|v3_noguard/<condition>/）
        _KNOWN = ("complete", "random_single", "random_block2",
                  "random_fixed3", "random_fixed4", "random_fixed5",
                  "random_block3", "random_block4", "random_block6",
                  "random_fixed3_ng", "random_fixed4_ng",
                  "random_block3_ng", "random_block4_ng", "random_block6_ng",
                  "uniform_hard_ng")
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

        # focal 局部坐标系：v1/v2 帧帧7恒可见；v3_noguard 下回退到最后可见帧，
        # 旋转角从最近的有效非零位移对回溯（无有效对则 theta=0）
        hist_valid_f = valid_mask[0, :obs_len]
        _fv = torch.nonzero(hist_valid_f).flatten()
        focal_last = int(_fv[-1].item()) if len(_fv) else obs_len - 1
        origin = positions[0, focal_last].clone()
        theta = torch.tensor(0.0)
        for t in range(obs_len - 1, 0, -1):
            if bool(valid_mask[0, t]) and bool(valid_mask[0, t - 1]):
                d = positions[0, t] - positions[0, t - 1]
                if torch.norm(d) >= 1e-4:
                    theta = torch.atan2(d[1], d[0])
                    break
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

        # 每 actor 最后可见历史位置（v1/v2 下恒为帧 7，与旧版一致）
        last_valid_idx = []
        for i in range(N):
            idxs = torch.nonzero(hist_valid[i]).flatten()
            last_valid_idx.append(int(idxs[-1].item()) if len(idxs) else obs_len - 1)
        x_centers = torch.stack([hist_pos[i, last_valid_idx[i]] for i in range(N)])

        # 每 actor 最近有效运动对朝向（无则 0：单帧可见/静止）；v1/v2 下 == x_angles[...,7]
        x_last_valid_angle = torch.zeros(N)
        for i in range(N):
            for t in range(obs_len - 1, 0, -1):
                if bool(diff_mask[i, t - 1]) and torch.norm(hist_pos[i, t] - hist_pos[i, t - 1]) >= 1e-4:
                    x_last_valid_angle[i] = torch.atan2(
                        hist_pos[i, t][1] - hist_pos[i, t - 1][1],
                        hist_pos[i, t][0] - hist_pos[i, t - 1][0])
                    break

        x_last_valid_idx = torch.tensor(last_valid_idx, dtype=torch.long)
        x_anchor_lag = (7 - x_last_valid_idx).clamp(min=0)
        x_forecast_gap = (8 - x_last_valid_idx).clamp(min=1)

        # 缺失感知特征（方案 §2.3/§2.4，任务 2）：仅由 hist_valid 掩码派生，
        # 与 ETH/UCY benchmark 路径同语义
        miss = build_missing_features(hist_valid)

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
            "x_last_valid_angle": x_last_valid_angle,
            "x_last_valid_idx": x_last_valid_idx,
            "x_anchor_lag_steps": x_anchor_lag,
            "x_forecast_gap_steps": x_forecast_gap,
            "x_gap_steps": miss["gap_steps"],
            "x_prev_valid_gap": miss["prev_valid_gap"],
            "x_motion_valid": miss["motion_valid"],
            "x_motion_run": miss["motion_run"],
            "x_missing_summary": miss["missing_summary"],
            "origin": origin.view(1, 2),
            "theta": theta.view(1),
            "scene_id": "SDD",
            "track_id": 0,
            "timestamp": torch.tensor([obs_len * 0.4]),
        }
