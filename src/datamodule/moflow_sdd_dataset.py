"""MoFlow 协议的 SDD (Stanford Drone Dataset) Dataset。

读取 MoFlow 预处理好的 SDD pkl（与 /home/lbh/MoFlow/data/dataloader_sdd.py 同源）：
    original/sdd_train.pkl  -> 训练场景列表 (8985)
    original/sdd_test.pkl   -> 测试场景列表 (2829)
每个场景是一个三元组 (past[8,2], fut[12,2], seq[20])，单智能体（A=1），
坐标在像素系（pixel），2.5 Hz（past 8 帧 = 3.2s，fut 12 帧 = 4.8s）。

输出的样本 dict 与 MoFlowEthUcyDataset 完全同构（同样的键、同样的局部坐标
变换），因此直接复用 moflow_ethucy_collate_fn 与现有模型/指标管线。
minADE/minFDE 在像素系计算（translate+rotate 刚性变换不改变 L2 距离），
与 SDD 文献惯例（像素单位）一致。

注意：SDD 无官方 val 划分，这里从 sdd_train.pkl 用固定 seed 做确定性
90/10 train/val 划分（val 仅用于 checkpoint 选点），held-out 测试用
sdd_test.pkl。单向/双向两臂用同一 seed，划分完全一致，保证对照公平。
"""

import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .ethucy_utils import compute_focal_rotation, transform_to_local


class MoFlowSddDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        obs_len: int = 8,
        pred_len: int = 12,
        val_ratio: float = 0.1,
        seed: int = 2024,
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root)
        self.split = split
        self.obs_len = obs_len
        self.pred_len = pred_len

        assert split in ("train", "val", "test"), f"bad split: {split}"
        pkl_path = (
            self.data_root / "original" / "sdd_test.pkl"
            if split == "test"
            else self.data_root / "original" / "sdd_train.pkl"
        )
        assert pkl_path.exists(), f"SDD pkl not found: {pkl_path}"
        with open(pkl_path, "rb") as f:
            scenes = pickle.load(f)  # list of (past[8,2], fut[12,2], seq[20])

        # train/val 确定性划分（与 arm 无关，两臂一致）
        if split in ("train", "val"):
            n = len(scenes)
            rng = np.random.default_rng(seed)
            perm = rng.permutation(n)
            n_val = int(n * val_ratio)
            if split == "val":
                scenes = [scenes[i] for i in perm[:n_val]]
            else:
                scenes = [scenes[i] for i in perm[n_val:]]

        self.scenes = scenes
        print(f"MoFlow SDD Dataset: split={split}, scenes={len(self.scenes)}")

    def __len__(self) -> int:
        return len(self.scenes)

    def __getitem__(self, index: int):
        past, fut, _seq = self.scenes[index]
        past = torch.as_tensor(past, dtype=torch.float)  # [8, 2]
        fut = torch.as_tensor(fut, dtype=torch.float)  # [12, 2]
        positions = torch.cat([past, fut], dim=0).unsqueeze(0)  # [1, 20, 2]
        valid_mask = torch.ones(1, positions.size(1), dtype=torch.bool)
        item = self.process(positions, valid_mask)
        item["scene_id"] = "SDD"
        item["track_id"] = index
        return item

    def process(self, positions: torch.Tensor, valid_mask: torch.Tensor) -> dict:
        """与 MoFlowEthUcyDataset.process 相同的字段构造（focal_idx=0）。"""
        obs_len = self.obs_len
        pred_len = self.pred_len
        N = positions.size(0)

        # focal 局部坐标系（平移 + 旋转，尺度保持像素制）
        origin = positions[0, obs_len - 1].clone()
        theta = compute_focal_rotation(
            positions[0, obs_len - 1], positions[0, obs_len - 2]
        )
        positions_local = transform_to_local(positions, valid_mask, origin, theta)

        hist_pos = positions_local[:, :obs_len]
        hist_valid = valid_mask[:, :obs_len]
        future_pos = positions_local[:, obs_len:]
        future_valid = valid_mask[:, obs_len:]

        velocity = torch.zeros(N, obs_len)
        for t in range(1, obs_len):
            velocity[:, t] = torch.norm(hist_pos[:, t] - hist_pos[:, t - 1], dim=-1)
        # 掩码感知（missing_history_v1 §5.3）：涉及缺失帧的相邻步速度置零
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

        x_centers = hist_pos[:, obs_len - 1].clone()

        # x_angles：涉及缺失帧的相邻步方向置零（missing_history_v1 §5.3）；
        # 模型 forward 只用 x_angles[..., -1]（帧 7←6，v1 下恒有效），不受影响
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
            "origin": origin.view(1, 2),
            "theta": theta.view(1),
            "scene_id": "SDD",
            "track_id": 0,
            "timestamp": torch.tensor([obs_len * 0.4]),
        }
