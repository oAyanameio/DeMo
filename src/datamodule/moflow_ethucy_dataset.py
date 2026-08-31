"""MoFlow 协议的 ETH/UCY Dataset。

标准 leave-one-out (LOO) 协议，直接读取 MoFlow (SocialGAN 预处理) 的 pkl 文件：
    data_root/<subset>/<subset>_train.pkl  /  <subset>_test.pkl  /  <subset>_val.pkl
其中 <subset> 为 held-out 测试场景；train/val pkl 包含其余 4 个场景的数据。
每个 pkl 包含:
    traj: [N, 20, 2]  世界坐标（米），20 = 8 obs + 12 pred，2.5 Hz
    seq_start_end: [G, 2] 邻居分组（同一时刻窗口的行人群）
    num_peds_in_seq: 每组人数

保证与 MoFlow 论文（及 SocialGAN 系）完全相同的数据划分与样本集合，
从而 DeMo 与 MoFlow 的指标可直接对比（K=20，米制）。
"""

import pickle
from pathlib import Path
from typing import Dict, List

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .ethucy_utils import compute_focal_rotation, transform_to_local

SUBSET_MAP = {
    "eth": "ETH",
    "hotel": "HOTEL",
    "univ": "UNIV",
    "zara1": "ZARA1",
    "zara2": "ZARA2",
}


class MoFlowEthUcyDataset(Dataset):
    """单折（一个 test subset）的 MoFlow 协议数据集。

    split:
        'train' / 'val' / 'test' 对应 MoFlow pkl 的后缀。
        MoFlow 训练用 <subset>_train.pkl，评测用 <subset>_test.pkl。
    """

    def __init__(
        self,
        data_root: str,
        subset: str,
        split: str = "train",
        obs_len: int = 8,
        pred_len: int = 12,
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root)
        self.subset = subset.lower()
        self.obs_len = obs_len
        self.pred_len = pred_len

        pkl_path = self.data_root / self.subset / f"{self.subset}_{split}.pkl"
        assert pkl_path.exists(), f"MoFlow pkl not found: {pkl_path}"
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        traj = torch.as_tensor(data["traj"], dtype=torch.float)  # [N, 20, 2]
        seq_start_end = torch.as_tensor(data["seq_start_end"], dtype=torch.long)

        # 预先构建样本索引: (group_idx, focal_local_idx)
        self.groups: List[Dict] = []
        for g_idx, (s, e) in enumerate(seq_start_end.tolist()):
            group_traj = traj[s:e]  # [M, 20, 2]，组内行人全部 20 帧完整
            self.groups.append(
                {
                    "positions": group_traj,
                    "valid_mask": torch.ones(
                        group_traj.shape[0], group_traj.shape[1], dtype=torch.bool
                    ),
                }
            )
        self.sample_index: List = []
        for g_idx, g in enumerate(self.groups):
            for focal_idx in range(g["positions"].shape[0]):
                self.sample_index.append((g_idx, focal_idx))

        print(
            f"MoFlow ETH/UCY Dataset: subset={self.subset}, split={split}, "
            f"groups={len(self.groups)}, samples={len(self.sample_index)}"
        )

    def __len__(self) -> int:
        return len(self.sample_index)

    def __getitem__(self, index: int):
        g_idx, focal_idx = self.sample_index[index]
        group = self.groups[g_idx]
        return self.process(group, focal_idx)

    def process(self, group: Dict, focal_idx: int) -> Dict:
        obs_len = self.obs_len
        pred_len = self.pred_len

        positions = group["positions"].clone()  # [N, T, 2]
        valid_mask = group["valid_mask"].clone()  # [N, T]
        N = positions.size(0)

        # focal agent 移到第 0 位
        perm = [focal_idx] + [i for i in range(N) if i != focal_idx]
        positions = positions[perm]
        valid_mask = valid_mask[perm]

        # focal 局部坐标系（与 DeMo 原始协议一致：平移+旋转，尺度保持米制）
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
            velocity[:, t] = torch.norm(
                hist_pos[:, t] - hist_pos[:, t - 1], dim=-1
            )

        x_positions_diff = torch.zeros_like(hist_pos)
        x_positions_diff[:, 1:] = hist_pos[:, 1:] - hist_pos[:, :-1]
        diff_mask = hist_valid[:, :-1] & hist_valid[:, 1:]
        x_positions_diff[:, 1:][~diff_mask] = 0.0

        # velocity_diff[t] 需要 t 与 t-1 两步的位移都有效（velocity[t-1] 对应步 t-2→t-1）
        velocity[:, 1:] = velocity[:, 1:] * diff_mask.float()
        x_velocity_diff = torch.zeros_like(velocity)
        x_velocity_diff[:, 1:] = velocity[:, 1:] - velocity[:, :-1]
        both_steps = diff_mask[:, 1:] & diff_mask[:, :-1]  # [N, obs_len-2]
        x_velocity_diff[:, 2:] = x_velocity_diff[:, 2:] * both_steps.float()

        x_centers = hist_pos[:, obs_len - 1].clone()

        x_angles = torch.zeros(N, obs_len)
        for t in range(1, obs_len):
            d = hist_pos[:, t] - hist_pos[:, t - 1]
            x_angles[:, t] = torch.atan2(d[:, 1], d[:, 0])
        x_angles[:, 0] = x_angles[:, 1]

        x_attr = torch.zeros(N, 3, dtype=torch.uint8)

        target = future_pos.clone()
        target_padded = torch.cat([hist_pos[:, -1:], target], dim=1)
        target_diff = target_padded[:, 1:] - target_padded[:, :-1]
        target_valid_padded = torch.cat(
            [hist_valid[:, -1:], future_valid], dim=1
        )
        target_diff_mask = (
            target_valid_padded[:, :-1] & target_valid_padded[:, 1:]
        )
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
            "scene_id": SUBSET_MAP.get(self.subset, self.subset.upper()),
            "track_id": focal_idx,
            "timestamp": torch.tensor([obs_len * 0.4]),
        }


def moflow_ethucy_collate_fn(batch):
    """与 ethucy_collate_fn 相同的 padding 逻辑。"""
    data = {}
    for key in [
        "x_positions_diff",
        "x_attr",
        "x_positions",
        "x_centers",
        "x_angles",
        "x_velocity",
        "x_velocity_diff",
    ]:
        data[key] = pad_sequence([b[key] for b in batch], batch_first=True)

    data["target"] = pad_sequence(
        [b["target"] for b in batch], batch_first=True
    )
    data["target_diff"] = pad_sequence(
        [b["target_diff"] for b in batch], batch_first=True
    )
    data["target_vel_diff"] = pad_sequence(
        [b["target_vel_diff"] for b in batch], batch_first=True
    )
    data["target_mask"] = pad_sequence(
        [b["target_mask"] for b in batch], batch_first=True, padding_value=False
    )
    data["x_valid_mask"] = pad_sequence(
        [b["x_valid_mask"] for b in batch], batch_first=True, padding_value=False
    )
    data["x_key_valid_mask"] = data["x_valid_mask"].any(-1)

    data["scene_id"] = [b["scene_id"] for b in batch]
    data["track_id"] = [b["track_id"] for b in batch]
    data["origin"] = torch.cat([b["origin"] for b in batch], dim=0)
    data["theta"] = torch.cat([b["theta"] for b in batch])
    data["timestamp"] = torch.cat([b["timestamp"] for b in batch])
    return data
