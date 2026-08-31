"""ETH/UCY 行人轨迹预测 Dataset。

不依赖车道线，使用 actor-only 模式。
每个预处理 .pt 文件是一个样本，Dataset 只需完成局部坐标变换。
"""

from pathlib import Path
from typing import List, Optional

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .ethucy_utils import compute_focal_rotation, transform_to_local


class EthUcyDataset(Dataset):
    """ETH/UCY 数据集。

    每个样本对应一个预处理后的 .pt 文件。
    __getitem__ 完成：
    1. 将 focal agent 移到第 0 个位置
    2. 局部坐标变换
    3. 构造历史位移、速度、速度差
    4. 构造未来目标和目标有效掩码
    """

    def __init__(
        self,
        data_root: Path,
        scene_names: List[str],
        obs_len: int = 8,
        pred_len: int = 12,
        radius: float = 150.0,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.scene_names = scene_names
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.radius = radius

        self.file_list = []
        for scene in scene_names:
            scene_dir = self.data_root / scene
            if scene_dir.exists():
                files = sorted(scene_dir.glob("*.pt"))
                self.file_list.extend(files)

        print(
            f"ETH/UCY Dataset: scenes={scene_names}, "
            f"total samples={len(self.file_list)}"
        )

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, index: int):
        data = torch.load(self.file_list[index], weights_only=False)
        return self.process(data)

    def process(self, data: dict) -> dict:
        obs_len = self.obs_len
        pred_len = self.pred_len
        total_len = obs_len + pred_len

        positions = data["positions"]  # [N, total_len, 2]
        valid_mask = data["valid_mask"]  # [N, total_len]
        agent_ids = data["agent_ids"]  # [N]
        focal_id = data["focal_id"]

        # 找到 focal agent 的索引
        focal_idx = (agent_ids == focal_id).nonzero(as_tuple=True)[0]
        if len(focal_idx) == 0:
            # 如果 focal 不在列表中，取第一个有完整历史的行为 focal
            focal_idx = torch.tensor([0])
        focal_idx = focal_idx[0].item()

        # 将 focal agent 移到第 0 个位置
        N = positions.size(0)
        perm = list(range(N))
        perm.remove(focal_idx)
        perm = [focal_idx] + perm
        positions = positions[perm]
        valid_mask = valid_mask[perm]
        agent_ids = agent_ids[perm]

        # focal 最后一帧作为局部坐标原点
        origin = positions[0, obs_len - 1].clone()
        # 旋转角
        focal_prev = positions[0, obs_len - 2]
        focal_last = positions[0, obs_len - 1]
        theta = compute_focal_rotation(focal_last, focal_prev)

        # 所有行人坐标转换到 focal 局部坐标
        positions_local = transform_to_local(positions, valid_mask, origin, theta)

        # 分离历史和未来
        hist_pos = positions_local[:, :obs_len]  # [N, obs_len, 2]
        hist_valid = valid_mask[:, :obs_len]  # [N, obs_len]
        future_pos = positions_local[:, obs_len:]  # [N, pred_len, 2]
        future_valid = valid_mask[:, obs_len:]  # [N, pred_len]

        # 速度（标量）：相邻帧间位移的模
        velocity = torch.zeros(N, obs_len, dtype=torch.float)
        for t in range(1, obs_len):
            diff = hist_pos[:, t] - hist_pos[:, t - 1]
            velocity[:, t] = torch.norm(diff, dim=-1)
        # 第一帧速度为 0

        # 构造历史位移 (x_positions_diff)
        # x_positions_diff[:, 0] = 0, x_positions_diff[:, t] = pos[t] - pos[t-1]
        x_positions_diff = torch.zeros_like(hist_pos)
        x_positions_diff[:, 1:] = hist_pos[:, 1:] - hist_pos[:, :-1]
        # 无效帧的位移设为 0
        diff_mask = hist_valid[:, :-1] & hist_valid[:, 1:]
        x_positions_diff[:, 1:][~diff_mask] = 0.0

        # 速度差：velocity_diff[t] 需要 t 与 t-1 两步位移都有效（velocity 本身先做掩码）
        velocity[:, 1:] = velocity[:, 1:] * diff_mask.float()
        x_velocity_diff = torch.zeros_like(velocity)
        x_velocity_diff[:, 1:] = velocity[:, 1:] - velocity[:, :-1]
        both_steps = diff_mask[:, 1:] & diff_mask[:, :-1]  # [N, obs_len-2]
        x_velocity_diff[:, 2:] = x_velocity_diff[:, 2:] * both_steps.float()

        # 中心：最后一帧有效位置
        x_centers = hist_pos[:, obs_len - 1].clone()

        # 角度：hist_pos 的位移方向
        x_angles = torch.zeros(N, obs_len, dtype=torch.float)
        for t in range(1, obs_len):
            d = hist_pos[:, t] - hist_pos[:, t - 1]
            x_angles[:, t] = torch.atan2(d[:, 1], d[:, 0])
        x_angles[:, 0] = x_angles[:, 1]  # 第一帧复制第二帧

        # 属性：类型为 pedestrian (0)
        x_attr = torch.zeros(N, 3, dtype=torch.uint8)
        x_attr[:, 2] = 0  # pedestrian type

        # 目标
        target = future_pos.clone()
        target_diff = torch.zeros_like(target)
        # target_diff[:, t] = target[:, t] - target[:, t-1], target_diff[:, 0] = target[:, 0] - hist_pos[:, -1]
        target_padded = torch.cat([hist_pos[:, -1:], target], dim=1)
        target_diff = target_padded[:, 1:] - target_padded[:, :-1]

        target_valid_padded = torch.cat(
            [hist_valid[:, -1:], future_valid], dim=1
        )
        target_diff_mask = (
            target_valid_padded[:, :-1] & target_valid_padded[:, 1:]
        )
        target_diff[~target_diff_mask] = 0.0

        # 目标速度差
        vel_future = torch.zeros(N, pred_len, dtype=torch.float)
        for t in range(pred_len):
            d = future_pos[:, t] - (
                hist_pos[:, -1] if t == 0 else future_pos[:, t - 1]
            )
            vel_future[:, t] = torch.norm(d, dim=-1)
        target_vel_diff = torch.zeros_like(vel_future)
        vel_padded = torch.cat([velocity[:, -1:], vel_future], dim=1)
        target_vel_diff = vel_padded[:, 1:] - vel_padded[:, :-1]
        target_vel_diff[~target_diff_mask] = 0.0

        # target_mask: focal 必须满足 target_mask[0].all()
        target_mask = future_valid.clone()
        # focal agent 的目标掩码必须全 True
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
            "scene_id": data["scene_id"],
            "track_id": focal_id,
            "timestamp": torch.tensor([obs_len * 0.4]),  # stride=10 -> 0.4s/frame
        }


def ethucy_collate_fn(batch):
    """ETH/UCY batch collate。

    对 N 不同的样本做 padding，不生成 lane 相关字段。
    """
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

    if batch[0]["target"] is not None:
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
            [b["target_mask"] for b in batch],
            batch_first=True,
            padding_value=False,
        )

    data["x_valid_mask"] = pad_sequence(
        [b["x_valid_mask"] for b in batch],
        batch_first=True,
        padding_value=False,
    )

    data["x_key_valid_mask"] = data["x_valid_mask"].any(-1)

    data["scene_id"] = [b["scene_id"] for b in batch]
    data["track_id"] = [b["track_id"] for b in batch]

    data["origin"] = torch.cat([b["origin"] for b in batch], dim=0)
    data["theta"] = torch.cat([b["theta"] for b in batch])
    data["timestamp"] = torch.cat([b["timestamp"] for b in batch])

    return data