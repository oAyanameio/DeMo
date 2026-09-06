"""ETH/UCY benchmark Dataset (fold/split 目录直读，不含 lane 字段)。

每个 .pt 是一个样本。__getitem__:
1. focal 移到 index 0
2. focal 观测末帧为原点，最后一个有效位移方向为旋转角（不足则前溯，仍无则 theta=0）
3. 生成 DeMo actor-only 字段；target_mask 反映真实有效性
"""

from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .missing_features import build_missing_features


def compute_theta(hist_pos: torch.Tensor, hist_valid: torch.Tensor, eps: float = 1e-4):
    """从 focal 历史末尾寻找最近的有效非零位移确定朝向。返回 (theta, degenerate).

    掩码感知（missing_history_v1，文档 §5.3 规则 3）：回溯时跳过涉及缺失帧的
    位移，缺失历史不参与朝向计算。complete 数据（全帧有效）行为与旧版完全一致。
    """
    T = hist_pos.size(0)
    for t in range(T - 1, 0, -1):
        if not (bool(hist_valid[t]) and bool(hist_valid[t - 1])):
            continue  # 该步涉及缺失帧：不参与朝向
        d = hist_pos[t] - hist_pos[t - 1]
        if torch.norm(d) >= eps:
            return torch.atan2(d[1], d[0]), False
    return torch.tensor(0.0), True


class EthUcyBenchmarkDataset(Dataset):
    def __init__(self, data_root: str, fold: str, split: str, obs_len: int = 8, pred_len: int = 12):
        super().__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.split_dir = Path(data_root) / f"fold_{fold}" / split
        self.file_list = []
        if self.split_dir.exists():
            for scene_dir in sorted(self.split_dir.iterdir()):
                if scene_dir.is_dir():
                    self.file_list.extend(sorted(scene_dir.glob("*.pt")))
        print(f"EthUcyBenchmarkDataset fold={fold} split={split}: {len(self.file_list)} samples")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index: int):
        data = torch.load(self.file_list[index], weights_only=False)
        return self.process(data)

    def process(self, data: dict) -> dict:
        obs_len, pred_len = self.obs_len, self.pred_len
        positions = data["positions"]  # [N, 20, 2]
        valid_mask = data["valid_mask"]  # [N, 20]
        agent_ids = data["agent_ids"]
        focal_id = data["focal_id"]

        matches = (agent_ids == focal_id).nonzero(as_tuple=True)[0]
        focal_idx = matches[0].item() if len(matches) else 0
        N = positions.size(0)
        perm = [focal_idx] + [i for i in range(N) if i != focal_idx]
        positions = positions[perm]
        valid_mask = valid_mask[perm]
        agent_ids = agent_ids[perm]

        # v3_noguard：focal 原点 = focal 最后可见历史位置（帧 7 可能缺失）；
        # v1/v2 下最后可见帧恒为 7，与旧版 positions[0, obs_len-1] 完全一致。
        focal_hist_valid = valid_mask[0, :obs_len]
        _fv = torch.nonzero(focal_hist_valid).flatten()
        focal_last = int(_fv[-1].item()) if len(_fv) else obs_len - 1
        origin = positions[0, focal_last].clone()
        theta, degenerate = compute_theta(positions[0, :obs_len], focal_hist_valid)

        # local transform (all actors are complete in strict mode)
        cos_t, sin_t = torch.cos(theta).item(), torch.sin(theta).item()
        rot = torch.tensor([[cos_t, -sin_t], [sin_t, cos_t]], dtype=torch.float64)
        local = torch.matmul((positions.double() - origin.double().view(1, 1, 2)), rot).float()

        hist_pos = local[:, :obs_len]
        hist_valid = valid_mask[:, :obs_len]
        future_pos = local[:, obs_len:]
        future_valid = valid_mask[:, obs_len:]

        # 位移/速度
        x_positions_diff = torch.zeros_like(hist_pos)
        x_positions_diff[:, 1:] = hist_pos[:, 1:] - hist_pos[:, :-1]
        diff_mask = hist_valid[:, :-1] & hist_valid[:, 1:]
        x_positions_diff[:, 1:] *= diff_mask.unsqueeze(-1)

        velocity = torch.zeros(N, obs_len)
        velocity[:, 1:] = torch.norm(hist_pos[:, 1:] - hist_pos[:, :-1], dim=-1)
        velocity[:, 1:] *= diff_mask.float()

        # velocity_diff[t] 需要 t 与 t-1 两步的位移都有效（velocity[t-1] 对应步 t-2→t-1）
        x_velocity_diff = torch.zeros_like(velocity)
        x_velocity_diff[:, 1:] = velocity[:, 1:] - velocity[:, :-1]
        both_steps = diff_mask[:, 1:] & diff_mask[:, :-1]  # [A, obs_len-2]
        x_velocity_diff[:, 2:] = x_velocity_diff[:, 2:] * both_steps.float()

        # x_centers / x_angles：只用有效帧
        last_valid_idx = []
        for i in range(N):
            idxs = torch.nonzero(hist_valid[i]).flatten()
            last_valid_idx.append(idxs[-1].item() if len(idxs) else obs_len - 1)
        x_centers = torch.stack([hist_pos[i, last_valid_idx[i]] for i in range(N)])
        # x_angles：涉及缺失帧的相邻步方向置零（missing_history_v1 §5.3）；
        # v3：帧 7 缺失时 x_angles[..., -1] 不再是有效朝向——模型改用
        # x_last_valid_angle（每 actor 最近有效运动对朝向，见 forward）；
        # v1/v2 下 x_last_valid_angle == x_angles[..., 7]，数值不变。
        x_angles = torch.zeros(N, obs_len)
        for t in range(1, obs_len):
            d = hist_pos[:, t] - hist_pos[:, t - 1]
            ang = torch.atan2(d[:, 1], d[:, 0])
            x_angles[:, t] = torch.where(diff_mask[:, t - 1], ang, torch.zeros_like(ang))
        x_angles[:, 0] = x_angles[:, 1]
        # 每 actor 最近有效朝向：从末帧向前找第一个 diff_mask 有效步；无则 0（单帧可见/静止）
        x_last_valid_angle = torch.zeros(N)
        for i in range(N):
            for t in range(obs_len - 1, 0, -1):
                if bool(diff_mask[i, t - 1]) and torch.norm(hist_pos[i, t] - hist_pos[i, t - 1]) >= 1e-4:
                    x_last_valid_angle[i] = torch.atan2(
                        hist_pos[i, t][1] - hist_pos[i, t - 1][1],
                        hist_pos[i, t][0] - hist_pos[i, t - 1][0])
                    break

        x_attr = torch.zeros(N, 3, dtype=torch.uint8)  # type 0 = pedestrian

        # 逐 actor 时间间隔字段（v3 §5.1.2；v1/v2 数据无该字段时按掩码现算，恒 0/1）
        x_last_valid_idx = torch.tensor(last_valid_idx, dtype=torch.long)
        x_anchor_lag = (7 - x_last_valid_idx).clamp(min=0)
        x_forecast_gap = (8 - x_last_valid_idx).clamp(min=1)

        # 缺失感知特征（方案 §2.3/§2.4，任务 2）：仅由 hist_valid 掩码派生
        miss = build_missing_features(hist_valid)
        x_gap_steps = miss["gap_steps"]
        x_prev_valid_gap = miss["prev_valid_gap"]
        x_motion_valid = miss["motion_valid"]
        x_motion_run = miss["motion_run"]
        x_missing_summary = miss["missing_summary"]

        # target
        target = future_pos.clone()
        target_padded = torch.cat([hist_pos[:, -1:], target], dim=1)
        target_diff = target_padded[:, 1:] - target_padded[:, :-1]
        tgt_valid_padded = torch.cat([hist_valid[:, -1:], future_valid], dim=1)
        tdiff_mask = tgt_valid_padded[:, :-1] & tgt_valid_padded[:, 1:]
        target_diff *= tdiff_mask.unsqueeze(-1)

        vel_future = torch.norm(target_diff, dim=-1)
        vel_padded = torch.cat([velocity[:, -1:], vel_future], dim=1)
        target_vel_diff = vel_padded[:, 1:] - vel_padded[:, :-1]
        target_vel_diff *= tdiff_mask.float()

        # 真实有效性掩码：focal 在 benchmark 中保证完整，无需强制 True
        target_mask = future_valid.clone()

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
            "x_gap_steps": x_gap_steps,
            "x_prev_valid_gap": x_prev_valid_gap,
            "x_motion_valid": x_motion_valid,
            "x_motion_run": x_motion_run,
            "x_missing_summary": x_missing_summary,
            "origin": origin.view(1, 2),
            "theta": theta.view(1),
            "degenerate_heading": degenerate,
            "scene_id": data["scene_id"],
            "track_id": focal_id,
            "timestamp": torch.tensor([obs_len * 0.4]),  # seconds, dt=0.4s/step
        }


def ethucy_benchmark_collate_fn(batch):
    data = {}
    for key in [
        "x_positions_diff", "x_attr", "x_positions", "x_centers",
        "x_angles", "x_velocity", "x_velocity_diff",
        "x_last_valid_angle", "x_last_valid_idx",
        "x_anchor_lag_steps", "x_forecast_gap_steps",
        "x_gap_steps", "x_prev_valid_gap", "x_motion_valid", "x_motion_run",
        "x_missing_summary",
    ]:
        data[key] = pad_sequence([b[key] for b in batch], batch_first=True)
    for key in ["target", "target_diff", "target_vel_diff"]:
        data[key] = pad_sequence([b[key] for b in batch], batch_first=True)
    data["target_mask"] = pad_sequence(
        [b["target_mask"] for b in batch], batch_first=True, padding_value=False
    )
    data["x_valid_mask"] = pad_sequence(
        [b["x_valid_mask"] for b in batch], batch_first=True, padding_value=False
    )
    data["x_key_valid_mask"] = data["x_valid_mask"].any(-1)
    data["x_key_valid_mask"][:, 0] = True  # focal always valid
    data["scene_id"] = [b["scene_id"] for b in batch]
    data["track_id"] = [b["track_id"] for b in batch]
    data["origin"] = torch.cat([b["origin"] for b in batch], dim=0)
    data["theta"] = torch.cat([b["theta"] for b in batch])
    data["timestamp"] = torch.cat([b["timestamp"] for b in batch])
    return data
