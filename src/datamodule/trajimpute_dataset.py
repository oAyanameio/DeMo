"""TrajImpute 官方 release -> DeMo actor-centric 直接缺失预测适配器。

官方字段语义（2026-09-07 release 审计 30/30，见 docs/research/选题.md §八.0.2）：
    obs_traj     [N, 8, 2] float32，缺失位置为 NaN
    pred_traj    [N, 12, 2] float32，完整（未来监督，永不做缺失填充）
    missing_mask [N, 8, 2] bool，True=缺失，与 obs_traj 的 NaN 严格一致
    seq_start_end [(s, e)] 每个场景序列的行人行范围

与历史 v1/v2/v3 相邻帧协议的关键差异（适配任务书 §六/§七/§八）：
    1. frame_missing = missing_mask.any(-1)；x_valid_mask = ~frame_missing
    2. 局部坐标原点 = focal 最后一个有效历史位置（末帧缺失不得取帧 7）
    3. 朝向 = 最近两个有效观测的连线（允许跨缺口）；单有效帧 -> 0（degenerate）
    4. 差分/速度跨缺口：有效帧 t 与前一个有效帧 s：diff = p[t] - p[s]，
       velocity = |diff| / (t - s)（按时间间隔归一化）
    5. target_diff 第一步从各 actor 的最后有效历史位置出发（显式 forecast gap）
    6. 未来轨迹永不参与任何历史输入特征；缺失历史坐标在局部变换后置 0 占位，
       由 x_valid_mask 明确区分，占位值不得被误认为真实观测
"""

from pathlib import Path
import pickle

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .missing_features import build_missing_features

SCENES = ("ETH-M", "HOTEL-M", "UNIV-M", "ZARA1-M", "ZARA2-M")
DIFFICULTIES = ("Easy", "Hard")
SPLITS = ("train", "val", "test")
REQUIRED_KEYS = ("obs_traj", "pred_traj", "missing_mask", "seq_start_end")

TRAJIMPUTE_ROOT = "/home/lbh/TrajImpute/dataset/TrajImpute"


# ---------------------------------------------------------------- pkl 读取与校验
def load_trajimpute_pkl(path, obs_len: int = 8, pred_len: int = 12):
    """读取并校验一个官方 pkl。返回 (obs, pred, frame_valid, seq_start_end)。

    校验失败直接抛错，绝不静默修复官方数据。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"TrajImpute pkl not found: {path}")
    with open(path, "rb") as f:
        raw = pickle.load(f)
    missing_keys = [k for k in REQUIRED_KEYS if k not in raw]
    if missing_keys:
        raise KeyError(f"{path}: missing keys {missing_keys}")

    def to_tensor(x, dtype):
        if torch.is_tensor(x):
            return x.to(dtype=dtype).cpu()
        return torch.from_numpy(np.ascontiguousarray(x)).to(dtype=dtype)

    obs = to_tensor(raw["obs_traj"], torch.float32)
    pred = to_tensor(raw["pred_traj"], torch.float32)
    mask = to_tensor(raw["missing_mask"], torch.bool)

    if obs.dim() != 3 or tuple(obs.shape[1:]) != (obs_len, 2):
        raise ValueError(f"{path}: obs_traj shape {tuple(obs.shape)} != [N,{obs_len},2]")
    if pred.dim() != 3 or tuple(pred.shape[1:]) != (pred_len, 2):
        raise ValueError(f"{path}: pred_traj shape {tuple(pred.shape)} != [N,{pred_len},2]")
    if mask.shape != obs.shape:
        raise ValueError(f"{path}: missing_mask shape {tuple(mask.shape)} != obs {tuple(obs.shape)}")
    # x/y 缺失语义必须逐帧一致；不一致直接报错
    if not torch.equal(mask[..., 0], mask[..., 1]):
        raise ValueError(f"{path}: x/y missing_mask inconsistent at same frame")
    # 官方语义：missing_mask=True <=> 该坐标为 NaN（严格一致）
    nan_mask = torch.isnan(obs[..., 0]) & torch.isnan(obs[..., 1])
    if not torch.equal(nan_mask, mask[..., 0]):
        raise ValueError(f"{path}: obs NaN pattern != missing_mask (官方语义被违反)")
    frame_valid = ~mask[..., 0]  # [N, obs_len]，True=该帧有效观测
    n_no_valid = int((~frame_valid.any(dim=-1)).sum())
    if n_no_valid > 0:
        raise ValueError(f"{path}: {n_no_valid} 个样本没有任何有效历史帧")
    if bool(torch.isnan(pred).any()):
        raise ValueError(f"{path}: pred_traj contains NaN（未来必须完整）")

    seq_start_end = [(int(s), int(e)) for s, e in raw["seq_start_end"]]
    for s, e in seq_start_end:
        if not (0 <= s < e <= obs.shape[0]):
            raise ValueError(f"{path}: invalid seq_start_end entry ({s},{e})")
    # 官方 test pkl 的 seq_start_end 为列表重复（未加块偏移）：确定性重建
    seq_start_end = rebuild_seq_start_end(seq_start_end, obs.shape[0], path)
    return obs, pred, frame_valid, seq_start_end


def rebuild_seq_start_end(seq_start_end, n_rows: int, path=None):
    """修复官方 test pkl 的 seq_start_end 平铺缺陷（不修改官方文件）。

    官方 generate_data.py 用 Python 列表乘法 `seq_start_end * 5`（Hard 为 4），
    复制条目但不加行偏移：块 k>=1 的 (s,e) 仍指向块 0 的行号，
    导致 181+ 行轨迹永远无法组到正确场景（train/val 未乘、平铺正确）。

    修复规则（确定性）：检测 len(se) == n_blocks * n_orig 且逐块完全重复后，
    块 k 的条目统一加偏移 k * block_size。无法确认重复模式时直接报错。
    """
    if not seq_start_end:
        raise ValueError(f"{path}: empty seq_start_end")
    last_end = seq_start_end[-1][1]
    if last_end == n_rows:  # 已正确平铺（train/val）
        return seq_start_end
    # 检测重复模式：n_orig 个条目为一组，重复 n_blocks 次
    for n_orig in range(1, len(seq_start_end) + 1):
        if len(seq_start_end) % n_orig:
            continue
        base = seq_start_end[:n_orig]
        if all(seq_start_end[k:k + n_orig] == base
               for k in range(0, len(seq_start_end), n_orig)):
            n_blocks = len(seq_start_end) // n_orig
            block_size = base[-1][1]  # 原始轨迹数 = 首块末行
            if block_size * n_blocks != n_rows:
                break
            fixed = [
                (s + b * block_size, e + b * block_size)
                for b in range(n_blocks)
                for (s, e) in base
            ]
            if fixed[-1][1] != n_rows:
                break
            return fixed
    raise ValueError(
        f"{path}: seq_start_end neither tiles [0,{n_rows}) nor repeats a base "
        f"pattern (官方平铺缺陷且无法确定性修复)"
    )


# ---------------------------------------------------------------- 跨缺口运动特征
def build_gap_aware_motion(hist_pos: torch.Tensor, hist_valid: torch.Tensor):
    """跨缺口差分/速度（任务书 §七）。

    对每个 actor 的每个有效帧 t：s = t 之前最近的有效帧，
        diff[t] = p[t] - p[s]（未归一化位移，跨缺口）
        velocity[t] = |diff[t]| / (t - s)（按时间间隔归一化的有效速度）
    缺失帧的 diff/velocity 置 0；首个有效帧（无前序有效帧）置 0。
    velocity_diff 只在相邻两个有效帧之间计算（有效速度 - 有效速度）。

    Args:
        hist_pos: [A, T, 2] 有限局部坐标（缺失处为占位 0）
        hist_valid: [A, T] bool
    Returns:
        diff [A, T, 2], velocity [A, T], velocity_diff [A, T]，全部 finite。
    """
    A, T, _ = hist_pos.shape
    diff = torch.zeros(A, T, 2, dtype=hist_pos.dtype)
    velocity = torch.zeros(A, T, dtype=hist_pos.dtype)
    velocity_diff = torch.zeros(A, T, dtype=hist_pos.dtype)
    for i in range(A):
        vidx = torch.nonzero(hist_valid[i]).flatten().tolist()
        for k in range(1, len(vidx)):
            t, s = vidx[k], vidx[k - 1]
            d = hist_pos[i, t] - hist_pos[i, s]
            gap = t - s
            diff[i, t] = d
            velocity[i, t] = torch.norm(d) / gap
            # 相邻有效帧对之间的有效速度差（k>=2 时 vel[s] 已定义）
            if k >= 2:
                velocity_diff[i, t] = velocity[i, t] - velocity[i, s]
    return diff, velocity, velocity_diff


def last_two_valid_pair(hist_valid_row: torch.Tensor):
    """返回 (t, s)：该 actor 最后两个有效帧索引；不足两个返回 None。"""
    vidx = torch.nonzero(hist_valid_row).flatten().tolist()
    if len(vidx) < 2:
        return None
    return vidx[-1], vidx[-2]


def compute_gap_aware_theta(hist_pos: torch.Tensor, hist_valid_row: torch.Tensor, eps: float = 1e-4):
    """朝向 = 最后两个有效观测连线（允许跨缺口）。返回 (theta, degenerate)。"""
    pair = last_two_valid_pair(hist_valid_row)
    if pair is None:
        return torch.tensor(0.0), True
    t, s = pair
    d = hist_pos[t] - hist_pos[s]
    if torch.norm(d) < eps:
        return torch.tensor(0.0), True
    return torch.atan2(d[1], d[0]), False


# ---------------------------------------------------------------- 单样本构造
def build_sample(
    hist_world: torch.Tensor,
    future_world: torch.Tensor,
    hist_valid: torch.Tensor,
    scene_id: str,
    track_id,
    obs_len: int = 8,
    pred_len: int = 12,
):
    """由一个场景（focal 已置于 actor 0）构造 DeMo batch 字段。

    Args:
        hist_world: [A, 8, 2] 世界坐标，缺失处 NaN
        future_world: [A, 12, 2] 世界坐标，完整（仅用于 target/评估）
        hist_valid: [A, 8] bool
    Returns:
        dict，字段与 EthUcyBenchmarkDataset 输出同构（新增 missing_count 等）。
    """
    A = hist_world.shape[0]
    positions = torch.cat([hist_world, future_world], dim=1).double()  # [A, 20, 2]

    # focal 原点/朝向：最后有效位置 + 最近两个有效观测连线（可跨缺口）
    focal_valid = hist_valid[0]
    fv = torch.nonzero(focal_valid).flatten()
    focal_last = int(fv[-1].item())
    origin = hist_world[0, focal_last].double().clone()
    theta, degenerate = compute_gap_aware_theta(hist_world[0].double(), focal_valid)

    cos_t, sin_t = float(torch.cos(theta)), float(torch.sin(theta))
    rot = torch.tensor([[cos_t, -sin_t], [sin_t, cos_t]], dtype=torch.float64)
    local = torch.matmul(positions - origin.view(1, 1, 2), rot)

    # 缺失历史坐标：NaN -> 0 占位（局部系下），由 x_valid_mask 区分；未来保持真实值
    hist_local = torch.nan_to_num(local[:, :obs_len], nan=0.0).float()
    future_local = local[:, obs_len:].float()  # 无 NaN（已校验）

    # 每 actor 最后有效帧 / 锚点间隔（任务书 §八）
    last_valid_idx = torch.zeros(A, dtype=torch.long)
    for i in range(A):
        vi = torch.nonzero(hist_valid[i]).flatten()
        last_valid_idx[i] = int(vi[-1].item())
    x_anchor_lag = (obs_len - 1 - last_valid_idx).clamp(min=0)
    x_forecast_gap = (obs_len - last_valid_idx).clamp(min=1)

    # 跨缺口运动特征
    diff, velocity, velocity_diff = build_gap_aware_motion(hist_local, hist_valid)
    x_positions_diff = diff
    x_velocity = velocity

    # x_centers：每 actor 最后有效历史位置（局部系，有限值）
    x_centers = hist_local[torch.arange(A), last_valid_idx].clone()

    # 相邻步角度（兼容字段；无效步为 0）；朝向输入使用 x_last_valid_angle
    x_angles = torch.zeros(A, obs_len)
    diff_mask = hist_valid[:, :-1] & hist_valid[:, 1:]
    for t in range(1, obs_len):
        d = hist_local[:, t] - hist_local[:, t - 1]
        ang = torch.atan2(d[:, 1], d[:, 0])
        x_angles[:, t] = torch.where(diff_mask[:, t - 1], ang, torch.zeros_like(ang))
    if obs_len >= 2:
        x_angles[:, 0] = x_angles[:, 1]
    # 每 actor 最近有效朝向：最后两个有效观测连线（与 focal theta 同规则）
    x_last_valid_angle = torch.zeros(A)
    for i in range(A):
        ang_i, _ = compute_gap_aware_theta(hist_local[i].double(), hist_valid[i])
        x_last_valid_angle[i] = ang_i

    x_attr = torch.zeros(A, 3, dtype=torch.uint8)  # type 0 = pedestrian

    # E1 缺失距离特征 + 模块一 GSM-lite 摘要（均 mask-only 纯函数派生）
    miss = build_missing_features(hist_valid)
    x_gap_steps = miss["gap_steps"]
    x_missing_summary = miss["missing_summary"]

    # target：未来完整；target_diff 第一步从各 actor 最后有效历史位置出发
    target = future_local.clone()
    start_pos = hist_local[torch.arange(A), last_valid_idx]  # [A, 2]
    padded = torch.cat([start_pos.unsqueeze(1), target], dim=1)
    target_diff = padded[:, 1:] - padded[:, :-1]
    vel_future = torch.norm(target_diff, dim=-1)  # [A, 12]
    vel_start = velocity[torch.arange(A), last_valid_idx]  # 最后有效帧的有效速度
    vel_padded = torch.cat([vel_start.unsqueeze(1), vel_future], dim=1)
    target_vel_diff = vel_padded[:, 1:] - vel_padded[:, :-1]
    target_mask = torch.ones(A, pred_len, dtype=torch.bool)  # TrajImpute 未来恒完整

    return {
        "target": target,
        "target_diff": target_diff,
        "target_vel_diff": target_vel_diff,
        "target_mask": target_mask,
        "x_positions_diff": x_positions_diff,
        "x_positions": hist_local,
        "x_attr": x_attr,
        "x_centers": x_centers,
        "x_angles": x_angles,
        "x_velocity": x_velocity,
        "x_velocity_diff": velocity_diff,
        "x_valid_mask": hist_valid.clone(),
        "x_key_valid_mask": hist_valid.any(-1),
        "x_last_valid_angle": x_last_valid_angle,
        "x_last_valid_idx": last_valid_idx,
        "x_anchor_lag_steps": x_anchor_lag,
        "x_forecast_gap_steps": x_forecast_gap,
        "x_gap_steps": x_gap_steps,
        "x_missing_summary": x_missing_summary,
        "origin": origin.float().view(1, 2),
        "theta": theta.view(1),
        "degenerate_heading": degenerate,
        "scene_id": scene_id,
        "track_id": track_id,
        "missing_count": int((~hist_valid[0]).sum().item()),
        "focal_last_valid_idx": int(focal_last),
        "timestamp": torch.tensor([obs_len * 0.4]),
    }


# ---------------------------------------------------------------- Dataset
class TrajImputeDataset(Dataset):
    """TrajImpute release 直读 Dataset。

    每个场景序列（seq_start_end 组）中的每个 actor 各生成一个 focal 样本；
    focal 置于 actor 维 0，其余 actor 按组内行序（确定性）作为交互上下文。
    scene_id = f"{scene}-{difficulty}-{split}-seq_{seq_index}"；
    track_id = 该 focal 在 pkl 中的全局行号（test 多缺失块间也唯一）。
    """

    def __init__(
        self,
        data_root: str,
        scene: str,
        difficulty: str,
        split: str,
        obs_len: int = 8,
        pred_len: int = 12,
        zero_missing_only: bool = False,
    ):
        super().__init__()
        if scene not in SCENES:
            raise ValueError(f"unknown scene {scene!r}, expected one of {SCENES}")
        if difficulty not in DIFFICULTIES:
            raise ValueError(f"unknown difficulty {difficulty!r}, expected one of {DIFFICULTIES}")
        if split not in SPLITS:
            raise ValueError(f"unknown split {split!r}, expected one of {SPLITS}")
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.scene = scene
        self.difficulty = difficulty
        self.split = split
        self.zero_missing_only = zero_missing_only

        path = Path(data_root) / scene / difficulty / f"data_{split}.pkl"
        obs, pred, frame_valid, seq_start_end = load_trajimpute_pkl(path, obs_len, pred_len)
        self.file_path = path
        self.obs = obs
        self.pred = pred
        self.frame_valid = frame_valid
        self.seq_start_end = seq_start_end
        self.missing_counts = (~frame_valid).sum(dim=1)  # [N]

        # focal 索引表：(seq_index, start, end, focal_global_row)
        self.samples = []
        for seq_i, (s, e) in enumerate(seq_start_end):
            for row in range(s, e):
                if zero_missing_only and int(self.missing_counts[row]) != 0:
                    continue
                self.samples.append((seq_i, s, e, row))
        print(
            f"TrajImputeDataset {scene}/{difficulty}/{split}: "
            f"{len(self.samples)} focal samples from {len(seq_start_end)} sequences "
            f"({'zero-missing only' if zero_missing_only else 'all actors'})"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        seq_i, s, e, focal_row = self.samples[index]
        hist_world = self.obs[s:e]
        future_world = self.pred[s:e]
        hist_valid = self.frame_valid[s:e]
        # focal 置 0，其余按组内原始行序（确定性、可复现）
        others = [r for r in range(e - s) if r != focal_row - s]
        order = [focal_row - s] + others
        sample = build_sample(
            hist_world[order],
            future_world[order],
            hist_valid[order],
            scene_id=f"{self.scene}-{self.difficulty}-{self.split}-seq_{seq_i}",
            track_id=focal_row,
            obs_len=self.obs_len,
            pred_len=self.pred_len,
        )
        sample["seq_index"] = seq_i
        sample["num_actors"] = e - s
        return sample


def trajimpute_collate_fn(batch):
    """TrajImpute collate：含 E1 缺失距离字段。"""
    data = {}
    for key in [
        "x_positions_diff", "x_attr", "x_positions", "x_centers",
        "x_angles", "x_velocity", "x_velocity_diff",
        "x_last_valid_angle", "x_last_valid_idx",
        "x_anchor_lag_steps", "x_forecast_gap_steps",
        "x_gap_steps", "x_missing_summary",
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
    data["x_key_valid_mask"][:, 0] = True  # focal 恒有效
    data["scene_id"] = [b["scene_id"] for b in batch]
    data["track_id"] = [b["track_id"] for b in batch]
    data["origin"] = torch.cat([b["origin"] for b in batch], dim=0)
    data["theta"] = torch.cat([b["theta"] for b in batch])
    data["timestamp"] = torch.cat([b["timestamp"] for b in batch])
    data["missing_count"] = torch.tensor([b["missing_count"] for b in batch], dtype=torch.long)
    data["focal_last_valid_idx"] = torch.tensor(
        [b["focal_last_valid_idx"] for b in batch], dtype=torch.long
    )
    data["seq_index"] = [b["seq_index"] for b in batch]
    return data


# ---------------------------------------------------------------- DataModule
from pytorch_lightning import LightningDataModule  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


class TrajImputeDataModule(LightningDataModule):
    """TrajImpute DataModule：test=True 时只加载 test split（与现有 eval 流程一致）。"""

    def __init__(
        self,
        data_root: str,
        scene: str,
        difficulty: str,
        obs_len: int = 8,
        pred_len: int = 12,
        train_batch_size: int = 64,
        val_batch_size: int = 64,
        test_batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True,
        test: bool = False,
        zero_missing_only: bool = False,
    ):
        super().__init__()
        self.data_root = data_root
        self.scene = scene
        self.difficulty = difficulty
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.test = test
        self.zero_missing_only = zero_missing_only

    def _dataset(self, split: str):
        return TrajImputeDataset(
            self.data_root, self.scene, self.difficulty, split,
            self.obs_len, self.pred_len, zero_missing_only=self.zero_missing_only,
        )

    def setup(self, stage=None):
        if self.test:
            self.test_dataset = self._dataset("test")
        else:
            self.train_dataset = self._dataset("train")
            self.val_dataset = self._dataset("val")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.train_batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=self.pin_memory,
            collate_fn=trajimpute_collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.val_batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=self.pin_memory,
            collate_fn=trajimpute_collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.test_batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=self.pin_memory,
            collate_fn=trajimpute_collate_fn,
        )


# ---------------------------------------------------------------- Clean-direct 数据源探查
def inspect_clean_source(data_root: str = TRAJIMPUTE_ROOT):
    """检查 release 是否存在独立 clean 数据。

    Easy 中的零缺失样本仅用于数据完整性诊断，不能替代完整 ETH/UCY
    benchmark 之外的自建数据，也不能作为 Clean 数据源（Clean=原始 ETH/UCY 直接引用原论文数字）。
    """
    root = Path(data_root)
    independent = []
    for scene in SCENES:
        for name in ("Clean", "clean", "CLEAN"):
            p = root / scene / name
            if p.exists():
                independent.append(str(p))
    info = {
        "independent_release_dir": independent,
        "type": "independent_release" if independent else "unavailable",
        "zero_missing_diagnostic_available": not bool(independent),
        "note": (
            "release 仅含 Easy/Hard；无独立 clean pkl。"
            "Easy 零缺失样本仅作诊断，不是 Clean-direct 数据源。"
        ) if not independent else "发现独立 clean 目录",
    }
    return info
