"""向量化等价性验收（第二阶段提速，2026-09-22）。

参考实现 = 逐帧/逐 actor 循环版（改造前原代码的逐字拷贝）。
验收标准：所有张量字段逐位相等（torch.equal / bitwise identical），
只允许两类文档化差异：
  1. 零有效帧 actor：旧版在 last_valid_idx 循环处 IndexError 崩溃，
     新版返回 0（官方数据加载时已校验不存在零有效帧样本，不可达路径）。
  2. 类型元数据（dtype 本身相同；仅标量 Python 包装差异不存在）。

覆盖：
  A. build_gap_aware_motion：穷举 2^8 全部有效掩码模式 × 多组位置
     （随机、退化位移 <eps、单有效帧、全有效），diff/velocity/velocity_diff
     逐位相等。
  B. _last_valid_indices：与逐 actor nonzero 循环一致。
  C. build_sample 全字段：合成场景（含 NaN 世界坐标、末帧缺失、单有效帧、
     近零位移退化朝向）逐字段 torch.equal。
  D. 真实 release 抽样：每场景×难度×split 取确定性子样本对比全字段。
"""

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.datamodule.trajimpute_dataset import (  # noqa: E402
    TRAJIMPUTE_ROOT, build_gap_aware_motion as build_gap_aware_motion_vec,
    build_sample as build_sample_vec,
    _last_valid_indices, _prev_valid_index,
)

HAVE_RELEASE = Path(TRAJIMPUTE_ROOT).exists()
needs_release = pytest.mark.skipif(not HAVE_RELEASE, reason="release 未下载")


# ---------------------------------------------------------------- 参考实现（改造前原代码）
def build_gap_aware_motion_ref(hist_pos: torch.Tensor, hist_valid: torch.Tensor):
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
            if k >= 2:
                velocity_diff[i, t] = velocity[i, t] - velocity[i, s]
    return diff, velocity, velocity_diff


def last_valid_idx_ref(hist_valid: torch.Tensor) -> torch.Tensor:
    A = hist_valid.shape[0]
    out = torch.zeros(A, dtype=torch.long)
    for i in range(A):
        vi = torch.nonzero(hist_valid[i]).flatten()
        if vi.numel() == 0:
            raise IndexError("ref: zero-valid actor (旧行为=崩溃)")
        out[i] = int(vi[-1].item())
    return out


def compute_gap_aware_theta_ref(hist_pos, hist_valid_row, eps: float = 1e-4):
    vidx = torch.nonzero(hist_valid_row).flatten().tolist()
    if len(vidx) < 2:
        return torch.tensor(0.0), True
    t, s = vidx[-1], vidx[-2]
    d = hist_pos[t] - hist_pos[s]
    if torch.norm(d) < eps:
        return torch.tensor(0.0), True
    return torch.atan2(d[1], d[0]), False


def build_sample_ref(hist_world, future_world, hist_valid, scene_id, track_id,
                     obs_len: int = 8, pred_len: int = 12):
    """改造前 build_sample 的逐字拷贝（循环版）。"""
    A = hist_world.shape[0]
    positions = torch.cat([hist_world, future_world], dim=1).double()

    focal_valid = hist_valid[0]
    fv = torch.nonzero(focal_valid).flatten()
    focal_last = int(fv[-1].item())
    origin = hist_world[0, focal_last].double().clone()
    theta, degenerate = compute_gap_aware_theta_ref(hist_world[0].double(), focal_valid)

    cos_t, sin_t = float(torch.cos(theta)), float(torch.sin(theta))
    rot = torch.tensor([[cos_t, -sin_t], [sin_t, cos_t]], dtype=torch.float64)
    local = torch.matmul(positions - origin.view(1, 1, 2), rot)

    hist_local = torch.nan_to_num(local[:, :obs_len], nan=0.0).float()
    future_local = local[:, obs_len:].float()

    last_valid_idx = last_valid_idx_ref(hist_valid)
    x_anchor_lag = (obs_len - 1 - last_valid_idx).clamp(min=0)
    x_forecast_gap = (obs_len - last_valid_idx).clamp(min=1)

    diff, velocity, velocity_diff = build_gap_aware_motion_ref(hist_local, hist_valid)

    x_centers = hist_local[torch.arange(A), last_valid_idx].clone()

    x_angles = torch.zeros(A, obs_len)
    diff_mask = hist_valid[:, :-1] & hist_valid[:, 1:]
    for t in range(1, obs_len):
        d = hist_local[:, t] - hist_local[:, t - 1]
        ang = torch.atan2(d[:, 1], d[:, 0])
        x_angles[:, t] = torch.where(diff_mask[:, t - 1], ang, torch.zeros_like(ang))
    if obs_len >= 2:
        x_angles[:, 0] = x_angles[:, 1]
    x_last_valid_angle = torch.zeros(A)
    for i in range(A):
        ang_i, _ = compute_gap_aware_theta_ref(hist_local[i].double(), hist_valid[i])
        x_last_valid_angle[i] = ang_i

    x_attr = torch.zeros(A, 3, dtype=torch.uint8)

    from src.datamodule.missing_features import build_missing_features
    miss = build_missing_features(hist_valid)
    x_gap_steps = miss["gap_steps"]

    target = future_local.clone()
    start_pos = hist_local[torch.arange(A), last_valid_idx]
    padded = torch.cat([start_pos.unsqueeze(1), target], dim=1)
    target_diff = padded[:, 1:] - padded[:, :-1]
    vel_future = torch.norm(target_diff, dim=-1)
    vel_start = velocity[torch.arange(A), last_valid_idx]
    vel_padded = torch.cat([vel_start.unsqueeze(1), vel_future], dim=1)
    target_vel_diff = vel_padded[:, 1:] - vel_padded[:, :-1]
    target_mask = torch.ones(A, pred_len, dtype=torch.bool)

    return {
        "target": target,
        "target_diff": target_diff,
        "target_vel_diff": target_vel_diff,
        "target_mask": target_mask,
        "x_positions_diff": diff,
        "x_positions": hist_local,
        "x_attr": x_attr,
        "x_centers": x_centers,
        "x_angles": x_angles,
        "x_velocity": velocity,
        "x_velocity_diff": velocity_diff,
        "x_valid_mask": hist_valid.clone(),
        "x_key_valid_mask": hist_valid.any(-1),
        "x_last_valid_angle": x_last_valid_angle,
        "x_last_valid_idx": last_valid_idx,
        "x_anchor_lag_steps": x_anchor_lag,
        "x_forecast_gap_steps": x_forecast_gap,
        "x_gap_steps": x_gap_steps,
        "origin": origin.float().view(1, 2),
        "theta": theta.view(1),
        "degenerate_heading": degenerate,
        "scene_id": scene_id,
        "track_id": track_id,
        "missing_count": int((~hist_valid[0]).sum().item()),
        "focal_last_valid_idx": int(focal_last),
        "timestamp": torch.tensor([obs_len * 0.4]),
    }


TENSOR_FIELDS = [
    "target", "target_diff", "target_vel_diff", "target_mask",
    "x_positions_diff", "x_positions", "x_attr", "x_centers",
    "x_angles", "x_velocity", "x_velocity_diff", "x_valid_mask",
    "x_key_valid_mask", "x_last_valid_angle", "x_last_valid_idx",
    "x_anchor_lag_steps", "x_forecast_gap_steps", "x_gap_steps",
    "origin", "theta", "timestamp",
]


def _assert_sample_equal(s_ref, s_new, ctx: str):
    for k in TENSOR_FIELDS:
        assert torch.equal(s_ref[k], s_new[k]), f"{ctx}: field {k} 不逐位相等"
    assert s_ref["degenerate_heading"] == s_new["degenerate_heading"], ctx
    assert s_ref["scene_id"] == s_new["scene_id"], ctx
    assert s_ref["track_id"] == s_new["track_id"], ctx
    assert s_ref["missing_count"] == s_new["missing_count"], ctx
    assert s_ref["focal_last_valid_idx"] == s_new["focal_last_valid_idx"], ctx


# ---------------------------------------------------------------- A. 穷举掩码模式
def _all_mask_patterns():
    return [tuple((m >> t) & 1 for t in range(8)) for m in range(1, 256)]  # 至少 1 有效帧


@pytest.mark.parametrize("pattern", _all_mask_patterns())
def test_motion_all_mask_patterns(pattern):
    valid = torch.tensor([pattern], dtype=torch.bool)
    for scale in (1.0, 3.7):
        g = torch.Generator().manual_seed(sum(pattern) + int(scale * 97))
        pos = torch.randn((1, 8, 2), generator=g) * scale
        r = build_gap_aware_motion_ref(pos, valid)
        v = build_gap_aware_motion_vec(pos, valid)
        for a, b in zip(r, v):
            assert torch.equal(a, b), f"pattern={pattern} scale={scale}"


def test_motion_degenerate_and_single_valid():
    # 单有效帧
    valid = torch.zeros(1, 8, dtype=torch.bool); valid[0, 4] = True
    pos = torch.randn(1, 8, 2)
    for a, b in zip(build_gap_aware_motion_ref(pos, valid), build_gap_aware_motion_vec(pos, valid)):
        assert torch.equal(a, b)
    # 近零位移（退化朝向路径）
    pos2 = torch.zeros(1, 8, 2)
    pos2[0, :, 0] = torch.linspace(0, 1e-5, 8)  # 相邻位移 << eps
    valid2 = torch.ones(1, 8, dtype=torch.bool)
    for a, b in zip(build_gap_aware_motion_ref(pos2, valid2), build_gap_aware_motion_vec(pos2, valid2)):
        assert torch.equal(a, b)
    # 多 actor 混合掩码
    g = torch.Generator().manual_seed(7)
    posm = torch.randn((6, 8, 2), generator=g)
    vm = torch.rand((6, 8), generator=g) > 0.4
    vm[:, 0] = True  # 每 actor 至少 1 有效帧
    for a, b in zip(build_gap_aware_motion_ref(posm, vm), build_gap_aware_motion_vec(posm, vm)):
        assert torch.equal(a, b)


def test_last_valid_indices_all_patterns():
    for pattern in _all_mask_patterns():
        valid = torch.tensor([pattern], dtype=torch.bool)
        assert torch.equal(last_valid_idx_ref(valid), _last_valid_indices(valid))


def test_prev_valid_index_semantics():
    valid = torch.tensor([[1, 0, 0, 1, 1, 0, 0, 1]], dtype=torch.bool)
    prev = _prev_valid_index(valid)
    assert prev.tolist() == [[-1, 0, 0, 0, 3, 4, 4, 4]]


# ---------------------------------------------------------------- C. build_sample 全字段（合成）
def _synth_scene(seed: int, n_actors: int = 5):
    g = torch.Generator().manual_seed(seed)
    hist = torch.randn((n_actors, 8, 2), generator=g)
    valid = torch.rand((n_actors, 8), generator=g) > 0.45
    valid[:, 0] = True
    valid[0] = torch.rand((8,), generator=g) > 0.4
    valid[0, 2] = True  # focal 至少 2 有效帧
    hist_w = hist.clone()
    hist_w[~valid] = float("nan")
    future = hist.nan_to_num(0)[:, -1:] + torch.randn((n_actors, 12, 2), generator=g).cumsum(1) * 0.3
    return hist_w, future, valid


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 11, 2024])
def test_build_sample_synthetic(seed):
    hist_w, future, valid = _synth_scene(seed)
    s_ref = build_sample_ref(hist_w, future, valid, scene_id="t", track_id=0)
    s_new = build_sample_vec(hist_w, future, valid, scene_id="t", track_id=0)
    _assert_sample_equal(s_ref, s_new, f"seed={seed}")


def test_build_sample_tail_missing_and_degenerate_heading():
    # 末帧缺失 + focal 近零位移（退化 theta）
    hist_w, future, valid = _synth_scene(42)
    valid[0, 6:] = False
    hist_w[0, 6:] = float("nan")
    # 退化：最后两个有效帧 (2,0)（3–7 缺失），位移 <eps
    hist_w[0, 2] = hist_w[0, 0] + 5e-5
    s_ref = build_sample_ref(hist_w, future, valid, scene_id="t", track_id=0)
    s_new = build_sample_vec(hist_w, future, valid, scene_id="t", track_id=0)
    _assert_sample_equal(s_ref, s_new, "tail_missing+degenerate")
    assert bool(s_new["degenerate_heading"]) is True


def test_build_sample_single_valid_focal():
    hist_w, future, valid = _synth_scene(5)
    valid[0, :] = False
    valid[0, 3] = True
    hist_w[0, :] = float("nan")
    hist_w[0, 3] = torch.tensor([1.0, -2.0])
    s_ref = build_sample_ref(hist_w, future, valid, scene_id="t", track_id=0)
    s_new = build_sample_vec(hist_w, future, valid, scene_id="t", track_id=0)
    _assert_sample_equal(s_ref, s_new, "single_valid_focal")
    assert bool(s_new["degenerate_heading"]) is True


def test_build_sample_full_valid():
    hist_w, future, valid = _synth_scene(9)
    valid[:] = True
    hist_w = hist_w.nan_to_num(0)
    s_ref = build_sample_ref(hist_w, future, valid, scene_id="t", track_id=0)
    s_new = build_sample_vec(hist_w, future, valid, scene_id="t", track_id=0)
    _assert_sample_equal(s_ref, s_new, "full_valid")


# ---------------------------------------------------------------- D. 真实 release 抽样
@needs_release
def test_build_sample_real_release_sampled():
    """5 场景 × Easy/Hard × train/val/test：每数据集确定性抽样 60 个 focal 样本
    （覆盖首/中/尾 + 均布），全字段逐位对比。全量 sweep 由独立脚本另行执行。"""
    from src.datamodule.trajimpute_dataset import TrajImputeDataset
    for scene in ("ETH-M", "HOTEL-M", "UNIV-M", "ZARA1-M", "ZARA2-M"):
        for diff in ("Easy", "Hard"):
            for split in ("train", "val", "test"):
                ds = TrajImputeDataset(TRAJIMPUTE_ROOT, scene, diff, split)
                n = len(ds)
                idxs = sorted(set(
                    list(range(0, min(20, n)))
                    + list(range(max(0, n // 2 - 10), max(0, n // 2 + 10)))
                    + list(range(max(0, n - 20), n))
                ))
                for i in idxs:
                    seq_i, s, e, focal_row = ds.samples[i]
                    hist_world = ds.obs[s:e]
                    future_world = ds.pred[s:e]
                    hist_valid = ds.frame_valid[s:e]
                    others = [r for r in range(e - s) if r != focal_row - s]
                    order = [focal_row - s] + others
                    args = (hist_world[order], future_world[order], hist_valid[order])
                    s_ref = build_sample_ref(*args, scene_id="t", track_id=focal_row)
                    s_new = build_sample_vec(*args, scene_id="t", track_id=focal_row)
                    _assert_sample_equal(s_ref, s_new, f"{scene}/{diff}/{split}[{i}]")
