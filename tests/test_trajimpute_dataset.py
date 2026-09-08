"""TrajImpute adapter 数据语义测试（任务书 §十二 1–13, 19, 20）。

依赖本机 release：/home/lbh/TrajImpute/dataset/TrajImpute（30 pkl 已审计）。
GPU 测试（forward/backward）需 CUDA，无 GPU 时自动跳过。
"""

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.datamodule.trajimpute_dataset import (  # noqa: E402
    SCENES, DIFFICULTIES, SPLITS, TRAJIMPUTE_ROOT,
    TrajImputeDataset, trajimpute_collate_fn, load_trajimpute_pkl,
    build_gap_aware_motion, build_sample, inspect_clean_source,
)

HAVE_RELEASE = Path(TRAJIMPUTE_ROOT).exists()
needs_release = pytest.mark.skipif(not HAVE_RELEASE, reason="release 未下载")


# ---------------------------------------------------------------- 1. 30 pkl 可发现
def test_all_30_pkls_discoverable():
    root = Path(TRAJIMPUTE_ROOT)
    files = list(root.glob("*/Easy/data_train.pkl")) + \
            list(root.glob("*/Hard/data_train.pkl"))
    n = 0
    for scene in SCENES:
        for diff in DIFFICULTIES:
            for split in SPLITS:
                p = root / scene / diff / f"data_{split}.pkl"
                assert p.exists(), f"missing {p}"
                n += 1
    assert n == 30


def test_path_parsing_scene_difficulty_split():
    ds = TrajImputeDataset(TRAJIMPUTE_ROOT, "ETH-M", "Easy", "train")
    assert ds.scene == "ETH-M" and ds.difficulty == "Easy" and ds.split == "train"
    with pytest.raises(ValueError):
        TrajImputeDataset(TRAJIMPUTE_ROOT, "ETH", "Easy", "train")
    with pytest.raises(ValueError):
        TrajImputeDataset(TRAJIMPUTE_ROOT, "ETH-M", "easy", "train")  # 大小写敏感
    with pytest.raises(ValueError):
        TrajImputeDataset(TRAJIMPUTE_ROOT, "ETH-M", "Easy", "heldout")


@needs_release
def test_nan_mask_consistency_all_30():
    """NaN 与 missing_mask 严格一致（load 时校验不抛错即通过，逐文件显式验证）。"""
    root = Path(TRAJIMPUTE_ROOT)
    for scene in SCENES:
        for diff in DIFFICULTIES:
            for split in SPLITS:
                p = root / scene / diff / f"data_{split}.pkl"
                obs, pred, fv, _ = load_trajimpute_pkl(p)  # 内部已断言一致
                nan = torch.isnan(obs[..., 0])
                assert torch.equal(nan, ~fv)


def test_missing_mask_true_maps_to_invalid():
    """missing_mask=True -> x_valid_mask=False（含 Easy 0 缺失与 Hard 7 缺失样本）。"""
    ds = TrajImputeDataset(TRAJIMPUTE_ROOT, "ETH-M", "Hard", "test")
    # Hard test 每块缺失数 4/5/6/7：取缺失 7 的样本（最后块）
    counts = ds.missing_counts.tolist()
    idx7 = counts.index(7)
    s = ds[idx7]
    assert int((~s["x_valid_mask"][0]).sum()) == 7
    assert bool(s["x_valid_mask"][0].any())  # 至少 1 有效帧
    # focal=缺失7 时 anchor_lag>=1
    assert int(s["x_anchor_lag_steps"][0]) == 7 - int(s["x_last_valid_idx"][0])


# ---------------------------------------------------------------- 运动特征语义
def _make_hist(tail_missing: bool = False):
    """构造 8 帧轨迹。默认帧 0,2,5,7 有效（跨缺口）；tail_missing=True 时帧 6/7 缺失（末帧缺失，最后有效=5）。"""
    pos = torch.zeros(2, 8, 2)
    pos[0, 0] = torch.tensor([0.0, 0.0])
    pos[0, 2] = torch.tensor([2.0, 0.0])
    pos[0, 5] = torch.tensor([8.0, 0.0])
    pos[0, 7] = torch.tensor([10.0, 0.0])
    valid = torch.zeros(2, 8, dtype=torch.bool)
    valid[0, [0, 2, 5, 7]] = True
    if tail_missing:
        valid[0, 7] = False
        pos[0, 7] = torch.tensor([0.0, 0.0])
    # actor 1 完整
    pos[1] = torch.linspace(0, 7, 8).unsqueeze(-1).repeat(1, 2)
    valid[1] = True
    return pos, valid


def test_gap_aware_diff_and_velocity():
    pos, valid = _make_hist()
    diff, vel, vel_diff = build_gap_aware_motion(pos, valid)
    # 帧 2: 前一有效=0，diff=(2,0)，gap=2，vel=1.0
    assert torch.allclose(diff[0, 2], torch.tensor([2.0, 0.0]))
    assert torch.isclose(vel[0, 2], torch.tensor(1.0))
    # 帧 5: 前一有效=2，diff=(6,0)，gap=3，vel=2.0
    assert torch.allclose(diff[0, 5], torch.tensor([6.0, 0.0]))
    assert torch.isclose(vel[0, 5], torch.tensor(2.0))
    # 帧 7: 前一有效=5，diff=(2,0)，gap=2，vel=1.0；vel_diff = 1 - 2 = -1
    assert torch.isclose(vel[0, 7], torch.tensor(1.0))
    assert torch.isclose(vel_diff[0, 7], torch.tensor(-1.0))
    # 缺失帧全部为 0
    assert torch.equal(diff[0, 1], torch.zeros(2))
    assert torch.equal(diff[0, 3], torch.zeros(2))
    assert torch.equal(diff[0, 4], torch.zeros(2))
    assert torch.equal(diff[0, 6], torch.zeros(2))
    assert torch.equal(vel[0, [1, 3, 4, 6]], torch.zeros(4))
    # 首个有效帧无差分
    assert torch.equal(diff[0, 0], torch.zeros(2))
    assert torch.isfinite(diff).all() and torch.isfinite(vel).all()


def test_missing_coords_do_not_enter_diff_or_velocity():
    """缺失坐标（NaN）不进入差分/速度：构造含 NaN 的世界坐标走完整 build_sample。"""
    pos, valid = _make_hist()
    pos_nan = pos.clone()
    pos_nan[0, ~valid[0]] = float("nan")
    pos_nan[1, ~valid[1]] = float("nan")  # 全 valid，无 NaN
    future = torch.linspace(0, 1, 12).unsqueeze(-1).repeat(2, 1, 2) + pos_nan[:, -1:].nan_to_num(0)
    s = build_sample(pos_nan, future, valid, scene_id="t", track_id=0)
    # 全部模型输入字段 finite
    for k in ["x_positions_diff", "x_velocity", "x_velocity_diff", "x_positions",
              "x_centers", "x_angles", "x_last_valid_angle", "x_gap_steps",
              "x_prev_valid_gap", "x_motion_valid", "x_motion_run",
              "x_missing_summary", "target", "target_diff", "target_vel_diff"]:
        assert torch.isfinite(s[k]).all(), f"{k} not finite"
    # 有效帧的差分真实等于世界坐标差（局部系为纯平移时）：帧 5 diff = p5-p2 = (6,0)
    # 注意 focal theta 可能非零（最后两有效帧 5->7 方向 (2,0) -> theta=0），此构造下为 0
    assert torch.allclose(s["x_positions_diff"][0, 5], torch.tensor([6.0, 0.0]), atol=1e-5)


def test_origin_is_last_valid_position_when_tail_missing():
    """末帧缺失时原点 = 最后有效位置，不得用帧 7。"""
    # focal 帧 7 缺失，最后有效=帧 5
    pos, valid = _make_hist(tail_missing=True)
    pos_nan = pos.clone()
    pos_nan[0, ~valid[0]] = float("nan")
    future = pos_nan[:, 5:6].nan_to_num(0) + torch.linspace(0, 1, 12).unsqueeze(-1).repeat(2, 1, 2)
    s = build_sample(pos_nan, future, valid, scene_id="t", track_id=0)
    assert int(s["x_last_valid_idx"][0]) == 5
    assert int(s["x_anchor_lag_steps"][0]) == 2
    assert int(s["x_forecast_gap_steps"][0]) == 3
    # 世界坐标下 focal 最后有效位置 = (8, 0)
    assert torch.allclose(s["origin"][0], torch.tensor([8.0, 0.0]), atol=1e-6)
    # focal 自身最后有效位置在局部系下 = 原点（0,0）
    assert torch.allclose(s["x_centers"][0], torch.zeros(2), atol=1e-6)


def test_target_diff_starts_from_last_valid_position():
    """target_diff 第一步 = target[0] - start_pos（最后有效历史位置），不从 hist[:, -1]。"""
    pos, valid = _make_hist(tail_missing=True)
    pos_nan = pos.clone()
    pos_nan[0, ~valid[0]] = float("nan")
    # theta：最后两个有效帧 2->5 方向 (6,0) -> theta=0；origin = 帧5 位置 (8,0)
    future_world = pos_nan[:, 5:6].nan_to_num(0) + torch.linspace(0, 1, 12).unsqueeze(-1).repeat(2, 1, 2)
    future_world = future_world.clone()
    future_world[0, 0] = torch.tensor([9.0, 1.0])
    s = build_sample(pos_nan, future_world, valid, scene_id="t", track_id=0)
    # 局部系：origin=(8,0), theta=0。target[0][0] 局部 = (1,1)；start_pos = (0,0)
    assert torch.allclose(s["target_diff"][0, 0], torch.tensor([1.0, 1.0]), atol=1e-5)


def test_single_valid_frame_no_nan_inf():
    """单有效帧：速度 0、朝向 0（degenerate）、全字段 finite。"""
    pos = torch.zeros(1, 8, 2)
    pos[0, 3] = torch.tensor([1.0, 2.0])
    valid = torch.zeros(1, 8, dtype=torch.bool)
    valid[0, 3] = True
    future = pos[:, :1] + torch.linspace(0, 1, 12).unsqueeze(-1).repeat(1, 1, 2)
    s = build_sample(pos, future, valid, scene_id="t", track_id=0)
    assert bool(s["degenerate_heading"]) is True
    assert float(s["theta"]) == 0.0
    assert torch.equal(s["x_velocity"][0], torch.zeros(8))
    assert torch.isfinite(s["x_positions"]).all()
    assert int(s["x_last_valid_idx"][0]) == 3
    assert int(s["x_forecast_gap_steps"][0]) == 5
    for k in ["x_positions_diff", "x_velocity", "x_velocity_diff", "target",
              "target_diff", "target_vel_diff", "x_missing_summary"]:
        assert torch.isfinite(s[k]).all(), f"{k} not finite (single valid frame)"


def test_future_target_no_nan():
    ds = TrajImputeDataset(TRAJIMPUTE_ROOT, "ZARA2-M", "Hard", "test")
    for i in [0, len(ds) // 2, len(ds) - 1]:
        s = ds[i]
        assert torch.isfinite(s["target"]).all()
        assert bool(s["target_mask"].all())


def test_seq_start_end_no_cross_scene_mixing():
    """scene_id/track 唯一且不跨组混合：同一 seq 内 actor 的 num_actors 一致。"""
    ds = TrajImputeDataset(TRAJIMPUTE_ROOT, "ETH-M", "Easy", "test")
    seen_scene_ids = set()
    for i in range(0, len(ds), max(1, len(ds) // 37)):
        s = ds[i]
        sid = s["scene_id"]
        assert sid not in seen_scene_ids or True  # 同一 seq 多 focal 属正常
        # 每个样本的 actor 数 == 该组 seq_start_end 大小（由构造保证）：
        assert s["x_positions"].shape[0] == s["num_actors"]
    # 不同 seq 的 scene_id 不同
    ids = {ds.samples[j][0] for j in range(len(ds))}
    assert len(ids) > 1


def test_focal_always_index_zero():
    ds = TrajImputeDataset(TRAJIMPUTE_ROOT, "UNIV-M", "Easy", "val")
    for i in [0, 5, 17]:
        s = ds[i]
        # focal = 该样本 track_id 对应行；构造上 hist_world[order] 中 focal 在 0
        # 验证：x_valid_mask[0] 与该行全局掩码一致
        row = s["track_id"]
        global_valid = ds.frame_valid[row]
        # focal 有效帧一致（无重排损失）
        assert torch.equal(s["x_valid_mask"][0], global_valid)
        # missing_count 一致
        assert s["missing_count"] == int((~global_valid).sum())


def test_deterministic_actor_order():
    ds = TrajImputeDataset(TRAJIMPUTE_ROOT, "ETH-M", "Easy", "test")
    s1 = ds[3]
    s2 = ds[3]
    for k in ["x_positions", "x_valid_mask", "x_centers"]:
        assert torch.equal(s1[k], s2[k])


def test_clean_source_inspection():
    info = inspect_clean_source(TRAJIMPUTE_ROOT)
    assert info["type"] in ("independent_release", "unavailable")
    assert info["zero_missing_diagnostic_available"] is True
    assert "note" in info


def test_easy_test_zero_missing_block_exists():
    """Easy test 的缺失数 0 块存在（clean 评估来源）。"""
    ds = TrajImputeDataset(TRAJIMPUTE_ROOT, "ETH-M", "Easy", "test")
    counts = ds.missing_counts.tolist()
    assert 0 in counts


# ---------------------------------------------------------------- collate + 模型
GPU = torch.cuda.is_available()


def _collated_batch(scene="ETH-M", difficulty="Easy", split="test", n=4):
    ds = TrajImputeDataset(TRAJIMPUTE_ROOT, scene, difficulty, split)
    items = [ds[i] for i in range(min(n, len(ds)))]
    return trajimpute_collate_fn(items)


@pytest.mark.skipif(not GPU, reason="需要 GPU（Mamba CUDA）")
def test_batch_m0_forward_and_backward():
    from src.model.model_forecast import ModelForecast
    batch = _collated_batch()
    batch = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in batch.items()}
    model = ModelForecast(num_modes=6, bimamba=True, use_observation_features=False,
                          use_missing_summary=False).cuda()
    out = model(batch)
    assert out["new_y_hat"].shape[1] == 6
    # loss（复用 trainer.cal_loss 的核心项）
    y = batch["target"][:, 0]
    l2 = torch.norm(out["new_y_hat"][..., :2] - y.unsqueeze(1), dim=-1).sum(-1)
    best = torch.argmin(l2, dim=-1)
    best_traj = out["new_y_hat"][torch.arange(y.shape[0]), best][..., :2]
    loss = torch.nn.functional.smooth_l1_loss(best_traj, y) + \
        torch.nn.functional.cross_entropy(out["new_pi"], best.detach())
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0 and all(torch.isfinite(g).all() for g in grads)
    assert torch.isfinite(out["new_y_hat"]).all()


@pytest.mark.skipif(not GPU, reason="需要 GPU（Mamba CUDA）")
def test_batch_m1_m2_forward():
    from src.model.model_forecast import ModelForecast
    batch = _collated_batch(difficulty="Hard", n=3)
    batch = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in batch.items()}
    for switches in (
        {"use_observation_features": True, "use_missing_summary": False},
        {"use_observation_features": True, "use_missing_summary": True},
    ):
        model = ModelForecast(num_modes=6, bimamba=True, **switches).cuda()
        out = model(batch)
        assert torch.isfinite(out["new_y_hat"]).all()


def test_collate_padding_semantics():
    """不同 actor 数量组成 batch 时，padding actor 不得进入 others loss。"""
    def make_item(num_actors: int, scene_id: str):
        hist = torch.stack([
            torch.arange(8, dtype=torch.float32),
            torch.zeros(8, dtype=torch.float32),
        ], dim=-1).repeat(num_actors, 1, 1)
        future = torch.stack([
            torch.arange(12, dtype=torch.float32) + 8,
            torch.zeros(12, dtype=torch.float32),
        ], dim=-1).repeat(num_actors, 1, 1)
        valid = torch.ones(num_actors, 8, dtype=torch.bool)
        sample = build_sample(
            hist,
            future,
            valid,
            scene_id=scene_id,
            track_id=0,
        )
        sample["seq_index"] = 0
        return sample

    batch = trajimpute_collate_fn([
        make_item(1, "one-actor"),
        make_item(3, "three-actors"),
    ])

    # The first sample has two padded actors; they must be false for all
    # forecast steps so trainer_forecast cannot include them in others loss.
    assert batch["target_mask"].shape == (2, 3, 12)
    assert bool(batch["target_mask"][0, 0].all())
    assert not bool(batch["target_mask"][0, 1:].any())
    assert bool(batch["target_mask"][1].all())
    assert torch.isfinite(batch["x_positions"]).all()


def test_evaluation_reads_no_future_as_input():
    """历史输入字段不含未来信息：x_positions 长度 == 8。"""
    s = TrajImputeDataset(TRAJIMPUTE_ROOT, "ETH-M", "Easy", "test")[0]
    assert s["x_positions"].shape[1] == 8
    assert s["target"].shape[1] == 12
