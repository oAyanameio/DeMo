"""B1（P0.5 运动证据增强）数据语义与模型前向测试。"""

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.datamodule.trajimpute_dataset import (  # noqa: E402
    TRAJIMPUTE_ROOT, TrajImputeDataset, trajimpute_collate_fn, build_sample,
)

GPU = torch.cuda.is_available()


def _hist():
    """帧 0,2,5,7 有效；速度序列 1.0 / 2.0 / 1.0（跨缺口归一化）。"""
    pos = torch.zeros(1, 8, 2)
    pos[0, 0] = torch.tensor([0.0, 0.0])
    pos[0, 2] = torch.tensor([2.0, 0.0])
    pos[0, 5] = torch.tensor([8.0, 0.0])
    pos[0, 7] = torch.tensor([10.0, 0.0])
    valid = torch.zeros(1, 8, dtype=torch.bool)
    valid[0, [0, 2, 5, 7]] = True
    return pos, valid


def test_b1_fields_present_and_finite():
    pos, valid = _hist()
    future = pos[:, 5:6] + torch.linspace(0, 1, 12).unsqueeze(-1).repeat(1, 1, 2)
    s = build_sample(pos, future, valid, scene_id="t", track_id=0)
    for k in ("x_velocity", "x_accel", "x_turn_rate", "x_motion_run"):
        assert k in s, f"{k} missing"
        assert torch.isfinite(s[k]).all(), f"{k} not finite"
    # 速度：帧2=2/2=1.0、帧5=6/3=2.0、帧7=2/2=1.0
    assert torch.isclose(s["x_velocity"][0, 2], torch.tensor(1.0))
    assert torch.isclose(s["x_velocity"][0, 5], torch.tensor(2.0))
    assert torch.isclose(s["x_velocity"][0, 7], torch.tensor(1.0))
    # accel（相邻有效速度差）：帧5=+1、帧7=-1；帧2 无前序速度=0
    assert torch.isclose(s["x_accel"][0, 5], torch.tensor(1.0))
    assert torch.isclose(s["x_accel"][0, 7], torch.tensor(-1.0))
    assert float(s["x_accel"][0, 2]) == 0.0


def test_b1_turn_rate_straight_is_zero_and_turn_detected():
    pos, valid = _hist()
    future = pos[:, 5:6] + torch.linspace(0, 1, 12).unsqueeze(-1).repeat(1, 1, 2)
    s = build_sample(pos, future, valid, scene_id="t", track_id=0)
    # 直线运动：所有有效步 turn_rate = 0
    assert float(s["x_turn_rate"].abs().sum()) == 0.0
    # 构造 90 度转向：帧 5 之后向上转
    pos2 = pos.clone()
    pos2[0, 7] = torch.tensor([8.0, 2.0])  # 5->7 方向 (0,2)
    future2 = pos2[:, 5:6] + torch.linspace(0, 1, 12).unsqueeze(-1).repeat(1, 1, 2)
    s2 = build_sample(pos2, future2, valid, scene_id="t", track_id=0)
    # 朝向变化：2->5 方向 (1,0)=0rad；5->7 方向 (0,1)=pi/2；差 pi/2，除以 gap 2
    assert torch.isclose(s2["x_turn_rate"][0, 7], torch.tensor(torch.pi / 4), atol=1e-5)


def test_b1_collate_fields():
    ds = TrajImputeDataset(TRAJIMPUTE_ROOT, "ETH-M", "Easy", "test")
    items = [ds[i] for i in range(3)]
    batch = trajimpute_collate_fn(items)
    for k in ("x_velocity", "x_accel", "x_turn_rate", "x_motion_run"):
        assert k in batch
        assert torch.isfinite(batch[k]).all()


def test_b1_clean_history_neutral():
    """完整历史（Clean-direct）：B1 字段 = 原始运动学量，无退化。"""
    ds = TrajImputeDataset(TRAJIMPUTE_ROOT, "ETH-M", "Easy", "test",
                           zero_missing_only=True)
    s = ds[0]
    assert bool(s["x_valid_mask"].all())
    # 完整历史 velocity[0]=0（首帧无差分），其余 > 0（真实运动）
    assert float(s["x_velocity"][0, 0]) == 0.0
    assert bool((s["x_velocity"][0, 1:] >= 0).all())


@pytest.mark.skipif(not GPU, reason="需要 GPU")
def test_b1_model_forward():
    from src.model.model_forecast import ModelForecast
    ds = TrajImputeDataset(TRAJIMPUTE_ROOT, "ETH-M", "Easy", "test")
    batch = trajimpute_collate_fn([ds[i] for i in range(3)])
    batch = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in batch.items()}
    model = ModelForecast(num_modes=6, bimamba=True, use_motion_features=True).cuda()
    out = model(batch)
    assert torch.isfinite(out["new_y_hat"]).all()
    assert model.hist_embed_mlp[0].in_features == 8  # 4 base + 4 motion


@pytest.mark.skipif(not GPU, reason="需要 GPU")
def test_b1_model_switch_off_identical_to_m0():
    """use_motion_features=False 时输入维度与 M0 一致（4）。"""
    from src.model.model_forecast import ModelForecast
    m0 = ModelForecast(num_modes=6, use_motion_features=False)
    assert m0.hist_embed_mlp[0].in_features == 4
