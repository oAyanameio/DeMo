"""B1 运动特征跨实现一致性测试（ethucy_benchmark vs trajimpute_dataset）。

2026-09-08 统一后两套 dataset 的 x_velocity/x_velocity_diff/x_turn_rate 均为
跨缺口语义（相邻有效观测、按步数归一化）。本测试锁定：
1. 相同输入（位置+valid 掩码）下两套实现数值一致；
2. 完整历史下跨缺口版与"朴素相邻帧"版数值一致（回归保障：Clean-direct
   上 B1 与旧实现等价，既有结论不受统一影响）；
3. 缺失场景下的预期值（跨缺口归一化）正确。
"""

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.datamodule.b1_motion_features import build_b1_motion_features  # noqa: E402
from src.datamodule.trajimpute_dataset import build_gap_aware_motion  # noqa: E402


def _make(A=2, T=8):
    """帧 0,2,5,7 有效的世界坐标 + 一个完整 actor。"""
    pos = torch.zeros(A, T, 2)
    pos[0, 0] = torch.tensor([0.0, 0.0])
    pos[0, 2] = torch.tensor([2.0, 0.0])
    pos[0, 5] = torch.tensor([8.0, 0.0])
    pos[0, 7] = torch.tensor([10.0, 0.0])
    valid = torch.zeros(A, T, dtype=torch.bool)
    valid[0, [0, 2, 5, 7]] = True
    pos[1] = torch.linspace(0, 7, T).unsqueeze(-1).repeat(1, 2)
    valid[1] = True
    return pos, valid


def test_cross_implementation_velocity_consistency():
    pos, valid = _make()
    # trajimpute 实现
    diff_t, vel_t, vdiff_t = build_gap_aware_motion(pos, valid)
    # ethucy 实现
    vel_e, vdiff_e, turn_e = build_b1_motion_features(pos, valid)
    assert torch.allclose(vel_t, vel_e, atol=1e-6), "x_velocity 两实现不一致"
    assert torch.allclose(vdiff_t, vdiff_e, atol=1e-6), "x_velocity_diff 两实现不一致"


def test_complete_history_equivalence_to_naive():
    """完整历史：跨缺口版 == 朴素相邻帧版（Clean-direct 回归保障）。"""
    torch.manual_seed(0)
    pos = torch.randn(3, 8, 2).cumsum(0)  # 平滑轨迹
    valid = torch.ones(3, 8, dtype=torch.bool)
    vel_e, vdiff_e, turn_e = build_b1_motion_features(pos, valid)
    # 朴素版
    naive_vel = torch.zeros(3, 8)
    naive_vel[:, 1:] = torch.norm(pos[:, 1:] - pos[:, :-1], dim=-1)
    naive_vd = torch.zeros(3, 8)
    naive_vd[:, 2:] = naive_vel[:, 2:] - naive_vel[:, 1:-1]
    assert torch.allclose(vel_e, naive_vel, atol=1e-5)
    assert torch.allclose(vdiff_e, naive_vd, atol=1e-5)


def test_gap_aware_values():
    pos, valid = _make()
    vel_e, vdiff_e, turn_e = build_b1_motion_features(pos, valid)
    # 帧2: |(2,0)|/2=1.0；帧5: |(6,0)|/3=2.0；帧7: |(2,0)|/2=1.0
    assert torch.isclose(vel_e[0, 2], torch.tensor(1.0))
    assert torch.isclose(vel_e[0, 5], torch.tensor(2.0))
    assert torch.isclose(vel_e[0, 7], torch.tensor(1.0))
    # 速度差：帧5=2-1=+1；帧7=1-2=-1
    assert torch.isclose(vdiff_e[0, 5], torch.tensor(1.0))
    assert torch.isclose(vdiff_e[0, 7], torch.tensor(-1.0))
    # 直线运动 turn_rate=0
    assert float(turn_e[0].abs().sum()) == 0.0


def test_turn_rate_gap_normalized():
    """跨缺口转向：2->5 方向 0rad，5->7 转向 pi/2，除以 gap=2 -> pi/4。"""
    pos, valid = _make()
    pos2 = pos.clone()
    pos2[0, 7] = torch.tensor([8.0, 2.0])
    _, _, turn_e = build_b1_motion_features(pos2, valid)
    assert torch.isclose(turn_e[0, 7], torch.tensor(torch.pi / 4), atol=1e-5)
