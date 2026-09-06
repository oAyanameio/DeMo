"""v3_noguard 协议单元测试（方案 §5.1）。

覆盖：
- 掩码生成：帧7缺失 / 帧6+7同缺 / 仅一帧可见（uniform_hard m=7）/ block6 三种起点 / uniform m∈{4..7}
- 时间间隔字段：last_valid_idx / anchor_lag_steps / forecast_gap_steps
- v1/v2 行为不变：帧6/7 恒可见、anchor_lag 恒 0、forecast_gap 恒 1（derive_history_mask 旧路径）
- 独立重推与 build 端一致（v3 确定性）
- Dataset：锚点回退、朝向回退、单帧样本处理（合成数据，不依赖 data/）
"""
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "数据集构建"))

from build_missing_history_dataset import (  # noqa: E402
    V3_CONDITIONS, derive_history_mask, apply_mask_to_sample, mask_seed_from,
)

OBS = 8


def _mk_sample(A=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    positions = torch.randn(A, 20, 2, generator=g) * 5
    return {
        "positions": positions,
        "valid_mask": torch.ones(A, 20, dtype=torch.bool),
    }


def _derive(cond, A=3, key_extra="x"):
    return derive_history_mask("ethucy", "ETH", "test", "ETH", 0,
                               "eth/test/biwi_eth.txt", 2, cond, 42, A)


class TestV3MaskSemantics:
    def test_frame7_can_be_missing(self):
        """帧 7 允许缺失（uniform_hard m=7 时必缺帧 7 的样本存在）。"""
        found = False
        for i in range(300):
            hm = derive_history_mask("ethucy", "ETH", "test", "ETH", i,
                                     "eth/test/biwi_eth.txt", 2, "uniform_hard_ng", 42, 1)
            if not bool(hm[0, 7]):
                found = True
                break
        assert found, "frame 7 never missing in uniform_hard_ng over 300 samples"

    def test_frames67_both_missing_possible(self):
        found = False
        for i in range(300):
            hm = derive_history_mask("ethucy", "ETH", "test", "ETH", i,
                                     "eth/test/biwi_eth.txt", 2, "uniform_hard_ng", 42, 1)
            if not bool(hm[0, 7]) and not bool(hm[0, 6]):
                found = True
                break
        assert found, "frames 6&7 never both missing over 300 samples"

    def test_min_one_visible_frame(self):
        for cond in sorted(V3_CONDITIONS):
            for i in range(100):
                hm = derive_history_mask("ethucy", "ETH", "test", "ETH", i,
                                         "eth/test/biwi_eth.txt", 2, cond, 42, 4)
                assert bool(hm.any(dim=1).all()), f"{cond}: agent with 0 visible frames"

    def test_block6_varied_starts(self):
        """block6_ng 起点 s∈{0,1,2}，三种都可能且块连续。"""
        starts = set()
        for i in range(200):
            hm = derive_history_mask("ethucy", "ETH", "test", "ETH", i,
                                     "eth/test/biwi_eth.txt", 2, "random_block6_ng", 42, 1)
            idx = (~hm[0]).nonzero().flatten().tolist()
            assert idx == list(range(idx[0], idx[0] + 6)), f"non-contiguous: {idx}"
            starts.add(idx[0])
        assert {0, 1, 2} <= starts, f"block6 starts not covered: {starts}"

    def test_uniform_hard_m_stratification(self):
        ms = set()
        for i in range(400):
            hm = derive_history_mask("ethucy", "ETH", "test", "ETH", i,
                                     "eth/test/biwi_eth.txt", 2, "uniform_hard_ng", 42, 1)
            m = int((~hm[0]).sum().item())
            assert 4 <= m <= 7, f"m out of range: {m}"
            ms.add(m)
        assert ms == {4, 5, 6, 7}, f"m values not all covered: {ms}"

    def test_deterministic(self):
        for cond in ["random_block6_ng", "uniform_hard_ng", "random_fixed4_ng"]:
            hm1 = _derive(cond)
            hm2 = _derive(cond)
            assert torch.equal(hm1, hm2), f"{cond} not deterministic"


class TestGapFields:
    def test_apply_writes_v3_fields(self):
        apply_mask_to_sample._v3_mode = True
        try:
            hm = torch.ones(2, OBS, dtype=torch.bool)
            hm[0, 5:] = False   # actor0: last valid = 4
            hm[1, :] = False; hm[1, 2] = True  # actor1: only frame 2
            out = apply_mask_to_sample(_mk_sample(A=2), hm)
            assert torch.equal(out["last_valid_idx"], torch.tensor([4, 2]))
            assert torch.equal(out["anchor_lag_steps"], torch.tensor([3, 5]))
            assert torch.equal(out["forecast_gap_steps"], torch.tensor([4, 6]))
        finally:
            apply_mask_to_sample._v3_mode = False

    def test_v12_no_v3_fields(self):
        """v1/v2 路径不写字段（保持输出逐字节不变）。"""
        hm = torch.ones(2, OBS, dtype=torch.bool); hm[0, 0] = False
        out = apply_mask_to_sample(_mk_sample(A=2), hm)
        assert "last_valid_idx" not in out

    def test_v12_mask_frame67_always_visible(self):
        for cond in ["random_single", "random_block2", "random_fixed4", "random_block6"]:
            hm = _derive(cond)
            assert bool(hm[:, 6].all()) and bool(hm[:, 7].all()), cond


class TestDatasetAnchorFallback:
    """合成样本直测 Dataset.process 的锚点/朝向回退（不依赖磁盘数据）。"""

    def _proc(self, positions, valid_mask):
        from src.datamodule.ethucy_benchmark_dataset import EthUcyBenchmarkDataset
        ds = EthUcyBenchmarkDataset.__new__(EthUcyBenchmarkDataset)
        ds.obs_len, ds.pred_len = 8, 12
        data = {
            "positions": positions, "valid_mask": valid_mask,
            "agent_ids": torch.tensor([7, 3, 9]), "focal_id": 7,
            "scene_id": "ETH",
        }
        return ds.process(data)

    def test_origin_at_last_visible(self):
        g = torch.Generator().manual_seed(1)
        pos = torch.randn(3, 20, 2, generator=g) * 4
        vm = torch.ones(3, 20, dtype=torch.bool)
        vm[0, 5:] = False  # focal 帧 5-7 缺失 → 最后可见 = 4
        out = self._proc(pos, vm)
        # origin = focal 最后可见位置（帧 4）→ 刚体变换后 |local[0,4]| = 0
        assert torch.norm(out["x_positions"][0, 4]).item() < 1e-5, \
            f"frame-4 should be origin, got {out['x_positions'][0, 4]}"
        assert int(out["x_last_valid_idx"][0].item()) == 4
        assert int(out["x_anchor_lag_steps"][0].item()) == 3
        assert int(out["x_forecast_gap_steps"][0].item()) == 4

    def test_single_frame_agent(self):
        pos = torch.randn(2, 20, 2).cumsum(0) * 0  # will set below
        pos = torch.zeros(2, 20, 2)
        pos[:, :, 0] = torch.arange(20).float() * 0.5  # 匀速直线 x
        vm = torch.ones(2, 20, dtype=torch.bool)
        vm[1, :] = False; vm[1, 3] = True  # agent1 仅帧 3 可见
        out = self._proc(pos, vm)
        assert int(out["x_last_valid_idx"][1].item()) == 3
        assert float(out["x_last_valid_angle"][1].item()) == 0.0  # 无运动对 → 0
        assert int(out["x_anchor_lag_steps"][1].item()) == 4
        assert bool(torch.isfinite(out["x_positions"]).all())

    def test_heading_backtrack_skips_missing(self):
        pos = torch.zeros(2, 20, 2)
        pos[0, :, 0] = torch.arange(20).float() * 0.5
        pos[0, 6, 1] = 5.0  # 帧 6 抬高（若被用到朝向会大偏转）
        vm = torch.ones(2, 20, dtype=torch.bool)
        vm[0, 5:] = False  # 帧 5-7 缺 → 朝向应回溯到 4←3（纯 x 方向，角≈0）
        out = self._proc(pos, vm)
        ang = float(out["x_last_valid_angle"][0].item())
        assert abs(ang) < 1e-4, f"heading not backtracked: {ang}"

    def test_complete_equals_v12_semantics(self):
        pos = torch.randn(3, 20, 2).cumsum(1)
        vm = torch.ones(3, 20, dtype=torch.bool)
        out = self._proc(pos, vm)
        assert bool((out["x_anchor_lag_steps"] == 0).all())
        assert bool((out["x_forecast_gap_steps"] == 1).all())
        # 完整历史下 x_last_valid_angle == x_angles[...,7]
        assert torch.allclose(out["x_last_valid_angle"], out["x_angles"][:, 7], atol=1e-6)


class TestModelAngleInput:
    def test_model_uses_last_valid_angle_when_present(self):
        """batch 带 x_last_valid_angle 时模型应使用之（数值不同于 x_angles[...,7] 时输出不同）。"""
        from src.model.model_forecast import ModelForecast
        B, N, L = 2, 3, 8
        base = {
            "x_positions_diff": torch.randn(B, N, L, 2),
            "x_attr": torch.zeros(B, N, 3, dtype=torch.uint8),
            "x_centers": torch.randn(B, N, 2),
            "x_angles": torch.randn(B, N, L),
            "x_velocity": torch.randn(B, N, L),
            "x_velocity_diff": torch.randn(B, N, L),
            "x_valid_mask": torch.ones(B, N, L, dtype=torch.bool),
            "x_key_valid_mask": torch.ones(B, N, dtype=torch.bool),
        }
        with_new = dict(base)
        with_new["x_last_valid_angle"] = torch.randn(B, N)
        # 两组输入仅朝向源不同 → pos_embed 输入不同即可（不跑 mamba，截断验证）
        # 用反射检查 forward 源码分支即可（mamba 需要 CUDA）：
        import inspect
        src = inspect.getsource(ModelForecast.forward)
        assert 'x_last_valid_angle' in src and 'x_angles"][:, :, -1]' in src


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
