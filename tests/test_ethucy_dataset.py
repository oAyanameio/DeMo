"""ETH/UCY Dataset 和 Collate 单元测试。"""

import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datamodule.ethucy_dataset import EthUcyDataset, ethucy_collate_fn
from src.datamodule.ethucy_utils import (
    load_ethucy_file,
    resample_scene,
    sliding_window_samples,
)


def _create_synthetic_data(num_frames=300, num_peds=5, seed=42):
    """创建合成数据并保存为 .pt 文件。"""
    import pandas as pd

    rng = np.random.RandomState(seed)
    rows = []
    for ped_id in range(num_peds):
        x = rng.randn() * 10
        y = rng.randn() * 10
        for frame in range(num_frames):
            x += rng.randn() * 0.5
            y += rng.randn() * 0.5
            rows.append(
                {"frame": frame, "ped_id": ped_id, "x": x, "y": y}
            )

    df = pd.DataFrame(rows)
    df = resample_scene(df, frame_stride=10)
    samples = sliding_window_samples(df, obs_len=8, pred_len=12)

    for s in samples:
        s["scene_id"] = "TEST"

    return samples


class TestEthUcyDataset:
    """测试 EthUcyDataset。"""

    def setup_method(self):
        """创建临时目录和合成数据。"""
        self.tmpdir = tempfile.mkdtemp()
        self.scene_dir = Path(self.tmpdir) / "TEST"
        self.scene_dir.mkdir()

        samples = _create_synthetic_data(num_frames=300, num_peds=5)
        for i, s in enumerate(samples):
            torch.save(s, self.scene_dir / f"{i:06d}.pt")

        self.dataset = EthUcyDataset(
            data_root=Path(self.tmpdir),
            scene_names=["TEST"],
            obs_len=8,
            pred_len=12,
        )

    def test_len(self):
        assert len(self.dataset) > 0

    def test_output_keys(self):
        sample = self.dataset[0]
        expected_keys = {
            "target",
            "target_diff",
            "target_vel_diff",
            "target_mask",
            "x_positions_diff",
            "x_positions",
            "x_attr",
            "x_centers",
            "x_angles",
            "x_velocity",
            "x_velocity_diff",
            "x_valid_mask",
            "origin",
            "theta",
            "scene_id",
            "track_id",
            "timestamp",
        }
        assert set(sample.keys()) == expected_keys

    def test_target_shape(self):
        sample = self.dataset[0]
        N = sample["x_positions"].size(0)
        assert sample["target"].shape == (N, 12, 2)
        assert sample["target_mask"].shape == (N, 12)

    def test_focal_at_index_zero(self):
        sample = self.dataset[0]
        # focal 的行人目标掩码全 True
        assert sample["target_mask"][0].all()
        # focal 的历史有效掩码全 True
        assert sample["x_valid_mask"][0].all()

    def test_focal_origin_at_zero(self):
        sample = self.dataset[0]
        # focal 最后一帧在局部坐标原点
        focal_last = sample["x_positions"][0, -1]
        assert torch.allclose(focal_last, torch.zeros(2), atol=1e-6)

    def test_x_attr_type(self):
        sample = self.dataset[0]
        # x_attr[..., 2] 应为 0 (pedestrian)
        assert (sample["x_attr"][:, 2] == 0).all()

    def test_no_lane_fields(self):
        sample = self.dataset[0]
        for lane_key in [
            "lane_positions",
            "lane_centers",
            "lane_angles",
            "lane_attr",
            "lane_valid_mask",
            "lane_key_valid_mask",
        ]:
            assert lane_key not in sample


class TestCollateFn:
    """测试 ethucy_collate_fn。"""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.scene_dir = Path(self.tmpdir) / "TEST"
        self.scene_dir.mkdir()

        samples = _create_synthetic_data(num_frames=300, num_peds=5)
        for i, s in enumerate(samples):
            torch.save(s, self.scene_dir / f"{i:06d}.pt")

        self.dataset = EthUcyDataset(
            data_root=Path(self.tmpdir),
            scene_names=["TEST"],
            obs_len=8,
            pred_len=12,
        )

    def test_batch_collate(self):
        loader = DataLoader(
            self.dataset,
            batch_size=4,
            collate_fn=ethucy_collate_fn,
        )
        batch = next(iter(loader))

        # 检查 batch 形状
        B = 4
        assert batch["target"].ndim == 4  # [B, N_pad, 12, 2]
        assert batch["target"].size(0) == B
        assert batch["target"].size(2) == 12
        assert batch["target"].size(3) == 2

        assert batch["x_valid_mask"].size(0) == B
        assert batch["x_positions_diff"].size(0) == B

        # x_key_valid_mask 应该存在
        assert "x_key_valid_mask" in batch
        assert batch["x_key_valid_mask"].shape[0] == B

        # 没有 lane 字段
        assert "lane_positions" not in batch
        assert "lane_key_valid_mask" not in batch

    def test_different_n_padding(self):
        """测试不同 N 的样本可以正确 padding。"""
        # 取 N 最大的和最小的样本
        samples = [self.dataset[i] for i in range(min(4, len(self.dataset)))]
        Ns = [s["x_positions"].size(0) for s in samples]

        if len(set(Ns)) > 1:
            batch = ethucy_collate_fn(samples)
            # 所有样本应该被 padding 到相同的 N
            N_padded = batch["x_positions"].size(1)
            assert N_padded == max(Ns)