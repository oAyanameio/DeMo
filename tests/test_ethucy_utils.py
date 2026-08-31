"""ETH/UCY 数据解析器单元测试。"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.datamodule.ethucy_utils import (
    DEFAULT_FRAME_STRIDE,
    DEFAULT_OBS_LEN,
    DEFAULT_PRED_LEN,
    compute_focal_rotation,
    load_ethucy_file,
    normalize_scene_name,
    resample_scene,
    sliding_window_samples,
    transform_to_local,
)


def _make_sample_df(num_frames=100, num_peds=5, seed=42):
    """创建合成 ETH/UCY 数据，确保每个行人都有完整帧序列。"""
    rng = np.random.RandomState(seed)
    rows = []
    for ped_id in range(num_peds):
        x = rng.randn() * 10
        y = rng.randn() * 10
        for frame in range(num_frames):
            x += rng.randn() * 0.5
            y += rng.randn() * 0.5
            rows.append({"frame": frame, "ped_id": ped_id, "x": x, "y": y})
    return pd.DataFrame(rows)


def _make_txt_file(df: pd.DataFrame, path: Path):
    """将 DataFrame 写入 ETH/UCY 格式 .txt 文件。"""
    with open(path, "w") as f:
        for _, row in df.iterrows():
            f.write(f"{int(row['frame'])}\t{int(row['ped_id'])}\t{row['x']:.6f}\t{row['y']:.6f}\n")


class TestLoadEthUcyFile:
    """测试 load_ethucy_file。"""

    def test_basic_parsing(self):
        df = _make_sample_df(num_frames=50, num_peds=3)
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
            _make_txt_file(df, Path(f.name))
            tmp_path = f.name

        loaded = load_ethucy_file(Path(tmp_path))
        Path(tmp_path).unlink()

        assert set(loaded.columns) == {"frame", "ped_id", "x", "y"}
        assert loaded["frame"].dtype in (np.int64, np.int32)
        assert loaded["ped_id"].dtype in (np.int64, np.int32)
        assert loaded["x"].dtype == np.float64
        assert loaded["y"].dtype == np.float64
        assert loaded["frame"].is_monotonic_increasing or loaded["frame"].is_monotonic_decreasing

    def test_skip_header_lines(self):
        content = "frame_id pedestrian_id pos_x pos_y\n1 1 0.5 0.3\n2 1 0.6 0.4\n"
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
            f.write(content)
            tmp_path = f.name

        loaded = load_ethucy_file(Path(tmp_path))
        Path(tmp_path).unlink()

        assert len(loaded) == 2
        assert loaded["frame"].iloc[0] == 1
        assert loaded["ped_id"].iloc[0] == 1

    def test_comma_separated(self):
        content = "1,1,0.5,0.3\n2,1,0.6,0.4\n"
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
            f.write(content)
            tmp_path = f.name

        loaded = load_ethucy_file(Path(tmp_path))
        Path(tmp_path).unlink()

        assert len(loaded) == 2

    def test_empty_file_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
            f.write("")
            tmp_path = f.name

        with pytest.raises(ValueError):
            load_ethucy_file(Path(tmp_path))
        Path(tmp_path).unlink()


class TestNormalizeSceneName:
    """测试 normalize_scene_name。"""

    def test_standard_names(self):
        assert normalize_scene_name(Path("eth.txt")) == "ETH"
        assert normalize_scene_name(Path("hotel.txt")) == "HOTEL"
        assert normalize_scene_name(Path("univ.txt")) == "UNIV"
        assert normalize_scene_name(Path("zara1.txt")) == "ZARA1"
        assert normalize_scene_name(Path("zara2.txt")) == "ZARA2"

    def test_variants(self):
        assert normalize_scene_name(Path("biwi_hotel.txt")) == "HOTEL"
        assert normalize_scene_name(Path("students003.txt")) == "UNIV"
        assert normalize_scene_name(Path("crowds_zara01.txt")) == "ZARA1"


class TestResampleScene:
    """测试 resample_scene。"""

    def test_stride_downsampling(self):
        df = _make_sample_df(num_frames=100, num_peds=3)
        resampled = resample_scene(df, frame_stride=10)

        # 帧数应该减少
        assert resampled["frame"].nunique() < df["frame"].nunique()

        # 新 frame 从 0 开始连续
        frames = sorted(resampled["frame"].unique())
        assert frames[0] == 0
        assert frames == list(range(len(frames)))


class TestSlidingWindow:
    """测试 sliding_window_samples。"""

    def test_window_count(self):
        df = _make_sample_df(num_frames=300, num_peds=3)
        df = resample_scene(df, frame_stride=10)
        samples = sliding_window_samples(df, obs_len=8, pred_len=12)
        # 至少有一些样本
        assert len(samples) > 0

    def test_sample_shape(self):
        df = _make_sample_df(num_frames=100, num_peds=5)
        df = resample_scene(df, frame_stride=10)
        samples = sliding_window_samples(df, obs_len=8, pred_len=12)

        for s in samples:
            pos = s["positions"]
            valid = s["valid_mask"]
            assert pos.ndim == 3  # [N, 20, 2]
            assert pos.shape[1] == 20  # 8+12
            assert pos.shape[2] == 2
            assert valid.shape == (pos.shape[0], 20)
            assert valid.dtype == torch.bool

    def test_focal_has_full_history(self):
        df = _make_sample_df(num_frames=100, num_peds=5)
        df = resample_scene(df, frame_stride=10)
        samples = sliding_window_samples(df, obs_len=8, pred_len=12)

        for s in samples:
            focal_id = s["focal_id"]
            focal_idx = (s["agent_ids"] == focal_id).nonzero(as_tuple=True)[0][0]
            # focal 的历史全部有效
            assert s["valid_mask"][focal_idx, :8].all()


class TestCoordinateTransform:
    """测试坐标变换。"""

    def test_local_transform_origin_at_zero(self):
        positions = torch.randn(3, 8, 2)
        valid_mask = torch.ones(3, 8, dtype=torch.bool)
        origin = positions[0, 7].clone()  # 最后一帧
        theta = torch.tensor(0.0)

        local = transform_to_local(positions, valid_mask, origin, theta)
        # focal 最后一帧应在原点
        assert torch.allclose(local[0, 7], torch.zeros(2), atol=1e-6)

    def test_small_displacement_rotation_zero(self):
        pos_prev = torch.tensor([0.0, 0.0])
        pos_last = torch.tensor([1e-6, 1e-6])
        theta = compute_focal_rotation(pos_last, pos_prev)
        assert theta.item() == 0.0

    def test_rotation_angle(self):
        pos_prev = torch.tensor([0.0, 0.0])
        pos_last = torch.tensor([1.0, 0.0])
        theta = compute_focal_rotation(pos_last, pos_prev)
        assert abs(theta.item()) < 1e-6  # 向右移动，角度为 0

        pos_last = torch.tensor([0.0, 1.0])
        theta = compute_focal_rotation(pos_last, pos_prev)
        assert abs(theta.item() - torch.pi / 2) < 1e-6  # 向上移动，角度为 pi/2