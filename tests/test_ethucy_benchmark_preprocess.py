import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.datamodule.ethucy_benchmark_preprocess import build_dense  # noqa: E402


def make_rows(n_frames=25, peds=("1", "2")):
    """两个行人，ped1 全程在场，ped2 只在前 10 帧在场。"""
    rows = []
    for t in range(n_frames):
        rows.append((t * 10, 1, float(t), 0.0))  # raw frame 间隔 10
        if t < 10:
            rows.append((t * 10, 2, 0.0, float(t)))
    return rows


def test_build_dense_timestep_mapping_and_windows():
    samples, nframes, npeds = build_dense(make_rows(), 8, 12)
    assert nframes == 25 and npeds == 2
    # 窗口数：ped1 需要 20 帧完整 -> start 范围 [0, 5]
    assert len(samples) == 6
    s = samples[0]
    assert s["focal_id"] == 1
    # strict-complete: ped2 不在任何样本中（不满足 20 帧完整）
    assert s["agent_ids"].tolist() == [1]
    assert s["valid_mask"].all()
    assert s["positions"].shape == (1, 20, 2)
    assert s["frame_ids"].tolist() == list(range(20))
    # raw frame id 保留原始编号（间隔 10）
    assert s["frame_ids_raw"].tolist() == [t * 10 for t in range(20)]


def test_focal_requires_full_future():
    # ped2 前 10 帧在场：窗口 start=0 时 ped2 只有 10 帧有效，不能成为 focal
    samples, _, _ = build_dense(make_rows(), 8, 12)
    assert all(s["focal_id"] == 1 for s in samples)


def test_short_file_yields_no_samples():
    samples, nframes, _ = build_dense(make_rows(n_frames=15), 8, 12)
    assert samples == []
    assert nframes == 15


def test_duplicate_row_raises():
    rows = make_rows(n_frames=21)
    rows.append(rows[0])  # duplicate (frame,ped)
    with pytest.raises(ValueError, match="Duplicate"):
        build_dense(rows, 8, 12)


def test_all_complete_actors_included():
    rows = make_rows(n_frames=25)
    # ped3 全程在场
    for t in range(25):
        rows.append((t * 10, 3, 5.0, 5.0))
    samples, _, _ = build_dense(rows, 8, 12)
    s = samples[0]
    assert sorted(s["agent_ids"].tolist()) == [1, 3]
    assert s["positions"].shape == (2, 20, 2)
    assert s["valid_mask"].all()
    assert torch.isfinite(s["positions"]).all()
