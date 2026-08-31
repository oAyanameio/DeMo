import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.datamodule.ethucy_benchmark_dataset import (  # noqa: E402
    EthUcyBenchmarkDataset,
    compute_theta,
    ethucy_benchmark_collate_fn,
)


def make_sample(N=2, T=20):
    """focal 沿 +x 走，其他人在附近。"""
    pos = torch.zeros(N, T, 2)
    for t in range(T):
        pos[0, t] = torch.tensor([float(t), 0.0])          # focal: right
        if N > 1:
            pos[1, t] = torch.tensor([float(t), 1.0])      # neighbor
    return {
        "scene_id": "ETH",
        "fold": "ETH",
        "split": "test",
        "source_file": "eth/test/biwi_eth.txt",
        "sample_index": 0,
        "focal_id": 100,
        "agent_ids": torch.tensor([100] + ([200] if N > 1 else [])),
        "positions": pos,
        "valid_mask": torch.ones(N, T, dtype=torch.bool),
        "frame_ids": torch.arange(T),
        "frame_ids_raw": torch.arange(T) * 10,
    }


def process(sample):
    ds = EthUcyBenchmarkDataset.__new__(EthUcyBenchmarkDataset)
    ds.obs_len, ds.pred_len = 8, 12
    return ds.process(sample)


def test_focal_first_and_local_frame():
    out = process(make_sample(N=2))
    # focal 局部坐标：origin 在 (7,0)，theta=0 -> focal hist 末帧为 (0,0)
    assert torch.allclose(out["x_positions"][0, 7], torch.zeros(2), atol=1e-6)
    assert out["target_mask"][0].all()
    assert torch.isfinite(out["target"]).all()
    assert out["x_attr"].shape == (2, 3) and (out["x_attr"][..., 2] == 0).all()


def test_roundtrip_right_motion():
    sample = make_sample(N=1)
    out = process(sample)
    rot = torch.tensor([[torch.cos(out["theta"][0]), -torch.sin(out["theta"][0])],
                        [torch.sin(out["theta"][0]), torch.cos(out["theta"][0])]])
    recovered = (out["target"][0] @ rot.T) + out["origin"][0]
    err = (recovered - sample["positions"][0, 8:]).abs().max()
    assert err < 1e-5, err


def test_roundtrip_up_motion():
    sample = make_sample(N=1)
    sample["positions"][0] = torch.stack([torch.zeros(20), torch.arange(20.0)], dim=-1)
    out = process(sample)
    rot = torch.tensor([[torch.cos(out["theta"][0]), -torch.sin(out["theta"][0])],
                        [torch.sin(out["theta"][0]), torch.cos(out["theta"][0])]])
    recovered = (out["target"][0] @ rot.T) + out["origin"][0]
    assert (recovered - sample["positions"][0, 8:]).abs().max() < 1e-5


def test_roundtrip_near_zero_last_displacement():
    sample = make_sample(N=1)
    sample["positions"][0, 6] = sample["positions"][0, 5]  # 末两帧位移为 0
    out = process(sample)
    rot = torch.tensor([[torch.cos(out["theta"][0]), -torch.sin(out["theta"][0])],
                        [torch.sin(out["theta"][0]), torch.cos(out["theta"][0])]])
    recovered = (out["target"][0] @ rot.T) + out["origin"][0]
    assert (recovered - sample["positions"][0, 8:]).abs().max() < 1e-5


def test_degenerate_heading():
    pos = torch.zeros(1, 20, 2)
    pos[0, :, 0] = 7.0  # 完全静止 -> degenerate, theta=0
    sample = make_sample(N=1)
    sample["positions"] = pos
    out = process(sample)
    assert out["theta"].item() == 0.0
    assert out["degenerate_heading"] is True


def test_collate_different_N():
    b1 = process(make_sample(N=1))
    b2 = process(make_sample(N=2))
    batch = ethucy_benchmark_collate_fn([b1, b2])
    # pad_sequence over [N_i,T,2] gives [B, N_max, T, 2]
    assert batch["x_positions"].shape == (2, 2, 8, 2)
    assert batch["target"].shape == (2, 2, 12, 2)
    assert batch["x_key_valid_mask"][0, 0] and batch["x_key_valid_mask"][1, 0]


def test_collate_single_actor():
    b = process(make_sample(N=1))
    batch = ethucy_benchmark_collate_fn([b])
    assert batch["x_positions"].shape == (1, 1, 8, 2)
