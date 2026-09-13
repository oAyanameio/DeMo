"""任务2：缺失特征接入 ETH/UCY benchmark 与 SDD missing 数据管线（方案 §2.2）。

合成样本测试（不依赖 data/ 目录）：
- Dataset 输出 5 个新字段及其中性/协议值
- collate 按 actor 维 padding；padding actor 新字段全 0 且 x_key_valid_mask=False
- 真实 actor 在 v1/v2 协议下 x_gap_steps[..., 6:8] == 0
- 新字段无 NaN/Inf
"""
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.datamodule.ethucy_benchmark_dataset import (  # noqa: E402
    EthUcyBenchmarkDataset,
    ethucy_benchmark_collate_fn,
)
from src.datamodule.sdd_missing_dataset import SddMissingDataset  # noqa: E402
from src.datamodule.moflow_ethucy_dataset import moflow_ethucy_collate_fn  # noqa: E402

NEW_TIME_KEYS = ("x_gap_steps", "x_prev_valid_gap", "x_motion_valid", "x_motion_run")
NEW_SUMMARY_KEY = "x_missing_summary"


def make_ethucy_sample(N=2, T=20, hist_mask=None):
    """focal 沿 +x 走；hist_mask 只作用于 focal 历史 [8]。"""
    pos = torch.zeros(N, T, 2)
    for t in range(T):
        pos[0, t] = torch.tensor([float(t), 0.0])
        if N > 1:
            pos[1, t] = torch.tensor([float(t), 1.0])
    valid = torch.ones(N, T, dtype=torch.bool)
    if hist_mask is not None:
        valid[0, :8] = hist_mask.bool()
    return {
        "scene_id": "ETH",
        "fold": "ETH",
        "split": "test",
        "source_file": "eth/test/biwi_eth.txt",
        "sample_index": 0,
        "focal_id": 100,
        "agent_ids": torch.tensor([100] + [200 + i for i in range(N - 1)]),
        "positions": pos,
        "valid_mask": valid,
        "frame_ids": torch.arange(T),
        "frame_ids_raw": torch.arange(T) * 10,
    }


def process_ethucy(sample):
    ds = EthUcyBenchmarkDataset.__new__(EthUcyBenchmarkDataset)
    ds.obs_len, ds.pred_len = 8, 12
    return ds.process(sample)


def make_sdd_item(hist_mask):
    """SddMissingDataset.process 的合成输入：单 actor [1,20,2]。"""
    pos = torch.zeros(1, 20, 2)
    for t in range(20):
        pos[0, t] = torch.tensor([float(t), 0.5 * t])
    valid = torch.ones(1, 20, dtype=torch.bool)
    valid[0, :8] = hist_mask.bool()
    ds = SddMissingDataset.__new__(SddMissingDataset)
    ds.obs_len, ds.pred_len = 8, 12
    return ds.process(pos, valid)


BLOCK6 = torch.tensor([False] * 6 + [True, True])


class TestEthUcyBenchmarkNewFields:
    def test_complete_history_neutral(self):
        out = process_ethucy(make_ethucy_sample())
        for k in NEW_TIME_KEYS + (NEW_SUMMARY_KEY,):
            assert k in out, k
        assert torch.equal(out["x_gap_steps"][0], torch.zeros(8))
        assert torch.equal(out["x_motion_valid"][0], torch.tensor([0., 1., 1., 1., 1., 1., 1., 1.]))
        assert torch.allclose(out["x_missing_summary"][0],
                              torch.tensor([0., 0., 0., 1., 1., 0.]))
        # 邻居全可见 -> 也中性
        assert torch.allclose(out["x_missing_summary"][1],
                              torch.tensor([0., 0., 0., 1., 1., 0.]))

    def test_block6_focal_features(self):
        out = process_ethucy(make_ethucy_sample(hist_mask=BLOCK6))
        assert torch.equal(out["x_gap_steps"][0],
                           torch.tensor([1., 2., 3., 4., 5., 6., 0., 0.]))
        assert torch.equal(out["x_motion_valid"][0],
                           torch.tensor([0., 0., 0., 0., 0., 0., 0., 1.]))
        assert torch.allclose(
            out["x_missing_summary"][0],
            torch.tensor([0.75, 0.75, 0.75, 1 / 7, 1 / 7, 21 / 64]),
        )

    def test_frames_67_gap_zero_for_real_actors(self):
        # v1/v2：帧 6/7 恒可见；即使前面有缺口，6/7 处 gap 必为 0
        mask = torch.tensor([False, True, False, True, False, True, True, True])
        out = process_ethucy(make_ethucy_sample(hist_mask=mask))
        for i in range(out["x_gap_steps"].size(0)):
            assert torch.equal(out["x_gap_steps"][i, 6:8], torch.zeros(2))

    def test_new_fields_finite(self):
        for mask in (BLOCK6, torch.ones(8, dtype=torch.bool),
                     torch.tensor([True, False, True, False, True, False, True, True])):
            out = process_ethucy(make_ethucy_sample(hist_mask=mask))
            for k in NEW_TIME_KEYS + (NEW_SUMMARY_KEY,):
                assert torch.isfinite(out[k]).all(), (mask, k)


class TestEthUcyBenchmarkCollate:
    def test_padding_actors_have_zero_summary_and_excluded(self):
        # 样本 0：2 actors；样本 1：5 actors。collate 后样本 0 的 actor 2..4 为 padding
        s0 = process_ethucy(make_ethucy_sample(N=2, hist_mask=BLOCK6))
        s1 = process_ethucy(make_ethucy_sample(N=5))
        batch = ethucy_benchmark_collate_fn([s0, s1])
        for k in NEW_TIME_KEYS:
            assert batch[k].shape == (2, 5, 8), k
            # padding 区（b=0, actor 2..4）全 0
            assert torch.equal(batch[k][0, 2:], torch.zeros(3, 8)), k
        assert batch[NEW_SUMMARY_KEY].shape == (2, 5, 6)
        assert torch.equal(batch[NEW_SUMMARY_KEY][0, 2:], torch.zeros(3, 6))
        # padding actor 被 key_valid 排除；样本 1 的 5 个真实 actor 全有效
        assert not batch["x_key_valid_mask"][0, 2:].any()
        assert batch["x_key_valid_mask"][1].all()
        assert batch["x_key_valid_mask"][:, 0].all()  # focal 恒有效

    def test_padded_batch_all_real_actors_finite(self):
        items = [process_ethucy(make_ethucy_sample(N=2 + i, hist_mask=BLOCK6)) for i in range(3)]
        batch = ethucy_benchmark_collate_fn(items)
        for k in NEW_TIME_KEYS + (NEW_SUMMARY_KEY,):
            assert torch.isfinite(batch[k]).all(), k


class TestSddMissingNewFields:
    def test_complete_neutral(self):
        item = make_sdd_item(torch.ones(8, dtype=torch.bool))
        for k in NEW_TIME_KEYS + (NEW_SUMMARY_KEY,):
            assert k in item, k
        assert torch.allclose(item["x_missing_summary"][0],
                              torch.tensor([0., 0., 0., 1., 1., 0.]))

    def test_block6_focal(self):
        item = make_sdd_item(BLOCK6)
        assert torch.equal(item["x_gap_steps"][0],
                           torch.tensor([1., 2., 3., 4., 5., 6., 0., 0.]))
        assert torch.allclose(
            item["x_missing_summary"][0],
            torch.tensor([0.75, 0.75, 0.75, 1 / 7, 1 / 7, 21 / 64]),
        )

    def test_collate_keeps_single_actor_dim(self):
        batch = moflow_ethucy_collate_fn([
            make_sdd_item(BLOCK6),
            make_sdd_item(torch.ones(8, dtype=torch.bool)),
        ])
        for k in NEW_TIME_KEYS:
            assert batch[k].shape == (2, 1, 8), k
        assert batch[NEW_SUMMARY_KEY].shape == (2, 1, 6)
        assert torch.isfinite(batch[NEW_SUMMARY_KEY]).all()

    def test_collate_semantics_match_ethucy(self):
        # 同掩码下 SDD 与 ETH/UCY 新字段语义一致（同 build_missing_features）
        sdd = make_sdd_item(BLOCK6)
        eth = process_ethucy(make_ethucy_sample(hist_mask=BLOCK6))
        for k in NEW_TIME_KEYS + (NEW_SUMMARY_KEY,):
            assert torch.allclose(sdd[k][0], eth[k][0]), k
