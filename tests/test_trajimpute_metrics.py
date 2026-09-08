"""TrajImpute 直接预测评估器测试（任务书 §十二 16–18, 11）。"""

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.evaluation.trajimpute_direct import (  # noqa: E402
    evaluate_predictions, DirectEvaluator,
)
from src.datamodule.trajimpute_dataset import (  # noqa: E402
    TRAJIMPUTE_ROOT, TrajImputeDataset,
)

HAVE_RELEASE = Path(TRAJIMPUTE_ROOT).exists()
needs_release = pytest.mark.skipif(not HAVE_RELEASE, reason="release 未下载")


# ---------------------------------------------------------------- 16. 人工构造预测
def test_minade_minfde_on_manual_predictions():
    """K=3 人工预测：minADE/minFDE 语义正确。"""
    B, K, T = 1, 3, 4
    target = torch.zeros(B, T, 2)
    target[0, :, 0] = torch.tensor([0.0, 0.0, 1.0, 2.0])  # x 随 t
    pred = torch.zeros(B, K, T, 2)
    pred[0, 0] = target[0]                                  # 完美模式
    pred[0, 1] = target[0] + 1.0                            # 每帧偏移 (1,1)
    pred[0, 2] = target[0] + 3.0                            # 每帧偏移 (3,3)
    prob = torch.tensor([[0.2, 0.5, 0.3]])                  # 模式 1 概率最高
    res = evaluate_predictions(pred, prob, target)
    # minADE_K：mode0=0，mode1=sqrt(2)，mode2=3*sqrt(2) -> min=0
    assert torch.isclose(res["minADE_K"][0], torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(res["minFDE_K"][0], torch.tensor(0.0), atol=1e-6)
    # ADE@1 = 最高概率模式（mode1）= sqrt(2)
    assert torch.isclose(res["ADE@1"][0], torch.tensor(2 ** 0.5), atol=1e-6)
    assert torch.isclose(res["FDE@1"][0], torch.tensor(2 ** 0.5), atol=1e-6)
    # MR：K 条中 mode0 FDE=0 <= 2.0 -> 不 miss
    assert res["MR"][0] == 0.0


def test_mr_all_modes_miss():
    B, K, T = 1, 2, 4
    target = torch.zeros(B, T, 2)
    pred = torch.full((B, K, T, 2), 10.0)  # 全部模式终点距离 sqrt(200)>2
    prob = torch.ones(B, K)
    res = evaluate_predictions(pred, prob, target, miss_threshold=2.0)
    assert res["MR"][0] == 1.0


def test_min_over_same_k_set():
    """minADE_K 与 minFDE_K 用同一组 K 条预测（不允许各自挑最优）。"""
    B, K, T = 2, 4, 3
    torch.manual_seed(0)
    target = torch.randn(B, T, 2)
    pred = torch.randn(B, K, T, 2)
    prob = torch.randn(B, K)
    res = evaluate_predictions(pred, prob, target)
    d = torch.norm(pred - target.unsqueeze(1), dim=-1)
    ade = d.mean(-1)
    fde = d[..., -1]
    assert torch.allclose(res["minADE_K"], ade.min(-1).values, atol=1e-6)
    assert torch.allclose(res["minFDE_K"], fde.min(-1).values, atol=1e-6)


def test_evaluator_grouping_manual():
    ev = DirectEvaluator()
    target = torch.zeros(1, 4, 2)
    pred = torch.zeros(1, 2, 4, 2)
    prob = torch.ones(1, 2)
    ev.update(pred, prob, target, "ETH-M", "Easy", "test", 0)
    pred2 = pred.clone()
    pred2[..., -1] += 10.0
    ev.update(pred2, prob, target, "ETH-M", "Easy", "test", 4)
    out = ev.compute()
    assert out["by_group"]["ETH-M/Easy/test/missing=0"]["n"] == 1
    assert out["by_group"]["ETH-M/Easy/test/missing=4"]["minFDE_K"] > 0
    assert out["overall"]["n"] == 2
    assert out["aggregation"] == "micro-average over all focal samples"


# ---------------------------------------------------------------- 17/18. Easy/Hard 分组统计
@needs_release
def test_easy_test_missing_count_groups_0_to_4():
    ds = TrajImputeDataset(TRAJIMPUTE_ROOT, "ETH-M", "Easy", "test")
    counts = ds.missing_counts.tolist()
    # 5 块：缺失数恰为 0/1/2/3/4（每块 = 全部测试轨迹，行数 N/5）
    import collections
    hist = collections.Counter(counts)
    assert set(hist.keys()) == {0, 1, 2, 3, 4}
    n_per_block = len(counts) // 5
    for k in range(5):
        assert hist[k] == n_per_block, f"missing={k} block size {hist[k]} != {n_per_block}"
    # 场景组数应为原始组数×5（seq_start_end 修复后覆盖全部行）
    total_rows = sum(e - s for s, e in ds.seq_start_end)
    assert total_rows == len(counts)


@needs_release
def test_hard_test_missing_count_groups_4_to_7():
    ds = TrajImputeDataset(TRAJIMPUTE_ROOT, "ETH-M", "Hard", "test")
    counts = ds.missing_counts.tolist()
    import collections
    hist = collections.Counter(counts)
    assert set(hist.keys()) == {4, 5, 6, 7}
    n_per_block = len(counts) // 4
    for k in range(4, 8):
        assert hist[k] == n_per_block, f"missing={k} block size {hist[k]} != {n_per_block}"
    total_rows = sum(e - s for s, e in ds.seq_start_end)
    assert total_rows == len(counts)


@needs_release
def test_evaluator_real_data_grouping_end_to_end():
    """用未训练随机模型 + 2 个 batch 的分组评估（管线端到端，结果无意义）。"""
    if not torch.cuda.is_available():
        pytest.skip("需要 GPU")
    from src.model.model_forecast import ModelForecast
    from src.datamodule.trajimpute_dataset import trajimpute_collate_fn
    ds = TrajImputeDataset(TRAJIMPUTE_ROOT, "ETH-M", "Hard", "test")
    model = ModelForecast(num_modes=6, bimamba=True).cuda().eval()
    ev = DirectEvaluator()
    with torch.no_grad():
        items = [ds[i] for i in range(0, 12)]
        # 分 3 批 collate
        for j in range(0, 12, 4):
            batch = trajimpute_collate_fn(items[j:j + 4])
            batch = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in batch.items()}
            out = model(batch)
            pred = out["new_y_hat"][..., :2].float().cpu()
            prob = out["new_pi"].float().cpu()
            tgt = batch["target"][:, 0].cpu()
            for i in range(pred.shape[0]):
                ev.update(pred[i:i+1], prob[i:i+1], tgt[i:i+1],
                          "ETH-M", "Hard", "test", int(batch["missing_count"][i]))
    res = ev.compute()
    assert res["overall"]["n"] == 12
    assert len(res["by_group"]) >= 1
    for g, entry in res["by_group"].items():
        assert entry["n"] > 0
        assert all(0 <= v < 1e6 for v in entry.values() if isinstance(v, float))
