"""逐样本懒缓存等价性门禁（2026-10-06）。

验证三点：
1. 缓存命中路径与未缓存路径逐张量 torch.equal（跨 Dataset 实例）；
2. 同一实例重复访问返回值逐位一致（缓存不自污染）；
3. TrajGapDataset 包装层（dict 浅拷贝改 scene_id）不污染底层缓存。

用法：PYTHONPATH=. python scripts/审计与校验/check_sample_cache_equivalence.py
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from src.datamodule.trajimpute_dataset import (  # noqa: E402
    TRAJIMPUTE_ROOT, TrajGapDataset, TrajImputeDataset,
)

TENSOR_KEYS_MAX = 12


def compare_samples(a, b, tag):
    assert set(a.keys()) == set(b.keys()), f"{tag}: key mismatch {set(a) ^ set(b)}"
    for k in a:
        va, vb = a[k], b[k]
        if torch.is_tensor(va):
            assert torch.is_tensor(vb), f"{tag}/{k}: type"
            assert va.dtype == vb.dtype, f"{tag}/{k}: dtype {va.dtype} vs {vb.dtype}"
            assert va.shape == vb.shape, f"{tag}/{k}: shape"
            assert torch.equal(va, vb), f"{tag}/{k}: values differ"
        else:
            assert va == vb, f"{tag}/{k}: scalar {va!r} vs {vb!r}"


def main():
    checks = passed = 0

    # ---- 1+2: TrajImputeDataset 跨实例与重复访问 ----
    for scene, diff, split, idxs in [
        ("ETH-M", "Easy", "train", [0, 1, 500, 5000, 29808]),
        ("ETH-M", "Hard", "val", [0, 17, 999]),
        ("ZARA2-M", "Hard", "test", [0, 3, 8888]),
    ]:
        ref = TrajImputeDataset(TRAJIMPUTE_ROOT, scene, diff, split)
        fresh = TrajImputeDataset(TRAJIMPUTE_ROOT, scene, diff, split)
        for i in idxs:
            a = ref[i]                    # 未缓存路径（ref 独立实例，各自首访）
            b1 = fresh[i]                 # 未缓存（首访，构建并存缓存）
            b2 = fresh[i]                 # 缓存命中
            compare_samples(a, b1, f"{scene}/{diff}/{split}[{i}] uncached-vs-uncached")
            compare_samples(a, b2, f"{scene}/{diff}/{split}[{i}] uncached-vs-cached")
            checks += 1
            passed += 1

    # ---- 3: TrajGapDataset 包装层不污染底层缓存 ----
    gap = TrajGapDataset(TRAJIMPUTE_ROOT, "ETH-M", "train")
    n = len(gap)
    probe = [0, 1, n // 2, n - 1]
    first = [dict(gap[i]) for i in probe]      # 首访（构建）
    second = [dict(gap[i]) for i in probe]     # 二访（缓存命中 + scene_id 改写后）
    for i, (a, b) in zip(probe, zip(first, second)):
        compare_samples(a, b, f"gap[{i}] wrap-pollution")
    # 底层 TrajImputeDataset 缓存原件的 scene_id 仍应是 Easy/Hard 原值
    for dataset in gap.datasets:
        assert len(dataset.samples) > 0
        cached = dataset._sample_cache[0]
        sid = cached["scene_id"]
        assert sid.startswith(f"ETH-M-{dataset.difficulty}-"), (
            f"底层缓存 scene_id 被包装层污染: {sid}"
        )
    checks += len(probe) + len(gap.datasets)
    passed += len(probe) + len(gap.datasets)

    print(f"[cache-equivalence] {passed}/{checks} checks passed")


if __name__ == "__main__":
    main()
