"""ETH/UCY benchmark smoke test（数据阶段 + 集成小训练阶段）。"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.datamodule.ethucy_benchmark_dataset import (  # noqa: E402
    EthUcyBenchmarkDataset,
    ethucy_benchmark_collate_fn,
)
from torch.utils.data import DataLoader  # noqa: E402


def stage_data(data_root, fold):
    ok = True
    for split in ("train", "val", "test"):
        ds = EthUcyBenchmarkDataset(data_root, fold, split)
        assert len(ds) > 0, f"{fold}/{split} empty"
        loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0,
                            collate_fn=ethucy_benchmark_collate_fn)
        batch = next(iter(loader))
        assert batch["x_positions"].shape[2] == 8, "obs len != 8"
        assert batch["target"].shape[-2] == 12, "pred len != 12"
        assert batch["target_mask"][:, 0].all(), "focal target_mask not all valid"
        for k in ("x_positions", "target", "x_velocity"):
            assert torch.isfinite(batch[k]).all(), f"NaN/Inf in {k}"
        assert not any(k.startswith("lane") or "lane" in k for k in batch), "lane fields present"
        print(f"  [ok] {fold}/{split}: {len(ds)} samples, batch shapes ok")
    print("STAGE DATA PASSED")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["data"], default="data")
    ap.add_argument("--data-root", default="data/ETHUCY_benchmark_v1")
    ap.add_argument("--fold", default="ETH")
    args = ap.parse_args()
    if args.stage == "data":
        stage_data(args.data_root, args.fold)
