import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.datamodule.ethucy_benchmark_dataset import (  # noqa: E402
    EthUcyBenchmarkDataset,
    ethucy_benchmark_collate_fn,
)


def load_runner():
    path = REPO / "scripts/训练与评估/run_trajimpute_experiments.py"
    spec = importlib.util.spec_from_file_location("trajimpute_experiments", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_summary():
    path = REPO / "scripts/结果分析/summarize_trajimpute.py"
    spec = importlib.util.spec_from_file_location("summarize_trajimpute", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_clean_protocol_uses_complete_ethucy_source():
    runner = load_runner()
    protocol = runner.PROTOCOLS["clean-direct"]

    assert protocol["dataset"] == "ethucy"
    assert protocol["data_root"] == "data/ETHUCY_benchmark_v1"
    assert protocol["zero_missing_only"] is False


def test_complete_ethucy_samples_expose_b1_motion_features():
    dataset = EthUcyBenchmarkDataset("data/ETHUCY_benchmark_v1", "ETH", "test")
    sample = dataset[0]

    for key in ("x_accel", "x_turn_rate", "x_motion_run"):
        assert key in sample
        assert torch.isfinite(sample[key]).all()
    assert bool(sample["x_valid_mask"][:, :8].all())

    batch = ethucy_benchmark_collate_fn([sample])
    for key in ("x_accel", "x_turn_rate", "x_motion_run"):
        assert key in batch
        assert torch.isfinite(batch[key]).all()


def test_summary_relabels_legacy_easy_zero_missing_clean_run(tmp_path):
    summary = load_summary()
    run_dir = tmp_path / "M0-current_ETH-M_clean-direct_seed2024"
    result_path = run_dir / "eval" / "epoch=0" / "results.json"
    result_path.parent.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(json.dumps({
        "protocol": "clean-direct",
        "clean_source": {"type": "easy_zero_missing_subset"},
    }))
    result_path.write_text(json.dumps({
        "meta": {
            "variant": "M0-current",
            "scene": "ETH-M",
            "seed": 2024,
            "K": 6,
        },
        "results": {
            "overall": {
                "n": 10,
                "minADE_K": 0.1,
                "minFDE_K": 0.2,
                "ADE@1": 0.3,
                "FDE@1": 0.4,
                "MR": 0.5,
            },
        },
    }))

    rows = summary.collect(tmp_path)

    assert rows[0]["protocol"] == "easy-zero-missing-diagnostic"


def test_direct_runner_passes_k20_to_clean_training():
    runner = load_runner()
    assert runner.DIRECT_NUM_MODES == 20
    args = SimpleNamespace(
        variant="M0-current",
        clean_data_root="data/ETHUCY_benchmark_v1",
        output_root="outputs/clean_ethucy",
        seed=2024,
        gpu=0,
        bimamba=True,
        batch_size=64,
        num_workers=4,
        epochs=100,
        precision="bf16",
        folds=runner.FOLDS,
        K=runner.DIRECT_NUM_MODES,
        lr=1e-3,
        weight_decay=1e-4,
    )

    command = runner.build_clean_runner_command(args)
    assert command[command.index("--num-modes") + 1] == "20"


def test_direct_runner_defaults_to_missing_data_retraining():
    runner = load_runner()

    assert runner.DEFAULT_PROTOCOL == "easy-direct"
    assert runner.DEFAULT_VARIANT == "M0-current"
    assert runner.DIRECT_NUM_MODES == 20
    assert runner.PROTOCOLS["easy-direct"]["zero_missing_only"] is False
    assert runner.PROTOCOLS["hard-direct"]["zero_missing_only"] is False
