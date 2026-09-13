"""Benchmark the ETH/UCY benchmark dataset CPU pipeline.

Measures per-sample: torch.load (weights_only=False vs True), process() full,
and inside process: rotation (float64 vs float32), build_b1_motion_features,
build_missing_features, and the Python loops (theta / last_valid_idx / x_last_valid_angle).
"""
import time
import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datamodule.ethucy_benchmark_dataset import (
    EthUcyBenchmarkDataset,
    compute_theta,
)
from src.datamodule.b1_motion_features import build_b1_motion_features
from src.datamodule.missing_features import build_missing_features

DATA_ROOT = "data/ETHUCY_benchmark_v1"
FOLD = "ETH"
N_SAMPLE = 3000  # number of samples to time

ds = EthUcyBenchmarkDataset(data_root=DATA_ROOT, fold=FOLD, split="train")
files = ds.file_list[:N_SAMPLE]
n = len(files)
print(f"timing over {n} samples (total {len(ds.file_list)})")

# 1) torch.load weights_only=False vs True
t0 = time.perf_counter()
for f in files:
    torch.load(f, weights_only=False)
t1 = time.perf_counter()
t_load_false = (t1 - t0) / n * 1000

t0 = time.perf_counter()
for f in files:
    torch.load(f, weights_only=True)
t1 = time.perf_counter()
t_load_true = (t1 - t0) / n * 1000

# 2) full __getitem__ (load + process)
t0 = time.perf_counter()
for i in range(n):
    ds.__getitem__(i)
t1 = time.perf_counter()
t_getitem = (t1 - t0) / n * 1000

# 3) process() alone (re-use one loaded sample)
sample_data = torch.load(files[0], weights_only=False)
# warm
ds.process(sample_data)
t0 = time.perf_counter()
for _ in range(n):
    ds.process(sample_data)
t1 = time.perf_counter()
t_process = (t1 - t0) / n * 1000

# 4) inside process: rotation float64 vs float32
positions = sample_data["positions"]
valid_mask = sample_data["valid_mask"]
obs_len = ds.obs_len
focal_hist_valid = valid_mask[0, :obs_len]
theta, _ = compute_theta(positions[0, :obs_len], focal_hist_valid)
cos_t, sin_t = torch.cos(theta).item(), torch.sin(theta).item()
origin = positions[0, obs_len - 1].clone()

def rot_f64():
    rot = torch.tensor([[cos_t, -sin_t], [sin_t, cos_t]], dtype=torch.float64)
    return torch.matmul((positions.double() - origin.double().view(1, 1, 2)), rot).float()

def rot_f32():
    rot = torch.tensor([[cos_t, -sin_t], [sin_t, cos_t]], dtype=torch.float32)
    return torch.matmul((positions.float() - origin.float().view(1, 1, 2)), rot)

for f in (rot_f64, rot_f32):
    f()
t0 = time.perf_counter()
for _ in range(n):
    rot_f64()
t1 = time.perf_counter()
t_rot64 = (t1 - t0) / n * 1000

t0 = time.perf_counter()
for _ in range(n):
    rot_f32()
t1 = time.perf_counter()
t_rot32 = (t1 - t0) / n * 1000

# 5) b1_motion_features + missing_features
hist_pos = positions[:obs_len] if positions.size(0) >= obs_len else positions
hv = valid_mask[:obs_len]
hist_pos = positions[:, :obs_len]
hist_valid = valid_mask[:, :obs_len]
build_b1_motion_features(hist_pos, hist_valid)
build_missing_features(hist_valid)
t0 = time.perf_counter()
for _ in range(n):
    build_b1_motion_features(hist_pos, hist_valid)
t1 = time.perf_counter()
t_b1 = (t1 - t0) / n * 1000

t0 = time.perf_counter()
for _ in range(n):
    build_missing_features(hist_valid)
t1 = time.perf_counter()
t_miss = (t1 - t0) / n * 1000

print("\n=== per-sample CPU time (ms) ===")
print(f"torch.load weights_only=False : {t_load_false:8.3f}")
print(f"torch.load weights_only=True  : {t_load_true:8.3f}")
print(f"__getitem__ (load+process)    : {t_getitem:8.3f}")
print(f"process() alone               : {t_process:8.3f}")
print(f"  rotation float64            : {t_rot64:8.3f}")
print(f"  rotation float32            : {t_rot32:8.3f}")
print(f"  build_b1_motion_features    : {t_b1:8.3f}")
print(f"  build_missing_features      : {t_miss:8.3f}")

# 6) extrapolate per-epoch (30307 samples) and per-step (bs=64)
n_total = len(ds.file_list)
print("\n=== per-epoch extrapolation (30307 samples) ===")
for name, t in [("torch.load f64", t_load_false), ("process()", t_process), ("__getitem__ total", t_getitem)]:
    print(f"  {name:20s}: {t * n_total / 1000:8.1f} s/epoch")
print(f"  per-step (bs=64, 474 steps): {(t_getitem * 64):8.3f} ms/step single-worker")
