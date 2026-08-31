"""第一轮缺失历史数据集制作脚本（missing_history_v1）。

按 docs/research/ETHUCY与SDD第一轮缺失历史数据集制作说明.md 生成：
  - data/ETHUCY_missing_v1/{complete,random_single,random_block2}/fold_*/{train,val,test}/<scene>/*.pt
  - data/SDD_missing_v1/{complete,random_single,random_block2}/{train,val,test}/*.pt

要点：
  1. 只对历史 8 帧施加缺失，未来 12 帧保持完整；
  2. 帧编号 0-7，第 6、7 帧强制可见，缺失只从第 0-5 帧中抽；
  3. 掩码种子由 dataset|fold|split|scene_id|source_index|focal_id|condition|mask_seed
     经 SHA256 派生，完全确定、可复现，与遍历顺序/进程数无关；
  4. complete 条件不调用随机数生成器；
  5. SDD train/val 划分严格复刻 moflow_sdd_dataset.py（default_rng(2024) 置换）；
  6. 缺失历史坐标写 0.0，valid_mask[:, :8] = history_mask，valid_mask[:, 8:] 全 True。

用法：
  python3 scripts/数据集构建/build_missing_history_dataset.py --dataset ethucy \
      --source-root data/ETHUCY_benchmark_v1 --output-root data/ETHUCY_missing_v1 \
      --conditions complete random_single random_block2 --mask-seed 42
  python3 scripts/数据集构建/build_missing_history_dataset.py --dataset sdd \
      --source-root data/sdd --output-root data/SDD_missing_v1 \
      --conditions complete random_single random_block2 --split-seed 2024 --mask-seed 42
"""

import argparse
import hashlib
import json
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

OBS_LEN = 8
PRED_LEN = 12
SEQ_LEN = 20
DT = 0.4
FOLDS = ["ETH", "HOTEL", "UNIV", "ZARA1", "ZARA2"]
SPLITS = ["train", "val", "test"]
# 可缺失帧范围：第 0-5 帧（第 6、7 帧强制可见）
MISSABLE = list(range(6))
CONDITIONS = ["complete", "random_single", "random_block2"]


def mask_seed_from(key: str) -> int:
    """由样本身份字符串派生确定性掩码种子。"""
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")


def derive_history_mask(
    dataset: str,
    fold,
    split: str,
    scene_id: str,
    source_index: int,
    source_file: str,
    focal_id: int,
    condition: str,
    mask_seed: int,
    num_agents: int,
) -> torch.Tensor:
    """按行人生成历史掩码 [A, 8]，True=可见。

    身份键含 source_file：同一 (fold,split,scene) 的不同源 .txt 样本即使
    sample_index 相同也获得独立掩码（防跨源掩码碰撞）。
    """
    if condition == "complete":
        return torch.ones(num_agents, OBS_LEN, dtype=torch.bool)

    key = "|".join(
        str(x) for x in [dataset, fold, split, scene_id, source_index, source_file, focal_id, condition, mask_seed]
    )
    rng = np.random.default_rng(mask_seed_from(key))
    mask = torch.ones(num_agents, OBS_LEN, dtype=torch.bool)
    if condition == "random_single":
        # 每个行人独立随机缺失 1 帧（从 0-5 中抽）
        for a in range(num_agents):
            t = int(rng.integers(0, len(MISSABLE)))
            mask[a, t] = False
    elif condition == "random_block2":
        # 每个行人独立随机缺失连续 2 帧（起点从 0-4 中抽，块为 {s, s+1}）
        for a in range(num_agents):
            s = int(rng.integers(0, len(MISSABLE) - 1))
            mask[a, s] = False
            mask[a, s + 1] = False
    else:
        raise ValueError(f"unknown condition: {condition}")
    # 强制第 6、7 帧可见（双保险）
    mask[:, 6] = True
    mask[:, 7] = True
    return mask


def apply_mask_to_sample(sample: dict, history_mask: torch.Tensor) -> dict:
    """将历史掩码应用到 canonical 样本：缺失处置 0.0，重写 valid_mask。"""
    positions = sample["positions"].clone()
    valid_mask = sample["valid_mask"].clone()
    positions[:, :OBS_LEN] = positions[:, :OBS_LEN] * history_mask.unsqueeze(-1).float()
    valid_mask[:, :OBS_LEN] = history_mask
    valid_mask[:, OBS_LEN:] = True
    out = dict(sample)
    out["positions"] = positions
    out["valid_mask"] = valid_mask
    out["history_mask"] = history_mask.clone()
    return out


# ---------------------------------------------------------------------------
# ETH/UCY
# ---------------------------------------------------------------------------

def list_ethucy_units(source_root: Path):
    """枚举 (fold, split, scene, files) 工作单元，顺序固定。"""
    units = []
    for fold in FOLDS:
        for split in SPLITS:
            split_dir = source_root / f"fold_{fold}" / split
            if not split_dir.exists():
                continue
            for scene_dir in sorted(split_dir.iterdir()):
                if not scene_dir.is_dir():
                    continue
                files = sorted(scene_dir.glob("*.pt"))
                if files:
                    units.append((fold, split, scene_dir.name, [str(f) for f in files]))
    return units


def build_ethucy_unit(args):
    """处理一个 (condition, fold, split, scene) 单元。在子进程中运行。"""
    condition, fold, split, scene, files, source_root, output_root, mask_seed = args
    out_dir = Path(output_root) / condition / f"fold_{fold}" / split / scene
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "samples": 0, "agents": 0, "missing_frames": 0,
        "rejected": 0, "reject_reasons": {},
        "source_index_min": None, "source_index_max": None,
        "sha256": [],  # (filename, digest)
    }
    for f in files:
        try:
            src = torch.load(f, weights_only=False)
            A, T, _ = src["positions"].shape
            if T != SEQ_LEN or A < 1:
                raise ValueError(f"bad shape {src['positions'].shape}")
            if not bool(src["valid_mask"].all()):
                raise ValueError("source valid_mask not all True")
        except Exception as e:  # noqa: BLE001
            stats["rejected"] += 1
            stats["reject_reasons"][str(e)[:80]] = stats["reject_reasons"].get(str(e)[:80], 0) + 1
            continue

        source_index = int(src["sample_index"])
        hmask = derive_history_mask(
            "ethucy", fold, split, src["scene_id"], source_index,
            str(src["source_file"]), int(src["focal_id"]), condition, mask_seed, A,
        )
        out = apply_mask_to_sample(src, hmask)
        out["condition"] = condition
        out["source_file"] = src["source_file"]
        out["source_index"] = source_index
        out["mask_seed"] = mask_seed
        out["obs_len"] = OBS_LEN
        out["pred_len"] = PRED_LEN
        out["dt"] = DT
        out["unit"] = "meter"

        name = Path(f).name
        out_path = out_dir / name
        torch.save(out, out_path)

        stats["samples"] += 1
        stats["agents"] += A
        stats["missing_frames"] += int((~hmask).sum())
        si = source_index
        stats["source_index_min"] = si if stats["source_index_min"] is None else min(stats["source_index_min"], si)
        stats["source_index_max"] = si if stats["source_index_max"] is None else max(stats["source_index_max"], si)
        h = hashlib.sha256()
        with open(out_path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 16), b""):
                h.update(chunk)
        stats["sha256"].append((name, h.hexdigest()))
    return (condition, fold, split, scene), stats


def build_ethucy(source_root: Path, output_root: Path, conditions, mask_seed, workers):
    units = list_ethucy_units(source_root)
    total_src = sum(len(u[3]) for u in units)
    print(f"[ethucy] source units={len(units)} files={total_src}", flush=True)

    tasks = [(c, u[0], u[1], u[2], u[3], str(source_root), str(output_root), mask_seed)
             for c in conditions for u in units]
    # sample_index：按固定遍历顺序赋值（fold,split,scene 文件名序），与条件无关
    manifest_stats = {}
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for key, stats in ex.map(build_ethucy_unit, tasks, chunksize=1):
            condition, fold, split, scene = key
            md = manifest_stats.setdefault(
                (condition, fold, split),
                {"samples": 0, "agents": 0, "missing_frames": 0, "history_frames": 0,
                 "rejected": 0, "reject_reasons": {}, "scenes": {},
                 "source_index_min": None, "source_index_max": None},
            )
            md["samples"] += stats["samples"]
            md["agents"] += stats["agents"]
            md["missing_frames"] += stats["missing_frames"]
            md["history_frames"] += stats["agents"] * OBS_LEN
            md["rejected"] += stats["rejected"]
            for k, v in stats["reject_reasons"].items():
                md["reject_reasons"][k] = md["reject_reasons"].get(k, 0) + v
            md["scenes"][scene] = {
                "samples": stats["samples"],
                "file_digests": dict(stats["sha256"]),
            }
            md["source_index_min"] = stats["source_index_min"] if md["source_index_min"] is None else min(md["source_index_min"], stats["source_index_min"])
            md["source_index_max"] = stats["source_index_max"] if md["source_index_max"] is None else max(md["source_index_max"], stats["source_index_max"])
            done += 1
            if done % 30 == 0 or done == len(tasks):
                rate = done / (time.time() - t0)
                print(f"[ethucy] {done}/{len(tasks)} units, {rate:.1f} units/s", flush=True)

    manifest = {
        "version": "missing_history_v1",
        "dataset": "ETHUCY",
        "obs_len": OBS_LEN,
        "pred_len": PRED_LEN,
        "dt": DT,
        "unit": "meter",
        "use_map": False,
        "mask_seed": mask_seed,
        "split_seed": None,
        "conditions": conditions,
        "history_visibility_rule": "frames_6_and_7_visible",
        "future_policy": "complete",
        "coordinate_policy": "raw_scene_coordinates",
        "feature_policy": "recompute_from_visible_history",
        "source_root": str(source_root),
        "mask_rng_key_format": "dataset|fold|split|scene_id|source_index|source_file|focal_id|condition|mask_seed -> sha256[:8] -> np.random.default_rng",
        "splits": {},
    }
    for (condition, fold, split), md in sorted(manifest_stats.items()):
        cd = manifest["splits"].setdefault(condition, {}).setdefault(f"fold_{fold}", {})
        cd[split] = {
            "samples": md["samples"],
            "agents_total": md["agents"],
            "missing_history_frames": md["missing_frames"],
            "history_frames_total": md["history_frames"],
            "actual_missing_rate": round(md["missing_frames"] / md["history_frames"], 6) if md["history_frames"] else 0.0,
            "source_index_range": [md["source_index_min"], md["source_index_max"]],
            "rejected_samples": md["rejected"],
            "reject_reasons": md["reject_reasons"],
            "scenes": {s: v["samples"] for s, v in md["scenes"].items()},
            "per_file_sha256": {s: v["file_digests"] for s, v in md["scenes"].items()},
        }
    with open(output_root / "manifest.json", "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print(f"[ethucy] manifest -> {output_root / 'manifest.json'}", flush=True)


# ---------------------------------------------------------------------------
# SDD
# ---------------------------------------------------------------------------

def sdd_split_indices(n: int, split_seed: int, val_ratio: float = 0.1):
    """严格复刻 moflow_sdd_dataset.py 的确定性 90/10 划分。

    返回 (train_indices, val_indices)，均为 sdd_train.pkl 列表中的原始索引，
    顺序与 MoFlowSddDataset 中的 scenes 顺序一致。
    """
    rng = np.random.default_rng(split_seed)
    perm = rng.permutation(n)
    n_val = int(n * val_ratio)
    return perm[n_val:], perm[:n_val]


def build_sdd_unit(args):
    """处理一个 (condition, split) 单元。在子进程中运行。"""
    condition, split, records, output_root, mask_seed = args
    out_dir = Path(output_root) / condition / split
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "samples": 0, "agents": 0, "missing_frames": 0,
        "rejected": 0, "reject_reasons": {},
        "source_index_min": None, "source_index_max": None,
        "sha256": {},
    }
    for sample_index, (orig_idx, past, fut) in enumerate(records):
        try:
            past = np.asarray(past, dtype=np.float32)
            fut = np.asarray(fut, dtype=np.float32)
            if past.shape != (OBS_LEN, 2) or fut.shape != (PRED_LEN, 2):
                raise ValueError(f"bad shapes {past.shape} {fut.shape}")
            if not (np.isfinite(past).all() and np.isfinite(fut).all()):
                raise ValueError("non-finite coordinates")
        except Exception as e:  # noqa: BLE001
            stats["rejected"] += 1
            stats["reject_reasons"][str(e)[:80]] = stats["reject_reasons"].get(str(e)[:80], 0) + 1
            continue

        positions = torch.from_numpy(np.concatenate([past, fut], axis=0)).unsqueeze(0)  # [1,20,2]
        src_file = "sdd_train.pkl" if split in ("train", "val") else "sdd_test.pkl"

        hmask = derive_history_mask(
            "sdd", None, split, "SDD", int(orig_idx), src_file, 0, condition, mask_seed, 1,
        )
        sample = {
            "positions": positions,
            "valid_mask": torch.ones(1, SEQ_LEN, dtype=torch.bool),
            "agent_ids": torch.zeros(1, dtype=torch.long),
            "focal_id": 0,
            "scene_id": "SDD",
            "fold": None,
            "split": split,
        }
        out = apply_mask_to_sample(sample, hmask)
        out["condition"] = condition
        out["source_file"] = src_file
        out["source_index"] = int(orig_idx)
        out["sample_index"] = sample_index
        out["mask_seed"] = mask_seed
        out["obs_len"] = OBS_LEN
        out["pred_len"] = PRED_LEN
        out["dt"] = DT
        out["unit"] = "pixel"

        out_path = out_dir / f"{sample_index:06d}.pt"
        torch.save(out, out_path)

        stats["samples"] += 1
        stats["agents"] += 1
        stats["missing_frames"] += int((~hmask).sum())
        stats["source_index_min"] = int(orig_idx) if stats["source_index_min"] is None else min(stats["source_index_min"], int(orig_idx))
        stats["source_index_max"] = int(orig_idx) if stats["source_index_max"] is None else max(stats["source_index_max"], int(orig_idx))
        h = hashlib.sha256()
        with open(out_path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 16), b""):
                h.update(chunk)
        stats["sha256"][out_path.name] = h.hexdigest()
    return (condition, split), stats


def build_sdd(source_root: Path, output_root: Path, conditions, split_seed, mask_seed, workers):
    with open(source_root / "original" / "sdd_train.pkl", "rb") as f:
        train_all = pickle.load(f)
    with open(source_root / "original" / "sdd_test.pkl", "rb") as f:
        test_all = pickle.load(f)
    print(f"[sdd] train.pkl={len(train_all)} test.pkl={len(test_all)}", flush=True)

    train_idx, val_idx = sdd_split_indices(len(train_all), split_seed)
    print(f"[sdd] split: train={len(train_idx)} val={len(val_idx)} (split_seed={split_seed})", flush=True)

    split_records = {
        "train": [(int(i), train_all[i][0], train_all[i][1]) for i in train_idx],
        "val": [(int(i), train_all[i][0], train_all[i][1]) for i in val_idx],
        "test": [(int(i), t[0], t[1]) for i, t in enumerate(test_all)],
    }
    tasks = [(c, s, split_records[s], str(output_root), mask_seed)
             for c in conditions for s in SPLITS]

    manifest_stats = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as ex:
        for key, stats in ex.map(build_sdd_unit, tasks, chunksize=1):
            condition, split = key
            md = manifest_stats.setdefault(
                (condition, split),
                {"samples": 0, "agents": 0, "missing_frames": 0, "history_frames": 0,
                 "rejected": 0, "reject_reasons": {},
                 "source_index_min": None, "source_index_max": None, "sha256": {}},
            )
            md["samples"] += stats["samples"]
            md["agents"] += stats["agents"]
            md["missing_frames"] += stats["missing_frames"]
            md["history_frames"] += stats["agents"] * OBS_LEN
            md["rejected"] += stats["rejected"]
            for k, v in stats["reject_reasons"].items():
                md["reject_reasons"][k] = md["reject_reasons"].get(k, 0) + v
            md["source_index_min"] = stats["source_index_min"] if md["source_index_min"] is None else min(md["source_index_min"], stats["source_index_min"])
            md["source_index_max"] = stats["source_index_max"] if md["source_index_max"] is None else max(md["source_index_max"], stats["source_index_max"])
            md["sha256"].update(stats["sha256"])

    manifest = {
        "version": "missing_history_v1",
        "dataset": "SDD",
        "obs_len": OBS_LEN,
        "pred_len": PRED_LEN,
        "dt": DT,
        "unit": "pixel",
        "use_map": False,
        "mask_seed": mask_seed,
        "split_seed": split_seed,
        "val_ratio": 0.1,
        "conditions": conditions,
        "history_visibility_rule": "frames_6_and_7_visible",
        "future_policy": "complete",
        "coordinate_policy": "raw_scene_coordinates",
        "feature_policy": "recompute_from_visible_history",
        "source_root": str(source_root / "original"),
        "mask_rng_key_format": "dataset|fold|split|scene_id|source_index|source_file|focal_id|condition|mask_seed -> sha256[:8] -> np.random.default_rng",
        "splits": {},
    }
    for (condition, split), md in sorted(manifest_stats.items()):
        manifest["splits"].setdefault(condition, {})[split] = {
            "samples": md["samples"],
            "agents_total": md["agents"],
            "missing_history_frames": md["missing_frames"],
            "history_frames_total": md["history_frames"],
            "actual_missing_rate": round(md["missing_frames"] / md["history_frames"], 6) if md["history_frames"] else 0.0,
            "source_index_range": [md["source_index_min"], md["source_index_max"]],
            "rejected_samples": md["rejected"],
            "reject_reasons": md["reject_reasons"],
            "per_file_sha256": md["sha256"],
        }
    with open(output_root / "manifest.json", "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print(f"[sdd] manifest -> {output_root / 'manifest.json'} ({time.time()-t0:.1f}s)", flush=True)


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Build missing-history v1 datasets")
    ap.add_argument("--dataset", choices=["ethucy", "sdd"], required=True)
    ap.add_argument("--source-root", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--conditions", nargs="+", default=CONDITIONS, choices=CONDITIONS)
    ap.add_argument("--mask-seed", type=int, default=42)
    ap.add_argument("--split-seed", type=int, default=2024)
    ap.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 8))
    ap.add_argument("--folds", nargs="+", default=FOLDS, help="ethucy only: subset of folds")
    ap.add_argument("--scenes", nargs="+", default=None, help="ethucy only: only these scene names (smoke test)")
    args = ap.parse_args()

    if args.folds != FOLDS:
        # 通过模块级约定传给子进程可见的枚举逻辑：直接改写本模块全局
        globals()["FOLDS"] = args.folds

    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.dataset == "ethucy":
        build_ethucy(source_root, output_root, args.conditions, args.mask_seed, args.workers)
    else:
        build_sdd(source_root, output_root, args.conditions, args.split_seed, args.mask_seed, args.workers)


if __name__ == "__main__":
    main()
