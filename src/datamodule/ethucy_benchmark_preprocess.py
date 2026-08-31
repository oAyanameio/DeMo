"""ETH/UCY benchmark 预处理器（SocialGAN 风格固定窗口提取）。

关键规则：
- 排序后的 unique raw frame 直接映射为连续 timestep 0,1,2,...（不做 frame//10 分箱）。
- 保留 raw frame id 到 frame_ids_raw。
- 滑窗 obs_len=8, pred_len=12, skip=1；focal 的 20 帧必须全部有效。
- strict-complete actor 策略：窗口内只保留 20 帧全部有效的 actor（含 focal）。
- 不插值，NaN 位置置 0 保存但 valid_mask 保留（strict-complete 下全 True）。
"""
import hashlib
import json
import os
import sys
from typing import Dict, List

import numpy as np
import torch

FOLDS = ["ETH", "HOTEL", "UNIV", "ZARA1", "ZARA2"]


def load_rows(path: str):
    """读取 raw .txt -> list of (raw_frame:int, ped_id:int, x, y)。"""
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            try:
                frame = int(float(parts[0]))
                ped = int(float(parts[1]))
                x = float(parts[2])
                y = float(parts[3])
            except ValueError:
                continue
            rows.append((frame, ped, x, y))
    return rows


def build_dense(rows: List, obs_len: int, pred_len: int):
    """将行数据转为稠密矩阵并生成 strict-complete 滑窗样本。

    返回 (samples, num_frames, num_peds)；每个 sample 为 dict。
    """
    total_len = obs_len + pred_len
    raw_frames = sorted({r[0] for r in rows})
    peds = sorted({r[1] for r in rows})
    if len(raw_frames) < total_len:
        return [], len(raw_frames), len(peds)
    frame_to_idx = {fr: i for i, fr in enumerate(raw_frames)}
    ped_to_idx = {p: i for i, p in enumerate(peds)}

    # 同一 (raw_frame, ped) 重复行 -> 报错（数据异常）
    seen = set()
    pos = np.full((len(peds), len(raw_frames), 2), np.nan, dtype=np.float32)
    for fr, ped, x, y in rows:
        key = (fr, ped)
        if key in seen:
            raise ValueError(f"Duplicate (frame, ped) row: {key}")
        seen.add(key)
        pos[ped_to_idx[ped], frame_to_idx[fr]] = (x, y)

    valid = ~np.isnan(pos[:, :, 0])

    samples = []
    for start in range(0, len(raw_frames) - total_len + 1):
        wv = valid[:, start : start + total_len]
        complete = wv.all(axis=1)
        if not complete.any():
            continue
        idxs = np.where(complete)[0]
        wpos = pos[idxs, start : start + total_len, :]
        wpos = np.nan_to_num(wpos, nan=0.0)
        wvalid = wv[idxs]
        assert wvalid.all()
        frame_ids_raw = [raw_frames[start + i] for i in range(total_len)]
        for k, actor_idx in enumerate(idxs):
            samples.append({
                "focal_id": int(peds[actor_idx]),
                "agent_ids": torch.tensor([peds[i] for i in idxs], dtype=torch.long),
                "positions": torch.from_numpy(wpos.copy()),
                "valid_mask": torch.from_numpy(wvalid.copy()),
                "frame_ids": torch.arange(total_len, dtype=torch.long),
                "frame_ids_raw": torch.tensor(frame_ids_raw, dtype=torch.long),
            })
    return samples, len(raw_frames), len(peds)


def preprocess(raw_root: str, output_root: str, obs_len: int = 8, pred_len: int = 12,
               dt: float = 0.4, source_manifest: str = None, folds=None):
    folds = folds or FOLDS
    if source_manifest is None or not os.path.exists(source_manifest):
        source_manifest = os.path.join(output_root, "source_manifest.json")
    if not os.path.exists(source_manifest):
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
        from build_ethucy_manifest import build_manifest as _bm
        os.makedirs(output_root, exist_ok=True)
        sm_folds = _bm(raw_root)
        # 包裹为与 build_ethucy_manifest.main() 一致的格式（preprocess 读 sm["folds"]）
        sm = {
            "version": "ethucy_benchmark_v1_source",
            "raw_root": raw_root,
            "folds": sm_folds,
        }
        with open(source_manifest, "w") as f:
            json.dump(sm, f, indent=2)
    else:
        with open(source_manifest) as f:
            sm = json.load(f)

    manifest = {
        "version": "ethucy_benchmark_v1",
        "obs_len": obs_len,
        "pred_len": pred_len,
        "sequence_length": obs_len + pred_len,
        "dt": dt,
        "frame_stride": 1,
        "context_policy": "strict_complete",
        "scenes": FOLDS,
        "folds": {f: {"train": [], "val": [], "test": []} for f in FOLDS},
    }

    for fold in folds:
        for split in ("train", "val", "test"):
            for item in sm["folds"][fold][split]:
                src_abs = item["absolute_path"]
                scene = item["scene_id"]
                rows = load_rows(src_abs)
                samples, nframes, npeds = build_dense(rows, obs_len, pred_len)
                out_dir = os.path.join(output_root, f"fold_{fold}", split, scene)
                os.makedirs(out_dir, exist_ok=True)
                # 修复覆盖 bug：同一 (fold,split,scene) 可能有多个源 .txt，
                # 文件名必须带源文件 stem，否则各源文件都从 000000.pt 编号互相覆盖。
                stem = os.path.splitext(os.path.basename(src_abs))[0]
                for i, s in enumerate(samples):
                    s.update({
                        "scene_id": scene,
                        "fold": fold,
                        "split": split,
                        "source_file": item["source_file"],
                        "sample_index": i,
                    })
                    torch.save(s, os.path.join(out_dir, f"{stem}__{i:06d}.pt"))
                manifest["folds"][fold][split].append({
                    "fold": fold,
                    "split": split,
                    "scene_id": scene,
                    "source_file": item["source_file"],
                    "num_source_rows": item["num_source_rows"],
                    "num_source_frames": nframes,
                    "num_peds": npeds,
                    "num_samples": len(samples),
                    "sha256": item["sha256"],
                })
                print(f"fold_{fold}/{split}/{scene}/{os.path.basename(src_abs)}: "
                      f"{len(samples)} samples ({nframes} frames, {npeds} peds)")

    with open(os.path.join(output_root, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("manifest written:", os.path.join(output_root, "manifest.json"))
    return manifest


if __name__ == "__main__":
    import argparse
    import sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--obs-len", type=int, default=8)
    ap.add_argument("--pred-len", type=int, default=12)
    ap.add_argument("--dt", type=float, default=0.4)
    ap.add_argument("--folds", nargs="+", default=FOLDS)
    args = ap.parse_args()
    preprocess(args.raw_root, args.output_root, args.obs_len, args.pred_len, args.dt,
               folds=args.folds)
