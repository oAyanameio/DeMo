"""Build explicit source manifest for the ETH/UCY benchmark.

Scans <raw_root>/<fold>/{train,val,test}/*.txt only (never raw/all_data),
validates the official split organization, and records size / rows / frames /
SHA-256 for every source file.

Returns dict: {"folds": {fold: {split: [item, ...]}}, ...} where each item has
fold, split, scene_id, source_file (relative), absolute_path,
num_source_rows, num_source_frames, size_bytes, sha256.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

FOLDS = ["ETH", "HOTEL", "UNIV", "ZARA1", "ZARA2"]
FOLD_DIRS = {"ETH": "eth", "HOTEL": "hotel", "UNIV": "univ", "ZARA1": "zara1", "ZARA2": "zara2"}

# Explicit filename stem -> standard scene mapping (no fuzzy matching).
SCENE_BY_STEM = {
    "biwi_eth": "ETH",
    "biwi_hotel": "HOTEL",
    "students001": "UNIV",
    "students003": "UNIV",
    "uni_examples": "UNIV",
    "crowds_zara01": "ZARA1",
    "crowds_zara02": "ZARA2",
    "crowds_zara03": "ZARA2",
}


def scene_of(filename: str) -> str:
    stem = Path(filename).stem
    stem = re.sub(r"_(train|val|test)$", "", stem)
    if stem not in SCENE_BY_STEM:
        raise ValueError(f"Unknown scene file (not in explicit mapping): {filename}")
    return SCENE_BY_STEM[stem]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def count_rows_and_frames(path: Path):
    rows = 0
    frames = set()
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p for p in line.replace(",", " ").split() if p]
            if len(parts) < 4:
                continue
            try:
                frame = int(float(parts[0]))
                float(parts[1])
            except ValueError:
                continue  # header
            rows += 1
            frames.add(frame)
    return rows, len(frames)


def build_manifest(raw_root, expected_folds=None):
    raw_root = Path(raw_root)
    expected_folds = expected_folds or FOLDS
    folds_out = {f: {"train": [], "val": [], "test": []} for f in expected_folds}
    seen = {}
    seen_name_in_fold = {}

    for fold in expected_folds:
        fold_dir = raw_root / FOLD_DIRS[fold]
        if not fold_dir.is_dir():
            raise FileNotFoundError(f"missing fold dir: {fold_dir}")
        for split in ["train", "val", "test"]:
            split_dir = fold_dir / split
            if not split_dir.is_dir():
                raise FileNotFoundError(f"missing split dir: {split_dir}")
            files = sorted(split_dir.glob("*.txt"))
            if not files:
                raise FileNotFoundError(f"empty split dir: {split_dir}")
            for f in files:
                key = f.resolve()
                if key in seen:
                    raise ValueError(
                        f"source file appears in multiple splits/folds: {f} "
                        f"(first seen at {seen[key]})"
                    )
                seen[key] = f
                base_key = (fold, f.name)
                if base_key in seen_name_in_fold:
                    raise ValueError(
                        f"source file appears in multiple splits: {f} "
                        f"(first seen at {seen_name_in_fold[base_key]})"
                    )
                seen_name_in_fold[base_key] = f
                scene = scene_of(f.name)
                if f.stat().st_size == 0:
                    raise ValueError(f"empty file: {f}")
                rows, frames = count_rows_and_frames(f)
                if rows == 0:
                    raise ValueError(f"no valid rows: {f}")
                folds_out[fold][split].append(
                    {
                        "fold": fold,
                        "split": split,
                        "scene_id": scene,
                        "source_file": str(f.relative_to(raw_root)),
                        "absolute_path": str(f.resolve()),
                        "num_source_rows": rows,
                        "num_source_frames": frames,
                        "size_bytes": f.stat().st_size,
                        "sha256": sha256_file(f),
                    }
                )

    return folds_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--expected-folds", nargs="+", default=FOLDS)
    args = ap.parse_args()
    manifest = build_manifest(args.raw_root, args.expected_folds)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    wrapped = {
        "version": "ethucy_benchmark_v1_source",
        "raw_root": args.raw_root,
        "folds": manifest,
        "scene_by_stem": SCENE_BY_STEM,
    }
    with open(out, "w") as f:
        json.dump(wrapped, f, indent=2)
    n = sum(len(v) for fd in manifest.values() for v in fd.values())
    print(f"wrote {n} manifest items to {out}")


if __name__ == "__main__":
    main()
