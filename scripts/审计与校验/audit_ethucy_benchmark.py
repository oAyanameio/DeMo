"""ETH/UCY benchmark 数据审计：验证 fold/split 完整性、focal 有效性、无 NaN、无跨 split 重复。"""
import argparse
import glob
import json
import os

import torch

FOLDS = ["ETH", "HOTEL", "UNIV", "ZARA1", "ZARA2"]


def audit(data_root: str, manifest_path: str, max_check: int = 0):
    with open(manifest_path) as f:
        manifest = json.load(f)

    lines = []
    ok = True

    def log(s=""):
        print(s)
        lines.append(s)

    log(f"# audit data_root={data_root}")
    src_seen = {}
    for fold in FOLDS:
        for split in ("train", "val", "test"):
            files = sorted(glob.glob(os.path.join(data_root, f"fold_{fold}", split, "*", "*.pt")))
            n_focal_invalid_hist = 0
            n_focal_invalid_fut = 0
            n_nan = 0
            actor_counts = []
            checked = 0
            for fp in files:
                if max_check and checked >= max_check:
                    break
                s = torch.load(fp, weights_only=False)
                checked += 1
                vm = s["valid_mask"]
                fi = (s["agent_ids"] == s["focal_id"]).nonzero()
                assert len(fi) == 1, f"focal not found in {fp}"
                f = fi[0, 0]
                if not vm[f, :8].all():
                    n_focal_invalid_hist += 1
                if not vm[f, 8:20].all():
                    n_focal_invalid_fut += 1
                if not torch.isfinite(s["positions"]).all():
                    n_nan += 1
                assert s["positions"].shape[1] == 20, fp
                assert s["frame_ids"].shape == (20,), fp
                if manifest.get("context_policy") == "strict_complete":
                    assert vm.all(), f"strict-complete violated: {fp}"
                actor_counts.append(int(s["positions"].size(0)))
            log(f"fold_{fold}/{split}: files={len(files)} checked={checked} "
                f"invalid_focal_hist={n_focal_invalid_hist} invalid_focal_fut={n_focal_invalid_fut} "
                f"nan={n_nan} actors[min/med/max]={min(actor_counts) if actor_counts else '-'}"
                f"/{sorted(actor_counts)[len(actor_counts)//2] if actor_counts else '-'}"
                f"/{max(actor_counts) if actor_counts else '-'}")
            if n_focal_invalid_hist or n_focal_invalid_fut or n_nan:
                ok = False
            # source file overlap check via manifest
            for item in manifest["folds"][fold][split]:
                key = item["source_file"]
                src_seen.setdefault(key, set()).add((fold, split))

    overlaps = {k: v for k, v in src_seen.items() if len(v) > 1}
    log(f"source_file cross-split overlaps within fold: {overlaps if overlaps else 'NONE'}")
    if overlaps:
        ok = False

    log(f"AUDIT {'PASSED' if ok else 'FAILED'}")
    return ok, "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output")
    ap.add_argument("--max-check", type=int, default=0, help="0 = check all files")
    args = ap.parse_args()
    ok, text = audit(args.data_root, args.manifest, args.max_check)
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            f.write(text + "\n")
    raise SystemExit(0 if ok else 1)
