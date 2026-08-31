"""缺失历史数据集 v1 验收审计（对应说明文档 §10 全部四类检查）。

用法：
  python3 scripts/审计与校验/audit_missing_history_dataset.py --dataset ethucy \
      --data-root data/ETHUCY_missing_v1 --source-root data/ETHUCY_benchmark_v1 [--full]
  python3 scripts/审计与校验/audit_missing_history_dataset.py --dataset sdd \
      --data-root data/SDD_missing_v1 --source-root data/sdd [--full]

默认快速模式：每个 (condition, fold, split, scene) 抽样检查 + 全量结构计数；
--full 对每个样本做逐样本校验（慢，约 30-60 分钟）。

退出码 0 = 全部通过；非 0 = 存在失败项（明细打印 + JSON 报告落盘）。
"""

import argparse
import glob
import hashlib
import json
import pickle
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

OBS_LEN, PRED_LEN, SEQ_LEN = 8, 12, 20
FOLDS = ["ETH", "HOTEL", "UNIV", "ZARA1", "ZARA2"]
SPLITS = ["train", "val", "test"]
CONDITIONS = ["complete", "random_single", "random_block2"]


class Audit:
    def __init__(self, name):
        self.name = name
        self.items = []  # (section, item, ok, detail)

    def add(self, section, item, ok, detail=""):
        self.items.append((section, item, ok, detail))
        if not ok:
            print(f"  [FAIL] {item}" + (f" — {detail}" if detail else ""), flush=True)

    def fail_count(self):
        return sum(1 for _, _, ok, _ in self.items if not ok)

    def report(self):
        return [
            {"section": s, "item": i, "ok": bool(ok), "detail": d}
            for s, i, ok, d in self.items
        ]


def mask_seed_from(key: str) -> int:
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")


def expected_mask(dataset, fold, split, scene_id, source_index, source_file, focal_id, condition, mask_seed, num_agents):
    """与 build 脚本完全一致的掩码推导（独立重实现，交叉验证）。"""
    if condition == "complete":
        return torch.ones(num_agents, OBS_LEN, dtype=torch.bool)
    key = "|".join(str(x) for x in [dataset, fold, split, scene_id, source_index, source_file, focal_id, condition, mask_seed])
    rng = np.random.default_rng(mask_seed_from(key))
    mask = torch.ones(num_agents, OBS_LEN, dtype=torch.bool)
    if condition == "random_single":
        for a in range(num_agents):
            t = int(rng.integers(0, 6))
            mask[a, t] = False
    else:
        for a in range(num_agents):
            s = int(rng.integers(0, 5))
            mask[a, s] = False
            mask[a, s + 1] = False
    mask[:, 6] = True
    mask[:, 7] = True
    return mask


def check_sample(audit, d, dataset, condition, fold, split, scene_dir_name, rel_path, source_root, split_seed=2024):
    """逐样本校验（掩码语义 + 坐标置零 + 与源样本对齐）。"""
    A = d["positions"].shape[0]
    hm, vm = d["history_mask"], d["valid_mask"]

    ok_shape = hm.shape == (A, 8) and vm.shape == (A, 20) and d["positions"].shape == (A, 20, 2)
    audit.add("10.2 掩码", f"形状 {rel_path}", ok_shape,
              "" if ok_shape else f"hm{tuple(hm.shape)} vm{tuple(vm.shape)} pos{tuple(d['positions'].shape)}")

    ok_match = torch.equal(vm[:, :8], hm) and bool(vm[:, 8:].all())
    audit.add("10.2 掩码", f"valid_mask 一致性 {rel_path}", ok_match)

    ok_zero = bool((d["positions"][:, :8][~hm] == 0).all())
    audit.add("10.2 掩码", f"缺失位置坐标为 0 {rel_path}", ok_zero)

    # 条件语义
    if condition == "complete":
        ok_c = bool(hm.all())
        det = "全可见"
    elif condition == "random_single":
        ok_c = bool(((~hm).sum(1) == 1).all()) and bool(hm[:, 6].all()) and bool(hm[:, 7].all())
        det = "每行人恰缺 1 帧, 帧6/7可见"
    else:
        rows_ok = True
        for row in ~hm:
            idx = row.nonzero().flatten().tolist()
            if sorted(idx) != idx or (idx and idx != list(range(idx[0], idx[0] + len(idx)))):
                rows_ok = False
        ok_c = bool(((~hm).sum(1) == 2).all()) and rows_ok and bool(hm[:, 6].all()) and bool(hm[:, 7].all())
        det = "每行人恰缺连续 2 帧"
    audit.add("10.2 掩码", f"条件语义 [{condition}] {rel_path}", ok_c, det)

    # 与源对齐
    if dataset == "ethucy":
        src_path = Path(source_root) / f"fold_{fold}" / split / scene_dir_name / Path(rel_path).name
        if src_path.exists():
            src = torch.load(src_path, weights_only=False)
            ok_fut = torch.equal(d["positions"][:, 8:], src["positions"][:, 8:])
            audit.add("10.3 泄漏", f"未来帧与源一致 {rel_path}", ok_fut)
            ok_vis = torch.equal(d["positions"][:, :8][hm], src["positions"][:, :8][hm])
            audit.add("10.3 泄漏", f"可见历史与源一致 {rel_path}", ok_vis)
    else:
        # SDD：按 source_index 回查 pkl
        fname = "sdd_train.pkl" if split in ("train", "val") else "sdd_test.pkl"
        with open(Path(source_root) / "original" / fname, "rb") as f:
            pass  # pkl 在调用方缓存，见 audit_sdd 内处理

    # 掩码确定性复推
    exp = expected_mask(dataset, fold if dataset == "ethucy" else None, split,
                        d["scene_id"], d["source_index"], d.get("source_file", ""), d["focal_id"], condition, d["mask_seed"], A)
    ok_det = torch.equal(hm, exp)
    audit.add("10.3 泄漏/复现", f"掩码可复现推 {rel_path}", ok_det)


def audit_ethucy(data_root, source_root, full=False, sample_frac=0.02, seed=0):
    audit = Audit("ETH/UCY")
    print("=== 10.1 结构检查 ===", flush=True)
    # 全量文件计数（快）
    src_counts = {}
    for fold in FOLDS:
        for split in SPLITS:
            src_counts[(fold, split)] = len(glob.glob(str(Path(source_root) / f"fold_{fold}" / split / "*" / "*.pt")))

    ok_counts = True
    for cond in CONDITIONS:
        for fold in FOLDS:
            for split in SPLITS:
                n = len(glob.glob(str(Path(data_root) / cond / f"fold_{fold}" / split / "*" / "*.pt")))
                if n != src_counts[(fold, split)]:
                    ok_counts = False
                    print(f"  [FAIL] count mismatch {cond} fold_{fold}/{split}: {n} vs src {src_counts[(fold, split)]}", flush=True)
    audit.add("10.1 结构", "5 fold × 3 split × 3 condition 样本数与源一致", ok_counts,
              f"total_src={sum(src_counts.values())} per_cond_expected={sum(src_counts.values())}")

    # manifest
    mpath = Path(data_root) / "manifest.json"
    ok_m = mpath.exists()
    audit.add("10.1 结构", "manifest.json 存在", ok_m)
    if ok_m:
        m = json.load(open(mpath))
        ok_fields = (m.get("version") == "missing_history_v1" and m.get("mask_seed") == 42
                     and m.get("conditions") == CONDITIONS and m.get("unit") == "meter")
        audit.add("10.1 结构", "manifest 关键字段", ok_fields)
        # manifest 计数 vs 实际
        ok_mc = True
        for cond in CONDITIONS:
            for fold in FOLDS:
                for split in SPLITS:
                    got = len(glob.glob(str(Path(data_root) / cond / f"fold_{fold}" / split / "*" / "*.pt")))
                    want = m["splits"][cond][f"fold_{fold}"][split]["samples"]
                    if got != want:
                        ok_mc = False
        audit.add("10.1 结构", "manifest 计数与磁盘一致", ok_mc)

    # 逐样本（抽样或全量）
    rng = random.Random(seed)
    if full:
        targets = []
        for cond in CONDITIONS:
            for fold in FOLDS:
                for split in SPLITS:
                    for f in sorted(glob.glob(str(Path(data_root) / cond / f"fold_{fold}" / split / "*" / "*.pt"))):
                        targets.append((cond, fold, split, f))
    else:
        targets = []
        for cond in CONDITIONS:
            for fold in FOLDS:
                for split in SPLITS:
                    scene_dirs = sorted((Path(data_root) / cond / f"fold_{fold}" / split).iterdir())
                    for sd in scene_dirs:
                        files = sorted(sd.glob("*.pt"))
                        k = max(1, int(len(files) * sample_frac))
                        for f in rng.sample(files, k):
                            targets.append((cond, fold, split, str(f)))

    print(f"=== 10.2/10.3 掩码与泄漏检查（{'全量' if full else f'抽样 {sample_frac:.0%}'}，{len(targets)} 样本）===", flush=True)
    t0 = time.time()
    for i, (cond, fold, split, f) in enumerate(targets):
        d = torch.load(f, weights_only=False)
        check_sample(audit, d, "ethucy", cond, fold, split, Path(f).parent.name, f, source_root)
        if (i + 1) % 2000 == 0:
            print(f"  ... {i+1}/{len(targets)} ({(i+1)/(time.time()-t0):.0f} samples/s)", flush=True)

    fails = audit.fail_count()
    print(f"\n=== ETH/UCY 审计完成：{len(audit.items)} 项检查，{fails} 失败 ===", flush=True)
    return audit, fails


def audit_sdd(data_root, source_root, full=False, sample_frac=0.02, seed=0):
    audit = Audit("SDD")
    print("=== 10.1 结构检查 ===", flush=True)
    with open(Path(source_root) / "original" / "sdd_train.pkl", "rb") as f:
        train_all = pickle.load(f)
    with open(Path(source_root) / "original" / "sdd_test.pkl", "rb") as f:
        test_all = pickle.load(f)

    rng_np = np.random.default_rng(2024)
    perm = rng_np.permutation(len(train_all))
    n_val = int(len(train_all) * 0.1)
    val_orig_idx = set(perm[:n_val].tolist())
    train_orig_idx = set(perm[n_val:].tolist())

    ok_counts = True
    for cond in CONDITIONS:
        for split in SPLITS:
            n = len(glob.glob(str(Path(data_root) / cond / split / "*.pt")))
            expect = {"train": len(train_all) - n_val, "val": n_val, "test": len(test_all)}[split]
            if n != expect:
                ok_counts = False
                print(f"  [FAIL] {cond}/{split}: {n} != {expect}", flush=True)
    audit.add("10.1 结构", "3 condition × 3 split 样本数", ok_counts)

    # 划分一致性：三个条件的 train/val source_index 集合完全一致
    sets = {}
    for cond in CONDITIONS:
        for split in ["train", "val"]:
            idxs = set()
            for f in glob.glob(str(Path(data_root) / cond / split / "*.pt")):
                d = torch.load(f, weights_only=False)
                idxs.add(d["source_index"])
            sets[(cond, split)] = idxs
    ok_split = (sets[("complete", "train")] == sets[("random_single", "train")] == sets[("random_block2", "train")] == train_orig_idx
                and sets[("complete", "val")] == sets[("random_single", "val")] == sets[("random_block2", "val")] == val_orig_idx)
    audit.add("10.1 结构", "train/val 划分三条件一致且与 seed=2024 对齐", ok_split)

    mpath = Path(data_root) / "manifest.json"
    ok_m = mpath.exists()
    audit.add("10.1 结构", "manifest.json 存在", ok_m)
    if ok_m:
        m = json.load(open(mpath))
        ok_fields = (m.get("version") == "missing_history_v1" and m.get("mask_seed") == 42
                     and m.get("split_seed") == 2024 and m.get("unit") == "pixel"
                     and m.get("conditions") == CONDITIONS)
        audit.add("10.1 结构", "manifest 关键字段", ok_fields)
        ok_mc = True
        for cond in CONDITIONS:
            for split in SPLITS:
                got = len(glob.glob(str(Path(data_root) / cond / split / "*.pt")))
                want = m["splits"][cond][split]["samples"]
                if got != want:
                    ok_mc = False
        audit.add("10.1 结构", "manifest 计数与磁盘一致", ok_mc)

    # 逐样本
    rng = random.Random(seed)
    print(f"=== 10.2/10.3 掩码与泄漏检查（{'全量' if full else f'抽样 {sample_frac:.0%}'}）===", flush=True)
    n_checked = 0
    for cond in CONDITIONS:
        for split in SPLITS:
            files = sorted(glob.glob(str(Path(data_root) / cond / split / "*.pt")))
            sel = files if full else rng.sample(files, max(1, int(len(files) * sample_frac)))
            for f in sel:
                d = torch.load(f, weights_only=False)
                # 基本掩码语义
                A = 1
                hm, vm = d["history_mask"], d["valid_mask"]
                rel = f"{cond}/{split}/{Path(f).name}"
                ok_shape = hm.shape == (A, 8) and vm.shape == (A, 20) and d["positions"].shape == (A, 20, 2)
                audit.add("10.2 掩码", f"形状 {rel}", ok_shape)
                audit.add("10.2 掩码", f"valid_mask 一致性 {rel}", torch.equal(vm[:, :8], hm) and bool(vm[:, 8:].all()))
                audit.add("10.2 掩码", f"缺失坐标为 0 {rel}", bool((d["positions"][:, :8][~hm] == 0).all()))
                if cond == "complete":
                    ok_c = bool(hm.all())
                elif cond == "random_single":
                    ok_c = bool(((~hm).sum(1) == 1).all()) and bool(hm[:, 6].all()) and bool(hm[:, 7].all())
                else:
                    rows_ok = True
                    for row in ~hm:
                        idx = row.nonzero().flatten().tolist()
                        if idx and idx != list(range(idx[0], idx[0] + len(idx))):
                            rows_ok = False
                    ok_c = bool(((~hm).sum(1) == 2).all()) and rows_ok and bool(hm[:, 6].all()) and bool(hm[:, 7].all())
                audit.add("10.2 掩码", f"条件语义 [{cond}] {rel}", ok_c)
                # 与源 pkl 对齐
                if split in ("train", "val"):
                    past, fut = train_all[d["source_index"]][0], train_all[d["source_index"]][1]
                else:
                    past, fut = test_all[d["source_index"]][0], test_all[d["source_index"]][1]
                src_pos = torch.from_numpy(np.concatenate([np.asarray(past), np.asarray(fut)], 0).astype(np.float32))
                ok_fut = torch.equal(d["positions"][0, 8:], src_pos[8:])
                audit.add("10.3 泄漏", f"未来帧与源一致 {rel}", ok_fut)
                ok_vis = torch.equal(d["positions"][0, :8][hm[0]], src_pos[:8][hm[0]])
                audit.add("10.3 泄漏", f"可见历史与源一致 {rel}", ok_vis)
                exp = expected_mask("sdd", None, split, d["scene_id"], d["source_index"], d.get("source_file", ""), d["focal_id"], cond, d["mask_seed"], 1)
                audit.add("10.3 泄漏/复现", f"掩码可复现推 {rel}", torch.equal(hm, exp))
                n_checked += 1
    print(f"  共检查 {n_checked} 个样本", flush=True)

    fails = audit.fail_count()
    print(f"\n=== SDD 审计完成：{len(audit.items)} 项检查，{fails} 失败 ===", flush=True)
    return audit, fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["ethucy", "sdd"], required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--source-root", required=True)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--sample-frac", type=float, default=0.02)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    t0 = time.time()
    if args.dataset == "ethucy":
        audit, fails = audit_ethucy(args.data_root, args.source_root, args.full, args.sample_frac)
        name = "ethucy"
    else:
        audit, fails = audit_sdd(args.data_root, args.source_root, args.full, args.sample_frac)
        name = "sdd"

    report = {
        "dataset": name,
        "mode": "full" if args.full else f"sample_{args.sample_frac}",
        "total_checks": len(audit.items),
        "failures": fails,
        "elapsed_s": round(time.time() - t0, 1),
        "items": audit.report(),
    }
    out = args.out or f"/tmp/audit_missing_{name}_{'full' if args.full else 'sample'}.json"
    with open(out, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(f"报告 -> {out}（failures={fails}）", flush=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
