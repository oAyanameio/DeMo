"""缺失历史数据集验收审计（v1 / v2_high 通用，对应方案 §7 与 v1 文档 §10）。

用法：
  python3 scripts/审计与校验/audit_missing_history_dataset.py --dataset ethucy \
      --data-root data/ETHUCY_missing_v1 --source-root data/ETHUCY_benchmark_v1 [--full]
  python3 scripts/审计与校验/audit_missing_history_dataset.py --dataset sdd \
      --data-root data/SDD_missing_v1 --source-root data/sdd [--full]

  v2_high（conditions 缺省时从 manifest 自动读取）：
  python3 scripts/审计与校验/audit_missing_history_dataset.py --dataset ethucy \
      --data-root data/ETHUCY_missing_v2_high --source-root data/ETHUCY_benchmark_v1 --full
  python3 scripts/审计与校验/audit_missing_history_dataset.py --dataset sdd \
      --data-root data/SDD_missing_v2_high --source-root data/sdd --full

默认快速模式：每个 (condition, fold, split, scene) 抽样检查 + 全量结构计数；
--full 对每个样本做逐样本校验（慢）。

检查内容（v2 扩展，方案 §7.1）：
  10.1 结构   样本数/文件名集合/源索引集合跨 condition 与源一致；manifest 字段；
              实际缺失率 == 名义缺失率（k/8）
  10.2 掩码   形状、valid_mask 一致性、缺失坐标为 0、条件语义
              （每行人恰缺 k 帧 ⊆ 0-5、连续条件要求唯一连续块、block6 == 0-5、
               帧 6/7 可见）
  10.3 泄漏   未来帧与源逐位一致；可见历史与源一致；掩码独立复推；
              同一源样本跨 condition 未来 12 帧逐位一致
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
from pathlib import Path

import numpy as np
import torch

OBS_LEN, PRED_LEN, SEQ_LEN = 8, 12, 20
FOLDS = ["ETH", "HOTEL", "UNIV", "ZARA1", "ZARA2"]
SPLITS = ["train", "val", "test"]
AUDIT_VERSION = "audit_missing_history_v2high_r1"

# 条件元数据：k=每行人缺失帧数；contiguous=是否要求唯一连续块
CONDITION_META = {
    "complete":      {"k": 0, "contiguous": None, "nominal_rate": 0.0},
    "random_single": {"k": 1, "contiguous": False, "nominal_rate": 0.125},
    "random_block2": {"k": 2, "contiguous": True, "nominal_rate": 0.25},
    "random_fixed3": {"k": 3, "contiguous": False, "nominal_rate": 0.375},
    "random_fixed4": {"k": 4, "contiguous": False, "nominal_rate": 0.5},
    "random_fixed5": {"k": 5, "contiguous": False, "nominal_rate": 0.625},
    "random_block3": {"k": 3, "contiguous": True, "nominal_rate": 0.375},
    "random_block4": {"k": 4, "contiguous": True, "nominal_rate": 0.5},
    "random_block6": {"k": 6, "contiguous": True, "nominal_rate": 0.75},
}
KNOWN_VERSIONS = ["missing_history_v1", "missing_history_v2_high"]


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
    elif condition == "random_block2":
        for a in range(num_agents):
            s = int(rng.integers(0, 5))
            mask[a, s] = False
            mask[a, s + 1] = False
    elif condition in ("random_fixed3", "random_fixed4", "random_fixed5"):
        k = int(condition[-1])
        for a in range(num_agents):
            ts = rng.choice(6, size=k, replace=False)
            for t in ts:
                mask[a, int(t)] = False
    elif condition in ("random_block3", "random_block4"):
        m = int(condition[-1])
        for a in range(num_agents):
            s = int(rng.integers(0, 6 - m + 1))
            mask[a, s:s + m] = False
    elif condition == "random_block6":
        mask[:, :6] = False
    else:
        raise ValueError(f"unknown condition: {condition}")
    mask[:, 6] = True
    mask[:, 7] = True
    return mask


def check_condition_semantics(hm: torch.Tensor, condition: str):
    """条件语义检查（方案 §7.1）。

    - 每个行人缺失帧数恰为 k；
    - 帧 6、7 可见（⇒ 缺失 ⊆ 帧 0-5）；
    - 连续条件：每行缺失下标构成唯一连续块（block6 恰为 0-5）；
    - 随机条件：缺失下标天然互异（布尔掩码），无需额外检查。
    """
    meta = CONDITION_META[condition]
    k = meta["k"]
    inv = ~hm
    ok = bool((inv.sum(1) == k).all())
    ok = ok and bool(hm[:, 6].all()) and bool(hm[:, 7].all())
    if ok and meta["contiguous"]:
        for row in inv:
            idx = row.nonzero().flatten().tolist()
            if k == 0:
                break
            if idx != list(range(idx[0], idx[0] + k)):
                ok = False
                break
    return ok


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

    ok_c = check_condition_semantics(hm, condition)
    det = f"k={CONDITION_META[condition]['k']}, contiguous={CONDITION_META[condition]['contiguous']}"
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

    # 掩码确定性复推
    exp = expected_mask(dataset, fold if dataset == "ethucy" else None, split,
                        d["scene_id"], d["source_index"], d.get("source_file", ""), d["focal_id"], condition, d["mask_seed"], A)
    ok_det = torch.equal(hm, exp)
    audit.add("10.3 泄漏/复现", f"掩码可复现推 {rel_path}", ok_det)


def audit_ethucy(data_root, source_root, full=False, sample_frac=0.02, seed=0, conds=None, folds=None, scenes=None):
    audit = Audit("ETH/UCY")
    data_root = Path(data_root)
    folds = folds or FOLDS
    print("=== 10.1 结构检查 ===", flush=True)

    # conditions：CLI 指定或从 manifest 读取
    mpath = data_root / "manifest.json"
    ok_m = mpath.exists()
    audit.add("10.1 结构", "manifest.json 存在", ok_m)
    m = json.load(open(mpath)) if ok_m else {}
    if conds is None:
        conds = m.get("conditions", ["complete", "random_single", "random_block2"])
    print(f"  conditions = {conds}", flush=True)

    ok_ver = m.get("version") in KNOWN_VERSIONS
    audit.add("10.1 结构", "manifest version 合法", ok_ver, str(m.get("version")))

    # 全量文件计数 + 文件名集合（快）。folds/scenes 过滤与 build 脚本对称，
    # 冒烟构建（部分场景）审计时传同样过滤；全量审计不传 = 严格全量比对。
    src_files = {}
    for fold in folds:
        for split in SPLITS:
            base = Path(source_root) / f"fold_{fold}" / split
            if not base.exists():
                continue
            for sd in sorted(base.iterdir()):
                if not sd.is_dir():
                    continue
                if scenes is not None and sd.name not in scenes:
                    continue
                src_files[(fold, split, sd.name)] = {f.name for f in sd.glob("*.pt")}

    ok_counts = True
    ok_names = True
    for cond in conds:
        for (fold, split, scene), names in src_files.items():
            out_dir = data_root / cond / f"fold_{fold}" / split / scene
            got = {f.name for f in out_dir.glob("*.pt")} if out_dir.exists() else set()
            if len(got) != len(names):
                ok_counts = False
                print(f"  [FAIL] count mismatch {cond} fold_{fold}/{split}/{scene}: {len(got)} vs src {len(names)}", flush=True)
            if got != names:
                ok_names = False
    audit.add("10.1 结构", "样本数与源一致（全部 condition/fold/split/scene）", ok_counts,
              f"scenes={len(src_files)} total_src={sum(len(v) for v in src_files.values())}")
    audit.add("10.1 结构", "文件名集合跨 condition 与源一致", ok_names)

    # 各 condition 共享同一源文件集合 ⇒ 源索引/样本数一致；额外核对 manifest 计数与实际缺失率
    if ok_m:
        ok_fields = (m.get("mask_seed") == 42 and m.get("conditions") == conds and m.get("unit") == "meter")
        audit.add("10.1 结构", "manifest 关键字段", ok_fields)
        ok_mc = True
        ok_rate = True
        for cond in conds:
            for fold in folds:
                for split in SPLITS:
                    node = m["splits"].get(cond, {}).get(f"fold_{fold}", {}).get(split)
                    if node is None:
                        continue
                    got = len(glob.glob(str(data_root / cond / f"fold_{fold}" / split / "*" / "*.pt")))
                    if got != node["samples"]:
                        ok_mc = False
                    nom = CONDITION_META[cond]["nominal_rate"]
                    if node["history_frames_total"] and abs(node["actual_missing_rate"] - nom) > 5e-7:
                        ok_rate = False
                        print(f"  [FAIL] nominal rate {cond} fold_{fold}/{split}: "
                              f"actual {node['actual_missing_rate']} != nominal {nom}", flush=True)
        audit.add("10.1 结构", "manifest 计数与磁盘一致", ok_mc)
        audit.add("10.1 结构", "实际缺失率 == 名义缺失率（全部 condition/fold/split）", ok_rate)
        # 有效行人数跨 condition 一致（同一源样本集 ⇒ agents_total 必须相同）
        ok_agents = True
        for fold in folds:
            for split in SPLITS:
                ref_a = None
                for cond in conds:
                    node = m["splits"].get(cond, {}).get(f"fold_{fold}", {}).get(split)
                    if node is None:
                        continue
                    if ref_a is None:
                        ref_a = node["agents_total"]
                    elif node["agents_total"] != ref_a:
                        ok_agents = False
                        print(f"  [FAIL] agents_total mismatch fold_{fold}/{split}: {cond} {node['agents_total']} != {ref_a}", flush=True)
        audit.add("10.1 结构", "有效行人数跨 condition 一致", ok_agents)

    # 逐样本（抽样或全量）
    rng = random.Random(seed)
    if full:
        targets = []
        for cond in conds:
            for fold in folds:
                for split in SPLITS:
                    for f in sorted(glob.glob(str(data_root / cond / f"fold_{fold}" / split / "*" / "*.pt"))):
                        targets.append((cond, fold, split, f))
    else:
        targets = []
        for cond in conds:
            for fold in folds:
                for split in SPLITS:
                    split_base = data_root / cond / f"fold_{fold}" / split
                    if not split_base.exists():
                        continue
                    for sd in sorted(split_base.iterdir()):
                        files = sorted(sd.glob("*.pt"))
                        k = max(1, int(len(files) * sample_frac))
                        for f in rng.sample(files, k):
                            targets.append((cond, fold, split, str(f)))

    print(f"=== 10.2/10.3 掩码与泄漏检查（{'全量' if full else f'抽样 {sample_frac:.0%}'}，{len(targets)} 样本）===", flush=True)
    t0 = time.time()
    for i, (cond, fold, split, f) in enumerate(targets):
        d = torch.load(f, weights_only=False)
        check_sample(audit, d, "ethucy", cond, fold, split, Path(f).parent.name, f, source_root)
        if (i + 1) % 5000 == 0:
            print(f"  ... {i+1}/{len(targets)} ({(i+1)/(time.time()-t0):.0f} samples/s)", flush=True)

    # 跨 condition 未来一致性：同一源文件在不同 condition 下未来 12 帧逐位一致
    print("=== 10.3 跨 condition 未来一致性 ===", flush=True)
    n_x = 0
    ok_x = True
    for (fold, split, scene), names in sorted(src_files.items()):
        pool = sorted(names)
        sel = pool if full else rng.sample(pool, max(1, int(len(pool) * sample_frac)))
        for name in sel:
            ref = None
            for cond in conds:
                p = data_root / cond / f"fold_{fold}" / split / scene / name
                if not p.exists():
                    continue
                fut = torch.load(p, weights_only=False)["positions"][:, 8:]
                if ref is None:
                    ref = (cond, fut)
                elif not torch.equal(ref[1], fut):
                    ok_x = False
                    print(f"  [FAIL] future mismatch {name}: {ref[0]} vs {cond} (fold_{fold}/{split}/{scene})", flush=True)
                n_x += 1
    audit.add("10.3 泄漏", "同一源样本跨 condition 未来 12 帧逐位一致", ok_x, f"{n_x} 加载次")

    fails = audit.fail_count()
    print(f"\n=== ETH/UCY 审计完成：{len(audit.items)} 项检查，{fails} 失败 ===", flush=True)
    return audit, fails


def audit_sdd(data_root, source_root, full=False, sample_frac=0.02, seed=0, conds=None):
    audit = Audit("SDD")
    data_root = Path(data_root)
    print("=== 10.1 结构检查 ===", flush=True)
    with open(Path(source_root) / "original" / "sdd_train.pkl", "rb") as f:
        train_all = pickle.load(f)
    with open(Path(source_root) / "original" / "sdd_test.pkl", "rb") as f:
        test_all = pickle.load(f)

    mpath = data_root / "manifest.json"
    ok_m = mpath.exists()
    audit.add("10.1 结构", "manifest.json 存在", ok_m)
    m = json.load(open(mpath)) if ok_m else {}
    if conds is None:
        conds = m.get("conditions", ["complete", "random_single", "random_block2"])
    print(f"  conditions = {conds}", flush=True)

    rng_np = np.random.default_rng(2024)
    perm = rng_np.permutation(len(train_all))
    n_val = int(len(train_all) * 0.1)
    val_orig_idx = set(perm[:n_val].tolist())
    train_orig_idx = set(perm[n_val:].tolist())

    ok_counts = True
    for cond in conds:
        for split in SPLITS:
            n = len(glob.glob(str(data_root / cond / split / "*.pt")))
            expect = {"train": len(train_all) - n_val, "val": n_val, "test": len(test_all)}[split]
            if n != expect:
                ok_counts = False
                print(f"  [FAIL] {cond}/{split}: {n} != {expect}", flush=True)
    audit.add("10.1 结构", "全部 condition × 3 split 样本数", ok_counts)

    # 划分一致性：所有 condition 的 train/val source_index 集合完全一致且与 seed=2024 对齐
    sets = {}
    for cond in conds:
        for split in ["train", "val"]:
            idxs = set()
            for f in glob.glob(str(data_root / cond / split / "*.pt")):
                d = torch.load(f, weights_only=False)
                idxs.add(d["source_index"])
            sets[(cond, split)] = idxs
    ok_split = all(sets[(c, "train")] == train_orig_idx for c in conds) and \
               all(sets[(c, "val")] == val_orig_idx for c in conds)
    audit.add("10.1 结构", "train/val 划分全 condition 一致且与 seed=2024 对齐", ok_split)
    # 源索引集合跨 condition 一致（含 test）
    ok_idx = all(sets[(c, "train")] == sets[(conds[0], "train")] for c in conds) and \
             all(sets[(c, "val")] == sets[(conds[0], "val")] for c in conds)
    audit.add("10.1 结构", "源索引集合跨 condition 一致", ok_idx)

    if ok_m:
        ok_ver = m.get("version") in KNOWN_VERSIONS
        audit.add("10.1 结构", "manifest version 合法", ok_ver, str(m.get("version")))
        ok_fields = (m.get("mask_seed") == 42 and m.get("split_seed") == 2024
                     and m.get("unit") == "pixel" and m.get("conditions") == conds)
        audit.add("10.1 结构", "manifest 关键字段", ok_fields)
        ok_mc = True
        ok_rate = True
        for cond in conds:
            for split in SPLITS:
                node = m["splits"].get(cond, {}).get(split)
                if node is None:
                    continue
                got = len(glob.glob(str(data_root / cond / split / "*.pt")))
                if got != node["samples"]:
                    ok_mc = False
                nom = CONDITION_META[cond]["nominal_rate"]
                if node["history_frames_total"] and abs(node["actual_missing_rate"] - nom) > 5e-7:
                    ok_rate = False
                    print(f"  [FAIL] nominal rate {cond}/{split}: actual {node['actual_missing_rate']} != {nom}", flush=True)
        audit.add("10.1 结构", "manifest 计数与磁盘一致", ok_mc)
        audit.add("10.1 结构", "实际缺失率 == 名义缺失率", ok_rate)
        # 有效行人数跨 condition 一致（SDD 恒为 A=1 ⇒ agents_total == samples）
        ok_agents = True
        for split in SPLITS:
            ref_a = None
            for cond in conds:
                node = m["splits"].get(cond, {}).get(split)
                if node is None:
                    continue
                if ref_a is None:
                    ref_a = node["agents_total"]
                elif node["agents_total"] != ref_a:
                    ok_agents = False
        audit.add("10.1 结构", "有效行人数跨 condition 一致", ok_agents)

    # 逐样本
    rng = random.Random(seed)
    print(f"=== 10.2/10.3 掩码与泄漏检查（{'全量' if full else f'抽样 {sample_frac:.0%}'}）===", flush=True)
    n_checked = 0
    for cond in conds:
        for split in SPLITS:
            files = sorted(glob.glob(str(data_root / cond / split / "*.pt")))
            sel = files if full else rng.sample(files, max(1, int(len(files) * sample_frac)))
            for f in sel:
                d = torch.load(f, weights_only=False)
                hm, vm = d["history_mask"], d["valid_mask"]
                rel = f"{cond}/{split}/{Path(f).name}"
                ok_shape = hm.shape == (1, 8) and vm.shape == (1, 20) and d["positions"].shape == (1, 20, 2)
                audit.add("10.2 掩码", f"形状 {rel}", ok_shape)
                audit.add("10.2 掩码", f"valid_mask 一致性 {rel}", torch.equal(vm[:, :8], hm) and bool(vm[:, 8:].all()))
                audit.add("10.2 掩码", f"缺失坐标为 0 {rel}", bool((d["positions"][:, :8][~hm] == 0).all()))
                audit.add("10.2 掩码", f"条件语义 [{cond}] {rel}", check_condition_semantics(hm, cond))
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

    # 跨 condition 未来一致性（按 sample_index 文件名对齐）
    print("=== 10.3 跨 condition 未来一致性 ===", flush=True)
    n_x = 0
    ok_x = True
    for split in SPLITS:
        names = sorted(p.name for p in (data_root / conds[0] / split).glob("*.pt"))
        pool = names if full else rng.sample(names, max(1, int(len(names) * sample_frac)))
        for name in pool:
            ref = None
            for cond in conds:
                p = data_root / cond / split / name
                if not p.exists():
                    continue
                fut = torch.load(p, weights_only=False)["positions"][:, 8:]
                if ref is None:
                    ref = (cond, fut)
                elif not torch.equal(ref[1], fut):
                    ok_x = False
                    print(f"  [FAIL] future mismatch {split}/{name}: {ref[0]} vs {cond}", flush=True)
                n_x += 1
    audit.add("10.3 泄漏", "同一源样本跨 condition 未来 12 帧逐位一致", ok_x, f"{n_x} 加载次")

    fails = audit.fail_count()
    print(f"\n=== SDD 审计完成：{len(audit.items)} 项检查，{fails} 失败 ===", flush=True)
    return audit, fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["ethucy", "sdd"], required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--source-root", required=True)
    ap.add_argument("--conditions", nargs="+", default=None, choices=list(CONDITION_META),
                    help="缺省从 manifest.json 读取")
    ap.add_argument("--folds", nargs="+", default=None, help="ethucy only：与 build --folds 对称（冒烟范围审计）")
    ap.add_argument("--scenes", nargs="+", default=None, help="ethucy only：与 build --scenes 对称（冒烟范围审计）")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--sample-frac", type=float, default=0.02)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    t0 = time.time()
    if args.dataset == "ethucy":
        audit, fails = audit_ethucy(args.data_root, args.source_root, args.full, args.sample_frac,
                                    conds=args.conditions, folds=args.folds, scenes=args.scenes)
        name = "ethucy"
    else:
        audit, fails = audit_sdd(args.data_root, args.source_root, args.full, args.sample_frac, conds=args.conditions)
        name = "sdd"

    report = {
        "dataset": name,
        "audit_script": AUDIT_VERSION,
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
