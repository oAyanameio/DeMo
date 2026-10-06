"""C8-0 ET/Anchor 离线诊断（不训练、不改模型、不改 checkpoint）。

按 docs/research/C8_ET输出空间Anchor实施规范.md §3-§6 实施：
  1) 每场景仅用 Mixed-train（Easy/train + Hard/train）未来 GT 建 ET/SVD 基
     （k=4 主配置 + k=8 消融）与 anchor 库（M=64 主配置 + M=32 消融，seed=2024）。
     DeMo 环境无 sklearn -> 使用 torch Lloyd k-means++ 全批实现，算法名如实写入产物。
  2) 加载正式 best-val checkpoint（epoch=*.ckpt，禁止 last.ckpt），Mixed-test 前向一次，
     对同一批预测做纯后处理变体（pi 原样保留）：
       m0 / etproj_k4 / etproj_k8 / anchor_hard_* / interp_*（缺失样本 λ>0，
       完整样本 λ=0 且逐位等于 M0）/ randanchor_/randinterp_ 随机 anchor 对照。
  3) 同时报告候选生成质量（minADE20/minFDE20/oracle-best ET distance/spread/residual）
     与概率排序质量（ADE@1/FDE@1/MR/top1_et/prob@best），按
     valid_count∈{1-2,3-4,5-6,7-8} 分层 + 完整层（valid=8）保护检查。
  4) 输出 outputs/c8_et_anchor/diagnostic/{scene}.json + {scene}_persample.npz
     + summary.json（含 Go/No-Go 判定表）。不写入 runner VARIANTS，不触碰 M0 代码。

用法（仓库根目录）：
  PYTHONNOUSERSITE=1 PYTHONPATH=. python scripts/结果分析/c8_et_anchor_diagnostic.py \
      --device cuda:1
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.datamodule.trajimpute_dataset import (  # noqa: E402
    TrajGapDataset, trajimpute_collate_fn, load_trajimpute_pkl,
    _last_valid_indices, _prev_valid_index,
)
from src.evaluation.trajimpute_direct import evaluate_predictions  # noqa: E402

SCENES = ("ETH-M", "HOTEL-M", "UNIV-M", "ZARA1-M", "ZARA2-M")
GPU_MAP = {"ETH-M": "gpu1", "HOTEL-M": "gpu1", "UNIV-M": "gpu1",
           "ZARA1-M": "gpu3", "ZARA2-M": "gpu3"}
TIERS = ((1, 2), (3, 4), (5, 6), (7, 8))
LAMBDAS = (0.10, 0.25, 0.50)
LAM_NAME = {0.10: "lam0.10", 0.25: "lam0.25", 0.50: "lam0.50"}
K8_VARIANTS = ("etproj_k8", "anchor_hard_k8_M64", "interp_k8_M64_lam0.25")
METRIC_KEYS = ("minADE20", "minFDE20", "ADE@1", "FDE@1", "MR", "best_et", "prob_at_best")


def get_git_revision():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "unknown"


def build_model(K: int):
    from src.model.trajectory_forecaster import TrajectoryForecaster
    return TrajectoryForecaster(
        embed_dim=128, future_steps=12, num_heads=8, mlp_ratio=4.0,
        qkv_bias=False, drop_path=0.2, num_actor_types=1,
        num_modes=K, bimamba=False, dt=0.4, obs_len=8,
    )


def load_checkpoint(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    cleaned = {k[len("net."):]: v for k, v in state.items() if k.startswith("net.")}
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    assert not missing and not unexpected, (
        f"checkpoint 与纯 M0 不匹配: missing={missing[:5]} unexpected={unexpected[:5]}")
    return ckpt.get("epoch", None)


def find_ckpt(ckpt_root, scene):
    run_dir = Path(ckpt_root) / f"m0_{GPU_MAP[scene]}" / f"M0_{scene}_mixed-direct_seed2024_uni"
    cands = sorted(run_dir.glob("train/checkpoints/epoch=*.ckpt"))
    if not cands:
        cands = sorted(Path(ckpt_root).rglob(
            f"M0_{scene}_*/train/checkpoints/epoch=*.ckpt"))
    assert cands, f"best-val checkpoint not found for {scene} ({run_dir})"
    return cands[-1], len(cands)


# ---------------------------------------------------------------- ET 基（仅 Mixed-train）
def local_futures_from_pkl(path):
    """所有行（每行都会成为一次 focal）的未来 GT -> focal 局部系 [N,12,2] float32。

    与 build_sample 的 focal 局部变换逐位等价：origin=最后有效历史位置（世界系），
    theta=最后两个有效观测连线（跨缺口，退化置 0），float64 计算后转 float32。
    """
    obs, pred, fv, _ = load_trajimpute_pkl(path)
    N = obs.shape[0]
    ar = torch.arange(N)
    t_last = _last_valid_indices(fv)                 # [N]
    prev = _prev_valid_index(fv)                     # [N,8]
    s_last_raw = prev[ar, t_last]
    o64, p64 = obs.double(), pred.double()
    origin = o64[ar, t_last]                         # [N,2] 有限（最后有效帧）
    d = o64[ar, t_last] - o64[ar, s_last_raw.clamp(min=0)]
    ok = (s_last_raw >= 0) & (torch.norm(d, dim=-1) >= 1e-4)
    theta = torch.where(ok, torch.atan2(d[:, 1], d[:, 0]),
                        torch.zeros(N, dtype=torch.float64))
    cos, sin = torch.cos(theta).unsqueeze(1), torch.sin(theta).unsqueeze(1)
    rel = p64 - origin.unsqueeze(1)                  # [N,12,2]
    x = rel[..., 0] * cos + rel[..., 1] * sin
    y = -rel[..., 0] * sin + rel[..., 1] * cos
    fut = torch.stack([x, y], dim=-1)
    assert torch.isfinite(fut).all(), f"non-finite local future: {path}"
    return fut.float()


def verify_against_dataset(data_root, scene, F, nE, n_verify, seed):
    """向量化局部未来 vs TrajGapDataset 真实 __getitem__ 输出，逐样本对账。"""
    ds = TrajGapDataset(data_root, scene, "train")
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(ds), size=min(n_verify, len(ds)), replace=False)
    worst = 0.0
    for i in idxs:
        s = ds[int(i)]
        tid = int(s["track_id"])
        fut = F["easy"] if int(i) < nE else F["hard"]
        t = s["target"]
        if t.dim() == 3:
            t = t[0]
        worst = max(worst, float((fut[tid].double() - t.double()).abs().max()))
    assert worst < 5e-5, f"{scene}: 向量化局部未来与 dataset 输出不一致 (max|d|={worst})"
    return len(idxs), worst


def kmeans_torch(X, M, seed, max_iter=200):
    """Lloyd k-means（k-means++ 初始化，全批更新，确定性 seed）。"""
    X = X.float().cpu()
    N, D = X.shape
    g = torch.Generator().manual_seed(int(seed))
    centers = torch.empty(M, D)
    centers[0] = X[torch.randint(N, (1,), generator=g)]
    d2 = ((X - centers[0]) ** 2).sum(-1)
    for m in range(1, M):
        p = (d2 / d2.sum().clamp(min=1e-12)).double()
        centers[m] = X[torch.multinomial(p, 1, generator=g)]
        d2 = torch.minimum(d2, ((X - centers[m]) ** 2).sum(-1))
    assign, it = None, 0
    for it in range(1, max_iter + 1):
        dist = torch.cdist(X, centers)
        new_assign = dist.argmin(-1)
        if assign is not None and torch.equal(new_assign, assign):
            break
        assign = new_assign
        sums = torch.zeros(M, D).index_add_(0, assign, X)
        cnt = torch.bincount(assign, minlength=M).float().view(M, 1)
        centers = torch.where(cnt > 0, sums / cnt.clamp(min=1), centers)
    dist = torch.cdist(X, centers)
    assign = dist.argmin(-1)
    counts = torch.bincount(assign, minlength=M)
    inertia = float(dist.min(-1).values.sum())
    return centers, counts, inertia, it


def build_or_load_basis(args, scene):
    basis_path = Path(args.basis_root) / f"{scene}.pt"
    if basis_path.exists() and not args.force_rebuild_basis:
        blob = torch.load(basis_path, map_location="cpu")
        assert blob.get("source_split") == "Mixed-train"
        print(f"[basis] 复用已有基: {basis_path}")
        return blob, basis_path
    t0 = time.time()
    F = {}
    for diff, key in (("Easy", "easy"), ("Hard", "hard")):
        p = Path(args.data_root) / scene / diff / "data_train.pkl"
        F[key] = local_futures_from_pkl(p)
    nE = F["easy"].shape[0]
    n_verified, worst = verify_against_dataset(
        args.data_root, scene, F, nE, args.verify_samples, seed=123 + SCENES.index(scene))
    Y = torch.cat([F["easy"], F["hard"]], dim=0).reshape(-1, 24)  # [N,24]
    N = Y.shape[0]
    assert Y.shape[1] == 24 and torch.isfinite(Y).all()
    Y64 = Y.double()
    mean = Y64.mean(dim=0)
    Xc = Y64 - mean
    _, S, Vt = torch.linalg.svd(Xc, full_matrices=False)
    ev = (S ** 2) / (S ** 2).sum()
    blob = {
        "scene": scene, "source_split": "Mixed-train",
        "difficulties": ["Easy", "Hard"], "n_train_rows": int(N),
        "n_easy": int(nE), "n_hard": int(N - nE),
        "git_revision": get_git_revision(),
        "created": datetime.now().isoformat(timespec="seconds"),
        "vectorized_local_future_verified": {
            "n_samples": n_verified, "max_abs_diff": worst},
    }
    for k, Ms in ((4, (64, 32)), (8, (64,))):
        B = Vt[:k]
        c_train = (Xc @ B.T).float()
        anchors = {}
        for M in Ms:
            centers, counts, inertia, iters = kmeans_torch(
                c_train, M, seed=args.anchor_seed,
                max_iter=args.kmeans_max_iters)
            anchors[str(M)] = {
                "anchors": centers, "anchor_counts": counts,
                "inertia": inertia, "iters": iters,
                "anchor_algorithm": (
                    "torch_lloyd_kmeanspp_fullbatch "
                    "(sklearn unavailable in DeMo env; spec §3.3 torch fallback)"),
                "anchor_seed": int(args.anchor_seed), "M": int(M),
            }
        blob[f"k{k}"] = {
            "k": k, "mean": mean.float(), "basis": B.float(),
            "singular_values": S.float(), "explained_variance": ev.float(),
            "ev_topk": float(ev[:k].sum()), "anchors": anchors,
        }
        print(f"[basis] {scene} k={k} EV@k={float(ev[:k].sum()):.4f} "
              f"anchors={ {m: round(a['inertia'], 2) for m, a in anchors.items()} } "
              f"inertia iters={ {m: a['iters'] for m, a in anchors.items()} }")
    basis_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, basis_path)
    print(f"[basis] {scene} N_train={N} 建基完成 ({time.time()-t0:.1f}s) -> {basis_path}")
    return blob, basis_path


# ---------------------------------------------------------------- 变体后处理
def variant_transforms(y, gate, packs, rand_idx):
    """y [B,K,24] cpu f32；gate [B] bool（True=有缺失）。

    返回 (dict name->y_var, dict name->identity_bool)。门控变体对完整样本逐位复制 M0
    （torch.where 选择原张量比特，非重投影）。
    """
    B, K, _ = y.shape
    g = gate.view(B, 1, 1)
    out = {"m0": y}
    ident = {}
    p4, p8 = packs[4], packs[8]

    def proj(p, c):
        return p["mean"][None, None, :] + c @ p["basis"]

    # C8-0A ET 投影（全样本，纯诊断）
    c4 = (y - p4["mean"]) @ p4["basis"].T
    c8 = (y - p8["mean"]) @ p8["basis"].T
    out["etproj_k4"] = proj(p4, c4)
    out["etproj_k8"] = proj(p8, c8)

    # C8-0B 最近 anchor 硬替换（全样本，纯诊断）
    dist4_64 = torch.cdist(c4, p4["anchors"]["64"])
    near4_64 = dist4_64.argmin(-1)
    c_a4_64 = p4["anchors"]["64"][near4_64]
    out["anchor_hard_k4_M64"] = proj(p4, c_a4_64)
    near4_32 = torch.cdist(c4, p4["anchors"]["32"]).argmin(-1)
    out["anchor_hard_k4_M32"] = proj(p4, p4["anchors"]["32"][near4_32])
    near8 = torch.cdist(c8, p8["anchors"]["64"]).argmin(-1)
    out["anchor_hard_k8_M64"] = proj(p8, p8["anchors"]["64"][near8])

    # C8-0C 固定 λ 插值（缺失样本生效，完整样本 λ=0 逐位保持）
    for lam in LAMBDAS:
        c_new = (1 - lam) * c4 + lam * c_a4_64
        y_t = proj(p4, c_new)
        y_v = torch.where(g, y_t, y)
        out[f"interp_k4_M64_{LAM_NAME[lam]}"] = y_v
        ident[f"interp_k4_M64_{LAM_NAME[lam]}"] = torch.equal(y_v[~gate], y[~gate])
    c_new = 0.75 * c4 + 0.25 * p4["anchors"]["32"][near4_32]
    y_v = torch.where(g, proj(p4, c_new), y)
    out["interp_k4_M32_lam0.25"] = y_v
    ident["interp_k4_M32_lam0.25"] = torch.equal(y_v[~gate], y[~gate])
    c8_new = 0.75 * c8 + 0.25 * p8["anchors"]["64"][near8]
    y_v = torch.where(g, proj(p8, c8_new), y)
    out["interp_k8_M64_lam0.25"] = y_v
    ident["interp_k8_M64_lam0.25"] = torch.equal(y_v[~gate], y[~gate])

    # 随机 anchor 对照（同一 anchor 库、随机选择；硬替换 + λ=0.25 插值）
    c_r = p4["anchors"]["64"][rand_idx]                    # [B,K,k]
    out["randanchor_hard_k4_M64"] = proj(p4, c_r)
    y_v = torch.where(g, proj(p4, 0.75 * c4 + 0.25 * c_r), y)
    out["randinterp_k4_M64_lam0.25"] = y_v
    ident["randinterp_k4_M64_lam0.25"] = torch.equal(y_v[~gate], y[~gate])
    return out, ident


VARIANT_NAMES = (
    ["m0", "etproj_k4", "etproj_k8",
     "anchor_hard_k4_M64", "anchor_hard_k4_M32", "anchor_hard_k8_M64"]
    + [f"interp_k4_M64_{LAM_NAME[l]}" for l in LAMBDAS]
    + ["interp_k4_M32_lam0.25", "interp_k8_M64_lam0.25",
       "randanchor_hard_k4_M64", "randinterp_k4_M64_lam0.25"]
)
GATED_VARIANTS = {v for v in VARIANT_NAMES if v.startswith(("interp", "randinterp"))}


def corr_np(a, b):
    if len(a) < 3 or float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return None
    return round(float(np.corrcoef(a, b)[0, 1]), 4)


def aggregate(arrs, vc):
    def st(mask):
        n = int(mask.sum())
        o: dict = {"n": n}
        for key, a in arrs.items():
            o[key] = round(float(a[mask].mean()), 6) if n else 0.0
        return o
    out = {"overall": st(np.ones(len(vc), dtype=bool)), "tiers": {}}
    for lo, hi in TIERS:
        mask = (vc >= lo) & (vc <= hi)
        if mask.any():
            out["tiers"][f"valid{lo}-{hi}"] = st(mask)
    mask = vc == 8
    if mask.any():
        out["complete_valid8"] = st(mask)
    return out


@torch.no_grad()
def run_scene(args, scene, packs):
    device = args.device
    ckpt, n_ckpt = find_ckpt(args.ckpt_root, scene)
    model = build_model(20)
    epoch = load_checkpoint(model, ckpt)
    model = model.to(device).eval()

    dataset = TrajGapDataset(args.data_root, scene, "test")
    n_test_total = len(dataset)
    stride = 1
    eval_dataset = dataset
    if args.max_test and n_test_total > args.max_test:
        stride = -(-n_test_total // args.max_test)   # ceil，等步抽样防 Easy/Hard 分块偏置
        from torch.utils.data import Subset
        eval_dataset = Subset(dataset, list(range(0, n_test_total, stride)))
    loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=trajimpute_collate_fn)

    p4, p8 = packs[4], packs[8]
    mean4, b4 = p4["mean"], p4["basis"]
    mean8, b8 = p8["mean"], p8["basis"]
    rng = np.random.default_rng(args.random_anchor_seed)

    var_arrays = {v: {k: [] for k in METRIC_KEYS} for v in VARIANT_NAMES}
    shared = {k: [] for k in (
        "valid_count", "missing_count", "best_anchor_distance", "spread_k4",
        "spread_k8", "resid_mean_k4", "resid_best_k4", "top1_et_k4", "top1_et_k8",
        "et_gap_k4", "original_top1_index", "best_mode_index_m0",
        "best_mode_index_interp025", "best_mode_index_anchorhard64",
        "best_et_k8_m0")}
    best_et_k8_extra = {v: [] for v in K8_VARIANTS}
    identity_flags = {}
    n_batches = 0
    t0 = time.time()
    for batch in loader:
        batch_dev = {k: (v.to(device) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
        out = model(batch_dev)
        pred = out["new_y_hat"] if out.get("new_y_hat") is not None else out["y_hat"]
        prob = out["new_pi"] if out.get("new_pi") is not None else out["pi"]
        pred = pred[..., :2].float()
        prob = prob.float()
        B = pred.shape[0]
        assert pred.shape == (B, 20, 12, 2), f"pred shape {tuple(pred.shape)}"
        assert prob.shape == (B, 20), f"prob shape {tuple(prob.shape)}"
        if not torch.allclose(prob.sum(-1), prob.new_ones(B), atol=1e-2):
            prob = torch.softmax(prob, dim=-1)
        pred, prob = pred.cpu(), prob.cpu()
        target = batch_dev["target"][:, 0].float().cpu()           # [B,12,2]
        valid = batch["x_valid_mask"][:, 0]                        # [B,8]
        vc = valid.sum(-1)
        gate = vc < 8
        y = pred.reshape(B, 20, 24)
        gt_flat = target.reshape(B, 24)
        assert torch.isfinite(gt_flat).all()
        rand_idx = torch.from_numpy(rng.integers(0, 64, size=(B, 20))).long()

        variants, ident = variant_transforms(y, gate, packs, rand_idx)
        for v, flag in ident.items():
            identity_flags.setdefault(v, True)
            identity_flags[v] = identity_flags[v] and bool(flag)

        cg4 = (gt_flat - mean4) @ b4.T
        cg8 = (gt_flat - mean8) @ b8.T
        for name, yv in variants.items():
            mets = evaluate_predictions(yv.view(B, 20, 12, 2), prob, target)
            c4v = (yv - mean4) @ b4.T
            d4 = (c4v - cg4[:, None, :]).norm(dim=-1)              # [B,K]
            best4 = d4.argmin(-1)                                   # [B]
            be = d4.gather(1, best4[:, None]).squeeze(1)
            pb = prob.gather(1, best4[:, None]).squeeze(1)
            var_arrays[name]["minADE20"].append(mets["minADE_K"].numpy())
            var_arrays[name]["minFDE20"].append(mets["minFDE_K"].numpy())
            var_arrays[name]["ADE@1"].append(mets["ADE@1"].numpy())
            var_arrays[name]["FDE@1"].append(mets["FDE@1"].numpy())
            var_arrays[name]["MR"].append(mets["MR"].numpy())
            var_arrays[name]["best_et"].append(be.numpy())
            var_arrays[name]["prob_at_best"].append(pb.numpy())
            if name in K8_VARIANTS:
                c8v = (yv - mean8) @ b8.T
                d8 = (c8v - cg8[:, None, :]).norm(dim=-1)
                best_et_k8_extra[name].append(
                    d8.gather(1, d8.argmin(-1)[:, None]).squeeze(1).numpy())

        # 原始 M0 的 ET 诊断量
        c4_m0 = (y - mean4) @ b4.T
        c8_m0 = (y - mean8) @ b8.T
        d4_m0 = (c4_m0 - cg4[:, None, :]).norm(dim=-1)
        d8_m0 = (c8_m0 - cg8[:, None, :]).norm(dim=-1)
        best4_m0 = d4_m0.argmin(-1)
        top1 = prob.argmax(-1)
        pd4 = torch.cdist(c4_m0, c4_m0)
        pd8 = torch.cdist(c8_m0, c8_m0)
        spread4 = (pd4.sum((1, 2)) - pd4.diagonal(dim1=1, dim2=2).sum(1)) / (20 * 19)
        spread8 = (pd8.sum((1, 2)) - pd8.diagonal(dim1=1, dim2=2).sum(1)) / (20 * 19)
        resid = (y - (mean4 + c4_m0 @ b4)).norm(dim=-1) / (
            (y - mean4).norm(dim=-1) + 1e-8)                        # [B,K]
        dist_a = torch.cdist(c4_m0, p4["anchors"]["64"])            # [B,K,64]
        bad = dist_a.gather(
            1, best4_m0[:, None, None].expand(-1, 1, 64)).squeeze(1).min(-1).values
        shared["valid_count"].append(vc.numpy())
        shared["missing_count"].append(batch["missing_count"].numpy())
        shared["best_anchor_distance"].append(bad.numpy())
        shared["spread_k4"].append(spread4.numpy())
        shared["spread_k8"].append(spread8.numpy())
        shared["resid_mean_k4"].append(resid.mean(-1).numpy())
        shared["resid_best_k4"].append(resid.gather(1, best4_m0[:, None]).squeeze(1).numpy())
        shared["top1_et_k4"].append(d4_m0.gather(1, top1[:, None]).squeeze(1).numpy())
        shared["top1_et_k8"].append(d8_m0.gather(1, top1[:, None]).squeeze(1).numpy())
        shared["et_gap_k4"].append(
            (d4_m0.gather(1, top1[:, None]) - d4_m0.gather(1, best4_m0[:, None])).squeeze(1).numpy())
        shared["original_top1_index"].append(top1.numpy())
        shared["best_mode_index_m0"].append(best4_m0.numpy())
        shared["best_mode_index_interp025"].append(
            (((variants["interp_k4_M64_lam0.25"] - mean4) @ b4.T
              - cg4[:, None, :]).norm(dim=-1).argmin(-1)).numpy())
        shared["best_mode_index_anchorhard64"].append(
            (((variants["anchor_hard_k4_M64"] - mean4) @ b4.T
              - cg4[:, None, :]).norm(dim=-1).argmin(-1)).numpy())
        shared["best_et_k8_m0"].append(
            d8_m0.gather(1, d8_m0.argmin(-1)[:, None]).squeeze(1).numpy())

        n_batches += 1
        if n_batches % 100 == 0:
            print(f"[fwd] {scene} batch={n_batches} ({time.time()-t0:.0f}s)", flush=True)
        if args.max_batches is not None and n_batches >= args.max_batches:
            break

    var_arrays = {v: {k: np.concatenate(lst).astype(np.float32)
                      for k, lst in d.items()} for v, d in var_arrays.items()}
    for v in K8_VARIANTS:
        var_arrays[v]["best_et_k8"] = np.concatenate(best_et_k8_extra[v]).astype(np.float32)
    shared = {k: np.concatenate(lst) for k, lst in shared.items()}
    vc_all = shared["valid_count"].astype(int)

    variants_json = {}
    for v in VARIANT_NAMES:
        entry = aggregate(var_arrays[v], vc_all)
        entry["corr_best_et_minFDE"] = corr_np(var_arrays[v]["best_et"],
                                               var_arrays[v]["minFDE20"])
        variants_json[v] = entry
    et_diag_arrays = {k: shared[k].astype(np.float32) for k in (
        "best_anchor_distance", "spread_k4", "spread_k8", "resid_mean_k4",
        "resid_best_k4", "top1_et_k4", "top1_et_k8", "et_gap_k4", "best_et_k8_m0")}
    et_diag_arrays["best_et_k4"] = var_arrays["m0"]["best_et"]
    et_diag_arrays["prob_at_best"] = var_arrays["m0"]["prob_at_best"]
    et_diag = aggregate(et_diag_arrays, vc_all)
    et_diag["corr_best_et_minFDE"] = corr_np(
        var_arrays["m0"]["best_et"], var_arrays["m0"]["minFDE20"])

    return {
        "var_arrays": var_arrays, "shared": shared, "vc": vc_all,
        "variants_json": variants_json, "et_diag": et_diag,
        "identity_flags": identity_flags,
        "ckpt": str(ckpt), "epoch": epoch, "n_ckpt_candidates": n_ckpt,
        "n_test": int(len(vc_all)),
        "n_test_total": int(n_test_total), "test_stride": int(stride),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/home/lbh/TrajImpute/dataset/TrajImpute")
    ap.add_argument("--ckpt-root", default="outputs/trajgap_bench_seed2024")
    ap.add_argument("--scenes", nargs="+", default=list(SCENES))
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--anchor-seed", type=int, default=2024)
    ap.add_argument("--random-anchor-seed", type=int, default=2024)
    ap.add_argument("--verify-samples", type=int, default=25)
    ap.add_argument("--kmeans-max-iters", type=int, default=200)
    ap.add_argument("--basis-root", default="outputs/c8_et_anchor/et_basis")
    ap.add_argument("--output-root", default="outputs/c8_et_anchor/diagnostic")
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--max-test", type=int, default=30000,
                    help="每场景最多评估样本数（等步抽样），0=全量；与 Step-1 诊断同规模")
    ap.add_argument("--force-rebuild-basis", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    for scene in args.scenes:
        if (out_root / f"{scene}.json").exists():
            raise SystemExit(f"拒绝覆盖已有产物: {out_root / f'{scene}.json'}")
    if (out_root / "summary.json").exists():
        raise SystemExit(f"拒绝覆盖已有产物: {out_root / 'summary.json'}")

    scene_results, scene_arrays = {}, {}
    for scene in args.scenes:
        t0 = time.time()
        blob, basis_path = build_or_load_basis(args, scene)
        packs = {}
        for k in (4, 8):
            entry = blob[f"k{k}"]
            packs[k] = {
                "mean": entry["mean"].float(), "basis": entry["basis"].float(),
                "anchors": {M: a["anchors"].float()
                            for M, a in entry["anchors"].items()},
            }
        r = run_scene(args, scene, packs)
        np.savez_compressed(
            out_root / f"{scene}_persample.npz",
            **{f"{v}__{m}": a for v, d in r["var_arrays"].items()
               for m, a in d.items()},
            **r["shared"])
        scene_json = {
            "scene": scene,
            "checkpoint": r["ckpt"], "epoch": r["epoch"],
            "n_ckpt_candidates": r["n_ckpt_candidates"],
            "n_test": r["n_test"],
            "n_test_total": r["n_test_total"], "test_stride": r["test_stride"],
            "basis": {
                "path": str(basis_path),
                "n_train_rows": blob["n_train_rows"],
                "n_easy": blob["n_easy"], "n_hard": blob["n_hard"],
                "ev_topk": {"k4": blob["k4"]["ev_topk"], "k8": blob["k8"]["ev_topk"]},
                "anchor_algorithm": blob["k4"]["anchors"]["64"]["anchor_algorithm"],
                "anchor_seed": blob["k4"]["anchors"]["64"]["anchor_seed"],
                "vectorized_local_future_verified":
                    blob["vectorized_local_future_verified"],
            },
            "variants": r["variants_json"],
            "et_diag_original": r["et_diag"],
            "identity_complete": r["identity_flags"],
            "meta": {
                "K": 20, "pi": "M0 原始 pi 原样保留（所有变体共享同一 top-1 排序）",
                "gate": "interp*/randinterp* 仅缺失样本(valid_count<8)应用 λ；"
                        "完整样本 torch.where 逐位复制 M0",
                "MR_threshold": 2.0,
                "best_anchor_distance_def":
                    "M0 oracle-best mode 的系数到最近 anchor(k4,M64) 的 L2 距离",
                "best_et_basis": "所有变体 best_et 统一用 k4 基（与 m0 可比）；"
                                 "k8 系变体另存 best_et_k8",
                "npz": str(out_root / f"{scene}_persample.npz"),
                "git_revision": get_git_revision(),
                "device": args.device, "seed": args.seed,
                "random_anchor_seed": args.random_anchor_seed,
            },
        }
        with open(out_root / f"{scene}.json", "w") as f:
            json.dump(scene_json, f, ensure_ascii=False, indent=2)
        scene_results[scene] = scene_json
        scene_arrays[scene] = (r["var_arrays"], r["vc"], r["shared"])
        print(f"[done] {scene} n_test={r['n_test']} ({time.time()-t0:.0f}s)")

    # ---------------- 跨场景 summary + Go/No-Go
    micro, macro, worst = {}, {}, {}
    for v in VARIANT_NAMES:
        cat = {m: np.concatenate([scene_arrays[s][0][v][m] for s in args.scenes])
               for m in METRIC_KEYS}
        vc_cat = np.concatenate([scene_arrays[s][1] for s in args.scenes])
        entry = aggregate({m: cat[m] for m in METRIC_KEYS}, vc_cat)
        hi = (vc_cat >= 1) & (vc_cat <= 2)
        entry["highmissing_valid1-2"] = {
            m: round(float(cat[m][hi].mean()), 6) for m in METRIC_KEYS}
        entry["highmissing_n"] = int(hi.sum())
        micro[v] = entry
        macro[v] = {m: round(float(np.mean(
            [scene_arrays[s][0][v][m].mean() for s in args.scenes])), 6)
            for m in METRIC_KEYS}
        worst[v] = {"minFDE20": min(
            ((s, round(float(scene_arrays[s][0][v]["minFDE20"].mean()), 6))
             for s in args.scenes),
            key=lambda t: -t[1])}

    # 完整层保护（门控变体必须逐位等于 M0）
    identity_all = all(
        all(scene_results[s]["identity_complete"].get(v, False) is True
            for s in args.scenes)
        for v in GATED_VARIANTS)
    complete_zero = {}
    for v in GATED_VARIANTS:
        diffs = []
        for s in args.scenes:
            va, vc_s, _ = scene_arrays[s]
            cm = vc_s == 8
            if cm.any():
                diffs.append(float(np.abs(
                    va[v]["minFDE20"][cm] - va["m0"]["minFDE20"][cm]).max()))
        complete_zero[v] = float(max(diffs)) if diffs else None

    # 随机 anchor 对照（高缺失层）
    controls = {}
    for s in args.scenes:
        va, vc_s, _ = scene_arrays[s]
        hi = (vc_s >= 1) & (vc_s <= 2)
        if not hi.any():
            controls[s] = {"note": "冒烟截断：无高缺失样本"}
            continue
        controls[s] = {
            "interp025_vs_randinterp025_minFDE20_highmissing": round(float(
                va["interp_k4_M64_lam0.25"]["minFDE20"][hi].mean()
                - va["randinterp_k4_M64_lam0.25"]["minFDE20"][hi].mean()), 6),
            "anchorhard64_vs_randanchor64_minFDE20_highmissing": round(float(
                va["anchor_hard_k4_M64"]["minFDE20"][hi].mean()
                - va["randanchor_hard_k4_M64"]["minFDE20"][hi].mean()), 6),
        }

    corr_orig = {s: scene_results[s]["et_diag_original"]["corr_best_et_minFDE"]
                 for s in args.scenes}

    go_nogo = {}
    RANK_REL_THRESHOLD = 0.02   # 分析阈值（规范未预注册数值，仅本报告标注）
    for cfg in [f"interp_k4_M64_{LAM_NAME[l]}" for l in LAMBDAS]:
        improved, be_improved = [], []
        for s in args.scenes:
            va, vc_s, _ = scene_arrays[s]
            hi = (vc_s >= 1) & (vc_s <= 2)
            if not hi.any():
                continue  # 冒烟截断时可能无高缺失样本，跳过该场景计数
            if (float(va[cfg]["minFDE20"][hi].mean())
                    < float(va["m0"]["minFDE20"][hi].mean()) - 1e-9 or
                float(va[cfg]["minADE20"][hi].mean())
                    < float(va["m0"]["minADE20"][hi].mean()) - 1e-9):
                improved.append(s)
            if float(va[cfg]["best_et"][hi].mean()) \
                    < float(va["m0"]["best_et"][hi].mean()) - 1e-9:
                be_improved.append(s)
        m0m, cm = micro["m0"]["overall"], micro[cfg]["overall"]
        rank_flags = {}
        for met in ("MR", "ADE@1", "FDE@1"):
            rel = (cm[met] - m0m[met]) / max(m0m[met], 1e-8)
            rank_flags[met] = {"delta_rel": round(float(rel), 4),
                               "flag": bool(rel > RANK_REL_THRESHOLD)}
        near_beats_rand = sum(
            1 for s in args.scenes
            if "interp025_vs_randinterp025_minFDE20_highmissing" in controls[s]
            and controls[s]["interp025_vs_randinterp025_minFDE20_highmissing"] < 0)
        c1 = len(improved) >= 3
        c2 = len(be_improved) >= 3
        c3 = identity_all and all(d == 0.0 or d is None
                                  for d in complete_zero.values())
        c5 = near_beats_rand >= 3
        c6 = sum(1 for c in corr_orig.values() if c is not None and c >= 0.5) >= 4
        c4_ok = not any(f["flag"] for f in rank_flags.values())
        go_nogo[cfg] = {
            "c1_highmissing_minFDE20_or_minADE20_improved_ge3of5": {
                "scenes": improved, "pass": c1},
            "c2_highmissing_best_et_improved_ge3of5": {
                "scenes": be_improved, "pass": c2},
            "c3_complete_layer_identity": {"pass": bool(c3),
                                           "per_variant_maxdiff": complete_zero},
            "c4_no_ranking_regression": {
                "pass": c4_ok, "threshold_rel": RANK_REL_THRESHOLD,
                "note": "阈值为本报告设定的分析阈值（规范未预注册数值）",
                "details": rank_flags},
            "c5_nearest_beats_random_ge3of5": {
                "scenes_count": near_beats_rand, "pass": c5},
            "c6_et_corr_with_minFDE": {"corr_by_scene": corr_orig, "pass": c6},
            "verdict": "GO" if all([c1, c2, c3, c4_ok, c5, c6]) else "NO-GO",
        }

    summary = {
        "scenes": {s: {v: scene_results[s]["variants"][v]["overall"]
                       for v in VARIANT_NAMES} for s in args.scenes},
        "macro_average": macro,
        "micro": micro,
        "worst_scene_minFDE20": worst,
        "highmissing_controls": controls,
        "corr_best_et_minFDE_original": corr_orig,
        "go_nogo": go_nogo,
        "variants_order": VARIANT_NAMES,
        "meta": {
            "protocol": "TrajGap-Bench mixed-direct, K=20, best-val ckpt, "
                        "C8-0 离线诊断（不训练）",
            "pi": "M0 pi 原样保留；top-1 排序与 M0 完全一致",
            "git_revision": get_git_revision(),
            "started": datetime.now().isoformat(timespec="seconds"),
            "tier_def": "valid_count∈{1-2,3-4,5-6,7-8}; complete=valid8",
            "note": "sklearn 缺失，anchor 聚类用 torch Lloyd k-means++ 全批实现"
                    "（seed=2024），算法名已写入 et_basis 与各 scene JSON",
        },
    }
    with open(out_root / "summary.json", "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n[summary] {out_root / 'summary.json'}")
    for cfg, g in go_nogo.items():
        print(f"[go-nogo] {cfg}: {g['verdict']} "
              f"c1={g['c1_highmissing_minFDE20_or_minADE20_improved_ge3of5']['pass']} "
              f"c2={g['c2_highmissing_best_et_improved_ge3of5']['pass']} "
              f"c5={g['c5_nearest_beats_random_ge3of5']['pass']}")


if __name__ == "__main__":
    main()
