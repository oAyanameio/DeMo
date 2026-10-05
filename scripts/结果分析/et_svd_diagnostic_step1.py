"""ET/SVD Step-1 只读诊断（C8 前置探针，不改任何训练代码）。

目的：判定 M0 高缺失失败属于"候选生成问题"还是"概率排序问题"。

做法：
  A. 每场景用 TrajGap Mixed-train 的 GT 未来轨迹（局部坐标系，与评估同源）
     建 ET/SVD 基与训练均值 —— 仅 train split，无测试泄漏；
  B. 加载现有 M0 checkpoint，在 Mixed-test 上前向，把 GT 未来与 K=20 条
     预测 mode 全部投影到 ET 系数空间；
  C. 按 valid_count 分层统计：
       - best_et     : K 条 mode 中与 GT 系数距离最小者（生成质量）
       - top1_et     : 最高概率 mode 的系数距离（排序质量）
       - spread      : K 条 mode 系数的平均两两距离（模态坍缩）
       - prob_at_best: 最优 mode 拿到的概率质量（排序失败签名）
       - resid_rel   : 子空间外残差占比（流形外漂移）
     并同时记录 minFDE20 / FDE@1 / MR（指标空间锚点）与
     best_et↔minFDE 相关性（ET 距离是否为有效代理）。

判定规则（写回 JSON conclusion 字段）：
  生成问题签名: 高缺失层 best_et 相对低缺失层显著膨胀 且 spread 坍缩
               => anchor 检索（C8 层2）有收益空间
  排序问题签名: best_et 跨层近似持平 但 top1_et-best_et 差值膨胀
               => 该投 F2 校准，anchor 无益
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.datamodule.trajimpute_dataset import (  # noqa: E402
    MIXED_DIFFICULTY, TrajGapDataset, trajimpute_collate_fn,
)

SCENES = ("ETH-M", "HOTEL-M", "UNIV-M", "ZARA1-M", "ZARA2-M")
TIERS = ((1, 2), (3, 4), (5, 6), (7, 8))  # valid_count 分层


def get_git_revision():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "unknown"


def build_model(K: int):
    from src.model.model_forecast import ModelForecast
    return ModelForecast(
        embed_dim=128, future_steps=12, num_heads=8, mlp_ratio=4.0,
        qkv_bias=False, drop_path=0.2, num_actor_types=1,
        num_modes=K, bimamba=False, dt=0.4, obs_len=8,
    )


def load_checkpoint(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    cleaned = {k[len("net."):]: v for k, v in state.items() if k.startswith("net.")}
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    missing = [k for k in missing]
    assert not missing and not unexpected, (
        f"checkpoint 与纯 M0 不匹配: missing={missing[:5]} unexpected={unexpected[:5]}")
    return ckpt.get("epoch", None)


def build_et_basis(dataset: TrajGapDataset, k: int, max_n: int = 60000, seed: int = 2024):
    """Mixed-train focal GT 未来 -> SVD 基。返回 mean[24], V[k,24], 奇异值, n。"""
    n_total = len(dataset)
    idx = np.arange(n_total)
    if n_total > max_n:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n_total, size=max_n, replace=False))
    feats = torch.zeros(len(idx), 24)
    for j, i in enumerate(idx):
        s = dataset[int(i)]
        tgt = s["target"]
        if tgt.dim() == 3:      # [A, T, 2] -> focal
            tgt = tgt[0]
        feats[j] = tgt.reshape(-1).float()
    mean = feats.mean(dim=0)
    Xc = feats - mean
    # SVD: Xc = U S V^T，行为样本 -> 基向量为右奇异向量 V 的行
    _, S, Vt = torch.linalg.svd(Xc, full_matrices=False)
    V = Vt[:k]
    # 能量谱：top-k 累计解释方差比
    ev = ((S ** 2) / (S ** 2).sum()).cpu()
    return mean, V, S, len(idx), float(ev[:k].sum()), ev.tolist()


@torch.no_grad()
def project_test(model, dataset, mean, V, device="cuda:1",
                 batch_size=64, num_workers=2, max_n=None):
    """Mixed-test 前向 + ET 投影，逐样本记录诊断量。"""
    model = model.to(device).eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, collate_fn=trajimpute_collate_fn)
    K = model.time_decoder.num_modes
    n_total = len(dataset)
    stride = max(1, -(-n_total // max_n)) if max_n else 1  # 等步抽样防 Easy/Hard 分块偏置
    recs = []
    n_seen = 0
    for batch in loader:
        batch_dev = {kk: (v.to(device) if torch.is_tensor(v) else v)
                     for kk, v in batch.items()}
        out = model(batch_dev)
        pred = out["new_y_hat"] if out.get("new_y_hat") is not None else out["y_hat"]
        prob = out["new_pi"] if out.get("new_pi") is not None else out["pi"]
        pred = pred[..., :2].float()          # [B,K,T,2]
        prob = prob.float()                    # [B,K]
        if not torch.allclose(prob.sum(-1), prob.new_ones(pred.shape[0]), atol=1e-2):
            prob = torch.softmax(prob, dim=-1)  # 未归一化得分时防御性归一
        target = batch_dev["target"][:, 0].float()   # [B,T,2]
        valid = batch["x_valid_mask"][:, 0]           # [B,T]
        for i in range(pred.shape[0]):
            if n_seen % stride:
                n_seen += 1
                continue
            n_seen += 1
            yk = pred[i].reshape(K, -1).cpu()          # [K,24]
            gt = target[i].reshape(-1).cpu()           # [24]
            cg = (gt - mean) @ V.T                     # [k]
            ck = (yk - mean) @ V.T                     # [K,k]
            d = torch.norm(ck - cg, dim=-1)            # [K]
            best = int(torch.argmin(d))
            top1 = int(torch.argmax(prob[i].cpu()))
            # K 条 mode 的两两距离（抽样 200 对控制计算量）
            rng = np.random.default_rng(12345)
            pi_, pj_ = rng.integers(0, K, 200), rng.integers(0, K, 200)
            spread = float(torch.norm(ck[pi_] - ck[pj_], dim=-1).mean())
            # 残差占比
            gt_norm = float(torch.norm(gt - mean)) + 1e-8
            gt_resid = float(torch.norm((gt - mean) - cg @ V)) / gt_norm
            bk_resid = float(torch.norm((yk[best] - mean) - ck[best] @ V)) / (
                float(torch.norm(yk[best] - mean)) + 1e-8)
            # 指标空间锚点（在折叠前按 [K,T,2] 计算；误差 = 到 GT 的距离）
            ykt = pred[i].cpu()                        # [K,T,2]
            gtt = target[i].cpu()                      # [T,2]
            fde = torch.norm(ykt[:, -1, :] - gtt[-1], dim=-1)   # [K] 终点误差
            ade = torch.norm(ykt - gtt.unsqueeze(0), dim=-1).mean(-1)
            minFDE = float(fde.min()); minADE = float(ade.min())
            FDE1 = float(fde[top1]); ADE1 = float(ade[top1])
            MR = float(fde.min() > 2.0)
            recs.append(dict(
                valid_count=int(valid[i].sum()),
                missing_count=int(batch["missing_count"][i]),
                minFDE=minFDE, minADE=minADE, FDE1=FDE1, ADE1=ADE1, MR=MR,
                best_et=float(d[best]), top1_et=float(d[top1]),
                et_gap=float(d[top1] - d[best]),
                prob_at_best=float(prob[i, best].cpu()),
                spread=spread, gt_resid_rel=gt_resid, best_resid_rel=bk_resid,
            ))
            n_seen += 1
    return recs


def aggregate(recs):
    tiers = {}
    for lo, hi in TIERS:
        sub = [r for r in recs if lo <= r["valid_count"] <= hi]
        if not sub:
            continue
        arr = {kk: float(np.mean([r[kk] for r in sub])) for kk in (
            "minFDE", "minADE", "FDE1", "ADE1", "MR",
            "best_et", "top1_et", "et_gap", "prob_at_best",
            "spread", "gt_resid_rel", "best_resid_rel")}
        arr["n"] = len(sub)
        tiers[f"valid{lo}-{hi}"] = arr
    # best_et 与 minFDE 的样本级相关性（ET 距离有效性）
    if len(recs) > 3:
        be = np.array([r["best_et"] for r in recs])
        mf = np.array([r["minFDE"] for r in recs])
        corr = float(np.corrcoef(be, mf)[0, 1])
    else:
        corr = None
    return tiers, corr


def conclude(tiers):
    """按预注册规则给生成/排序判定。"""
    lo = tiers.get("valid7-8"); hi = tiers.get("valid1-2")
    if not lo or not hi:
        return "insufficient_tiers"
    best_ratio = hi["best_et"] / max(lo["best_et"], 1e-8)
    gap_ratio = hi["et_gap"] / max(lo["et_gap"], 1e-8)
    spread_ratio = hi["spread"] / max(lo["spread"], 1e-8)
    gen_sig = best_ratio > 1.3 and spread_ratio < 0.9
    rank_sig = gap_ratio > 1.3 and best_ratio < 1.15
    verdict = []
    if gen_sig:
        verdict.append("GENERATION")
    if rank_sig:
        verdict.append("RANKING")
    if not verdict:
        verdict.append("MIXED/NEITHER")
    return dict(best_et_ratio=best_ratio, et_gap_ratio=gap_ratio,
                spread_ratio=spread_ratio,
                verdict="+".join(verdict))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/home/lbh/TrajImpute/dataset/TrajImpute")
    ap.add_argument("--ckpt-root", default="outputs/trajgap_bench_seed2024")
    ap.add_argument("--scenes", nargs="+", default=list(SCENES))
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--max-test", type=int, default=30000,
                    help="每场景诊断最多样本数（等步抽样），0=全量")
    ap.add_argument("--output-root", default="outputs/et_diagnostic_step1")
    args = ap.parse_args()

    out_root = Path(args.output_root); out_root.mkdir(parents=True, exist_ok=True)
    summary = {}
    for scene in args.scenes:
        gpu = "gpu1" if scene in ("ETH-M", "HOTEL-M", "UNIV-M") else "gpu3"
        run_dir = Path(args.ckpt_root) / f"m0_{gpu}" / f"M0_{scene}_mixed-direct_seed2024_uni"
        # 正式协议：best-val checkpoint（epoch=N.ckpt），非 last.ckpt
        best = sorted(run_dir.glob("train/checkpoints/epoch=*.ckpt"))
        ckpt = best[-1] if best else run_dir / "train/checkpoints/last.ckpt"
        if not ckpt.exists():
            hits = list(Path(args.ckpt_root).rglob(f"M0_{scene}_*/train/checkpoints/epoch=*.ckpt"))
            if not hits:
                hits = list(Path(args.ckpt_root).rglob(f"M0_{scene}_*/train/checkpoints/last.ckpt"))
            assert hits, f"checkpoint not found for {scene}"
            ckpt = hits[0]
        train_ds = TrajGapDataset(args.data_root, scene, "train")
        test_ds = TrajGapDataset(args.data_root, scene, "test")

        mean, V, S, n_train, ev_k, ev_all = build_et_basis(train_ds, args.k)
        torch.save({"mean": mean, "V": V, "S": S, "k": args.k,
                    "n_train": n_train, "scene": scene},
                   out_root / f"et_basis_{scene}.pt")

        model = build_model(20)
        epoch = load_checkpoint(model, ckpt)

        n_test = len(test_ds)
        max_n = args.max_test if args.max_test > 0 else None
        recs = project_test(model, test_ds, mean, V, device=args.device, max_n=max_n)
        tiers, corr = aggregate(recs)
        concl = conclude(tiers)
        scene_out = dict(
            scene=scene, checkpoint=str(ckpt), epoch=epoch,
            k=args.k, n_train_basis=n_train, n_test_total=n_test,
            n_test_diag=len(recs),
            explained_variance_topk=ev_k,
            explained_variance_k4=float(sum(ev_all[:4])),
            explained_variance_k8=float(sum(ev_all[:8])),
            singular_top8=[float(x) for x in S[:8]],
            best_et_vs_minFDE_corr=corr,
            tiers=tiers, conclusion=concl,
        )
        summary[scene] = scene_out
        with open(out_root / f"et_diagnosis_{scene}.json", "w") as f:
            json.dump(scene_out, f, ensure_ascii=False, indent=2)
        print(f"\n=== {scene} (k={args.k}, EV@k={ev_k:.3f}, corr(best_et,minFDE)={corr:.3f}) ===")
        for tname, t in tiers.items():
            print(f"  {tname}: n={t['n']:6d} minFDE={t['minFDE']:.3f} FDE1={t['FDE1']:.3f} "
                  f"MR={t['MR']:.3f} | best_et={t['best_et']:.3f} top1_et={t['top1_et']:.3f} "
                  f"gap={t['et_gap']:.3f} spread={t['spread']:.3f} p@best={t['prob_at_best']:.3f}")
        print(f"  conclusion: {concl}")

    summary["_meta"] = dict(
        git_revision=get_git_revision(), K=20, protocol="mixed-direct",
        purpose="ET/SVD Step-1 生成vs排序诊断（只读）",
        rule="best_et_ratio>1.3&spread_ratio<0.9=>GENERATION; "
             "et_gap_ratio>1.3&best_et_ratio<1.15=>RANKING",
    )
    with open(out_root / "et_diagnosis_summary.json", "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n[summary] {out_root / 'et_diagnosis_summary.json'}")


if __name__ == "__main__":
    main()
