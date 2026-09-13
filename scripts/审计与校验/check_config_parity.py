#!/usr/bin/env python
"""校正版:对比 uni vs bi 训练 run(取最新有 metrics 者)的 resolved config。

2026-09 扩展：追加 Missing-Aware M0/M1/M2 配置一致性检查（不改变原有
uni/bi 检查逻辑；无 outputs 历史 run 依赖，独立可运行）。
"""
import yaml
from pathlib import Path
ROOT = Path("/home/lbh/DeMo")
SUBS = ["eth","hotel","univ","zara1","zara2"]

def training_run(base):
    cands = [rd for rd in base.glob("*/") if list((rd/"logs").glob("version_*/metrics.csv"))]
    return sorted(cands, key=lambda p: p.stat().st_mtime)[-1] if cands else None

# 先打印 uni/bi eth 各 run 的 epochs & 复用
for tag in ["moflow_eth","moflow_bimamba_eth"]:
    base=ROOT/"outputs"/tag
    if not base.exists():
        print(f"== {tag} == (目录不存在，跳过)")
        continue
    print(f"== {tag} ==")
    for rd in sorted(base.glob("*/"), key=lambda p:p.stat().st_mtime):
        cfg = rd/".hydra/config.yaml"
        ep = yaml.safe_load(cfg.read_text())["epochs"] if cfg.exists() else "?"
        mcsv = list((rd/"logs").glob("version_*/metrics.csv"))
        maxep = "?"
        if mcsv:
            import csv
            with open(mcsv[0]) as f:
                es=[int(r["epoch"]) for r in csv.DictReader(f) if r.get("epoch")]
            maxep = max(es) if es else "?"
        print(f"   {Path(rd).name:20s} cfg.epochs={ep!r:6s} metrics_max_epoch={maxep}")

print("\n=== 修正后的 uni/bi 对比(取最新有 metrics 的训练 run)===")
for s in SUBS:
    un=training_run(ROOT/"outputs"/f"moflow_{s}")
    bi=training_run(ROOT/"outputs"/f"moflow_bimamba_{s}")
    if un is None or bi is None:
        print(f"--- {s} --- 跳过（缺 run: uni={un is not None} bi={bi is not None}）")
        continue
    du,_db = yaml.safe_load((un/".hydra/config.yaml").read_text()), yaml.safe_load((bi/".hydra/config.yaml").read_text())
    db=_db
    print(f"--- {s} ---  runs: uni={Path(un).name} bi={Path(bi).name}")
    for f in ["seed","epochs","batch_size","lr","weight_decay","warmup_epochs","precision","gpus","num_workers","monitor","save_top_k"]:
        a,b=du.get(f),db.get(f)
        print(f"   {f:14s} uni={a!r:10s} bi={b!r:10s} {'SAME' if a==b else '!!DIFF'}")
    mu=du["model"]["target"]["model"]; mb=db["model"]["target"]["model"]
    for f in ["type","embed_dim","num_heads","future_steps","mlp_ratio","qkv_bias","drop_path","num_actor_types","num_modes","dt"]:
        a,b=mu.get(f),mb.get(f)
        print(f"   model.{f:14s} uni={a!r:10s} bi={b!r:10s} {'SAME' if a==b else '!!DIFF'}")
    # bimamba 全配置树找
    def findbi(d):
        if isinstance(d,dict):
            if "bimamba" in d: return d["bimamba"]
            for v in d.values():
                r=findbi(v)
                if r is not None: return r
        return None
    print(f"   bimamba(anywhere)     uni={findbi(du)!r:10s} bi={findbi(db)!r:10s}")
    # datamodule
    dd=du["datamodule"]["target"]; db2=db["datamodule"]["target"]
    same_dm=all(dd.get(k)==db2.get(k) for k in ["data_root","obs_len","pred_len","num_workers","pin_memory"]) and dd.get("subset")==db2.get("subset")
    print(f"   datamodule 除subset相同: {same_dm} | data_root={dd.get('data_root')} obs/pred={dd.get('obs_len')}/{dd.get('pred_len')} subset={dd.get('subset')}")

# ============================================================
# 当前主链配置检查（TrajImpute direct；历史 missing_aware_* 配置已随
# 旧自定义缺失数据废弃删除，2026-09-13）
# ============================================================
print("\n=== 主链配置检查 (config_missing_aware_trajimpute) ===")
ma_fail = 0
path = ROOT / "conf" / "config_missing_aware_trajimpute.yaml"
if not path.exists():
    print(f"[FAIL] config_missing_aware_trajimpute: 配置文件缺失")
    ma_fail += 1
else:
    import yaml as _yaml
    c = _yaml.safe_load(path.read_text())
    expect = {
        "data_root": "/home/lbh/TrajImpute/dataset/TrajImpute",
        "monitor": "val_minFDE20",
        "bimamba": False,
        "num_modes-check": c["model"]["target"]["model"]["num_modes"] == 20,
    }
    problems = []
    if c.get("data_root") != expect["data_root"]:
        problems.append(f"data_root={c.get('data_root')!r}")
    if c.get("monitor") != expect["monitor"]:
        problems.append(f"monitor={c.get('monitor')!r}")
    if c.get("bimamba") != expect["bimamba"]:
        problems.append(f"bimamba={c.get('bimamba')!r} (期望单向 false)")
    if not expect["num_modes-check"]:
        problems.append("num_modes != 20")
    status = "PASS" if not problems else "FAIL"
    if problems:
        ma_fail += 1
    print(f"[{status}] config_missing_aware_trajimpute: " +
          ("; ".join(problems) if problems else "data_root/monitor/bimamba/num_modes 全部符合"))

print(f"\n配置检查失败项: {ma_fail}")