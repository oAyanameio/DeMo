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
# Missing-Aware M0/M1/M2 配置检查（新增段，独立于上面 outputs 依赖）
# ============================================================
print("\n=== Missing-Aware 配置检查 (yaml 静态解析) ===")
MA_EXPECT = {
    "embed_dim": 128, "future_steps": 12, "dt": 0.4, "obs_len": 8, "bimamba": True,
    "use_observation_features": False, "use_missing_summary": False,
    "use_gap_condition": False,
}
MA_FORBIDDEN = ["condition_state_query", "condition_mode_query", "condition_hybrid"]
ma_fail = 0
ma_models = {}
for name, modes in (("missing_aware_ethucy", 6), ("missing_aware_sdd", 20)):
    path = ROOT / "conf" / "model" / f"{name}_model_forecast.yaml"
    if not path.exists():
        print(f"[FAIL] {name}: 配置文件缺失 {path}")
        ma_fail += 1
        continue
    text = path.read_text()
    m = yaml.safe_load(text)["target"]["model"]
    problems = []
    for k, v in MA_EXPECT.items():
        if m.get(k) != v:
            problems.append(f"{k}={m.get(k)!r} (期望 {v!r})")
    if m.get("num_modes") != modes:
        problems.append(f"num_modes={m.get('num_modes')!r} (期望 {modes})")
    for k in MA_FORBIDDEN:
        if k in m:  # 只检查实际配置键，不误伤注释中的提及
            problems.append(f"包含未实现参数 {k}")
    ma_models[name] = m
    status = "PASS" if not problems else "FAIL"
    if problems:
        ma_fail += 1
    print(f"[{status}] {name}: " + ("; ".join(problems) if problems else f"主参数全部符合（num_modes={modes}）"))

# 两套配置除 num_modes（及数据相关 monitor）外模型主参数一致
if len(ma_models) == 2:
    e, s = ma_models["missing_aware_ethucy"], ma_models["missing_aware_sdd"]
    diffs = [k for k in set(e) | set(s)
             if k not in ("num_modes",) and e.get(k) != s.get(k)]
    if diffs:
        print(f"[FAIL] ETH/UCY vs SDD 模型主参数不一致: {diffs}")
        ma_fail += 1
    else:
        print("[PASS] ETH/UCY vs SDD 除 num_modes 外模型主参数完全一致")

# 顶层配置 monitor 检查
for name, mon in (("config_missing_aware_ethucy", "val_minFDE6"),
                  ("config_missing_aware_sdd", "val_minFDE20")):
    path = ROOT / "conf" / f"{name}.yaml"
    if not path.exists():
        print(f"[FAIL] {name}: 配置文件缺失")
        ma_fail += 1
        continue
    c = yaml.safe_load(path.read_text())
    ok = c.get("monitor") == mon
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: monitor={c.get('monitor')!r} (期望 {mon!r})")
    if not ok:
        ma_fail += 1

print(f"\nMissing-Aware 检查失败项: {ma_fail}")