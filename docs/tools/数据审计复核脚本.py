#!/usr/bin/env python
"""audit: 重跑 held-out test 评估并落盘可查证输出(逐折 test_minADE20/minFDE20 + N)。"""
import os, re, subprocess, sys, time
from pathlib import Path
GPU = sys.argv[1] if len(sys.argv) > 1 else "2"
D = Path("/home/lbh/DeMo"); PY = f"{Path.home()}/.conda/envs/DeMo/bin/python"
LOG = D/"docs"/"_audit_heldout_rerun.log"
JOBS = [
 ("config_moflow_ethucy","eth","moflow_eth","20260826-183752",82),
 ("config_moflow_ethucy","hotel","moflow_hotel","20260826-215931",61),
 ("config_moflow_ethucy","univ","moflow_univ","20260827-020739",92),
 ("config_moflow_ethucy","zara1","moflow_zara1","20260827-035014",69),
 ("config_moflow_ethucy","zara2","moflow_zara2","20260827-080537",89),
 ("config_moflow_ethucy_bimamba","eth","moflow_bimamba_eth","20260826-212037",98),
 ("config_moflow_ethucy_bimamba","hotel","moflow_bimamba_hotel","20260827-020642",99),
 ("config_moflow_ethucy_bimamba","univ","moflow_bimamba_univ","20260827-073801",99),
 ("config_moflow_ethucy_bimamba","zara1","moflow_bimamba_zara1","20260827-085216",67),
 ("config_moflow_ethucy_bimamba","zara2","moflow_bimamba_zara2","20260827-130643",96),
]
env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU)
with open(LOG,"w") as log:
    log.write(f"# held-out test 重跑审计 {time.strftime('%Y-%m-%d %H:%M:%S')} GPU={GPU}\n")
    log.write(f"# CUDA_VISIBLE_DEVICES={GPU} {PY} -u eval.py --config-name=<cfg> gpus=1 test=true datamodule.target.subset=<sub> checkpoint=<best ckpt>\n")
    for cfg,sub,outdir,ts,ep in JOBS:
        ck = D/"outputs"/outdir/ts/"checkpoints"/f"epoch={ep}.ckpt"
        link = ck.parent/"best_for_eval.ckpt"; link.unlink(missing_ok=True); link.symlink_to(ck)
        cmd=[PY,"-u","eval.py",f"--config-name={cfg}","gpus=1","test=true",f"datamodule.target.subset={sub}",f"checkpoint={link}"]
        r = subprocess.run(cmd, cwd=D, env=env, capture_output=True, text=True)
        link.unlink(missing_ok=True)
        n = re.search(r"Total predictions:\s+(\d+)", r.stdout)
        a = re.search(r"test_minADE20[^0-9]*([0-9.]+)", r.stdout)
        f = re.search(r"test_minFDE20[^0-9]*([0-9.]+)", r.stdout)
        ok = "OK" if "Loaded model weights" in r.stdout else "WARN-LOAD"
        line=f"{'BI' if 'bimamba' in outdir else 'UNI'} {sub:6s} cfg={cfg} ckpt={Path(ck).name:<16s} " \
             f"N={n.group(1) if n else '?'} minADE20={a.group(1) if a else '?'} minFDE20={f.group(1) if f else '?'} [{ok}] rc={r.returncode}"
        print(line, flush=True); log.write(line+"\n")
        if r.returncode!=0: log.write("  STDERR: "+r.stderr[-500:].replace("\n"," ")+"\n")
    log.write("# done\n")
print("saved:", LOG)