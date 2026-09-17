"""生成配对初始化 checkpoint（R1 修正 3：M0/S1 同初始 backbone）。

用法：
  PYTHONPATH=. python scripts/训练与评估/make_paired_init.py \
      --seed 2024 --out outputs/paired_init/m0_init_seed2024.ckpt

以 seed 2024 构造 M0（全开关关），把 initialize_weights 后的完整
state_dict 存成 Lightning 兼容格式（含 "state_dict" 键、"net." 前缀）。
S1 训练时经 pretrained_weights 加载：共享参数逐位与 M0 初始一致，
新分支保持自身零初始化（load_state_dict strict=False 跳过缺失键）。
"""
import argparse
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--embed-dim", type=int, default=128)
    ap.add_argument("--num-modes", type=int, default=20)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from src.model.model_forecast import ModelForecast

    torch.manual_seed(args.seed)
    m0 = ModelForecast(
        embed_dim=args.embed_dim, num_modes=args.num_modes,
        future_steps=12, bimamba=False, dt=0.4, obs_len=8,
        use_gap_scaling=False, use_missing_summary=False,
    )
    sd = {f"net.{k}": v for k, v in m0.state_dict().items()}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": sd, "hyper_parameters": {"paired_init": True}}, out)
    print(f"saved {sum(v.numel() for v in sd.values())} params -> {out}")


if __name__ == "__main__":
    main()
