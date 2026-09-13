"""收缩后验证：模型实例化（无 map/lane/stream 属性）。
只读验证，不训练、不写数据。用法:
    cd /home/lbh/DeMo && ~/.conda/envs/DeMo/bin/python scripts/审计与校验/verify_actor_only.py
"""
import copy
import logging
import sys
from pathlib import Path

logging.getLogger().addHandler(logging.NullHandler())
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def check_instantiation():
    from src.model.trainer_forecast import Trainer

    def build(cfgdict, label):
        m = copy.deepcopy(cfgdict)
        net = Trainer(model=m)
        bad = [a for a in ["use_map", "lane_embed", "lane_type_embed", "interaction",
                           "mode_fusion", "stream_loc", "traj_embed", "pose_dim",
                           "use_stream_encoder", "use_stream_decoder"]
               if hasattr(net.net, a)]
        n = sum(p.numel() for p in net.net.parameters())
        print(f"[{label}] 实例化 OK, 参数量={n/1e6:.2f}M, 违禁属性={bad or '无'}")
        assert not bad

    build({"type": "ModelForecast", "embed_dim": 128, "future_steps": 12, "num_heads": 8,
           "mlp_ratio": 4.0, "qkv_bias": False, "drop_path": 0.2, "num_actor_types": 1,
           "num_modes": 6, "dt": 0.4}, "ETH/UCY uni 6modes")
    build({"type": "ModelForecast", "embed_dim": 128, "future_steps": 12, "num_heads": 8,
           "mlp_ratio": 4.0, "qkv_bias": False, "drop_path": 0.2, "num_actor_types": 1,
           "num_modes": 20, "dt": 0.4}, "SDD 20modes")

    import src.model.model_forecast as mf
    import src.model.trainer_forecast as tf
    print("StreamModelForecast 已移除:", not hasattr(mf, "StreamModelForecast"))
    print("StreamTrainer 已移除:", not hasattr(tf, "StreamTrainer"))
    assert not hasattr(mf, "StreamModelForecast") and not hasattr(tf, "StreamTrainer")


if __name__ == "__main__":
    check_instantiation()
    print("ALL VERIFICATIONS PASSED")
