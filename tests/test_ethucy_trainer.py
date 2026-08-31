"""ETH/UCY Trainer 单元测试。

测试 Trainer 的损失函数和训练步骤。
"""

import torch

from src.model.trainer_forecast import Trainer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _make_dummy_batch_actor_only(B=2, N=10, obs_len=8, pred_len=12):
    """构造 actor-only dummy batch（mamba 需要 GPU）。"""
    data = {
        "x_positions_diff": torch.randn(B, N, obs_len, 2, device=DEVICE),
        "x_positions": torch.randn(B, N, obs_len, 2, device=DEVICE),
        "x_attr": torch.zeros(B, N, 3, dtype=torch.uint8, device=DEVICE),
        "x_centers": torch.randn(B, N, 2, device=DEVICE),
        "x_angles": torch.randn(B, N, obs_len, device=DEVICE),
        "x_velocity": torch.randn(B, N, obs_len, device=DEVICE),
        "x_velocity_diff": torch.randn(B, N, obs_len, device=DEVICE),
        "x_valid_mask": torch.ones(B, N, obs_len, dtype=torch.bool, device=DEVICE),
        "x_key_valid_mask": torch.ones(B, N, dtype=torch.bool, device=DEVICE),
        "target": torch.randn(B, N, pred_len, 2, device=DEVICE),
        "target_mask": torch.ones(B, N, pred_len, dtype=torch.bool, device=DEVICE),
        "origin": torch.zeros(B, 1, 2, device=DEVICE),
        "theta": torch.zeros(B, 1, device=DEVICE),
        "timestamp": torch.zeros(B, 1, device=DEVICE),
    }
    return data


def _make_trainer():
    trainer = Trainer(
        model={
            "type": "ModelForecast",
            "embed_dim": 128,
            "future_steps": 12,
            "use_map": False,
            "num_actor_types": 1,
        },
        lr=0.001,
        epochs=100,
        warmup_epochs=5,
        submission_type="ethucy",
    )
    return trainer.to(DEVICE)


class TestTrainerEthUcy:
    """测试 ETH/UCY 模式下的 Trainer。"""

    def test_trainer_init(self):
        trainer = _make_trainer()
        assert trainer.submission_type == "ethucy"
        assert trainer.net.use_map is False
        assert trainer.net.future_steps == 12

    def test_training_step_no_nan(self):
        trainer = _make_trainer()
        trainer.train()
        data = _make_dummy_batch_actor_only(B=2, N=10, obs_len=8, pred_len=12)

        loss = trainer.training_step(data, batch_idx=0)

        assert loss is not None
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_validation_step(self):
        trainer = _make_trainer()
        trainer.eval()
        data = _make_dummy_batch_actor_only(B=2, N=10, obs_len=8, pred_len=12)

        # 验证步骤不应抛出异常
        trainer.validation_step(data, batch_idx=0)

    def test_loss_no_other_agents(self):
        """测试没有其他行人目标时损失仍为有限值。"""
        trainer = _make_trainer()
        trainer.train()

        # 只有 focal agent，没有其他行人
        data = _make_dummy_batch_actor_only(B=2, N=1, obs_len=8, pred_len=12)
        loss = trainer.training_step(data, batch_idx=0)

        assert not torch.isnan(loss)
        assert not torch.isinf(loss)