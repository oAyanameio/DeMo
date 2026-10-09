import datetime
from pathlib import Path
import time
import pickle
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MetricCollection
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.metrics import MR, minADE, minFDE, brier_minFDE
from src.utils.optim import WarmupCosLR
from src.utils.LaplaceNLLLoss import LaplaceNLLLoss
from .trajectory_forecaster import TrajectoryForecaster, ModelForecast


def _log_detach(x):
    """日志标量统一 detach：tensor 去梯度，标量原样返回（避免对 0 调 .detach）。"""
    return x.detach() if torch.is_tensor(x) else x


class ForecastingLightningModule(pl.LightningModule):
    def __init__(
        self,
        model: dict,
        pretrained_weights: str = None,
        lr: float = 1e-3,
        warmup_epochs: int = 10,
        epochs: int = 60,
        weight_decay: float = 1e-4,
    ) -> None:
        super(Trainer, self).__init__()
        self.warmup_epochs = warmup_epochs
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.save_hyperparameters()
        from src.utils.submission_ethucy import SubmissionEthUcy
        self.submission_handler = SubmissionEthUcy()

        model_type = model.pop('type')

        self.net = self.get_model(model_type)(**model)

        if pretrained_weights is not None:
            self.net.load_from_checkpoint(pretrained_weights)
            print('Pretrained weights have been loaded.')

        num_modes = model.get('num_modes', 6)

        metrics = MetricCollection(
            {
                "minADE1": minADE(k=1),
                f"minADE{num_modes}": minADE(k=num_modes),
                "minFDE1": minFDE(k=1),
                f"minFDE{num_modes}": minFDE(k=num_modes),
                "MR": MR(),
                f"b-minFDE{num_modes}": brier_minFDE(k=num_modes),
            }
        )
        self.laplace_loss = LaplaceNLLLoss()
        self.val_metrics = metrics.clone(prefix="val_")
        self.val_metrics_new = metrics.clone(prefix="val_new_")
        # M1：calibrated 最终输出的同一组指标（raw/cal 并存，方案 §6.4）
        self.val_metrics_cal = metrics.clone(prefix="val_cal_")

    def get_model(self, model_type):
        model_dict = {
            'TrajectoryForecaster': TrajectoryForecaster,
            'ModelForecast': ModelForecast,  # legacy config name
        }
        assert model_type in model_dict, f"Unknown model type: {model_type}"
        return model_dict[model_type]

    def forward(self, data):
        return self.net(data)

    def predict(self, data):
        """推理入口：与 trainer.test_step / direct evaluator 同一分支。

        轨迹优先 new_y_hat；概率优先 calibrated new_pi_cal（M1），
        无校准输出时回退 new_pi（M0/旧模型），再回退 mode-query pi。
        """
        predictions = []
        probs = []
        for i in range(len(data)):
            cur_data = data[i]
            out = self(cur_data)
            y_hat = out.get("new_y_hat")
            if y_hat is None:
                y_hat = out["y_hat"]
            pi = out.get("new_pi_cal")
            if pi is None:
                pi = out.get("new_pi", out["pi"])
            prediction, prob = self.submission_handler.format_data(
                cur_data, y_hat, pi, inference=True)
            predictions.append(prediction)
            probs.append(prob)

        return predictions, probs

    def cal_loss(self, out, data, tag=''):
        y_hat, pi, y_hat_others = out["y_hat"], out["pi"], out["y_hat_others"]
        scal, scal_new = out["scal"], out["scal_new"]
        new_y_hat = out.get("new_y_hat", None)
        new_pi = out.get("new_pi", None)
        dense_predict = out.get("dense_predict", None)

        # gt
        y, y_others = data["target"][:, 0], data["target"][:, 1:]

        # loss for output of state query
        if dense_predict is not None:
            dense_reg_loss = F.smooth_l1_loss(dense_predict, y)
        else:
            dense_reg_loss = 0

        # loss for output of mode query
        l2_norm = torch.norm(y_hat[..., :2] - y.unsqueeze(1), dim=-1).sum(dim=-1)
        best_mode = torch.argmin(l2_norm, dim=-1)
        y_hat_best = y_hat[torch.arange(y_hat.shape[0]), best_mode]
        agent_reg_loss = F.smooth_l1_loss(y_hat_best[..., :2], y)
        agent_cls_loss = F.cross_entropy(pi, best_mode.detach(), label_smoothing=0.2)

        # loss for final output
        best_mode_new = None
        if new_y_hat is not None:
            l2_norm_new = torch.norm(new_y_hat[..., :2] - y.unsqueeze(1), dim=-1).sum(dim=-1)
            best_mode_new = torch.argmin(l2_norm_new, dim=-1)
            new_y_hat_best = new_y_hat[torch.arange(new_y_hat.shape[0]), best_mode_new]
            new_agent_reg_loss = F.smooth_l1_loss(new_y_hat_best[..., :2], y)
        else:
            new_agent_reg_loss = 0
        # M1 校准（方案 §6.4）：最终分支的 CE 与 Laplace NLL 使用 calibrated
        # 输出；mode-query raw 分支损失保持原样，校准头不单独承担全部信号。
        new_pi_raw = out.get("new_pi", None)
        new_pi_final = out.get("new_pi_cal", new_pi_raw)
        scal_new_raw = out.get("scal_new", None)
        scal_new_final = out.get("scal_cal", scal_new_raw)
        if new_pi_final is not None and best_mode_new is not None:
            new_pi_reg_loss = F.cross_entropy(new_pi_final, best_mode_new.detach(), label_smoothing=0.2)
        else:
            new_pi_reg_loss = 0

        # loss for other agents
        others_reg_mask = data["target_mask"][:, 1:]
        if others_reg_mask.any():
            others_reg_loss = F.smooth_l1_loss(
                y_hat_others[others_reg_mask], y_others[others_reg_mask]
            )
        else:
            others_reg_loss = y_hat_others.new_zeros(())

        # Laplace loss, which is not necessary
        predictions = {}
        predictions['traj'] = y_hat
        predictions['scale'] = scal
        predictions['probs'] = pi
        laplace_loss = self.laplace_loss.compute(predictions, y)

        # M1：最终分支 Laplace NLL 使用 calibrated 输出（calibration 关闭时即 raw）
        predictions['traj'] = new_y_hat
        predictions['scale'] = scal_new_final
        predictions['probs'] = new_pi_final
        laplace_loss_new = self.laplace_loss.compute(predictions, y)

        # total loss
        loss = agent_reg_loss + agent_cls_loss + others_reg_loss + \
                new_agent_reg_loss + dense_reg_loss + new_pi_reg_loss
        loss = loss + laplace_loss + laplace_loss_new

        # 日志标量保持 detached tensor，避免每 batch 多次 GPU->CPU 同步（提速方案 §2.3）；
        # 字段名与数值定义不变，只在 epoch 结束由 logger 统一 .item() 一次。
        disp_dict = {
            f"{tag}loss": loss.detach(),
            f"{tag}reg_loss": agent_reg_loss.detach(),
            f"{tag}cls_loss": agent_cls_loss.detach(),
            f"{tag}others_reg_loss": others_reg_loss.detach(),
            f"{tag}laplace_loss": laplace_loss.detach(),
            f"{tag}laplace_loss_new": laplace_loss_new.detach(),
        }
        if new_y_hat is not None:
            disp_dict[f"{tag}reg_loss_refine"] = _log_detach(new_agent_reg_loss)
        if new_pi is not None:
            disp_dict[f"{tag}reg_loss_new_pi"] = _log_detach(new_pi_reg_loss)
        if dense_predict is not None:
            disp_dict[f"{tag}reg_loss_dense"] = _log_detach(dense_reg_loss)

        return loss, disp_dict

    def training_step(self, data, batch_idx):
        out = self(data)
        loss, loss_dict = self.cal_loss(out, data)

        for k, v in loss_dict.items():
            self.log(
                f"train/{k}",
                v,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                sync_dist=False,  # 单卡无 all-reduce；多卡再改回 True
            )

        return loss

    def validation_step(self, data, batch_idx):
        out = self(data)
        _, loss_dict = self.cal_loss(out, data)
        # 校准敏感验证信号（方案 §6.4）：最终分支 Laplace NLL——M1 下
        # cal_loss 内部已用 calibrated 输出计算，M0 下即 raw final NLL。
        for key in ("laplace_loss_new", "reg_loss_new_pi"):
            if key in loss_dict:
                self.log(
                    f"val_{key}",
                    loss_dict[key],
                    on_step=False,
                    on_epoch=True,
                    batch_size=1,
                    sync_dist=True,
                )
        metrics = self.val_metrics(out, data['target'][:, 0])
        metrics_cal = None
        if out['new_y_hat'] is not None:
            out['y_hat'] = out['new_y_hat']
            out['pi'] = out['new_pi']
            metrics_new = self.val_metrics_new(out, data['target'][:, 0])

        # M1：calibrated 输出指标（calibration 关闭时 new_pi_cal 不存在，跳过）
        if out.get('new_pi_cal') is not None:
            out_cal = dict(out)
            out_cal['y_hat'] = out['new_y_hat']
            out_cal['pi'] = out['new_pi_cal']
            metrics_cal = self.val_metrics_cal(out_cal, data['target'][:, 0])
            # 诊断：mode entropy（高缺失证据下排序可靠性）
            self.log(
                "val_cal/mode_entropy",
                -(torch.softmax(out['new_pi_cal'].float(), dim=-1)
                  * torch.log_softmax(out['new_pi_cal'].float(), dim=-1)).sum(-1).mean(),
                on_step=False, on_epoch=True, batch_size=1, sync_dist=True,
            )
            if out.get("calibration_log_scale") is not None:
                self.log(
                    "val_cal/log_scale_mean",
                    out["calibration_log_scale"].float().mean(),
                    on_step=False, on_epoch=True, batch_size=1, sync_dist=True,
                )
                self.log(
                    "val_cal/log_tau_mean",
                    out["calibration_log_tau"].float().mean(),
                    on_step=False, on_epoch=True, batch_size=1, sync_dist=True,
                )

        self.log_dict(
            metrics,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=1,
            sync_dist=True,
        )
        if out['new_y_hat'] is not None:
            self.log_dict(
                metrics_new,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                batch_size=1,
                sync_dist=True,
            )
        if out.get('new_pi_cal') is not None:
            self.log_dict(
                metrics_cal,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                batch_size=1,
                sync_dist=True,
            )

    def on_test_start(self) -> None:
        save_dir = Path("./submission")
        save_dir.mkdir(exist_ok=True)
        from src.utils.submission_ethucy import SubmissionEthUcy
        self.submission_handler = SubmissionEthUcy(save_dir=str(save_dir))
        self.test_metrics = self.val_metrics.clone(prefix="test_")
        # M1：raw final 与 calibrated final 双套测试指标（方案 §6.5/§9.2）
        self.test_metrics_new = self.val_metrics_new.clone(prefix="test_new_")
        self.test_metrics_cal = self.val_metrics_cal.clone(prefix="test_cal_")

    def test_step(self, data, batch_idx) -> None:
        out = self(data)
        if out['new_y_hat'] is not None:
            out['y_hat'] = out['new_y_hat']
            out['pi'] = out['new_pi']
        # M1：正式主结果使用 calibrated final output（方案 §6.5）；
        # submission 同步使用校准概率，raw 指标作为诊断保留。
        if out.get('new_pi_cal') is not None:
            out['y_hat'] = out['new_y_hat']
            out['pi'] = out['new_pi_cal']
        self.submission_handler.format_data(data, out["y_hat"], out["pi"])
        metrics = self.test_metrics(out, data['target'][:, 0])
        self.log_dict(
            metrics,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=1,
        )
        # raw final / calibrated final 双套
        if out.get('new_y_hat') is not None:
            out_new = dict(out)
            out_new['y_hat'] = out['new_y_hat']
            out_new['pi'] = out['new_pi']
            metrics_new = self.test_metrics_new(out_new, data['target'][:, 0])
            self.log_dict(
                metrics_new,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                batch_size=1,
            )
        if out.get('new_pi_cal') is not None:
            out_cal = dict(out)
            out_cal['y_hat'] = out['new_y_hat']
            out_cal['pi'] = out['new_pi_cal']
            metrics_cal = self.test_metrics_cal(out_cal, data['target'][:, 0])
            self.log_dict(
                metrics_cal,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                batch_size=1,
            )

    def on_test_end(self) -> None:
        self.submission_handler.generate_submission_file()
        print("TEST METRICS:", self.test_metrics.compute())

    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (
            nn.Linear,
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.MultiheadAttention,
            nn.LSTM,
            nn.GRU,
        )
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                full_param_name = (
                    "%s.%s" % (module_name, param_name) if module_name else param_name
                )
                if "bias" in param_name:
                    no_decay.add(full_param_name)
                elif "weight" in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    else:
                        # Custom mixers/RMSNorms are not all covered by the
                        # module lists; trainable weights must never be omitted.
                        no_decay.add(full_param_name)
                elif not ("weight" in param_name or "bias" in param_name):
                    no_decay.add(full_param_name)
        param_dict = {
            param_name: param for param_name, param in self.named_parameters()
        }
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0

        optim_groups = [
            {
                "params": [
                    param_dict[param_name] for param_name in sorted(list(decay))
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    param_dict[param_name] for param_name in sorted(list(no_decay))
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optim_groups, lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = WarmupCosLR(
            optimizer=optimizer,
            lr=self.lr,
            min_lr=1e-5,
            warmup_epochs=self.warmup_epochs,
            epochs=self.epochs,
        )
        return [optimizer], [scheduler]


# Legacy class name kept for old Hydra targets and scripts.
Trainer = ForecastingLightningModule
