"""ETH/UCY 提交处理器。

保存预测结果并支持反变换到全局坐标。
"""

import pickle
import time
from pathlib import Path

import torch
from torch import Tensor


class SubmissionEthUcy:
    """ETH/UCY 预测结果处理器。

    保存格式：
        {
            "scene_id": str,
            "track_id": int,
            "predictions": Tensor[6, 12, 2],
            "probabilities": Tensor[6],
            "origin": Tensor[2],
            "theta": Tensor[1],
        }
    """

    def __init__(self, save_dir: str = "") -> None:
        self.save_dir = Path(save_dir) if save_dir else Path(".")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.results: list = []

    def format_data(
        self,
        data: dict,
        trajectory: Tensor,
        probability: Tensor,
        normalized_probability: bool = False,
        inference: bool = False,
    ):
        """将模型输出转换为 ETH/UCY 格式。

        trajectory: (B, M, 12, 2)
        probability: (B, M)
        """
        batch = len(data["track_id"])

        origin = data["origin"].view(batch, 1, 1, 2).double()
        theta = data["theta"].double()

        rotate_mat = torch.stack(
            [
                torch.cos(theta),
                torch.sin(theta),
                -torch.sin(theta),
                torch.cos(theta),
            ],
            dim=1,
        ).reshape(batch, 2, 2)

        with torch.no_grad():
            global_trajectory = (
                torch.matmul(
                    trajectory[..., :2].double(), rotate_mat.unsqueeze(1)
                )
                + origin
            )
            if not normalized_probability:
                probability = torch.softmax(probability.double(), dim=-1)

        global_trajectory = global_trajectory.detach().cpu()
        probability = probability.detach().cpu()

        if inference:
            return global_trajectory, probability

        for i, (scene_id, track_id) in enumerate(
            zip(data["scene_id"], data["track_id"])
        ):
            self.results.append(
                {
                    "scene_id": scene_id,
                    "track_id": track_id,
                    "predictions": global_trajectory[i],
                    "probabilities": probability[i],
                    "origin": origin[i].cpu(),
                    "theta": theta[i].cpu(),
                }
            )

    def generate_submission_file(self):
        """保存所有预测结果到 pickle 文件。"""
        stamp = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
        save_file = self.save_dir / f"ethucy_predictions_{stamp}.pkl"
        with open(save_file, "wb") as f:
            pickle.dump(self.results, f)
        print(f"ETH/UCY predictions saved to {save_file}")
        print(f"Total predictions: {len(self.results)}")