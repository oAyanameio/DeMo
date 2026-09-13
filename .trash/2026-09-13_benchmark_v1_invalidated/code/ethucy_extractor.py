"""ETH/UCY 数据预处理提取器。

将原始 .txt 文件转换为按场景组织的 .pt 样本文件。
"""

import traceback
from pathlib import Path
from typing import List, Optional

import torch

from .ethucy_utils import (
    DEFAULT_FRAME_STRIDE,
    DEFAULT_OBS_LEN,
    DEFAULT_PRED_LEN,
    load_ethucy_file,
    normalize_scene_name,
    resample_scene,
    sliding_window_samples,
)


class EthUcyExtractor:
    """ETH/UCY 数据预处理提取器。"""

    def __init__(
        self,
        save_path: Path,
        frame_stride: int = DEFAULT_FRAME_STRIDE,
        obs_len: int = DEFAULT_OBS_LEN,
        pred_len: int = DEFAULT_PRED_LEN,
    ) -> None:
        self.save_path = save_path
        self.frame_stride = frame_stride
        self.obs_len = obs_len
        self.pred_len = pred_len

    def save(self, file: Path) -> None:
        """解析单个 .txt 文件并保存为 .pt 样本。"""
        assert self.save_path is not None
        try:
            scene_name = normalize_scene_name(file)
            df = load_ethucy_file(file)
            df = resample_scene(df, frame_stride=self.frame_stride)
            samples = sliding_window_samples(
                df,
                obs_len=self.obs_len,
                pred_len=self.pred_len,
            )

            # 为每个样本设置 scene_id
            for s in samples:
                s["scene_id"] = scene_name

            scene_dir = self.save_path / scene_name
            scene_dir.mkdir(parents=True, exist_ok=True)

            # 每个样本保存为一个 .pt 文件
            for i, sample in enumerate(samples):
                save_file = scene_dir / f"{i:06d}.pt"
                torch.save(sample, save_file)

            print(f"  {scene_name}: {len(samples)} samples from {file.name}")
        except Exception:
            print(traceback.format_exc())
            print(f"  Error extracting data from {file}")

    @staticmethod
    def glob_files(data_root: Path) -> List[Path]:
        """递归收集所有 .txt 文件。"""
        return sorted(data_root.rglob("*.txt"))