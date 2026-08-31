"""ETH/UCY DataModule for PyTorch Lightning。

支持 Leave-One-Scene-Out 的数据划分。
"""

from pathlib import Path
from typing import List, Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader as TorchDataLoader

from .ethucy_dataset import EthUcyDataset, ethucy_collate_fn


class EthUcyDataModule(LightningDataModule):
    """ETH/UCY 数据模块。

    参数：
        data_root: 预处理后的数据根目录
        train_scenes: 训练场景列表
        val_scenes: 验证场景列表
        test_scenes: 测试场景列表
        obs_len: 历史帧数
        pred_len: 预测帧数
        batch_size: 批大小
        num_workers: 数据加载线程数
    """

    def __init__(
        self,
        data_root: str,
        train_scenes: Optional[List[str]] = None,
        val_scenes: Optional[List[str]] = None,
        test_scenes: Optional[List[str]] = None,
        obs_len: int = 8,
        pred_len: int = 12,
        train_batch_size: int = 32,
        val_batch_size: int = 32,
        test_batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        test: bool = False,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.train_scenes = train_scenes or ["HOTEL", "UNIV", "ZARA1", "ZARA2"]
        self.val_scenes = val_scenes or ["HOTEL"]
        self.test_scenes = test_scenes or ["ETH"]
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.test = test

    def setup(self, stage: Optional[str] = None) -> None:
        if not self.test:
            self.train_dataset = EthUcyDataset(
                data_root=self.data_root,
                scene_names=self.train_scenes,
                obs_len=self.obs_len,
                pred_len=self.pred_len,
            )
            self.val_dataset = EthUcyDataset(
                data_root=self.data_root,
                scene_names=self.val_scenes,
                obs_len=self.obs_len,
                pred_len=self.pred_len,
            )
        else:
            self.test_dataset = EthUcyDataset(
                data_root=self.data_root,
                scene_names=self.test_scenes,
                obs_len=self.obs_len,
                pred_len=self.pred_len,
            )

    def train_dataloader(self):
        return TorchDataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=ethucy_collate_fn,
        )

    def val_dataloader(self):
        return TorchDataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=ethucy_collate_fn,
        )

    def test_dataloader(self):
        return TorchDataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=ethucy_collate_fn,
        )