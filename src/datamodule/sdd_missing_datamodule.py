"""SDD missing-history v1 DataModule。

data_root 指向 data/SDD_missing_v1，condition 决定读取哪个子目录。
train/val/test 的样本清单由数据制作阶段固定（split_seed=2024），
本模块只按目录读取，不做任何再划分。
"""

from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader as TorchDataLoader

from .moflow_ethucy_dataset import moflow_ethucy_collate_fn
from .sdd_missing_dataset import SddMissingDataset


class SddMissingDataModule(LightningDataModule):
    def __init__(
        self,
        data_root: str,
        condition: str,
        obs_len: int = 8,
        pred_len: int = 12,
        train_batch_size: int = 32,
        val_batch_size: int = 32,
        test_batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        test: bool = False,
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.condition = condition
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.test = test
        self.name = f"sdd_missing_{condition}"

    def _make(self, split: str):
        return SddMissingDataset(
            data_root=self.data_root,
            condition=self.condition,
            split=split,
            obs_len=self.obs_len,
            pred_len=self.pred_len,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        if not self.test:
            self.train_dataset = self._make("train")
            self.val_dataset = self._make("val")
        else:
            self.test_dataset = self._make("test")

    def _loader(self, ds, bs, shuffle):
        return TorchDataLoader(
            ds,
            batch_size=bs,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=moflow_ethucy_collate_fn,
        )

    def train_dataloader(self):
        return self._loader(self.train_dataset, self.train_batch_size, True)

    def val_dataloader(self):
        return self._loader(self.val_dataset, self.val_batch_size, False)

    def test_dataloader(self):
        return self._loader(self.test_dataset, self.test_batch_size, False)
