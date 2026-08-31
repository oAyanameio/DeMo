"""ETH/UCY benchmark DataModule: fold/split 显式目录，test=True 时只加载 test。"""

from pathlib import Path
from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from .ethucy_benchmark_dataset import EthUcyBenchmarkDataset, ethucy_benchmark_collate_fn


class EthUcyBenchmarkDataModule(LightningDataModule):
    def __init__(
        self,
        data_root: str,
        fold: str,
        obs_len: int = 8,
        pred_len: int = 12,
        train_batch_size: int = 64,
        val_batch_size: int = 64,
        test_batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True,
        test: bool = False,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.fold = fold
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.test = test

    def setup(self, stage: Optional[str] = None) -> None:
        if self.test:
            self.test_dataset = EthUcyBenchmarkDataset(
                self.data_root, self.fold, "test", self.obs_len, self.pred_len
            )
        else:
            self.train_dataset = EthUcyBenchmarkDataset(
                self.data_root, self.fold, "train", self.obs_len, self.pred_len
            )
            self.val_dataset = EthUcyBenchmarkDataset(
                self.data_root, self.fold, "val", self.obs_len, self.pred_len
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.train_batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=self.pin_memory,
            collate_fn=ethucy_benchmark_collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.val_batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=self.pin_memory,
            collate_fn=ethucy_benchmark_collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.test_batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=self.pin_memory,
            collate_fn=ethucy_benchmark_collate_fn,
        )
