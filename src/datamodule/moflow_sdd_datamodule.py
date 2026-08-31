"""MoFlow 协议的 SDD DataModule。

train/val 来自 sdd_train.pkl 的确定性 90/10 划分（seed 固定，两臂一致），
test 来自 sdd_test.pkl（held-out）。checkpoint 选点 monitor=val_minFDE20，
最终报告 held-out test 指标（test_minADE20/test_minFDE20，像素系）。
"""

from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader as TorchDataLoader

from .moflow_sdd_dataset import MoFlowSddDataset
from .moflow_ethucy_dataset import moflow_ethucy_collate_fn


class MoFlowSddDataModule(LightningDataModule):
    def __init__(
        self,
        data_root: str,
        obs_len: int = 8,
        pred_len: int = 12,
        train_batch_size: int = 32,
        val_batch_size: int = 32,
        test_batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        test: bool = False,
        val_ratio: float = 0.1,
        seed: int = 2024,
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.test = test
        self.val_ratio = val_ratio
        self.seed = seed
        self.name = "moflow_sdd"

    def _make(self, split: str):
        return MoFlowSddDataset(
            data_root=self.data_root,
            split=split,
            obs_len=self.obs_len,
            pred_len=self.pred_len,
            val_ratio=self.val_ratio,
            seed=self.seed,
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
