"""MoFlow 协议的 ETH/UCY DataModule。

标准 leave-one-out (LOO)：每个 subset（eth/hotel/univ/zara1/zara2）为一折，
该 subset 作 test，其余 4 个场景作 train/val（SocialGAN 官方划分，
与 MoFlow 论文、SGAN/Social-STGCNN/AgentFormer/LED 等一致）。
"""

from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader as TorchDataLoader

from .moflow_ethucy_dataset import MoFlowEthUcyDataset, moflow_ethucy_collate_fn


class MoFlowEthUcyDataModule(LightningDataModule):
    def __init__(
        self,
        data_root: str,
        subset: str = "eth",
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
        self.subset = subset
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.test = test
        self.name = "moflow_ethucy"

    def _make(self, split: str):
        return MoFlowEthUcyDataset(
            data_root=self.data_root,
            subset=self.subset,
            split=split,
            obs_len=self.obs_len,
            pred_len=self.pred_len,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        if not self.test:
            self.train_dataset = self._make("train")
            # val pkl 若不存在则退回 test pkl（MoFlow 论文以 test 集报告指标）
            try:
                self.val_dataset = self._make("val")
            except AssertionError:
                self.val_dataset = self._make("test")
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
