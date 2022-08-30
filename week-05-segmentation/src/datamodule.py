import typing as tp
from os import path as osp

import numpy as np
import pandas as pd
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from src.config import Config
from src.constants import DATA_PATH
from src.dataset import SteelDataset


class SteelDM(LightningDataModule):
    def __init__(self, config: Config):
        super().__init__()
        self._batch_size = config.batch_size
        self._n_workers = config.num_workers
        self._train_size = config.train_size
        self._train_augs = config.train_augmentations
        self._test_augs = config.test_augmentations

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: tp.Optional[str] = None):
        df = _read_steel_df()
        train_df, other_df = _multilabel_stratified_split(df, self._train_size)
        val_df, test_df = _multilabel_stratified_split(other_df, 0.5)
        self.train_dataset = SteelDataset(train_df, self._train_augs)
        self.val_dataset = SteelDataset(val_df, self._test_augs)
        self.test_dataset = SteelDataset(test_df, self._test_augs)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self._batch_size,
            num_workers=self._n_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self._batch_size,
            num_workers=self._n_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self._batch_size,
            num_workers=self._n_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )


def _read_steel_df() -> pd.DataFrame:
    df = pd.read_csv(osp.join(DATA_PATH, 'train.csv'))
    df = df.pivot(index='ImageId', columns='ClassId', values='EncodedPixels')
    df = pd.DataFrame(df.to_records())
    df = df.rename({'ImageId': 'filename', '1': 'mask_0', '2': 'mask_1', '3': 'mask_2', '4': 'mask_3'}, axis=1)
    df = df.where(pd.notnull(df), None)
    return df


def _multilabel_stratified_split(df: pd.DataFrame, train_size: float) -> tp.Tuple[pd.DataFrame, pd.DataFrame]:
    n_folds = _get_num_folds_from_train_size(train_size)
    mskf = MultilabelStratifiedKFold(n_splits=n_folds, random_state=144, shuffle=True)
    targets = _get_targets_for_each_row(df)
    split = mskf.split(df, targets)
    train_index, other_index = list(split)[0]
    train_df = pd.DataFrame(df.values[train_index], columns=df.columns)
    other_df = pd.DataFrame(df.values[other_index], columns=df.columns)
    return train_df, other_df


def _get_num_folds_from_train_size(train_size: float) -> int:
    val_size = 1 - train_size
    return int(train_size / val_size) + 1


def _get_targets_for_each_row(df: pd.DataFrame) -> np.ndarray:
    targets = df.fillna(-1).drop('filename', axis=1).values
    targets[targets != -1] = 1
    return targets
