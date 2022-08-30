import typing as tp
from os import path as osp

import albumentations as albu
import cv2
import numpy as np
import pandas as pd
from pydantic import BaseModel
from torch.utils.data import Dataset

from src.config import NUM_CLASSES
from src.constants import TRAIN_IMAGES_PATH


class DFRow(BaseModel):
    filename: str
    mask_0: tp.Optional[str]
    mask_1: tp.Optional[str]
    mask_2: tp.Optional[str]
    mask_3: tp.Optional[str]

    @property
    def masks(self) -> tp.Tuple[tp.Optional[str], tp.Optional[str], tp.Optional[str], tp.Optional[str]]:
        return self.mask_0, self.mask_1, self.mask_2, self.mask_3

    @property
    def target_vector(self) -> np.ndarray:
        target = np.zeros(4, dtype=np.float32)
        for idx, mask in enumerate(self.masks):
            if mask is None:
                continue
            target[idx] = 1
        return target


class SteelDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transforms: albu.Compose):
        self._df = df
        self._transforms = transforms

    def __getitem__(self, idx: int) -> tp.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        row = DFRow(**self._df.iloc[idx])
        image = read_rgb_image(osp.join(TRAIN_IMAGES_PATH, row.filename))
        height, width = image.shape[:2]
        mask = _get_mask_from_row(row, width, height)
        transformed = self._transforms(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask']
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        mask = np.transpose(mask, (2, 0, 1)).astype(np.float32)
        return image, mask, row.target_vector

    def __len__(self) -> int:
        return len(self._df)


def read_rgb_image(img_path: str) -> np.ndarray:
    img = cv2.imread(img_path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _get_mask_from_row(row: DFRow, width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width, NUM_CLASSES), dtype=np.float32)
    for mask_idx, rle_mask in enumerate(row.masks):
        if rle_mask is None:
            continue
        mask[:, :, mask_idx] = _get_mask_from_rle_mask(rle_mask, width, height)
    return mask


def _get_mask_from_rle_mask(rle_mask: str, width: int, height: int) -> np.ndarray:
    mask = np.zeros(height * width, dtype=np.uint8)
    rle_mask = rle_mask.split(' ')
    start_positions = map(int, rle_mask[0::2])  # noqa: WPS349
    seq_lens = map(int, rle_mask[1::2])
    for curr_position, curr_len in zip(start_positions, seq_lens):
        curr_position -= 1
        mask[curr_position:curr_position + curr_len] = 1  # noqa: WPS362
    return mask.reshape((height, width), order='F')
