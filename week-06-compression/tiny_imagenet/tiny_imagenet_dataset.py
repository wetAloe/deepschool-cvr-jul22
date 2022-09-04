import os

import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

MEAN = np.array([0.406, 0.456, 0.485], dtype=np.float32)
STD = np.array([0.225, 0.224, 0.229], dtype=np.float32)


def get_valid_transform(input_size, mean, std):
    return transforms.Compose([
        transforms.Resize(int(input_size / 0.875)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std, inplace=True),
    ])


class CsvDataset(Dataset):

    def __init__(self, csv_file, data_dir, transform=None, use_cash=True, random_seed: int = 42):
        """Reads CSV annotation with fields "filepath": str, "label": int.

        """
        self.df = pd.read_csv(csv_file, dtype={'filepath': str, 'label': int})
        self.df = self.df.reindex(np.random.RandomState(seed=random_seed).permutation(self.df.index))
        self.data_dir = data_dir
        self.transform = transform
        self.cash = {} if use_cash else None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        if not isinstance(idx, int):
            raise NotImplementedError('Slicing is not implemented')

        path, label = self.df.iloc[idx]
        path = os.path.join(self.data_dir, path)

        if self.cash is None:
            image = Image.open(path).convert('RGB')
        elif idx not in self.cash:
            image = Image.open(path).convert('RGB')
            self.cash[idx] = image
        else:
            image = self.cash[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label
