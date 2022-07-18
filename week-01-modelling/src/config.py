from datetime import datetime
from functools import partial
import os

import albumentations as albu
import torch
from src.base_config import Config
from src.utils import preprocess_imagenet
from torch.nn import BCEWithLogitsLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau

SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 256
N_EPOCHS = 10
ROOT_PATH = os.path.join(os.environ.get("ROOT_PATH"))

augmentations = albu.Compose(
    [
        albu.HorizontalFlip(),
        albu.VerticalFlip(),
        albu.HueSaturationValue(
            hue_shift_limit=20,
            sat_shift_limit=30,
            val_shift_limit=20,
            p=0.5,
        ),
        albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        albu.ShiftScaleRotate(),
        albu.GaussianBlur(),
    ]
)


config = Config(
    num_workers=4,
    seed=SEED,
    loss=BCEWithLogitsLoss(),
    optimizer=torch.optim.Adam,
    optimizer_kwargs={
        "lr": 1e-3,
        "weight_decay": 5e-4,
    },
    scheduler=ReduceLROnPlateau,
    scheduler_kwargs={
        "mode": "min",
        "factor": 0.1,
        "patience": 5,
    },
    img_size=IMG_SIZE,
    augmentations=augmentations,
    preprocessing=partial(preprocess_imagenet, img_size=IMG_SIZE),
    batch_size=BATCH_SIZE,
    n_epochs=N_EPOCHS,
    early_stop_patience=10,
    model_kwargs={"model_name": "resnet18", "pretrained": True},
    log_metrics=["auc", "f1"],
    binary_thresh=0.1,
    valid_metric="auc",
    minimize_metric=False,
    images_dir=os.path.join(ROOT_PATH, "Images"),
    train_dataset_path=os.path.join(ROOT_PATH, "train_df.csv"),
    valid_dataset_path=os.path.join(ROOT_PATH, "valid_df.csv"),
    test_dataset_path=os.path.join(ROOT_PATH, "test_df.csv"),
    project_name="[Classification]MovieGenre",
    experiment_name=f'experiment_1_{datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}',
)
