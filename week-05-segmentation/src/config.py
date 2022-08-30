import typing as tp
import json
from dataclasses import asdict, dataclass

import albumentations as albu
import torch
import segmentation_models_pytorch as smp
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR
from pytorch_lightning.callbacks import Callback, EarlyStopping, LearningRateMonitor
from segmentation_models_pytorch.utils.base import Metric


NUM_CLASSES = 4
INPUT_H = 256
INPUT_W = 1600

METRICS_ACTIVATION = 'sigmoid'
MONITOR_METRIC = 'val_iou'
MONITOR_MODE = 'max'


@dataclass
class NamedMetric:
    name: str
    metric: Metric


@dataclass
class NamedLoss:
    name: str
    weight: float
    loss: nn.Module


@dataclass
class Config:
    project_name: str
    experiment_name: str

    model: nn.Module

    epochs: int
    num_workers: int
    batch_size: int
    train_size: float

    train_augmentations: albu.Compose
    test_augmentations: albu.Compose

    optimizer: type(Optimizer)
    optimizer_kwargs: dict
    scheduler: type(_LRScheduler)
    scheduler_kwargs: dict
    seg_losses: tp.List[NamedLoss]
    cls_losses: tp.List[NamedLoss]

    monitor_metric: str
    monitor_mode: str
    seg_metrics: tp.List[NamedMetric]
    cls_metrics: tp.List[NamedMetric]

    callbacks: tp.List[Callback]

    def to_dict(self) -> dict:
        res = {}
        for k, v in asdict(self).items():
            try:
                if isinstance(v, torch.nn.Module):
                    res[k] = v.__class__.__name__
                elif isinstance(v, dict):  # noqa: WPS220
                    res[k] = json.dumps(v, indent=4)
                else:
                    res[k] = str(v)
            except Exception:
                res[k] = str(v)
        return res


def get_config() -> Config:
    return Config(
        project_name='<your-project-name-in-wandb>',
        experiment_name='<your-experiment-name>',

        model=smp.Unet(
            encoder_name='resnet18',
            encoder_weights='imagenet',
            classes=NUM_CLASSES,
            aux_params={'pooling': 'avg', 'dropout': 0.2, 'classes': NUM_CLASSES},
        ),

        epochs=30,
        num_workers=6,
        batch_size=8,
        train_size=0.8,

        train_augmentations=albu.Compose([
            albu.HorizontalFlip(p=0.5),
            albu.VerticalFlip(p=0.5),
            albu.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            albu.ShiftScaleRotate(),
            albu.GaussianBlur(),
            albu.Resize(height=INPUT_H, width=INPUT_W),
            albu.Normalize(),
        ]),
        test_augmentations=albu.Compose([
            albu.Resize(height=INPUT_H, width=INPUT_W),
            albu.Normalize(),
        ]),

        optimizer=torch.optim.AdamW,
        optimizer_kwargs={'lr': 1e-3, 'weight_decay': 5e-5},
        scheduler=CosineAnnealingLR,
        scheduler_kwargs={'T_max': 10},
        seg_losses=[NamedLoss(name='dice', weight=0.7, loss=smp.losses.DiceLoss(mode='binary'))],
        cls_losses=[NamedLoss(name='bce', weight=0.3, loss=nn.BCEWithLogitsLoss())],

        seg_metrics=[
            NamedMetric(name='iou', metric=smp.utils.metrics.IoU(activation=METRICS_ACTIVATION)),
            NamedMetric(name='accuracy', metric=smp.utils.metrics.Accuracy(activation=METRICS_ACTIVATION)),
            NamedMetric(name='f1', metric=smp.utils.metrics.Fscore(activation=METRICS_ACTIVATION)),
        ],
        cls_metrics=[
            NamedMetric(
                name='accuracy', metric=smp.utils.metrics.Accuracy(activation=METRICS_ACTIVATION, threshold=0.7)
            ),
            NamedMetric(name='f1', metric=smp.utils.metrics.Fscore(activation=METRICS_ACTIVATION, threshold=0.7)),
        ],
        monitor_metric=MONITOR_METRIC,
        monitor_mode=MONITOR_MODE,

        callbacks=[
            EarlyStopping(monitor=MONITOR_METRIC, patience=3, mode=MONITOR_MODE),
            LearningRateMonitor(logging_interval='epoch'),
        ],
    )
