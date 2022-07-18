import logging
import typing as tp

import timm
import torch
from catalyst import dl
from catalyst.core.callback import Callback
from src.base_config import Config
from src.config import config
from src.const import IMAGES, LOGITS, PREDICTS, SCORES, TARGETS, VALID
from src.dataset import get_class_names, get_loaders
from src.loggers import ClearMLLogger
from src.utils import set_global_seed


def get_base_callbacks(class_names: tp.List[str]) -> tp.List[Callback]:
    return [
        dl.BatchTransformCallback(
            transform=torch.sigmoid,
            scope='on_batch_end',
            input_key=LOGITS,
            output_key=SCORES,
        ),
        dl.BatchTransformCallback(
            transform=lambda x: x > config.binary_thresh,
            scope='on_batch_end',
            input_key=SCORES,
            output_key=PREDICTS,
        ),
        dl.AUCCallback(input_key=SCORES, target_key=TARGETS, compute_per_class_metrics=True),
        dl.MultilabelPrecisionRecallF1SupportCallback(
            input_key=PREDICTS,
            target_key=TARGETS,
            num_classes=len(class_names),
            log_on_batch=False,
            compute_per_class_metrics=True,
        ),
    ]


def get_train_callbacks(class_names: tp.List[str]) -> tp.List[Callback]:
    callbacks = get_base_callbacks(class_names)
    callbacks.extend(
        [
            dl.CheckpointCallback(
                logdir=config.checkpoints_dir,
                loader_key=VALID,
                metric_key=config.valid_metric,
                minimize=config.minimize_metric,
            ),
            dl.EarlyStoppingCallback(
                patience=config.early_stop_patience,
                loader_key=VALID,
                metric_key=config.valid_metric,
                minimize=config.minimize_metric,
            ),
        ]
    )
    return callbacks


def train(config: Config):
    loaders, infer_loader = get_loaders(config)
    class_names = get_class_names(config)

    model = timm.create_model(num_classes=len(class_names), **config.model_kwargs)
    optimizer = config.optimizer(params=model.parameters(), **config.optimizer_kwargs)
    scheduler = config.scheduler(optimizer=optimizer, **config.scheduler_kwargs)
    clearml_logger = ClearMLLogger(config, class_names)

    runner = dl.SupervisedRunner(
        input_key=IMAGES,
        output_key=LOGITS,
        target_key=TARGETS,
    )

    runner.train(
        model=model,
        criterion=config.loss,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,
        callbacks=get_train_callbacks(class_names),
        loggers={'_clearml': clearml_logger},
        num_epochs=config.n_epochs,
        valid_loader=VALID,
        valid_metric=config.valid_metric,
        minimize_valid_metric=config.minimize_metric,
        seed=config.seed,
        verbose=True,
        load_best_on_end=True,
    )

    metrics = runner.evaluate_loader(
        model=model,
        loader=infer_loader['infer'],
        callbacks=get_base_callbacks(class_names),
        verbose=True,
        seed=config.seed,
    )

    clearml_logger.log_metrics(metrics, scope='loader', runner=runner, infer=True)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    set_global_seed(config.seed)
    train(config)
