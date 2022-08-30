import os
from os import path as osp

import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.seed import seed_everything

from src.config import get_config, Config
from src.constants import EXPERIMENTS_PATH
from src.datamodule import SteelDM
from src.model import SteelModule


def _train(config: Config):
    datamodule = SteelDM(config)
    model = SteelModule(config)
    experiment_save_path = osp.join(EXPERIMENTS_PATH, config.experiment_name)
    os.makedirs(experiment_save_path, exist_ok=True)
    monitor_metric = config.monitor_metric
    checkpoint_callback = ModelCheckpoint(
        experiment_save_path,
        monitor=monitor_metric,
        mode=config.monitor_mode,
        save_top_k=1,
        filename=f'epoch_{{epoch:02d}}-{monitor_metric}_{{{monitor_metric}:.3f}}',
    )
    trainer = pl.Trainer(
        max_epochs=config.epochs,
        accelerator='gpu',
        devices=[0],
        logger=WandbLogger(
            project=config.project_name, name=config.experiment_name, log_model=False, config=config.to_dict(),
        ),
        callbacks=[checkpoint_callback, *config.callbacks],
        deterministic=True,
    )
    trainer.fit(model, datamodule=datamodule)
    trainer.test(ckpt_path=checkpoint_callback.best_model_path, datamodule=datamodule)


if __name__ == '__main__':
    seed_everything(42, workers=True)
    config = get_config()
    _train(config)
