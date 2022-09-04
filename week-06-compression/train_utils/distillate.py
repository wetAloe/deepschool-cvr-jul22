from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional
from typing import Type
from typing import Union

import tensorboardX
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from tqdm import tqdm

_DEFAULT_SGD_OPTIMIZER_PARAMS = {
    'lr': 0.001,
    'momentum': 0.9,
}

_DEFAULT_SCHEDULER_PARAMS = {
    'T_0': 25000 // 32,
}


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, yhat: torch.tensor, y: torch.Tensor) -> torch.Tensor:
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss


class RunningMeanLogger:
    """Track Running mean in tracker_value"""

    def __init__(self, alpha: float = 0.9):
        self._tracked_value = None
        self._alpha = alpha

    @property
    def tracked_value(self):
        return self._tracked_value

    @tracked_value.setter
    def tracked_value(self, new_val: float):
        if self._tracked_value is not None:
            self._tracked_value = new_val * (1 - self._alpha) + self._tracked_value * self._alpha
        else:
            self._tracked_value = new_val


class ImageNetDistiller:
    """Class handle distillation for classification dataset. """

    def __init__(
            self,
            quantized_model: nn.Module,
            regular_model: nn.Module,
            train_data_loader: DataLoader,
            logdir: Union[Path, str],
            optimizer: Type[Optimizer] = SGD,
            optimizer_params: Optional[Dict[str, Any]] = None,
            scheduler: Optional = CosineAnnealingWarmRestarts,
            scheduler_params: Optional[Dict[str, Any]] = None,
            distillation_criterion: nn.Module = RMSELoss,
            number_of_epoch=2,
            save_every: int = 100,
            device: str = 'cuda:0'
    ):
        self.quantized_model = quantized_model
        self.regular_model = regular_model
        self.train_data_loader = train_data_loader
        self.logdir = Path(logdir)
        self.distillation_criterion = distillation_criterion()
        self.number_of_epoch = number_of_epoch
        self.save_every = save_every
        self.device = torch.device(device)

        self.quantized_model.to(self.device)
        self.regular_model.to(self.device)
        # Вообще в дистиляции студент должен быть в train(),
        # Но при квантовании лучше вообще от всех этих фишек избавится
        # Чтобы схема fake quant была максимальна близка к итоговой модели
        self.quantized_model.eval()
        self.regular_model.eval()

        # Initialize optimizer
        self.optimizer_params = optimizer_params if optimizer_params is not None else _DEFAULT_SGD_OPTIMIZER_PARAMS
        self.optimizer = optimizer(self.quantized_model.parameters(), **self.optimizer_params)
        self.scheduler_params = scheduler_params if scheduler_params is not None else _DEFAULT_SCHEDULER_PARAMS
        self.scheduler = scheduler(self.optimizer, **self.scheduler_params)

        # Initialize loggers
        self.logdir.mkdir(exist_ok=True, parents=True)
        self.tensorboard_logger = tensorboardX.SummaryWriter(
            log_dir=(self.logdir / 'tensorboard_logs').as_posix()
        )
        self.loss_logger = RunningMeanLogger()
        self.model_logdir = self.logdir / 'saved_models'
        self.model_logdir.mkdir(exist_ok=True, parents=True)
        self.minimal_loss_value = 1e9

    def train(self):
        total_steps = 0
        for epoch in range(self.number_of_epoch):
            tqdm_iterator = tqdm(self.train_data_loader)
            for images, _ in tqdm_iterator:
                self.optimizer.zero_grad()

                # Forward And Backward Passes
                input_tensor = images.to(self.device)
                quantized_output = self.quantized_model(input_tensor)
                # Обратите внимание, нет никакого смысла считать градиенты для учителя
                # Тут вообще не обязательно чтобы это была реальная сетка
                # Хоть с диска вектора в pickle читайте.
                with torch.no_grad():
                    regular_output = self.regular_model(input_tensor)

                loss = self.distillation_criterion(quantized_output, regular_output)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                total_steps += 1

                # Model logging
                if total_steps % self.save_every == 1:
                    torch.save(
                        self.quantized_model.state_dict(),
                        self.model_logdir / f'model_{total_steps}.pth'
                    )

                self.loss_logger.tracked_value = loss.item()
                if self.minimal_loss_value > self.loss_logger.tracked_value:
                    self.minimal_loss_value = self.loss_logger.tracked_value
                    torch.save(
                        self.quantized_model.state_dict(),
                        self.model_logdir / 'best_model.pth'
                    )

                # Values logging
                tqdm_iterator.set_description(f'loss = {self.loss_logger.tracked_value:.5f}')
                self.tensorboard_logger.add_scalar('train_loss', loss.item(), global_step=total_steps)
                self.tensorboard_logger.add_scalar('lr', self.scheduler.get_last_lr()[0], global_step=total_steps)

        self.quantized_model.load_state_dict(
            torch.load(self.model_logdir / 'best_model.pth')
        )
        return self.model_logdir / 'best_model.pth'
