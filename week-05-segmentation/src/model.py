import torch
import pytorch_lightning as pl

from src.config import Config


class SteelModule(pl.LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self._config = config
        self.model = config.model
        self.save_hyperparameters(self._config.to_dict())

    def configure_optimizers(self):
        optimizer = self._config.optimizer(self.model.parameters(), **self._config.optimizer_kwargs)
        scheduler = self._config.scheduler(optimizer, **self._config.scheduler_kwargs)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': self._config.monitor_metric,
                'interval': 'epoch',
                'frequency': 1,
            },
        }

    def training_step(self, batch, batch_idx):
        images, gt_masks, gt_targets = batch
        pred_masks_logits, pred_labels_logits = self.model(images)
        return self._calculate_loss(pred_masks_logits, pred_labels_logits, gt_masks, gt_targets, 'train')

    def validation_step(self, batch, batch_idx):
        images, gt_masks, gt_targets = batch
        pred_masks_logits, pred_labels_logits = self.model(images)
        self._calculate_loss(pred_masks_logits, pred_labels_logits, gt_masks, gt_targets, 'val')
        self._calculate_metrics(pred_masks_logits, pred_labels_logits, gt_masks, gt_targets, 'val')

    def test_step(self, batch, batch_idx):
        images, gt_masks, gt_targets = batch
        pred_masks_logits, pred_labels_logits = self.model(images)
        self._calculate_loss(pred_masks_logits, pred_labels_logits, gt_masks, gt_targets, 'test')
        self._calculate_metrics(pred_masks_logits, pred_labels_logits, gt_masks, gt_targets, 'test')

    def _calculate_loss(
        self,
        pred_masks_logits: torch.Tensor,
        pred_labels_logits: torch.Tensor,
        gt_masks: torch.Tensor,
        gt_targets: torch.Tensor,
        prefix: str,
    ) -> torch.Tensor:
        total_loss = 0
        for seg_loss in self._config.seg_losses:
            loss = seg_loss.loss(pred_masks_logits, gt_masks)
            total_loss += seg_loss.weight * loss
            self.log(f'{prefix}_{seg_loss.name}', loss.item(), batch_size=self._config.batch_size)
        for cls_loss in self._config.cls_losses:
            loss = cls_loss.loss(pred_labels_logits, gt_targets)
            total_loss += cls_loss.weight * loss
            self.log(f'{prefix}_{cls_loss.name}_loss', loss.item(), batch_size=self._config.batch_size)
        self.log(f'{prefix}_total_loss', total_loss.item(), batch_size=self._config.batch_size)
        return total_loss

    def _calculate_metrics(
        self,
        pred_masks_logits: torch.Tensor,
        pred_labels_logits: torch.Tensor,
        gt_masks: torch.Tensor,
        gt_targets: torch.Tensor,
        prefix: str,
    ):
        for seg_metric in self._config.seg_metrics:
            metric_value = seg_metric.metric(pred_masks_logits, gt_masks)
            self.log(f'{prefix}_{seg_metric.name}', metric_value, batch_size=self._config.batch_size)
        for cls_metric in self._config.cls_metrics:
            metric_value = cls_metric.metric(pred_labels_logits, gt_targets)
            self.log(f'{prefix}_{cls_metric.name}', metric_value, batch_size=self._config.batch_size)
