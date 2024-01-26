import os
from typing import Any, Dict, List, Tuple

import oyaml as yaml
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics


class SegmentationNetwork(pl.LightningModule):

  def __init__(self, network: nn.Module, criterion: nn.Module, learning_rate: float, weight_decay: float):
    super().__init__()

    self.network = network
    self.criterion = criterion
    self.learning_rate = learning_rate
    self.weight_decay = weight_decay

    self.save_hyperparameters("learning_rate", "weight_decay")

    # evaluation metrics
    self.metric_train_iou = torchmetrics.JaccardIndex(network.num_classes, reduction=None)
    self.metric_val_iou = torchmetrics.JaccardIndex(network.num_classes, reduction=None)
    self.metric_test_iou = torchmetrics.JaccardIndex(network.num_classes, reduction=None)

  def compute_loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """ Compute loss based on logits and ground-truths.

    Args:
        logits (torch.Tensor): logits [B x num_classes x H x W]
        y (torch.Tensor): ground-truth [B x H x W]

    Returns:
        torch.Tensor: loss
    """
    return self.criterion(logits, y)

  def forward(self, img_batch: torch.Tensor) -> torch.Tensor:
    """ Forward pass of semantic segmentation network.

    Args:
        img_batch (torch.Tensor): input image(s) [B x C x H x W]

    Returns:
        torch.Tensor: prediction(s) [B x num_classes x H x W]
    """
    logits = self.network(img_batch)

    return logits

  def training_step(self, batch: dict, batch_idx: int) -> Dict[str, Any]:
    logits = self.forward(batch['input_image'])

    # objective
    loss = self.compute_loss(logits, batch['anno'])

    # update training metrics
    self.metric_train_iou(torch.argmax(logits.detach(), dim=1) , batch['anno'].detach())

    return {'loss': loss, 'logits': logits.detach(), 'anno': batch['anno'].detach()}

  def training_epoch_end(self, training_step_outputs: List) -> None:
    # compute loss over all batches
    losses = torch.stack([x['loss'] for x in training_step_outputs])
    train_loss_avg = losses.mean().detach()

    # logging
    epoch = self.trainer.current_epoch
    self.logger.experiment.add_scalars('loss', {'train': train_loss_avg}, epoch)
    self.log("train_loss", train_loss_avg, on_epoch=True, sync_dist=False)

    # compute final metrics over all batches
    iou_per_class = self.metric_train_iou.compute().detach()
    self.metric_train_iou.reset()

    for class_index, iou_class in enumerate(iou_per_class):
      self.logger.experiment.add_scalars(f"iou_class_{class_index}", {'train': iou_class}, epoch)
    self.logger.experiment.add_scalars("mIoU", {'train': iou_per_class.mean()}, epoch)

    path_to_dir = os.path.join(self.trainer.log_dir, 'train', 'evaluation', f'epoch-{epoch:06d}')
    save_iou_metric(iou_per_class, path_to_dir)

  def validation_step(self, batch: dict, batch_idx: int) -> Dict[str, Any]:
    # predictions
    logits = self(batch['input_image']).detach()

    # objective
    loss = self.compute_loss(logits, batch['anno']).detach()

    # update validation metrics
    self.metric_val_iou(torch.argmax(logits, dim=1), batch['anno'])

    return {'loss': loss, 'logits': logits, 'anno': batch['anno'].detach()}

  def validation_epoch_end(self, validation_step_outputs: List) -> None:
    # compute loss over all batches
    losses = torch.stack([x['loss'] for x in validation_step_outputs])
    val_loss_avg = losses.mean()

    # logging
    epoch = self.trainer.current_epoch
    self.logger.experiment.add_scalars('loss', {'val': val_loss_avg}, epoch)
    self.log("val_loss", val_loss_avg, on_epoch=True, sync_dist=False)

    # compute final metrics over all batches
    iou_per_class = self.metric_val_iou.compute().detach()
    self.metric_val_iou.reset()

    for class_index, iou_class in enumerate(iou_per_class):
      self.logger.experiment.add_scalars(f"iou_class_{class_index}", {'val': iou_class}, epoch)
    self.logger.experiment.add_scalars("mIoU", {'val': iou_per_class.mean()}, epoch)

    path_to_dir = os.path.join(self.trainer.log_dir, 'val', 'evaluation', f'epoch-{epoch:06d}')
    save_iou_metric(iou_per_class, path_to_dir)

  def test_step(self, batch: dict, batch_idx: int) -> Dict[str, Any]:
    # predictions
    logits = self(batch['input_image'])

    self.metric_test_iou(torch.argmax(logits.detach(), dim=1), batch['anno'])

    return {'logits': logits.detach(), 'anno': batch['anno'].detach()}

  def test_epoch_end(self, test_step_outputs: List) -> None:
    # compute final metrics over all batches
    final_metric_val_iou = self.metric_test_iou.compute().detach()
    self.metric_test_iou.reset()

    epoch = self.trainer.current_epoch
    path_to_dir = os.path.join(self.trainer.log_dir, 'evaluation', f'epoch-{epoch:06d}')
    save_iou_metric(final_metric_val_iou, path_to_dir)

  def predict_step(self, batch, batch_idx):
    # predictions
    logits = self.forward(batch['input_image'])

    return {'logits': logits}

  def configure_optimizers(self) -> Tuple[List[optim.Optimizer], List[optim.lr_scheduler.LambdaLR]]:
    optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    lambda1 = lambda epoch: pow((1 - ((epoch - 1) / self.trainer.max_epochs)), 0.9)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    return [optimizer], [scheduler]


def save_iou_metric(metrics: torch.Tensor, path_to_dir: str) -> None:
  if not os.path.exists(path_to_dir):
    os.makedirs(path_to_dir)

  iou_info = {}
  for cls_index, iou_metric in enumerate(metrics):
    iou_info[f'class_{cls_index}'] = round(float(iou_metric), 5)

  iou_info['mIoU'] = round(float(metrics.mean()), 5)

  fpath = os.path.join(path_to_dir, "iou.yaml")
  with open(fpath, 'w') as ostream:
    yaml.dump(iou_info, ostream)
