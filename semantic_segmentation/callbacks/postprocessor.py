import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List

import numpy as np
import tifffile
import torch
import torch.nn.functional as functional
from pytorch_lightning.callbacks import Callback


class Postprocessor(ABC):
  """ Basic representation of postprocessor. """

  @abstractmethod
  def process_logits(self, logits: torch.Tensor) -> torch.Tensor:
    """ Perform a post-processing of the logits predicted by a semantic segmentation network.

    Args:
        logits (torch.Tensor): logits of shape [batch_size x num_classes x H x W]

    Returns:
        torch.Tensor: post-processed logits of shape [batch_size x ? x H x W] (? := depends on implementation)
    """
    raise NotImplementedError

  def save(self, processed_outputs: torch.Tensor, path_to_dir: str, fnames: List[str]) -> None:
    """ Save processed outputs created by this class.

    Args:
        processed_outputs (torch.Tensor): post-processed logits of shape [batch_size x ? x H x W] (? := depends on implementation)
        path_to_dir (str): path to output directory
        fnames (List[str]): raw filenames with fileformat (len(fnames)==batch_size)
    """
    raise NotImplementedError


class KeepLogitsPostprocessor(Postprocessor):
  """ Just keep the predicted logits and do not perform any further processing."""

  def __init__(self):
    self.name = 'logits'

  def process_logits(self, logits: torch.Tensor) -> torch.Tensor:
    """ This method just returns the logits and does not perform any processing.

    Args:
        logits (torch.Tensor): logits of shape [batch_size x num_classes x H x W]

    Returns:
        torch.Tensor: logits of shape [batch_size x num_classes x H x W]
    """
    assert len(logits.shape) == 4

    return logits

  def save(self, processed_outputs: torch.Tensor, path_to_dir: str, fnames: List[str]) -> None:

    path_to_dir = os.path.join(path_to_dir, self.name)
    if not os.path.exists(path_to_dir):
      os.makedirs(path_to_dir, exist_ok=True)

    if not (processed_outputs.device == torch.device('cpu')):
      processed_outputs = processed_outputs.cpu()  # [batch_size x num_classes x H x W]

    with torch.no_grad():
      processed_outputs = processed_outputs.numpy().astype(np.float32)

    # save each batch to disk
    for i, output in enumerate(processed_outputs):
      # output is of shape [H x W x num_classes]
      fname = fnames[i].split('.')[0] + ".tif"
      fpath = os.path.join(path_to_dir, fname)

      assert len(output.shape) == 3
      tifffile.imsave(fpath, output)


class ProbablisticSoftmaxPostprocessor(Postprocessor):
  """ Convert the predicted logits into softmax probabilities. """

  def __init__(self):
    self.name = 'probabilities'

  def process_logits(self, logits: torch.Tensor) -> torch.Tensor:
    """ Convert the predicted logits into softmax probabilities.

    Args:
        logits (torch.Tensor): logits of shape [batch_size x num_classes x H x W]

    Returns:
        torch.Tensor: class probabilities of shape [batch_size x num_classes x H x W]
    """
    assert len(logits.shape) == 4

    softmax_probs = functional.softmax(logits, dim=1)  # [batch_size x num_classes x H x W]

    return softmax_probs

  def save(self, processed_outputs: torch.Tensor, path_to_dir: str, fnames: List[str]) -> None:
    """ Save predicted probabilites for each image as a tiff image.

    Args:
        processed_outputs (torch.Tensor): post-processed logits of shape [batch_size x num_classes x H x W]
        path_to_dir (str): path to output directory
        fnames (List[str]): raw filenames with fileformat
    """
    assert len(fnames) == int(processed_outputs.shape[0])

    path_to_dir = os.path.join(path_to_dir, self.name)
    if not os.path.exists(path_to_dir):
      os.makedirs(path_to_dir, exist_ok=True)

    if not (processed_outputs.device == torch.device('cpu')):
      processed_outputs = processed_outputs.cpu()  # [batch_size x num_classes x H x W]

    with torch.no_grad():
      processed_outputs = processed_outputs.numpy().astype(np.float32)

    # save each batch to disk
    for i, output in enumerate(processed_outputs):
      # output is of shape [H x W x num_classes]
      fname = fnames[i].split('.')[0] + ".tif"
      fpath = os.path.join(path_to_dir, fname)

      assert len(output.shape) == 3
      tifffile.imsave(fpath, output)


def get_postprocessors(cfg: Dict) -> List[Postprocessor]:
  postprocessors = []

  try:
    cfg['postprocessors'].keys()
  except KeyError:
    return postprocessors

  for postprocessors_name in cfg['postprocessors'].keys():
    if postprocessors_name == 'keep_logits_postprocessor':
      postprocessor = KeepLogitsPostprocessor()
      postprocessors.append(postprocessor)
    if postprocessors_name == 'probablistic_softmax_postprocessor':
      postprocessor = ProbablisticSoftmaxPostprocessor()
      postprocessors.append(postprocessor)

  return postprocessors


class PostprocessorrCallback(Callback):
  """ Callback to visualize semantic segmentation.
  """

  def __init__(self, postprocessors: List[Postprocessor], postprocess_train_every_x_epochs: int = 1, postprocess_val_every_x_epochs: int = 1):
    """ Constructor.

    Args:
        postprocess_train_every_x_epochs (int): Frequency of train postprocessing. Defaults to 1.
        postprocess_val_every_x_epochs (int): Frequency of val postprocessing. Defaults to 1.
    """
    super().__init__()
    self.postprocessors = postprocessors
    self.postprocess_train_every_x_epochs = postprocess_train_every_x_epochs
    self.postprocess_val_every_x_epochs = postprocess_val_every_x_epochs

  def on_train_batch_end(self, trainer, pl_module, outputs: Dict[str, Any], batch, batch_idx, dataloader_idx):
    # visualize
    epoch = trainer.current_epoch
    if (epoch % self.postprocess_train_every_x_epochs) == 0 and (epoch != 0):
      path = os.path.join(trainer.log_dir, 'train', 'postprocess', f'epoch-{epoch:06d}')

      for postprocessor in self.postprocessors:
        processed = postprocessor.process_logits(outputs['logits'])
        postprocessor.save(processed, path, batch['fname'])

  def on_validation_batch_end(self, trainer, pl_module, outputs: Dict[str, Any], batch, batch_idx, dataloader_idx):
    # visualize
    epoch = trainer.current_epoch
    if ((epoch + 1) % self.postprocess_val_every_x_epochs) == 0 and (epoch != 0):
      path = os.path.join(trainer.log_dir, 'val', 'postprocess', f'epoch-{epoch:06d}')

      for postprocessor in self.postprocessors:
        processed = postprocessor.process_logits(outputs['logits'])
        postprocessor.save(processed, path, batch['fname'])

  def on_test_batch_end(self, trainer, pl_module, outputs: Dict[str, Any], batch, batch_idx, dataloader_idx):
    # visualize
    path = os.path.join(trainer.log_dir, 'postprocess')

    for postprocessor in self.postprocessors:
      processed = postprocessor.process_logits(outputs['logits'])
      postprocessor.save(processed, path, batch['fname'])

  def on_predict_batch_end(self, trainer, pl_module, outputs: Dict[str, Any], batch, batch_idx, dataloader_idx):
    # visualize
    path = os.path.join(trainer.log_dir, 'postprocess')

    for postprocessor in self.postprocessors:
      processed = postprocessor.process_logits(outputs['logits'])
      postprocessor.save(processed, path, batch['fname'])
