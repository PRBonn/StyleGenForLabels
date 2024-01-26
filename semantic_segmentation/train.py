""" Train semantic segmentation model.
"""
import argparse
import os
import pdb
from typing import Dict

import git
import oyaml as yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from callbacks import (ConfigCallback, PostprocessorrCallback,
                       VisualizerCallback, get_postprocessors, get_visualizers)
from datasets import get_data_module
from modules import get_criterion, module
from modules.erfnet import ERFNetModel


def get_git_commit_hash() -> str:
  repo = git.Repo(search_parent_directories=True)
  sha = repo.head.object.hexsha

  return sha

def parse_args() -> Dict[str, str]:
  parser = argparse.ArgumentParser()
  parser.add_argument("--export_dir", required=True, help="Path to export dir which saves logs, metrics, etc.")
  parser.add_argument("--config", required=True, help="Path to configuration file (*.yaml)")
  parser.add_argument("--ckpt_path", required=False, default=None, help="Provide *.ckpt file to continue training.")

  args = vars(parser.parse_args())

  return args


def load_config(path_to_config_file: str) -> Dict:
  assert os.path.exists(path_to_config_file)

  with open(path_to_config_file) as istream:
    config = yaml.safe_load(istream)

  config['git-commit'] = get_git_commit_hash()

  return config


def main():
  args = parse_args()
  cfg = load_config(args['config'])

  datasetmodule = get_data_module(cfg)
  criterion = get_criterion(cfg)

  n_classes = len(cfg['train']['class_weights'])
  network = ERFNetModel(n_classes)

  seg_module = module.SegmentationNetwork(network, criterion, cfg['train']['learning_rate'],
                                          cfg['train']['weight_decay'])

  # Add callbacks
  lr_monitor = LearningRateMonitor(logging_interval='epoch')
  checkpoint_saver_every_n = ModelCheckpoint(filename='esv-{epoch:02d}-{step}-{val_loss:.4f}', monitor="step", mode="max", every_n_train_steps=50000, save_top_k=-1, save_last=True)
  checkpoint_saver_val_loss = ModelCheckpoint(
      monitor='val_loss', filename=cfg['experiment']['id'] + '_{epoch:02d}_{val_loss:.4f}', mode='min', save_last=True)
  checkpoint_saver_train_loss = ModelCheckpoint(
      monitor='train_loss', filename=cfg['experiment']['id'] + '_{epoch:02d}_{train_loss:.4f}', mode='min', save_last=False)
  visualizer_callback = VisualizerCallback(get_visualizers(cfg), cfg['train']['vis_train_every_x_epochs'], cfg['val']['vis_val_every_x_epochs'])
  postprocessor_callback = PostprocessorrCallback(
      get_postprocessors(cfg), cfg['train']['postprocess_train_every_x_epochs'], cfg['val']['postprocess_val_every_x_epochs'])
  config_callback = ConfigCallback(cfg)

  # Setup trainer
  trainer = Trainer(
      benchmark=cfg['train']['benchmark'],
      gpus=cfg['train']['n_gpus'],
      default_root_dir=args['export_dir'],
      max_epochs=cfg['train']['max_epoch'],
      check_val_every_n_epoch=cfg['val']['check_val_every_n_epoch'],
      callbacks=[checkpoint_saver_val_loss, checkpoint_saver_train_loss, checkpoint_saver_every_n, lr_monitor, visualizer_callback, postprocessor_callback, config_callback])

  if args['ckpt_path'] is None:
    trainer.fit(seg_module, datasetmodule)
  else:
    trainer.fit(seg_module, datasetmodule, ckpt_path=args['ckpt_path'])


if __name__ == '__main__':
  main()
