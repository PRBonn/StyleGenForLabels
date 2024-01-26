import os
from typing import Dict, List, Optional, Callable

import numpy as np
from datasets import image_normalizer
import pytorch_lightning as pl
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import datasets.common as common
from datasets.image_normalizer import ImageNormalizer, get_image_normalizer
from datasets.augmentations_geometry import GeometricDataAugmentation, get_geometric_augmentations
from datasets.augmentations_color import get_color_augmentations


class UAVDataset(Dataset):
  """ Represents the UAV Bonn dataset.

  The directory structure is as following:
  ├── annotations
  └── images
      └── rgb
  └── split.yaml
  """

  def __init__(self, path_to_dataset: str, filenames: List[str], mode: str, img_normalizer: ImageNormalizer,
               augmentations_geometric: List[GeometricDataAugmentation], augmentations_color: List[Callable]):
    """ Get the path to all images and its corresponding annotations.

    Args:
        path_to_dataset (str): Path to dir that contains the images and annotations
        filenames (List[str]): List of filenames which are considered to be part of this dataset, e.g, [filename01.png, filename02.png, ...]
        mode(str): Train, val, or test
        img_normalizer (ImageNormalizer): Specifies how to normalize the input images
        augmentations_geometric (List[GeometricDataAugmentation]): Geometric data augmentations applied to the image and its annotations
        augmentations_color (List[Callable]): Color data augmentations applied to the image
    """

    assert os.path.exists(path_to_dataset), f"The path to the dataset does not exist: {path_to_dataset}."
    assert filenames, "The dataset is empty."

    super().__init__()

    self.filenames = filenames

    assert mode in ['train', 'val', 'test', 'predict']
    self.mode = mode

    self.img_normalizer = img_normalizer
    self.augmentations_geometric = augmentations_geometric
    self.augmentations_color = augmentations_color

    # get path to all RGB images
    self.path_to_images = os.path.join(path_to_dataset, "images", "rgb")
    assert os.path.exists(path_to_dataset)

    # get path to all annotations
    self.path_to_annos = os.path.join(path_to_dataset, "annotations")
    if mode != 'predict':
      assert os.path.exists(self.path_to_annos)

    # specify image transformations
    self.img_to_tensor = transforms.ToTensor()

  def __getitem__(self, idx: int):
    path_to_current_img = os.path.join(self.path_to_images, self.filenames[idx])
    img_pil = Image.open(path_to_current_img)
    img = self.img_to_tensor(img_pil)  # [C x H x W]

    for augmentor_color_fn in self.augmentations_color:
      img = augmentor_color_fn(img)

    if self.mode != 'predict':
      path_to_current_anno = os.path.join(self.path_to_annos, self.filenames[idx])
      anno = np.array(Image.open(path_to_current_anno))  # dtype: int32
      if len(anno.shape) > 2:
        anno = anno[:, :, 0]
      anno = anno.astype(np.int64)
      anno = torch.Tensor(anno).type(torch.int64)  # [H x W]
      anno = anno.unsqueeze(0)  # [1 x H x W]
    else:
      # create dummy anno to make everything conistent
      anno = torch.zeros((1, img.shape[1], img.shape[2]), dtype=torch.int64)

    for augmentor_geometric in self.augmentations_geometric:
      img, anno = augmentor_geometric(img, anno)

    anno = anno.squeeze(0)  # [H x W]
    crop_mask = anno == 10000
    anno[crop_mask]=1
    crop_mask = anno == 16
    anno[crop_mask]=1

    img_before_norm = img.clone()
    img = self.img_normalizer.normalize(img)

    return {'input_image_before_norm': img_before_norm, 'input_image': img, 'anno': anno, 'fname': self.filenames[idx]}

  def __len__(self) -> int:
    return len(self.filenames)


class UAVBonnDataModule(pl.LightningDataModule):
  """ Encapsulates all the steps needed to process data from UAV Bonn.
  """

  def __init__(self, cfg: Dict):
    super().__init__()

    self.cfg = cfg

  def setup(self, stage: Optional[str] = None):
    """ Data operations we perform on every GPU.

    Here we define the how to split the dataset.

    Args:
        stage (Optional[str], optional): _description_. Defaults to None.
    """
    path_to_dataset = self.cfg['data']['path_to_dataset']
    image_normalizer = get_image_normalizer(self.cfg)

    path_to_split_file = os.path.join(self.cfg['data']['path_to_dataset'], 'split.yaml')
    if stage != 'predict':
      if self.cfg['data']['check_data_split']:
        split_file_is_valid = common.check_split_file(path_to_split_file)
        assert split_file_is_valid, "The train, val, and test splits 'split.yaml' are not mutually exclusive."

      with open(path_to_split_file) as istream:
        split_info = yaml.safe_load(istream)

      if (stage == 'fit') or (stage == 'validate') or (stage is None):
        # ----------- TRAIN -----------
        train_filenames = split_info['train']
        train_filenames.sort()

        if self.cfg['train']['dataset_size'] is not None:
          train_filenames = train_filenames[:self.cfg['train']['dataset_size']]

        train_augmentations_geometric = get_geometric_augmentations(self.cfg, 'train')
        train_augmentations_color = get_color_augmentations(self.cfg, 'train')
        self._uav_train = UAVDataset(
            path_to_dataset,
            train_filenames,
            mode='train',
            img_normalizer=image_normalizer,
            augmentations_geometric=train_augmentations_geometric,
            augmentations_color=train_augmentations_color)

        # ----------- VAL -----------
        val_filenames = split_info['valid']
        val_filenames.sort()

        if self.cfg['val']['dataset_size'] is not None:
          val_filenames = val_filenames[:self.cfg['val']['dataset_size']]

        val_augmentations_geometric = get_geometric_augmentations(self.cfg, 'val')
        self._uav_val = UAVDataset(
            path_to_dataset,
            val_filenames,
            mode='val',
            img_normalizer=image_normalizer,
            augmentations_geometric=val_augmentations_geometric,
            augmentations_color=[])

      if stage == 'test' or stage is None:
        # ----------- TEST -----------
        test_filenames = split_info['test']
        test_filenames.sort()

        if self.cfg['test']['dataset_size'] is not None:
          test_filenames = test_filenames[:self.cfg['test']['dataset_size']]

        test_augmentations_geometric = get_geometric_augmentations(self.cfg, 'test')
        self._uav_test = UAVDataset(
            path_to_dataset,
            test_filenames,
            mode='test',
            img_normalizer=image_normalizer,
            augmentations_geometric=test_augmentations_geometric,
            augmentations_color=[])

    if stage == "predict":
      predict_augmentations_geometric = get_geometric_augmentations(self.cfg, 'predict')

      predict_filenames = []
      path_to_images = os.path.join(path_to_dataset, 'images', 'rgb')

      for fname in os.listdir(path_to_images):
        if common.is_image(fname):
          predict_filenames.append(fname)

      self._uav_predict = UAVDataset(
          path_to_dataset,
          predict_filenames,
          mode='predict',
          img_normalizer=image_normalizer,
          augmentations_geometric=predict_augmentations_geometric,
          augmentations_color=[])

  def train_dataloader(self) -> DataLoader:
    # Return DataLoader for Training Data here
    shuffle: bool = self.cfg['train']['shuffle']
    batch_size: int = self.cfg['train']['batch_size']
    n_workers: int = self.cfg['data']['num_workers']

    loader = DataLoader(self._uav_train, batch_size=batch_size, shuffle=shuffle, num_workers=n_workers, drop_last=True, pin_memory=True)

    return loader

  def val_dataloader(self) -> DataLoader:
    batch_size: int = self.cfg['val']['batch_size']
    n_workers: int = self.cfg['data']['num_workers']

    loader = DataLoader(self._uav_val, batch_size=batch_size, num_workers=n_workers, shuffle=False, drop_last=True, pin_memory=True)

    return loader

  def test_dataloader(self) -> DataLoader:
    batch_size: int = self.cfg['test']['batch_size']
    n_workers: int = self.cfg['data']['num_workers']

    loader = DataLoader(self._uav_test, batch_size=batch_size, num_workers=n_workers, shuffle=False, pin_memory=True)

    return loader

  def predict_dataloader(self):
    batch_size: int = self.cfg['predict']['batch_size']
    n_workers: int = self.cfg['data']['num_workers']

    loader = DataLoader(self._uav_predict, batch_size=batch_size, num_workers=n_workers, shuffle=False, pin_memory=True)

    return loader
