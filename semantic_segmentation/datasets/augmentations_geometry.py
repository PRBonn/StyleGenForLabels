""" Define a set of geometric data augmentations which can be applied to the input image and its corresponding annotations.

This is relevant for the task of semantic segmentation since the input image and its annotation need to be treated in the same way.
"""

import math
import random
from abc import ABC, abstractmethod
from statistics import mode
from typing import List, Optional, Tuple

import torch
from torchvision.transforms import transforms
import torchvision.transforms.functional as functional
import torchvision.transforms as transforms


class GeometricDataAugmentation(ABC):
  """ General transformation which can be applied simultaneously to the input image and its corresponding anntations.
  """

  @abstractmethod
  def __call__(self, image: torch.Tensor, anno: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Apply a geometric transformation to a given image and its corresponding annotation.

    Args:
      image (torch.Tensor): input image to be transformed.
      anno (torch.Tensor): annotation to be transformed.

    Returns:
      Tuple[torch.Tensor, torch.Tensor]: transformed image and its corresponding annotation
    """
    raise NotImplementedError

class RandomRotationTransform(GeometricDataAugmentation):
  """ Randomly rotate an image and its annotation by a random angle.
  """
  def __init__(self, min_angle_in_deg :float = 0, max_angle_in_deg: float = 360):
    assert min_angle_in_deg >= 0
    assert max_angle_in_deg <= 360 
    assert min_angle_in_deg > max_angle_in_deg

    self.min_angle_in_deg = min_angle_in_deg
    self.max_angle_in_deg = max_angle_in_deg

  def __call__(self, image: torch.Tensor, anno: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # dimension of each input should be identical
    assert image.shape[1] == anno.shape[1], "Dimensions of all input should be identical."
    assert image.shape[2] == anno.shape[2], "Dimensions of all input should be identical."

    angle = random.uniform(self.min_angle_in_deg, self.max_angle_in_deg)

    image_rotated = functional.rotate(image, angle=angle, interpolation=transforms.InterpolationMode.BILINEAR)
    anno_rotated = functional.rotate(anno, angle=angle, interpolation=transforms.InterpolationMode.NEAREST)

    return image_rotated, anno_rotated

class RandomHorizontalFlipTransform(GeometricDataAugmentation):
  """ Apply random horizontal flipping.
  """

  def __init__(self, prob: float = 0.5):
    """ Apply random horizontal flipping.

    Args:
        prob (float, optional): probability of the image being flipped. Defaults to 0.5.
    """
    assert prob >= 0.0
    assert prob <= 1.0
    self.prob = prob

  def __call__(self, image: torch.Tensor, anno: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # dimension of each input should be identical
    assert image.shape[1] == anno.shape[1], "Dimensions of all input should be identical."
    assert image.shape[2] == anno.shape[2], "Dimensions of all input should be identical."

    if random.random() <= self.prob:
      image_hz_flipped = functional.hflip(image)
      anno_hz_flipped = functional.hflip(anno)

      return image_hz_flipped, anno_hz_flipped
    else:
      return image, anno


class RandomVerticalFlipTransform(GeometricDataAugmentation):
  """ Apply random vertical flipping.
  """

  def __init__(self, prob: float = 0.5):
    """ Apply random vertical flipping.

    Args:
        prob (float, optional): probability of the image being flipped. Defaults to 0.5.
    """
    assert prob >= 0.0
    assert prob <= 1.0
    self.prob = prob

  def __call__(self, image: torch.Tensor, anno: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # dimension of each input should be identical
    assert image.shape[1] == anno.shape[1], "Dimensions of all input should be identical."
    assert image.shape[2] == anno.shape[2], "Dimensions of all input should be identical."

    if random.random() <= self.prob:
      image_v_flipped = functional.vflip(image)
      anno_v_flipped = functional.vflip(anno)

      return image_v_flipped, anno_v_flipped
    else:
      return image, anno


class CenterCropTransform(GeometricDataAugmentation):
  """ Extract a patch from the image center.
  """

  def __init__(self, crop_height: Optional[int] = None, crop_width: Optional[int] = None):
    """ Set height and width of cropping region.

    Args:
        crop_height (Optional[int], optional): Height of cropping region. Defaults to None.
        crop_width (Optional[int], optional): Width of cropping region. Defaults to None.
    """
    self.crop_height = crop_height
    self.crop_width = crop_width

  def __call__(self, image: torch.Tensor, anno: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # dimension of each input should be identical
    assert image.shape[1] == anno.shape[1], "Dimensions of all input should be identical."
    assert image.shape[2] == anno.shape[2], "Dimensions of all input should be identical."

    if (self.crop_height is None) or (self.crop_width is None):
      return image, anno

    img_chans, img_height, img_width = image.shape[:3]
    anno_chans = anno.shape[0]

    if (self.crop_width > img_width):
      raise ValueError("Width of cropping region must not be greather than img width")
    if (self.crop_height > img_height):
      raise ValueError("Height of cropping region must not be greather than img height.")

    image_cropped = functional.center_crop(image, [self.crop_height, self.crop_width])
    anno_cropped = functional.center_crop(anno, [self.crop_height, self.crop_width])

    assert image_cropped.shape[0] == img_chans, "Cropped image has an unexpected number of channels."
    assert image_cropped.shape[1] == self.crop_height, "Cropped image has not the desired size."
    assert image_cropped.shape[2] == self.crop_width, "Cropped image has not the desired width."

    assert anno_cropped.shape[0] == anno_chans, "Cropped anno has an unexpected number of channels."
    assert anno_cropped.shape[1] == self.crop_height, "Cropped anno has not the desired size."
    assert anno_cropped.shape[2] == self.crop_width, "Cropped anno has not the desired width."

    return image_cropped, anno_cropped


class RandomCropTransform(GeometricDataAugmentation):
  """ Extract a random patch from a given image and its corresponding annnotation.
    """

  def __init__(self, crop_height: Optional[int] = None, crop_width: Optional[int] = None):
    """ Set height and width of cropping region.

      Args:
          crop_height (Optional[int], optional): Height of cropping region. Defaults to None.
          crop_width (Optional[int], optional): Width of cropping region. Defaults to None.
      """
    self.crop_height = crop_height
    self.crop_width = crop_width

  def __call__(self, image: torch.Tensor, anno: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # dimension of each input should be identical
    assert image.shape[1] == anno.shape[1], "Dimensions of all input should be identical."
    assert image.shape[2] == anno.shape[2], "Dimensions of all input should be identical."

    if (self.crop_height is None) or (self.crop_width is None):
      return image, anno

    img_chans, img_height, img_width = image.shape[:3]
    anno_chans = anno.shape[0]

    if (self.crop_width > img_width):
      raise ValueError("Width of cropping region must not be greather than img width")
    if (self.crop_height > img_height):
      raise ValueError("Height of cropping region must not be greather than img height.")

    max_x = img_width - self.crop_width
    x_start = random.randint(0, max_x)

    max_y = img_height - self.crop_height
    y_start = random.randint(0, max_y)

    assert (x_start + self.crop_width) <= img_width, "Cropping region (width) exceeds image dims."
    assert (y_start + self.crop_height) <= img_height, "Cropping region (height) exceeds image dims."

    image_cropped = functional.crop(image, y_start, x_start, self.crop_height, self.crop_width)
    anno_cropped = functional.crop(anno, y_start, x_start, self.crop_height, self.crop_width)

    assert image_cropped.shape[0] == img_chans, "Cropped image has an unexpected number of channels."
    assert image_cropped.shape[1] == self.crop_height, "Cropped image has not the desired size."
    assert image_cropped.shape[2] == self.crop_width, "Cropped image has not the desired width."

    assert anno_cropped.shape[0] == anno_chans, "Cropped anno has an unexpected number of channels."
    assert anno_cropped.shape[1] == self.crop_height, "Cropped anno has not the desired size."
    assert anno_cropped.shape[2] == self.crop_width, "Cropped anno has not the desired width."

    return image_cropped, anno_cropped

class MyRandomShearTransform(GeometricDataAugmentation):
  """ Apply random shear along x- and y-axis.
  """
  def __init__(self, max_x_shear: float, max_y_shear: float, prob: float = 0.5):
    """ Apply random shear.

    Args:
        x_shear (float): maximum shear along x-axis (in degrees).
        y_shear (float): maximum shear along y-axis (in degrees).
        prob (float, optional): probability of the image being sheared. Defaults to 0.5.
    """
    assert prob >= 0.0
    assert prob <= 1.0
    
    self.max_x_shear = max_x_shear
    self.max_y_shear = max_y_shear
    self.prob = prob

  def __call__(self, image: torch.Tensor, anno: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # dimension of each input should be identical
    assert image.shape[1] == anno.shape[1], "Dimensions of all input should be identical."
    assert image.shape[2] == anno.shape[2], "Dimensions of all input should be identical."

    img_chans, img_height, img_width = image.shape[0], image.shape[1], image.shape[2]
    anno_chans = anno.shape[0]

    if random.random() < self.prob:
      x_shear = random.uniform(-self.max_x_shear, self.max_x_shear)
      
      image = functional.affine(image, angle=0, translate=[0,0], scale=1.0, shear=[x_shear,0], interpolation=transforms.InterpolationMode.BILINEAR)
      anno = functional.affine(anno, angle=0, translate=[0,0], scale=1.0, shear=[x_shear,0], interpolation=transforms.InterpolationMode.NEAREST)
    
    if random.random() < self.prob:
      y_shear = random.uniform(-self.max_y_shear, self.max_y_shear)
      
      image = functional.affine(image, angle=0, translate=[0,0], scale=1.0, shear=[0, y_shear], interpolation=transforms.InterpolationMode.BILINEAR)
      anno = functional.affine(anno, angle=0, translate=[0,0], scale=1.0, shear=[0, y_shear], interpolation=transforms.InterpolationMode.NEAREST)

    assert img_chans == image.shape[0]
    assert img_height == image.shape[1]
    assert img_width == image.shape[2]

    assert anno_chans == anno.shape[0]
    assert img_height == anno.shape[1]
    assert img_width == anno.shape[2]

    return image, anno

def get_geometric_augmentations(cfg, stage: str) -> List[GeometricDataAugmentation]:
  assert stage in ['train', 'val', 'test', 'predict']

  geometric_augmentations = []

  for tf_name in cfg[stage]['geometric_data_augmentations'].keys():
    if tf_name == 'random_hflip':
      augmentor = RandomHorizontalFlipTransform()
      geometric_augmentations.append(augmentor)

    if tf_name == 'random_vflip':
      augmentor = RandomVerticalFlipTransform()
      geometric_augmentations.append(augmentor)

    if tf_name == 'random_rotate':
      min_angle_in_deg = cfg[stage]['geometric_data_augmentations'][tf_name]['min_angle_in_deg']
      max_angle_in_deg = cfg[stage]['geometric_data_augmentations'][tf_name]['max_angle_in_deg']
      augmentor = RandomRotationTransform(min_angle_in_deg, max_angle_in_deg)
      geometric_augmentations.append(augmentor)

    if tf_name == 'random_shear':
      max_x_shear = cfg[stage]['geometric_data_augmentations'][tf_name]['max_x_shear']
      max_y_shear = cfg[stage]['geometric_data_augmentations'][tf_name]['max_y_shear']
      augmentor = MyRandomShearTransform(max_x_shear, max_y_shear)
      geometric_augmentations.append(augmentor)

    if tf_name == 'center_crop':
      crop_height = cfg[stage]['geometric_data_augmentations'][tf_name]['height']
      crop_width = cfg[stage]['geometric_data_augmentations'][tf_name]['width']
      augmentor = CenterCropTransform(crop_height, crop_width)
      geometric_augmentations.append(augmentor)

    if tf_name == 'random_crop':
      crop_height = cfg[stage]['geometric_data_augmentations'][tf_name]['height']
      crop_width = cfg[stage]['geometric_data_augmentations'][tf_name]['width']
      augmentor = RandomCropTransform(crop_height, crop_width)
      geometric_augmentations.append(augmentor)

  return geometric_augmentations