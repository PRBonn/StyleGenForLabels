""" Define a set of geometric color data augmentations which can be applied to the input image.

Note that the annotations is not affected in any way.
"""

import math
import random
from typing import Callable, List

import torch
from numpy import imag


def rgb_to_hsv(image: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
  """ Convert an image from RGB to HSV.

    .. image:: _static/img/rgb_to_hsv.png

    The image data is assumed to be in the range of (0, 1).

    Args:
        image: RGB Image to be converted to HSV with shape of :math:`(*, 3, H, W)`.
        eps: scalar to enforce numarical stability.

    Returns:
        HSV version of the image with shape of :math:`(*, 3, H, W)`.
        The H channel values are in the range 0..2pi. S and V are in the range 0..1.

    .. note::
       See a working example `here <https://kornia-tutorials.readthedocs.io/en/latest/
       color_conversions.html>`__.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_hsv(input)  # 2x3x4x5
  """
  if not isinstance(image, torch.Tensor):
    raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

  if len(image.shape) < 3 or image.shape[-3] != 3:
    raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

  max_rgb, argmax_rgb = image.max(-3)
  min_rgb, argmin_rgb = image.min(-3)
  deltac = max_rgb - min_rgb

  v = max_rgb
  s = deltac / (max_rgb + eps)

  deltac = torch.where(deltac == 0, torch.ones_like(deltac), deltac)
  rc, gc, bc = torch.unbind((max_rgb.unsqueeze(-3) - image), dim=-3)

  h1 = (bc - gc)
  h2 = (rc - bc) + 2.0 * deltac
  h3 = (gc - rc) + 4.0 * deltac

  h = torch.stack((h1, h2, h3), dim=-3) / deltac.unsqueeze(-3)
  h = torch.gather(h, dim=-3, index=argmax_rgb.unsqueeze(-3)).squeeze(-3)
  h = (h / 6.0) % 1.0
  h = 2. * math.pi * h  # we return 0/2pi output

  return torch.stack((h, s, v), dim=-3)


def hsv_to_rgb(image: torch.Tensor) -> torch.Tensor:
  """ Convert an image from HSV to RGB.

    The H channel values are assumed to be in the range 0..2pi. S and V are in the range 0..1.

    Args:
        image: HSV Image to be converted to HSV with shape of :math:`(*, 3, H, W)`.

    Returns:
        RGB version of the image with shape of :math:`(*, 3, H, W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = hsv_to_rgb(input)  # 2x3x4x5
    """
  if not isinstance(image, torch.Tensor):
    raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

  if len(image.shape) < 3 or image.shape[-3] != 3:
    raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

  h: torch.Tensor = image[..., 0, :, :] / (2 * math.pi)
  s: torch.Tensor = image[..., 1, :, :]
  v: torch.Tensor = image[..., 2, :, :]

  hi: torch.Tensor = torch.floor(h * 6) % 6
  f: torch.Tensor = ((h * 6) % 6) - hi
  one: torch.Tensor = torch.tensor(1.0, device=image.device, dtype=image.dtype)
  p: torch.Tensor = v * (one - s)
  q: torch.Tensor = v * (one - f * s)
  t: torch.Tensor = v * (one - (one - f) * s)

  hi = hi.long()
  indices: torch.Tensor = torch.stack([hi, hi + 6, hi + 12], dim=-3)
  out = torch.stack((v, q, p, p, t, v, t, v, v, q, p, p, p, p, t, v, v, q), dim=-3)
  out = torch.gather(out, -3, indices)

  return out


def jiiter_soil_saturation(image: torch.Tensor) -> torch.Tensor:

  img_tensor_hsv = rgb_to_hsv(image)
  h, s, v = img_tensor_hsv[0], img_tensor_hsv[1], img_tensor_hsv[2]

  # get range of red tones
  mask_lower_red = (h <= (45 * (math.pi / 180)))
  mask_upper_red = (h >= (315 * (math.pi / 180)))
  mask = mask_lower_red | mask_upper_red

  # randomly adjust saturation of red tones
  s[mask] = s[mask] * random.uniform(0.4, 1.6)
  s = torch.clip(s, 0, 1)

  # build new image
  out_img_hsv = torch.stack([h, s, v], dim=0)
  out_img_rgb = hsv_to_rgb(out_img_hsv)
  out_img_rgb = torch.clip(out_img_rgb, 0, 1)

  return out_img_rgb


def jiiter_soil_contrast_brightness(image: torch.Tensor) -> torch.Tensor:

  img_tensor_hsv = rgb_to_hsv(image)
  h, _, _ = img_tensor_hsv[0], img_tensor_hsv[1], img_tensor_hsv[2]

  # get range of red tones
  mask_lower_red = (h <= (45 * (math.pi / 180)))
  mask_upper_red = (h >= (315 * (math.pi / 180)))
  mask = mask_lower_red | mask_upper_red

  random_contrast = random.uniform(0.6, 1.2)
  random_brightness = random.uniform(-0.2, 0.2)

  image[:, mask] = image[:, mask] * random_contrast + random_brightness

  out_img_rgb = torch.clip(image, 0, 1)

  return out_img_rgb


def jiiter_vegetation_saturation(image: torch.Tensor) -> torch.Tensor:

  img_tensor_hsv = rgb_to_hsv(image)
  h, s, v = img_tensor_hsv[0], img_tensor_hsv[1], img_tensor_hsv[2]

  # get range of green tones
  mask_lower_green = (h >= (50 * (math.pi / 180)))
  mask_upper_green = (h <= (175 * (math.pi / 180)))
  mask = mask_lower_green & mask_upper_green

  # randomly adjust saturation of green tones
  s[mask] = s[mask] * random.uniform(0.5, 1.5)  # random.uniform(0.5, 1.5)
  s = torch.clip(s, 0, 1)

  # build new image
  out_img_hsv = torch.stack([h, s, v], dim=0)
  out_img_rgb = hsv_to_rgb(out_img_hsv)
  out_img_rgb = torch.clip(out_img_rgb, 0, 1)

  return out_img_rgb


def jiiter_vegetation_hue(image: torch.Tensor) -> torch.Tensor:

  img_tensor_hsv = rgb_to_hsv(image)
  h, s, v = img_tensor_hsv[0], img_tensor_hsv[1], img_tensor_hsv[2]

  # get range of green tones
  mask_lower_green = (h >= (50 * (math.pi / 180)))
  mask_upper_green = (h <= (175 * (math.pi / 180)))
  mask = mask_lower_green & mask_upper_green

  # randomly adjust hue
  h[mask] = h[mask] + (random.uniform((-30 * (math.pi / 180)), (+30 * (math.pi / 180))))
  # no clipping since this operation won't produce values < 0 or > 2pi

  # build new image
  out_img_hsv = torch.stack([h, s, v], dim=0)
  out_img_rgb = hsv_to_rgb(out_img_hsv)
  out_img_rgb = torch.clip(out_img_rgb, 0, 1)

  return out_img_rgb


def jiiter_vegetation_contrast_brightness(image: torch.Tensor) -> torch.Tensor:

  img_tensor_hsv = rgb_to_hsv(image)
  h, _, _ = img_tensor_hsv[0], img_tensor_hsv[1], img_tensor_hsv[2]

  # get range of green tones
  mask_lower_green = (h >= (50 * (math.pi / 180)))
  mask_upper_green = (h <= (175 * (math.pi / 180)))
  mask = mask_lower_green & mask_upper_green

  random_contrast = random.uniform(0.6, 1.4)
  random_brightness = random.uniform(-0.2, 0.2)

  image[:, mask] = image[:, mask] * random_contrast + random_brightness

  out_img_rgb = torch.clip(image, 0, 1)

  return out_img_rgb


def vegetation_specific_color_augmentations(image: torch.Tensor) -> torch.Tensor:
  assert torch.max(image) <= 1.0, f"{torch.max(image)}"
  assert torch.min(image) >= 0.0, f"{torch.min(image)}"

  if random.random() < 0.5:
    image = jiiter_soil_saturation(image)
  if random.random() < 0.5:
    image = jiiter_vegetation_saturation(image)

  if random.random() < 0.5:
    image = jiiter_soil_contrast_brightness(image)

  if random.random() < 0.5:
    image = jiiter_vegetation_contrast_brightness(image)

  if random.random() < 0.5:
    image = jiiter_vegetation_hue(image)

  assert torch.max(image) <= 1.0, f"{torch.max(image)}"
  assert torch.min(image) >= 0.0, f"{torch.min(image)}"

  return image


def get_color_augmentations(cfg, stage: str) -> List[Callable]:
  assert stage in ['train', 'val', 'test', 'predict']

  color_augmentations = []
  try:
    cfg[stage]['color_data_augmentations'].keys()
  except KeyError:
    return color_augmentations

  for cf_name in cfg[stage]['color_data_augmentations'].keys():
    if cf_name == 'vegetation_specific':
      augmentor_fn = vegetation_specific_color_augmentations
      color_augmentations.append(augmentor_fn)

  return color_augmentations
