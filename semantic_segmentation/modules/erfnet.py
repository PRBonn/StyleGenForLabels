import torch
import torch.nn as nn
import torch.nn.functional as F

class DownsamplerBlock(nn.Module):

  def __init__(self, ninput, noutput):
    super().__init__()

    self.conv = nn.Conv2d(ninput, noutput - ninput, (3, 3), stride=2, padding=1, bias=True)
    self.pool = nn.MaxPool2d(2, stride=2)
    self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

  def forward(self, input):
    output = torch.cat([self.conv(input), self.pool(input)], 1)
    output = self.bn(output)
    return F.relu(output)

class non_bottleneck_1d(nn.Module):

  def __init__(self, chann, dropprob, dilated):
    super().__init__()

    self.conv3x1_1 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1, 0), bias=True)

    self.conv1x3_1 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1), bias=True)

    self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)

    self.conv3x1_2 = nn.Conv2d(
        chann, chann, (3, 1), stride=1, padding=(1 * dilated, 0), bias=True, dilation=(dilated, 1))

    self.conv1x3_2 = nn.Conv2d(
        chann, chann, (1, 3), stride=1, padding=(0, 1 * dilated), bias=True, dilation=(1, dilated))

    self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)

    self.dropout = nn.Dropout2d(dropprob)

  def forward(self, input):

    output = self.conv3x1_1(input)
    output = F.relu(output)
    output = self.conv1x3_1(output)
    output = self.bn1(output)
    output = F.relu(output)

    output = self.conv3x1_2(output)
    output = F.relu(output)
    output = self.conv1x3_2(output)
    output = self.bn2(output)

    if (self.dropout.p != 0):
      output = self.dropout(output)

    return F.relu(output + input)  #+input = identity (residual connection)

class UpsamplerBlock(nn.Module):

  def __init__(self, ninput, noutput):
    super().__init__()
    self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
    self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

  def forward(self, input):
    output = self.conv(input)
    output = self.bn(output)
    return F.relu(output)

class ERFNetModel(nn.Module):

  def __init__(self, num_classes: int):
    super().__init__()
    self.num_classes = num_classes

    # ---------------- ENCODER ----------------
    self.downsampler_block_01 = DownsamplerBlock(3, 16)
    self.downsampler_block_02 = DownsamplerBlock(16, 64)

    self.enc_non_bottleneck_1d_01 = non_bottleneck_1d(64, 0.03, 1)
    self.enc_non_bottleneck_1d_02 = non_bottleneck_1d(64, 0.03, 1)
    self.enc_non_bottleneck_1d_03 = non_bottleneck_1d(64, 0.03, 1)
    self.enc_non_bottleneck_1d_04 = non_bottleneck_1d(64, 0.03, 1)
    self.enc_non_bottleneck_1d_05 = non_bottleneck_1d(64, 0.03, 1)

    self.downsampler_block_03 = DownsamplerBlock(64, 128)

    self.enc_non_bottleneck_1d_06 = non_bottleneck_1d(128, 0.3, 2)
    self.enc_non_bottleneck_1d_07 = non_bottleneck_1d(128, 0.3, 4)
    self.enc_non_bottleneck_1d_08 = non_bottleneck_1d(128, 0.3, 8)
    self.enc_non_bottleneck_1d_09 = non_bottleneck_1d(128, 0.3, 16)

    self.enc_non_bottleneck_1d_10 = non_bottleneck_1d(128, 0.3, 2)
    self.enc_non_bottleneck_1d_11 = non_bottleneck_1d(128, 0.3, 4)
    self.enc_non_bottleneck_1d_12 = non_bottleneck_1d(128, 0.3, 8)
    self.enc_non_bottleneck_1d_13 = non_bottleneck_1d(128, 0.3, 16)

    # ---------------- DECODER ----------------
    self.upsampler_block_01 = UpsamplerBlock(128, 64)

    self.dec_non_bottleneck_1d_01  = non_bottleneck_1d(64, 0, 1)
    self.dec_non_bottleneck_1d_02 = non_bottleneck_1d(64, 0, 1)

    self.upsampler_block_02 = UpsamplerBlock(64, 16)
    
    self.dec_non_bottleneck_1d_03 = non_bottleneck_1d(16, 0, 1)
    self.dec_non_bottleneck_1d_04 = non_bottleneck_1d(16, 0, 1)

    self.output_conv = nn.ConvTranspose2d(16, self.num_classes, 2, stride=2, padding=0, output_padding=0, bias=True)


  def forward(self, img: torch.Tensor) -> torch.Tensor:
    # ---------------- ENCODER ----------------
    img = self.downsampler_block_01(img)
    img = self.downsampler_block_02(img)

    img = self.enc_non_bottleneck_1d_01(img)
    img = self.enc_non_bottleneck_1d_02(img)
    img = self.enc_non_bottleneck_1d_03(img)
    img = self.enc_non_bottleneck_1d_04(img)
    img = self.enc_non_bottleneck_1d_05(img)

    img = self.downsampler_block_03(img)

    img = self.enc_non_bottleneck_1d_06(img)
    img = self.enc_non_bottleneck_1d_07(img)
    img = self.enc_non_bottleneck_1d_08(img)
    img = self.enc_non_bottleneck_1d_09(img)

    img = self.enc_non_bottleneck_1d_10(img)
    img = self.enc_non_bottleneck_1d_11(img)
    img = self.enc_non_bottleneck_1d_12(img)
    img = self.enc_non_bottleneck_1d_13(img)

    # ---------------- DECODER ----------------
    img = self.upsampler_block_01(img)

    img = self.dec_non_bottleneck_1d_01(img)
    img = self.dec_non_bottleneck_1d_02(img)

    img = self.upsampler_block_02(img)

    img = self.dec_non_bottleneck_1d_03(img)
    img = self.dec_non_bottleneck_1d_04(img)

    img = self.output_conv(img)

    return img
