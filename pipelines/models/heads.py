"""
This module contains the implementation of the segmentation heads.
The first heads is single size head, while the second head is a multi-size head.
The second heads can be  any size from nano (n), small (s), medium (m) and large (l).
"""
import torch.nn as nn
from segmentation_models_pytorch.base.modules import Activation


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1):
        conv2d_1 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        conv2d_2 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        activation = Activation(activation)
        super().__init__(conv2d_1, conv2d_2, upsampling, activation)


class SegmentationHead2(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1, head_size='n'):
        # in_channels from resnet34 is 16, out_channels are number of classes (mostly 1 because of binary segmentation)
        # n, s, m and l are nano, small, midium and heads sizes respectively
        layers = [nn.Conv2d(in_channels, 64, kernel_size=kernel_size, padding=kernel_size // 2),
                  nn.BatchNorm2d(64),
                  nn.ReLU(),
                  nn.Conv2d(64, 64, kernel_size=kernel_size, padding=kernel_size // 2),
                  nn.BatchNorm2d(64),
                  nn.ReLU(),
                  nn.Conv2d(64, 64, kernel_size=kernel_size, padding=kernel_size // 2),
                  nn.BatchNorm2d(64),
                  nn.ReLU(),
                  nn.Conv2d(64, 64, kernel_size=kernel_size, padding=kernel_size // 2)
                  ]
        if head_size == 'n':
            layers = layers[:1]
        if head_size == 's':
            layers = layers[:4]
        if head_size == 'm':
            layers = layers[:7]
        elif head_size == 'l':
            layers = layers

        layers.append(nn.Conv2d(64, out_channels, kernel_size=kernel_size, padding=kernel_size // 2))
        layers.append(nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity())
        layers.append(Activation(activation))
        super().__init__(*layers)
