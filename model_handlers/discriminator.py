# Copyright (c) 2026-present, Mohamed Chtourou.
# All rights reserved.
# This module defines the DiscriminatorModel class, which implements the discriminator component of a GAN.

# --------------------------
# Importing necessary libraries
# --------------------------
import torch
import torch.nn as nn

class DiscriminatorModel(nn.Module):
    """
    Flexible Discriminator for GANs supporting multiple image sizes.
    """
    def __init__(self, image_size: int = 64):
        super().__init__()
        # Decide starting channels based on image size
        if image_size == 64:
            channels = [64, 128, 256, 512]
        elif image_size == 128:
            channels = [32, 64, 128, 256, 512]
        else:
            raise ValueError("Unsupported image_size. Use 64 or 128.")
        layers = []
        in_channels = 3
        for out_channels in channels:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False))
            if out_channels != channels[-1] or image_size == 128:  # Apply BatchNorm except first layer
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_channels = out_channels
        # Final conv to get a single output
        # H -> Height, W -> Width, Dataset is transformed to be a squared matrix, meaning H=W
        # to apply the convultion to an image with dimension (H,W), the following formula is applied to calculated the new height/width 
        # H_output​=( H_input​ + 2×padding - kernel_size )/stride      
        layers.append(nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=0, bias=False))
        layers.append(nn.Flatten())
        self.discriminator = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.discriminator(x)