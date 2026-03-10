# Copyright (c) 2026-present, Mohamed Chtourou.
# All rights reserved.
# This module defines the GeneratorModel class, which implements the generator component of a GAN.

# --------------------------
# Importing necessary libraries
# --------------------------
import torch
import torch.nn as nn

from utilities.model_helper import LATENT_DIM

class GeneratorModel(nn.Module):
    """
    Flexible Generator for GANs supporting multiple image sizes.
    """
    def __init__(self, image_size: int = 64, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.latent_dim = latent_dim
        # Define channel progression based on image size
        if image_size == 64:
            channels = [512, 256, 128, 64, 3]
        elif image_size == 128:
            channels = [512, 256, 128, 64, 32, 3]
        else:
            raise ValueError("Unsupported image_size. Use 64 or 128.")
        layers = []
        in_channels = latent_dim
        for i, out_channels in enumerate(channels):
            # H -> Height, W -> Width, Dataset is transformed to be a squared matrix, meaning H=W
            # to apply the convultion transpose to an image with dimension (H,W), the following formula is applied to calculated the new height/width 
            # H_output​=(H_input​−1)×stride−2×padding+kernel_size+output_padding
            if i == 0:
                layers.append(nn.ConvTranspose2d(
                    in_channels, out_channels, kernel_size=4, stride=1, padding=0, bias=False
                ))
            else:
                layers.append(nn.ConvTranspose2d(
                    in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False
                ))
            # Apply BatchNorm + ReLU for all except the last layer
            if i < len(channels) - 1:
                layers.append(nn.BatchNorm2d(out_channels))
                layers.append(nn.ReLU(True))
            else:
                layers.append(nn.Tanh())  # output in [-1, 1]
            in_channels = out_channels
        
        self.generator = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.generator(x)