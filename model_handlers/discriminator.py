# Copyright (c) 2026-present, Mohamed Chtourou.
# All rights reserved.
# This module defines the DiscriminatorModel class, which implements the discriminator component of a GAN.

import torch
import torch.nn as nn

class DiscriminatorModel64(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.discriminator = nn.Sequential(
            # Input: (3,64,64)
            # H out​=( H in​ + 2×padding - kernel_size )/stride
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),  # (64,32,32)
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),  # (128,16,16)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),  # (256,8,8)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False),  # (512,4,4)
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False),   # (1,1,1)
            nn.Flatten()  # Output: (batch_size, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.discriminator(x)


class DiscriminatorModel128(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.discriminator = nn.Sequential(
            # Input: (3,128,128)
            # H out​=( H in​ + 2×padding - kernel_size )/stride
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2, padding=1, bias=False),      # (32,64,64)
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),     # (64,32,32)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),    # (128,16,16)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),   # (256,8,8)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False),   # (512,4,4)
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False),     # (1,1,1)
            nn.Flatten()  # Output: (batch_size, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.discriminator(x)