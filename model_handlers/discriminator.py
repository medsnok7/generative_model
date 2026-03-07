# Copyright (c) 2026-present, Mohamed Chtourou.
# All rights reserved.
# This module defines the DiscriminatorModel class, which implements the discriminator component of a GAN.

import torch
import torch.nn as nn

class DiscriminatorModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.discriminator = nn.Sequential(
            # Input: (3,64,64)
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),  # (64,32,32)
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),  # (128,16,16)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),  # (256,8,8)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1, bias=False),  # (512,4,4)
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4, 1, 0, bias=False),   # (1,1,1)
            nn.Flatten()  # Output: (batch_size, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.discriminator(x)