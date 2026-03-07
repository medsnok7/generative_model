# Copyright (c) 2026-present, Mohamed Chtourou.
# All rights reserved.
# This module defines the GeneratorModel class, which implements the generator component of a GAN.
import torch
import torch.nn as nn

class GeneratorModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.generator = nn.Sequential(
            # Input: (128,1,1)
            nn.ConvTranspose2d(128, 512, 4, 1, 0, bias=False),  # (512,4,4)
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),  # (256,8,8)
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),  # (128,16,16)
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),   # (64,32,32)
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),     # (3,64,64)
            nn.Tanh()  # output in [-1,1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.generator(x)