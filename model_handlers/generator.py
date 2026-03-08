# Copyright (c) 2026-present, Mohamed Chtourou.
# All rights reserved.
# This module defines the GeneratorModel class, which implements the generator component of a GAN.
import torch
import torch.nn as nn

class GeneratorModel64(nn.Module):
    def __init__(self):
        super().__init__()
        self.generator = nn.Sequential(
            # Input: (128,1,1)
            # Hout​=(Hin​−1)×stride−2×padding+kernel_size+output_padding
            nn.ConvTranspose2d(in_channels=128, out_channels=512, kernel_size=4, stride=1, padding=0, bias=False),  # (512,4,4)
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),  # (256,8,8)
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),  # (128,16,16)
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),   # (64,32,32)
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=4, stride=2, padding=1, bias=False),     # (3,64,64)
            nn.Tanh()  # output in [-1,1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.generator(x)

class GeneratorModel128(nn.Module):
    def __init__(self):
        super().__init__()
        self.generator = nn.Sequential(
            # Input: (128,1,1)
            # H out​=(H in​−1)×stride−2×padding+kernel_size+output_padding
            nn.ConvTranspose2d(in_channels=128, out_channels=512, kernel_size=4, stride=1, padding=0, bias=False),  # (512,4,4)
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),  # (256,8,8)
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),  # (128,16,16)
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),   # (64,32,32)
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1, bias=False),    # (32,64,64)
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=4, stride=2, padding=1, bias=False),     # (3,128,128)
            nn.Tanh()  # output in [-1,1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.generator(x)