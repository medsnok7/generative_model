# Copyright (c) 2026-present, Mohamed Chtourou.
# All rights reserved.
# This module defines utility functions for the GAN-based image generator, including denormalization of images, visualization of generated samples, and weight initialization for the generator and discriminator models. These functions are used to facilitate training and evaluation of the GAN models implemented in the generative_model.py module.


import os
import matplotlib.pyplot as plt
import torch.nn as nn
from torchvision.utils import save_image, make_grid


# --------------------------
# Settings and hyperparameters
# --------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))  # if inside model_handlers/
DATASET_DIR = os.path.join(PROJECT_ROOT, "animefacedataset")
IMAGE_SIZE = 64
BATCH_SIZE = 128
STATS = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

# --------------------------
# Utility functions
# --------------------------

def denormalize(image_tensor):


    return image_tensor * STATS[1][0] + STATS[0][0]


def show_images(images, nmax=64):
    _, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks([]); ax.set_yticks([])
    grid = make_grid(denormalize(images.detach()[:nmax]), nrow=8)
    ax.imshow(grid.permute(1, 2, 0))
    plt.show()


def show_batch(dl, nmax=64):
    for images, _ in dl:
        show_images(images, nmax)
        break


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
