# Copyright (c) 2026-present, Mohamed Chtourou.
# All rights reserved.
# This module defines utility functions for the GAN-based image generator, including denormalization of images, visualization of generated samples, and weight initialization for the generator and discriminator models. These functions are used to facilitate training and evaluation of the GAN models implemented in the generative_model.py module.


import os
import logging
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision.transforms as T

from torchvision.utils import save_image, make_grid
from typing import Union

# --------------------------
# Settings and hyperparameters
# --------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))  # if inside model_handlers/
DATASET_DIR = os.path.join(PROJECT_ROOT, "animefacedataset")
IMAGE_SIZE = 64
BATCH_SIZE = 128
STATS = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
NOISE_PARAM = 0.05
# --------------------------
# Utility functions
# --------------------------

def create_transformer(image_size, stats) -> T.Compose: 
    return T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(*stats)
        ])


def create_folders( paths: Union[str, list]):
    if isinstance(paths, str):
        os.makedirs(paths, exist_ok=True)
    if isinstance (paths, list):
        for path in paths:
            os.makedirs(path, exist_ok=True)



def init_logger(logger_name: str, logging_path: str) :
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(levelname)s][%(asctime)s] %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    fh = logging.FileHandler(f"{logging_path}/{logger_name}.txt")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


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
