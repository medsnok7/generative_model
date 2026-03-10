# Copyright (c) 2026-present, Mohamed Chtourou.
# All rights reserved.
# This module defines utility functions for the GAN-based image generator, including denormalization of images, visualization of generated samples, and weight initialization for the generator and discriminator models. These functions are used to facilitate training and evaluation of the GAN models implemented in the image_generator.py module.


# --------------------------
# Importing necessary libraries
# --------------------------
import os
import logging
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import torchvision.transforms as T

from torchvision.utils import save_image, make_grid
from typing import Union

# --------------------------
# Settings and hyperparameters
# --------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))  # if inside model_handlers/
LOGGING_FOLDER = "logging"
BATCH_SIZE = 128
NOISE_PARAM = 0.01
LATENT_DIM = 1024
GENERATOR_LEARNING_RATE = 0.0003
DISCRIMINATOR_LEARNING_RATE = 0.0001
BETAS = (0.5, 0.999)



# --------------------------
# Utility functions
# --------------------------
def get_defaul_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    

def create_transformer(image_size, is_complexe_image:bool = False) -> T.Compose: 
    if is_complexe_image:
     return T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(),
            T.RandomRotation(10),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.RandomResizedCrop(image_size, scale=(0.8,1.0)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet-like normalization
                                 std=[0.229, 0.224, 0.225])
        ]), ((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
    else:
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]), ((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))



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


def denormalize(image_tensor, stats):
    return image_tensor * stats[1][0] + stats[0][0]


def show_images(images,stats, nmax=64):
    _, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks([]); ax.set_yticks([])
    grid = make_grid(denormalize(images.detach()[:nmax],stats), nrow=8)
    ax.imshow(grid.permute(1, 2, 0))
    plt.show()


def show_batch(dl,stats, nmax=64):
    for images, _ in dl:
        show_images(images, stats, nmax)
        break


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
