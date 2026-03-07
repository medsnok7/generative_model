# Copyright (c) 2026-present, Mohamed Chtourou.
# All rights reserved.
# This module defines the main entry point for training the GAN-based image generator. It imports the ImageGenerator class from the generative_model module and calls its fit method to train the generator and discriminator models using the specified learning rates and number of epochs.

import argparse

from model_handlers.generative_model import ImageGenerator

# --------------------------
# CLI arguments
# --------------------------
parser = argparse.ArgumentParser(description="Train GAN image generator")
parser.add_argument("--lr_generator", type=float, default=0.0003,
                    help="Learning rate for the generator")
parser.add_argument("--lr_discriminator", type=float, default=0.0001,
                    help="Learning rate for the discriminator")
parser.add_argument("--epochs", type=int, default=20,
                    help="Number of training epochs")
args = parser.parse_args()

# --------------------------
# Training discriminator and generator
# --------------------------

print("******************** [INFO] Starting training with the following hyperparameters: ********************")
print(f"******************** [INFO] Generator Learning Rate: {args.lr_generator} ********************")
print(f"******************** [INFO] Discriminator Learning Rate: {args.lr_discriminator} ********************")
print(f"******************** [INFO] Number of Epochs: {args.epochs} ********************")

image_generator = ImageGenerator()
image_generator.fit(
    args.epochs,
    args.lr_generator,
    args.lr_discriminator
)