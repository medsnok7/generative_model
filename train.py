# Copyright (c) 2026-present, Mohamed Chtourou.
# All rights reserved.
# This module defines the main entry point for training the GAN-based image generator. It imports the ImageGenerator class from the generative_model module and calls its fit method to train the generator and discriminator models using the specified learning rates and number of epochs.

import argparse
import os

from model_handlers.generative_model import ImageGenerator

# --------------------------
# CLI arguments
# --------------------------
parser = argparse.ArgumentParser(description="Train GAN image generator")
parser.add_argument("--stats",type=float,default=0.5,
                    help="stats number ")
parser.add_argument("--batch_size",type=int,default=128,
                    help="batch size ")
parser.add_argument("--ds_folder_name", type=str, default="jellyfish-types",
                    help="name of the dataset folder ")
parser.add_argument("--is_cmplx", type=int, default=0,
                    help="is image complexe, False:0/True:1, if True use models with 128x128 resolution else 64x64")
parser.add_argument("--gen_lr", type=float, default=0.0003,
                    help="Learning rate for the generator")
parser.add_argument("--dis_lr", type=float, default=0.0001,
                    help="Learning rate for the discriminator")
parser.add_argument("--epochs", type=int, default=20,
                    help="Number of training epochs")
args = parser.parse_args()

# --------------------------
# Training discriminator and generator
# --------------------------
if args.is_cmplx:
    size = 128
else: 
    size = 64
image_generator = ImageGenerator(size=size, stats=args.stats, batch_size=args.batch_size, is_complex_image=args.is_cmplx)

dataset_path = image_generator.prepare_dataset(args.ds_folder_name)
if not os.path.exists(dataset_path):
    image_generator.log.info(f" Unable to find {dataset_path}")
else:
    image_generator.log.info(f" Starting training with the following hyperparameters: ")
    image_generator.log.info(f" Generator Learning Rate: {args.gen_lr} ")
    image_generator.log.info(f" Discriminator Learning Rate: {args.dis_lr} ")
    image_generator.log.info(f" Number of Epochs: {args.epochs} ")
    image_generator.fit(
        args.epochs,
        args.gen_lr,
        args.dis_lr
    )