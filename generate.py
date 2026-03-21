# Copyright (c) 2026-present, Mohamed Chtourou.
# All rights reserved.
# This module defines the main entry point for generating images using the GAN-based image generator. It imports the ImageGenerator class from the image generator module and calls its generate method to produce new images based on the trained generator model.

# --------------------------
# Importing necessary libraries
# --------------------------
import argparse

# --------------------------
# Importing helper functions and models
# --------------------------
from model_handlers.image_generator import ImageGenerator


# --------------------------
# CLI arguments
# --------------------------
parser = argparse.ArgumentParser(description="Train GAN image generator")
parser.add_argument("--img_size", type=int, choices=[64, 128], required=True, help="choose 64 or 128 Use models with 128x128 resolution else 64x64")
parser.add_argument("--ds_name", type=str, default="default", required= True,
                    help="name of the dataset")
parser.add_argument("--latent_dim",type=int,default=256, required=True,
                    help="latent dimension, choose based on image input image dimension")
parser.add_argument("--img_name", type=str, default="default_name",
                    help="name of the generated image ")
parser.add_argument("--gen_img_batch", type=int, default=10,
                    help="generated image batch size")

args = parser.parse_args()

# --------------------------
# Generating Images
# --------------------------
image_generator = ImageGenerator(size=args.img_size,batch_size=args.gen_img_batch, latent_dim= args.latent_dim)
image_generator.set_models_folder_name(args.ds_name)
image_generator.generate(args.img_name)