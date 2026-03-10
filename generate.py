# Copyright (c) 2026-present, Mohamed Chtourou.
# All rights reserved.
# This module defines the main entry point for generating images using the GAN-based image generator. It imports the ImageGenerator class from the image generator module and calls its generate method to produce new images based on the trained generator model.

import argparse

from model_handlers.image_generator import ImageGenerator


# --------------------------
# CLI arguments
# --------------------------
parser = argparse.ArgumentParser(description="Train GAN image generator")
parser.add_argument("--img_name", type=str, default="default_name",
                    help="name of the generated image ")
parser.add_argument("--ds_folder_name", type=str, default="default",
                    help="name of the dataset folder ")
parser.add_argument("--is_cmplx", type=int, default=0,
                    help="is image complexe, False:0/True:1, if True use models with 128x128 resolution else 64x64")
parser.add_argument("--latent_dim",type=int,default=1024,
                    help="latent dimension, choose based on image input image dimension")

args = parser.parse_args()

# --------------------------
# Generating Images
# --------------------------
if args.is_cmplx:
    size = 128
else: 
    size = 64
image_generator = ImageGenerator(size=size, latent_dim= args.latent_dim,is_complex_image=args.is_cmplx)
image_generator.prepare_dataset(args.ds_folder_name)
image_generator.generate(args.img_name)