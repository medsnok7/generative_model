# Copyright (c) 2026-present, Mohamed Chtourou.
# All rights reserved.
# This module defines the main entry point for generating images using the GAN-based image generator. It imports the ImageGenerator class from the generative_model module and calls its generate method to produce new images based on the trained generator model.

import argparse

from model_handlers.generative_model import ImageGenerator


# --------------------------
# CLI arguments
# --------------------------
parser = argparse.ArgumentParser(description="Train GAN image generator")
parser.add_argument("--img_name", type=str, default="default_name",
                    help="name of the generated image ")
parser.add_argument("--ds_folder_name", type=str, default="default",
                    help="name of the dataset folder ")
parser.add_argument("--size", type=int, default=64,
                    help="image size ")
parser.add_argument("--stats", type=float, default=0.5,
                    help="stats ")
parser.add_argument("--is_cmplx", type=int, default=0,
                    help="is image complexe, False:0/True:1 ")

args = parser.parse_args()

# --------------------------
# Generating Images
# --------------------------

image_generator = ImageGenerator(size=args.size, stats=args.stats,is_complex_image=args.is_cmplx)
image_generator.prepare_dataset(args.ds_folder_name)
image_generator.generate(args.img_name)