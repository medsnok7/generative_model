# Copyright (c) 2026-present, Mohamed Chtourou.
# All rights reserved.
# This module defines the main entry point for generating images using the GAN-based image generator. It imports the ImageGenerator class from the generative_model module and calls its generate method to produce new images based on the trained generator model.

import argparse

from model_handlers.generative_model import ImageGenerator


# --------------------------
# CLI arguments
# --------------------------
parser = argparse.ArgumentParser(description="Train GAN image generator")
parser.add_argument("--name", type=str, default="default_name",
                    help="name of the generated image ")
args = parser.parse_args()

# --------------------------
# Generating Images
# --------------------------

image_generator = ImageGenerator()
image_generator.generate(args.name)