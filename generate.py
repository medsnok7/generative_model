# Copyright (c) 2026-present, Mohamed Chtourou.
# All rights reserved.
# This module defines the main entry point for generating images using the GAN-based image generator. It imports the ImageGenerator class from the generative_model module and calls its generate method to produce new images based on the trained generator model.

from model_handlers.generative_model import ImageGenerator


# --------------------------
# Generating Images
# --------------------------

image_generator = ImageGenerator()
image_generator.generate()