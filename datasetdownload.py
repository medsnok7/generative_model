# Copyright (c) 2026-present, Mohamed Chtourou.
# All rights reserved.
# This module defines the main entry point for downloading the dataset used for training the GAN-based image generator. It uses the opendatasets library to download the dataset from Kaggle, which contains anime face images that will be used to train the generator and discriminator models.   

import opendatasets as od 
import argparse
from utilities.model_helper import init_logger, create_folders
from utilities.model_helper import LOGGING_FOLDER

# --------------------------
# Dataset Download
# --------------------------
DEFAULT_URL = "https://www.kaggle.com/datasets/splcher/animefacedataset"


parser = argparse.ArgumentParser(description="Download dataset for GAN image generator")
parser.add_argument("--url", type=str, default = DEFAULT_URL,
                    help="the url of the dataset to download")

args = parser.parse_args()

create_folders(LOGGING_FOLDER)
logger = init_logger("Dataset", LOGGING_FOLDER)

logger.info(f" Dataset URL: {args.url} ")
if od.download(args.url):
    logger.info(" Dataset downloaded successfully! ")
else:
    logger.warning(f"It was not possible to download dataset from url: { args.url}")