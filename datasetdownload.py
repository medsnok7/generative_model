# Copyright (c) 2026-present, Mohamed Chtourou.
# All rights reserved.
# This module defines the main entry point for downloading the dataset used for training the GAN-based image generator. It uses the opendatasets library to download the dataset from Kaggle, which contains anime face images that will be used to train the generator and discriminator models.   

import opendatasets as od 
import argparse


# --------------------------
# Dataset Download
# --------------------------

parser = argparse.ArgumentParser(description="Download dataset for GAN image generator")
parser.add_argument("--url", type=str, default="https://www.kaggle.com/datasets/splcher/animefacedataset",
                    help="the url of the dataset to download")

args = parser.parse_args()

print(f"******************** [INFO] Dataset URL: {args.url} ********************")
od.download(args.url)
print("******************** [INFO] Dataset downloaded successfully! ********************")
