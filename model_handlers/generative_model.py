# Copyright (c) 2026-present, Mohamed Chtourou.
# All rights reserved.
# This module defines the ImageGenerator class, which encapsulates the training and generation logic for a GAN-based image generator. It includes methods for training the discriminator and generator, saving generated samples, and loading/saving model weights. The class uses PyTorch for model implementation and training, and torchvision for data handling and image processing.

# --------------------------
# Importing necessary libraries
# --------------------------
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# --------------------------
# Importing helper functions and models
# --------------------------
from .generator import GeneratorModel
from .discriminator import DiscriminatorModel
from .model_helper import denormalize, weights_init, show_images, show_batch
from .model_helper import (PROJECT_ROOT, DATASET_DIR, BATCH_SIZE)
import todevice as dv


# --------------------------
# ImageGenerator Class
# --------------------------
class ImageGenerator:
    # This class encapsulates the training and generation logic for the GAN-based image generator.
    def __init__(self):
        self.device = dv.get_defaul_device()
        self.generator = GeneratorModel().to(self.device)
        self.discriminator = DiscriminatorModel().to(self.device)
        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)
        self.loss_fn = nn.BCEWithLogitsLoss().to(self.device)
        self.generated_training = "generated_training"
        self.generator_images = "generator_images"
        self.models_dir = "models"
        self.fixed_latent = torch.randn(BATCH_SIZE, 128, 1, 1, device=self.device)
        self.train_dataset = ImageFolder(DATASET_DIR, transform=self.transformer)
        self.train_loader = DataLoader(self.train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.generated_training, exist_ok=True)
        os.makedirs(self.generator_images, exist_ok=True)


    def train_discriminator(self, real_images, optimizer):
        # Train the discriminator on real and fake images, and return the loss and scores for logging.
        optimizer.zero_grad()
        bs = real_images.size(0)
        # Real images
        real_images = real_images + 0.05 * torch.randn_like(real_images)
        real_preds = self.discriminator(real_images)
        real_targets = torch.full((bs,1), 0.9, device=self.device)
        real_loss = self.loss_fn(real_preds, real_targets)
        real_score = real_preds.mean().item()
        # Fake images
        latent = torch.randn(bs, 128, 1, 1, device=self.device)
        fake_images = self.generator(latent).detach()
        fake_preds = self.discriminator(fake_images)
        fake_targets = torch.rand(bs,1,device=self.device)*0.1
        fake_loss = self.loss_fn(fake_preds, fake_targets)
        fake_score = fake_preds.mean().item()
        # Total loss
        loss = real_loss + fake_loss
        loss.backward()
        optimizer.step()
        return loss.item(), real_score, fake_score
    

    def train_generator(self, optimizer, bs):
        # Train the generator to produce images that can fool the discriminator, and return the loss for logging.
        optimizer.zero_grad()
        # Generate fake images
        latent = torch.randn(bs, 128, 1, 1, device=self.device)
        fake_images = self.generator(latent)
        # Get discriminator predictions
        preds = self.discriminator(fake_images)
        # Generator wants discriminator to think fakes are real
        targets = torch.ones(bs, 1, device=self.device)
        loss = self.loss_fn(preds, targets)
        loss.backward()
        optimizer.step()
        return loss.item()


    def save_samples(self, index, latent_tensors, dir_path, show=True):
        # Generate and save sample images from the generator using the provided latent tensors, and optionally display them.
        fake_images = self.generator(latent_tensors)
        filename = f'generated-images-{index:04d}.png'
        save_image(denormalize(fake_images), os.path.join(dir_path, filename), nrow=8)
        if show:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.set_xticks([]); ax.set_yticks([])
            grid = make_grid(fake_images.cpu().detach(), nrow=8)
            ax.imshow(grid.permute(1, 2, 0))
            plt.show()


    def fit(self, epochs, lr_g,lr_d, start_idx=1):
        # Train the generator and discriminator models for the specified number of epochs, using the provided learning rates. The method also handles model saving and logging of training progress.
        torch.cuda.empty_cache()        
        # Load trained weights if exists
        models_dir = os.path.join(PROJECT_ROOT, self.models_dir)
        gen_path = os.path.join(models_dir, "generator.pth")
        disc_path = os.path.join(models_dir, "discriminator.pth")
        if os.path.exists(gen_path) and os.path.exists(disc_path):
            print("******************** [INFO] Loading Pretrained models ********************")
            self.generator.load_state_dict(torch.load(gen_path, map_location=self.device))
            self.discriminator.load_state_dict(torch.load(disc_path, map_location=self.device))
        #train 
        self.generator.train()
        self.discriminator.train()
        # Optimizers
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
        print(f"******************** [INFO] Training models with learning rate of discriminator {lr_d} and learning rate of generator {lr_g} ********************")
        print(f"******************** [INFO] Training models with {epochs} Epoches ********************")
        for epoch in range(epochs):
            for real_images, _ in tqdm(self.train_loader):
                real_images = real_images.to(self.device)
                bs = real_images.size(0)
                for _ in range(2):
                    loss_d, real_score, fake_score = self.train_discriminator(real_images, opt_d)
                loss_g = self.train_generator(opt_g, bs)
            # Logging
            print(f"[INFO] Epoch [{epoch+1}/{epochs}], loss_g: {loss_g:.4f}, loss_d: {loss_d:.4f}, "
                  f"real_score: {real_score:.4f}, fake_score: {fake_score:.4f}")

            self.save_samples(epoch + start_idx, self.fixed_latent, self.generated_training, show=False)
            # Save models
            print(f"******************** [INFO][EPOCH: {epoch+1}/{epochs}]  Finished training, Saving Models ********************")

            torch.save(self.generator.state_dict(), gen_path)
            torch.save(self.discriminator.state_dict(), disc_path)
        print("******************** [INFO] Loading Pretrained models ********************")


    def generate(self):
        # Generate images using the trained generator model, and save them to the specified directory. The method also handles loading of model weights if they exist.
        torch.cuda.empty_cache()
        # Load trained weights if exists
        models_dir = os.path.join(PROJECT_ROOT, self.models_dir)
        gen_path = os.path.join(models_dir, "generator.pth")
        disc_path = os.path.join(models_dir, "discriminator.pth")
        if os.path.exists(gen_path) and os.path.exists(disc_path):
            print(f"******************** [INFO] Generating using models, Saving Models ********************")
            self.generator.load_state_dict(torch.load(gen_path, map_location=self.device))
            self.discriminator.load_state_dict(torch.load(disc_path, map_location=self.device))
            self.save_samples(0, self.fixed_latent, self.generator_images, show=False)
            print(f"******************** [INFO] Finished generating, please check under {self.generator_images} ********************")
        else:
            print(f"******************** [ERROR] Cannot generate, please train your model first ********************")
 