# Copyright (c) 2026-present, Mohamed Chtourou.
# All rights reserved.
# This module defines the ImageGenerator class, which encapsulates the training and generation logic for a GAN-based image generator. It includes methods for training the discriminator and generator, saving generated samples, and loading/saving model weights. The class uses PyTorch for model implementation and training, and torchvision for data handling and image processing.

# --------------------------
# Importing necessary libraries
# --------------------------
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random
import numpy as np
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image, make_grid
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from typing import Union
# --------------------------
# Importing helper functions and models
# --------------------------
from .generator import GeneratorModel
from .discriminator import DiscriminatorModel
from utilities.model_helper import (denormalize, weights_init, init_logger, create_folders, create_transformer, get_defaul_device)
from utilities.model_helper import (PROJECT_ROOT, BATCH_SIZE, NOISE_PARAM, LATENT_DIM, DISCRIMINATOR_LEARNING_RATE, GENERATOR_LEARNING_RATE, BETAS)


# --------------------------
# ImageGenerator Class
# --------------------------
class ImageGenerator:
    """
    This class encapsulates the training and generation logic for the GAN-based image generator.
    """
    def __init__(self, size: int, latent_dim: int = LATENT_DIM, batch_size:int = BATCH_SIZE, is_complex_image: int = 0, seed:int = 42):
        self.size = size
        self.latent_dim = latent_dim 
        self.batch_size = batch_size
        self.is_complex_image = is_complex_image
        self.seed = seed
        # Set seed for reproducibility
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        # set device (gpu, Cpu)
        self.device = get_defaul_device()
        # set transformer based on the complexity of dataset
        self.transformer, self.stats = create_transformer(self.size, self.is_complex_image)
        # create necessary folders if they don't exist
        self.generated_training = "generated_training"
        self.generator_images = "generator_images"
        self.models_dir = "models"
        self.logging_folder = "logging"
        create_folders(paths=[self.models_dir, self.generated_training, self.generator_images, self.logging_folder])
        # set model to be used based on the complexity of the dataset
        if self.is_complex_image == 0:
            self.generator = GeneratorModel(64,self.latent_dim).to(self.device)
            self.discriminator = DiscriminatorModel(64).to(self.device)
        elif self.is_complex_image == 1:
            self.generator = GeneratorModel(128, self.latent_dim).to(self.device)
            self.discriminator = DiscriminatorModel(128).to(self.device)
        else: 
            raise RuntimeError("Can't choose model, please provide valid args, is_cmplx must be either 0:False/1:True")
        
        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)
        self.loss_fn = nn.BCEWithLogitsLoss().to(self.device)
        self.log = init_logger("ImageGenerator",self.logging_folder)
        self.dataset_path = ""
        self.dataset_name = ""
        self.fixed_latent = torch.randn(self.batch_size, self.latent_dim, 1, 1, device=self.device)
        
    def prepare_dataset(self, dataset_name:str = "default"):
        root = os.path.dirname(os.path.dirname(__file__))
        self.dataset_path = os.path.join(root, dataset_name)
        self.dataset_name = dataset_name
        create_folders(paths=[f"{self.models_dir}/{self.dataset_name}"])
        if os.path.exists(self.dataset_path):
            self.train_dataset = ImageFolder(self.dataset_path, transform=self.transformer)
            self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, prefetch_factor=2)
        return self.dataset_path
        
    def train_discriminator(self, real_images, optimizer):
        # Train the discriminator on real and fake images, and return the loss and scores for logging.
        optimizer.zero_grad()
        batch_size = real_images.size(0)
        # Real images
        real_images = real_images + NOISE_PARAM * torch.randn_like(real_images)
        real_preds = self.discriminator(real_images)
        real_targets = torch.full((batch_size,1), 0.9, device=self.device)
        real_loss = self.loss_fn(real_preds, real_targets)
        real_score = real_preds.mean().item()
        # Fake images
        latent = torch.randn(batch_size, self.latent_dim, 1, 1, device=self.device)
        fake_images = self.generator(latent).detach()
        fake_preds = self.discriminator(fake_images)
        fake_targets = torch.rand(batch_size,1,device=self.device)*0.1
        fake_loss = self.loss_fn(fake_preds, fake_targets)
        fake_score = fake_preds.mean().item()
        # Total loss
        loss = real_loss + fake_loss
        loss.backward()
        optimizer.step()
        return loss.item(), real_score, fake_score
    

    def train_generator(self, optimizer, batch_size):
        # Train the generator to produce images that can fool the discriminator, and return the loss for logging.
        optimizer.zero_grad()
        # Generate fake images
        latent = torch.randn(batch_size, self.latent_dim, 1, 1, device=self.device)
        fake_images = self.generator(latent)
        # Get discriminator predictions
        preds = self.discriminator(fake_images)
        # Generator wants discriminator to think fakes are real
        targets = torch.ones(batch_size, 1, device=self.device)
        loss = self.loss_fn(preds, targets)
        loss.backward()
        optimizer.step()
        return loss.item()


    def save_samples(self, name: Union[int,str], latent_tensors, dir_path):
        # Generate and save sample images from the generator using the provided latent tensors, and optionally display them.
        fake_images = self.generator(latent_tensors)
        if isinstance(name,int):
            filename = f'generated-images-{name:04d}.png'
        elif isinstance(name,str):
            filename = f'{name}.png'
        else:
            filename = "default_image.png"
        save_image(denormalize(fake_images, self.stats), os.path.join(dir_path, filename), nrow=8)


    def fit(self, epochs, lr_g:int = GENERATOR_LEARNING_RATE,lr_d: int = DISCRIMINATOR_LEARNING_RATE, start_idx=1):
        # Train the generator and discriminator models for the specified number of epochs, using the provided learning rates. The method also handles model saving and logging of training progress.
        if not self.train_loader:
            self.log.error(f" Cannot start training, please download dataset first")
            return
        torch.cuda.empty_cache()        
        profiler_activities = [torch.profiler.ProfilerActivity.CPU]
        if self.device.type == "cuda":
            profiler_activities.append(torch.profiler.ProfilerActivity.CUDA)
        # Load trained weights if exists
        models_dir = os.path.join(PROJECT_ROOT, self.models_dir)
        gen_path = os.path.join(models_dir, f"{self.dataset_name}/generator.pth")
        disc_path = os.path.join(models_dir, f"{self.dataset_name}/discriminator.pth")
        if os.path.exists(gen_path) and os.path.exists(disc_path):
            self.log.info(" Loading Pretrained models ...")
            checkpoint_gen = torch.load(gen_path, map_location=self.device)
            model_dict = self.generator.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint_gen.items() if k in model_dict and v.size() == model_dict[k].size()}
            model_dict.update(pretrained_dict)
            self.generator.load_state_dict(model_dict)

            checkpoint_dis = torch.load(disc_path, map_location=self.device)
            model_dict = self.discriminator.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint_dis.items() if k in model_dict and v.size() == model_dict[k].size()}
            model_dict.update(pretrained_dict)
            self.discriminator.load_state_dict(model_dict)

        #train 
        self.generator.train()
        self.discriminator.train()
        # Optimizers
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr_d, betas=BETAS)
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr_g, betas=BETAS)
        self.log.info(f" Training models with learning rate of discriminator {lr_d} and learning rate of generator {lr_g} ")
        self.log.info(f" Training models with {epochs} Epoches ")
        for epoch in range(epochs):
            with torch.profiler.profile(
                activities=profiler_activities,
                record_shapes=True,
                with_stack=True,
                acc_events=True
            ) as prof:
                for real_images, _ in tqdm(self.train_loader):
                    real_images = real_images.to(self.device)
                    batch_size = real_images.size(0)
                    for _ in range(3):
                        loss_d, real_score, fake_score = self.train_discriminator(real_images, opt_d)
                    loss_g = self.train_generator(opt_g, batch_size)
            print(prof.key_averages().table(sort_by="cpu_time_total" if self.device.type=="cpu" else "cuda_time_total", row_limit=10))
            # Save profiler log
            log_file = os.path.join(self.logging_folder, f"profiler_epoch_{epoch+1}.txt")
            with open(log_file, "w") as f:
                f.write(prof.key_averages().table(sort_by="cpu_time_total" if self.device.type=="cpu" else "cuda_time_total", row_limit=10))
            # Logging
            self.log.info(f" Epoch [{epoch+1}/{epochs}], loss_g: {loss_g:.4f}, loss_d: {loss_d:.4f}, "
                  f"real_score: {real_score:.4f}, fake_score: {fake_score:.4f}")
            self.save_samples(epoch + start_idx, self.fixed_latent, self.generated_training, show=False)
            # Save models
            self.log.info(f" [EPOCH: {epoch+1}/{epochs}]  Finished training, Saving Models")
            torch.save(self.generator.state_dict(), gen_path)
            torch.save(self.discriminator.state_dict(), disc_path)
        self.log.info(" Finished training all epochs...")


    def generate(self, name: str):
        # Generate images using the trained generator model, and save them to the specified directory. The method also handles loading of model weights if they exist.
        torch.cuda.empty_cache()
        # Load trained weights if exists
        models_dir = os.path.join(PROJECT_ROOT, self.models_dir)
        gen_path = os.path.join(models_dir, f"{self.dataset_name}/generator.pth")
        if not os.path.exists(gen_path):
            self.log.error(f"Was not able to find models, Cannot generate, please train your models first ")
        else:
            self.log.info(f" Generating using existing models ")
            checkpoint_gen = torch.load(gen_path, map_location=self.device)
            model_dict = self.generator.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint_gen.items() if k in model_dict and v.size() == model_dict[k].size()}
            model_dict.update(pretrained_dict)
            self.generator.load_state_dict(model_dict)
            self.save_samples(name, self.fixed_latent, self.generator_images)
            self.log.info(f" Finished generating, please check under {self.generator_images} ")
 