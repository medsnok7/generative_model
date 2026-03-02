import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# Import custom modules
from generator import GeneratorModel
from discriminator import DiscriminatorModel
import todevice as dv  # Assumed custom device helper

# --------------------------
# Settings and hyperparameters
# --------------------------
data_dir = "./animefacedataset"
image_size = 64
batch_size = 128
stats = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

# --------------------------
# Utility functions
# --------------------------
def denormalize(image_tensor):
    return image_tensor * stats[1][0] + stats[0][0]

def show_images(images, nmax=64):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks([]); ax.set_yticks([])
    grid = make_grid(denormalize(images.detach()[:nmax]), nrow=8)
    ax.imshow(grid.permute(1, 2, 0))
    plt.show()

def show_batch(dl, nmax=64):
    for images, _ in dl:
        show_images(images, nmax)
        break

# --------------------------
# Dataset and DataLoader
# --------------------------
transformer = T.Compose([
    T.Resize((image_size, image_size)),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(*stats)
])

train_dataset = ImageFolder(data_dir, transform=transformer)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

show_batch(train_loader)

device = dv.get_defaul_device()
print(f"Using device: {device}")

# --------------------------
# ImageGenerator Class
# --------------------------
class ImageGenerator:
    sample_dir = "generated"

    def __init__(self):
        self.generator = GeneratorModel()
        self.discriminator = DiscriminatorModel()
        self.loss_fn = nn.BCELoss()  # assuming binary classification for GAN

    def train_discriminator(self, real_images, optimizer):
        optimizer.zero_grad()

        # Real images
        real_preds = self.discriminator(real_images)
        real_targets = torch.ones(real_images.size(0), 1)
        real_loss = self.loss_fn(real_preds, real_targets)
        real_score = real_preds.mean().item()

        # Fake images
        latent = torch.randn(real_images.size(0), 128, 1, 1)
        fake_images = self.generator(latent)
        fake_preds = self.discriminator(fake_images)
        fake_targets = torch.zeros(fake_images.size(0), 1)
        fake_loss = self.loss_fn(fake_preds, fake_targets)
        fake_score = fake_preds.mean().item()

        # Total loss
        loss = real_loss + fake_loss
        loss.backward()
        optimizer.step()
        return loss.item(), real_score, fake_score

    def train_generator(self, optimizer):
        optimizer.zero_grad()
        latent = torch.randn(128, 128, 1, 1)
        fake_images = self.generator(latent)
        preds = self.discriminator(fake_images)
        targets = torch.ones(fake_images.size(0), 1)
        loss = self.loss_fn(preds, targets)
        loss.backward()
        optimizer.step()
        return loss.item()

    def save_samples(self, index, latent_tensors, show=True):
        os.makedirs(self.sample_dir, exist_ok=True)
        fake_images = self.generator(latent_tensors)
        filename = f'generated-images-{index:04d}.png'
        save_image(denormalize(fake_images), os.path.join(self.sample_dir, filename), nrow=8)

        if show:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.set_xticks([]); ax.set_yticks([])
            grid = make_grid(fake_images.cpu().detach(), nrow=8)
            ax.imshow(grid.permute(1, 2, 0))
            plt.show()

    def fit(self, epochs, lr, start_idx=1):
        torch.cuda.empty_cache()

        # Optimizers
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))

        fixed_latent = torch.randn(64, 128, 1, 1)

        for epoch in range(epochs):
            for real_images, _ in tqdm(train_loader):
                # Train discriminator
                loss_d, real_score, fake_score = self.train_discriminator(real_images, opt_d)

                # Train generator
                loss_g = self.train_generator(opt_g)

            # Logging
            print(f"Epoch [{epoch+1}/{epochs}], loss_g: {loss_g:.4f}, loss_d: {loss_d:.4f}, "
                  f"real_score: {real_score:.4f}, fake_score: {fake_score:.4f}")

            self.save_samples(epoch + start_idx, fixed_latent, show=False)

        # Save models
        torch.save(self.generator.state_dict(), "./models/generator.pth")
        torch.save(self.discriminator.state_dict(), "./models/discriminator.pth")

# --------------------------
# Training
# --------------------------
lr = 0.0002
epochs = 10
fixed_latent = torch.randn(64, 128, 1, 1)

image_generator = ImageGenerator()
image_generator.fit(epochs, lr)