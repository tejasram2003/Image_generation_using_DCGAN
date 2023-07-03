from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image
import torch.nn.functional as F
from tqdm import tqdm
import pickle


def data_loader(DATA_DIR, image_size, batch_size, stats):
    train_ds = ImageFolder(
        DATA_DIR,
        transform=T.Compose([
            T.Resize(image_size),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(*stats)
        ])
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=3,
        pin_memory=True
    )

    return train_dl


def get_discriminator(device):
    discriminator = nn.Sequential(
        # in: 3 x 256 x 256

        nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(32),
        nn.LeakyReLU(0.2, inplace=True),
        # out: 32 x 128 x 128

        nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(0.2, inplace=True),
        # out: 64 x 64 x 64

        nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(0.2, inplace=True),
        # out: 128 x 32 x 32

        nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(256),
        nn.LeakyReLU(0.2, inplace=True),
        # out: 256 x 16 x 16

        nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(512),
        nn.LeakyReLU(0.2, inplace=True),
        # out: 512 x 8 x 8

        nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(1024),
        nn.LeakyReLU(0.2, inplace=True),
        # out: 1024 x 4 x 4

        nn.Conv2d(1024, 1, kernel_size=4, stride=1, padding=0, bias=False),
        # out: 1 x 1 x 1

        nn.Flatten(),
        nn.Sigmoid())

    discriminator = discriminator.to(device)
    return discriminator



def get_generator(latent_size,device):
    generator = nn.Sequential(
        # in: latent_size x 1 x 1

        nn.ConvTranspose2d(latent_size, 1024, kernel_size=4, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(1024),
        nn.ReLU(True),
        # out: 1024 x 4 x 4

        nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(512),
        nn.ReLU(True),
        # out: 512 x 8 x 8

        nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(256),
        nn.ReLU(True),
        # out: 256 x 16 x 16

        nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU(True),
        # out: 128 x 32 x 32

        nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(True),
        # out: 64 x 64 x 64

        nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(32),
        nn.ReLU(True),
        # out: 32 x 128 x 128

        nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1, bias=False),
        nn.Tanh()
        # out: 3 x 256 x 256
    )

    generator = generator.to(device)
    return generator



def denorm(img_tensors, stats=((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))):
    return img_tensors * stats[1][0] + stats[0][0]


def show_images(images, nmax=64):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(make_grid(denorm(images.detach()[:nmax]), nrow=8).permute(1, 2, 0))


def train_discriminator(real_images, opt_d, discriminator, generator, device, latent_size=1024):
    # Clear discriminator gradients
    opt_d.zero_grad()

    # Pass real images through discriminator
    real_preds = discriminator(real_images)
    real_targets = torch.ones(real_images.size(0), 1, device=device)
    real_loss = F.binary_cross_entropy(real_preds, real_targets)
    real_score = torch.mean(real_preds).item()

    # Generate fake images
    latent = torch.randn(batch_size, latent_size, 1, 1, device=device)
    fake_images = generator(latent)

    # Pass fake images through discriminator
    fake_targets = torch.zeros(fake_images.size(0), 1, device=device)
    fake_preds = discriminator(fake_images)
    fake_loss = F.binary_cross_entropy(fake_preds, fake_targets)
    fake_score = torch.mean(fake_preds).item()

    # Update discriminator weights
    loss = real_loss + fake_loss
    loss.backward()
    opt_d.step()
    return loss.item(), real_score, fake_score


def train_generator(opt_g, discriminator, generator, device, latent_size=1024):
    # Clear generator gradients
    opt_g.zero_grad()

    # Generate fake images
    latent = torch.randn(batch_size, latent_size, 1, 1, device=device)
    fake_images = generator(latent)

    # Try to fool the discriminator
    preds = discriminator(fake_images)
    targets = torch.ones(batch_size, 1, device=device)
    loss = F.binary_cross_entropy(preds, targets)

    # Update generator weights
    loss.backward()
    opt_g.step()

    return loss.item()


def save_samples(index, latent_tensors, generator, show=True):
    sample_dir = 'dog_generated_256'
    os.makedirs(sample_dir, exist_ok=True)
    fake_images = generator(latent_tensors)
    fake_fname = 'generated-images-{0:0=4d}.png'.format(index)
    save_image(denorm(fake_images), os.path.join(sample_dir, fake_fname), nrow=8)
    print('Saving', fake_fname)
    if show:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(fake_images.cpu().detach(), nrow=8).permute(1, 2, 0))


def fit(epochs, lr, generator, discriminator, train_dl, device, start_idx=1):
    torch.cuda.empty_cache()

    losses_g = []
    losses_d = []
    real_scores = []
    fake_scores = []

    opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))

    for epoch in range(epochs):
        for real_images, _ in tqdm(train_dl):
            real_images = real_images.to(device)
            loss_d, real_score, fake_score = train_discriminator(real_images, opt_d, discriminator, generator, device)
            loss_g = train_generator(opt_g, discriminator, generator, device)

        losses_g.append(loss_g)
        losses_d.append(loss_d)
        real_scores.append(real_score)
        fake_scores.append(fake_score)

        print("Epoch [{}/{}], loss_g: {:.4f}, loss_d: {:.4f}, real_score: {:.4f}, fake_score: {:.4f}".format(
            epoch + 1, epochs, loss_g, loss_d, real_score, fake_score))

        save_samples(epoch + start_idx, fixed_latent, generator, show=False)
    return losses_g, losses_d, real_scores, fake_scores


if __name__ == "__main__":
    DATA_DIR = 'German_Shepherds_256'
    image_size = 256
    batch_size = 128
    stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    latent_size = 1024
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dl = data_loader(DATA_DIR, image_size, batch_size, stats)
    lr = 0.0002
    epochs = 100
    generator = get_generator(latent_size, device)
    discriminator = get_discriminator(device)
    fixed_latent = torch.randn(64, latent_size, 1, 1, device=device)
    save_samples(0, fixed_latent, generator)
    history = fit(epochs, lr, generator, discriminator, train_dl, device, start_idx=1)
    with open('generator_dog', 'wb') as generator_file:
        pickle.dump(generator, generator_file)

    with open('discriminator_dog', 'wb') as discriminator_file:
        pickle.dump(discriminator, discriminator_file)
