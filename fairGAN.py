import torch
import torchvision.datasets as datasets
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn, optim


# Configuration
BATCH_SIZE = 64
LEARNING_RATE = 0.05
EPOCHS = 1
Z_DIM = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
# DataLoader
dataset = datasets.MNIST(root = "dataset/", train = True, transform = transforms, download = True)
train_loader = DataLoader(dataset= dataset, batch_size = BATCH_SIZE, shuffle = True)

# FairGAN
# def gen dis
class Discriminator(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.disc(x)

class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, img_dim),
            nn.Tanh(),  # normalize inputs to [-1, 1] so make outputs [-1, 1]
        )

    def forward(self, x):
        return self.gen(x)

# LatentNet
class Latent(nn.Module):
    def __init__(self, in_features):
        super.__init__()
        self.in_feat = nn.Linear(in_features,128)
        self.BN = nn.BatchNorm2d()
        self.drop_out = nn.Dropout(0.2)
        self.out_feat = nn.Linear(128, in_features)

    def forward(self, x):
        h = self.drop_out(self.BN(self.in_feat(x)))
        return self.out_feat(h)

# Feature Recognition Network
class FeatRecog(nn.Module):
    pass
# Optimization
Gen = Generator(Z_DIM, 784).to(DEVICE)
Disc = Discriminator(784).to(DEVICE)
fixed_noise = torch.randn((BATCH_SIZE, Z_DIM)).to(DEVICE)



optimizer_gen = optim.Adam(Gen.parameters(), lr = LEARNING_RATE)
optimizer_disc = optim.Adam(Disc.parameters(), lr = LEARNING_RATE)
loss_GAN = nn.BCELoss()
loss_ft = nn.MSELoss()
loss_latent = nn.L1Loss()

for epoch in range(EPOCHS):
    for i, (x, _) in tqdm(enumerate(train_loader)):
        real = x.view(-1,784).to(DEVICE)
        batch_size = real.shape[0]

        noise = torch.randn(batch_size, Z_DIM).to(DEVICE)
        fake = Gen(noise)
        disc_real = Disc(real).view(-1)
        lossD_real = loss_GAN(disc_real, torch.ones_like(disc_real))
        disc_fake = Disc(fake).view(-1)
        lossD_fake = loss_GAN(disc_fake, torch.zeros_like(disc_fake))
        lossD = (lossD_real + lossD_fake) / 2
        Disc.zero_grad()
        lossD.backward(retain_graph=True)
        optimizer_disc.step()
        output = Disc(fake).view(-1)
        lossG = loss_GAN(output, torch.ones_like(output))
        Gen.zero_grad()
        lossG.backward()
        optimizer_gen.step()
#%%
