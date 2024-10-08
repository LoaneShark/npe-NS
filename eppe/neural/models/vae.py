"""
Vanilla VAE with 2D latent space
First working network of this project
"""

import torch
from torch import nn
import torch.nn.functional as F

# FIXME: trying using more sequential to organize the layers

class VAEEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        # FIXME: try to compute the dimensions given input size
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 32, 3),
            nn.ReLU(), 
            nn.MaxPool1d(kernel_size=3)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 32, 3),
            nn.ReLU(), 
            nn.MaxPool1d(kernel_size=3)
        )
        self.linear1_mu = nn.Linear(676, 20)
        self.linear2_mu = nn.Linear(20, 2)
        self.linear1_logvar = nn.Linear(676, 20)
        self.linear2_logvar = nn.Linear(20, 2)

    def forward(self, x, labels):
        # y = x.reshape(x.shape[0], 1, -1)
        y = x
        y = torch.selu(self.conv1(y))
        y = torch.selu(self.conv2(y))
        y = y.reshape(y.shape[0], -1)
        y = torch.cat((y, labels), dim=-1)
        mu = F.selu(self.linear1_mu(y))
        mu = self.linear2_mu(mu)
        logvar = F.selu(self.linear1_logvar(y))
        logvar = self.linear2_logvar(logvar)
        return mu, logvar


class VAEDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        # FIXME: try to compute the dimensions given input size
        self.linear1 = nn.Linear(6, 64)
        self.linear2 = nn.Linear(64, 512)
        self.conv2 = nn.ConvTranspose1d(16, 16, kernel_size=8, stride=2, padding=4)
        self.conv3 = nn.ConvTranspose1d(16, 4, kernel_size=8, stride=2, padding=8)
        self.conv4 = nn.ConvTranspose1d(4, 1, kernel_size=4, stride=2, padding=15)

    def forward(self, z, labels):
        x = torch.cat((z, labels), dim=-1)
        x = F.selu(self.linear1(x))
        x = F.selu(self.linear2(x))
        x = x.reshape(x.shape[0], 16, 32)
        x = F.selu(self.conv2(x))
        x = F.selu(self.conv3(x))
        x = self.conv4(x)
        # x = x.reshape(x.shape[0], -1)
        return x


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = VAEEncoder()
        self.decoder = VAEDecoder()

    def forward(self, x, labels):
        mu, logvar = self.encoder(x, labels)
        z = mu + torch.exp(0.5*logvar) * torch.randn_like(mu)
        x_recon = self.decoder(z, labels)
        return x_recon, mu, logvar

