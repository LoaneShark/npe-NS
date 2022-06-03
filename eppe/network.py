import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset


THEORY_CATEGORICAL_MAP = dict(GR=0, EDGB=1, dCS=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collate_batch(batch):
    label_list, value_list, theory_list = [], [], []
    
    for theory, label, phasing in batch:
        label_list.append(label)
        value_list.append(phasing)
        theory_list.append(
            THEORY_CATEGORICAL_MAP[theory]
        )
    
    label_list = np.array(label_list, dtype=np.float32)
    value_list = np.array(value_list, dtype=np.float32)
    theory_list = np.array(theory_list, dtype=np.float32)

    return label_list, value_list, theory_list


class PhasingDataset(Dataset):
    label_keys = 'm1,m2,s1z,s2z,length_scale'.split(',')
    value_keys = 'freqs,phases'.split(',')

    def __init__(self, dataset_filename, theory=None) -> None:
        """
        Dataloader for waveforms
        
        Parameters
        ----------
        dataset_filename : str
            filename storing waveform phasing (in pickle format)
        theory : str
            One of 'GR', 'EDGB', 'dCS'
        """
        self.waveform_dataset = pd.read_pickle(dataset_filename)
        self.theory = theory
        # augment dataset with theory categorical label
        self.waveform_dataset['theory'] = self.theory


    def __len__(self):
        return len(self.waveform_dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        entry = self.waveform_dataset.iloc[idx]
        label = entry.loc[self.label_keys].values
        value = entry.loc['phases']

        return self.theory, label, value


class VAEEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 12, 7)
        self.conv2 = nn.Conv2d(12, 6, 7)
        self.linear1 = nn.Linear(16*16*6, 5)
        self.linear2 = nn.Linear(16*16*6, 5)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 16*16*6)
        # use linear1 to map to mu; linear2 to map to sigma
        mu = self.linear1(x)
        sigma = self.linear2(x)
        return mu, sigma


class VAEDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(5, 16*16*6)
        self.deconv1 = nn.ConvTranspose2d(6, 12, 7)
        self.deconv2 = nn.ConvTranspose2d(12, 1, 7)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = x.view(-1, 6, 16, 16)
        x = F.relu(self.deconv1(x))
        x = torch.sigmoid(self.deconv2(x))
        return x


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = VAEEncoder()
        self.decoder = VAEDecoder()

    def forward(self, x):
        mu, log_sigma = self.encoder(x)
        sigma = torch.exp(log_sigma)
        # sample a point
        z = mu + sigma*torch.rand_like(sigma)
        res = self.decoder(z)
        return res, mu, sigma


class ResidualBlock(nn.Module):
    """Residual block from He et. al. (2016)
    """
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, dilation=1):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        assert self.out_channels >= self.in_channels
        
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            padding=dilation
        )
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            padding=dilation
        )


    def forward(self, x):
        orig_x = x.clone()
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        # if input and output channels differ, pad zero rows accordingly
        pad_x = F.pad(orig_x, (0, 0, 0, self.out_channels - self.in_channels))
        # add to original input
        x += pad_x
        x = F.relu(x)
        return x

