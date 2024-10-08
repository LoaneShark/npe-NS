import pytest

import torch
from eppe.neural import models


def test_vae_io_shapes(batch_size=10):
    """Check the tensor dimensions of the input/output of VAE"""
    network = models.vae.VAE()
    v = torch.empty(batch_size, 1, 200)
    labels = torch.empty(batch_size, 4)
    v_recon, mu, logvar = network(v, labels)
    assert v_recon.shape == v.shape
    assert mu.shape == (batch_size, 2)
    assert logvar.shape == (batch_size, 2)