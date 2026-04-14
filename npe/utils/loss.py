import torch
import numpy as np
from utils.network import NetworkType

def mean_squared_error(x1, x2, dim=None):
    """
    Mimics torch.nn.functional.mse_loss.
    `dim` refers to the dimension(s) for reduction.
    """
    err = torch.square(x1 - x2)
    if dim is None:
        err = torch.mean(err)
    else:
        err = torch.mean(err, dim=dim)
    return err

def mean_2m2cos_error(x1, x2, omega=1., dim=None):
    """
    2 * omega ** (-2) * (1 - cos(omega * (x1 - x2))).
    Approaches mse when omega -> 0.
    `dim` refers to the dimension(s) for reduction.
    """
    if abs(omega) < 1e-6:
        # FIXME: this threshold needs to be tested out
        return mean_squared_error(x1, x2, dim=dim)
    err = 2. / omega ** 2 * (1. - torch.cos(omega * (x1 - x2)))
    if dim is None:
        err = torch.mean(err)
    else:
        err = torch.mean(err, dim=dim)
    return err

def kl_div_diagonal_gaussian_to_standard_gaussian(mu, logvar, dim=None):
    """
    As name suggested.
    `dim` refers to the dimension(s) of Gaussian.
    """
    kld = -0.5 * (1 + logvar - torch.square(mu) - torch.exp(logvar))
    if dim is None:
        kld = torch.sum(kld)
    else:
        kld = torch.sum(kld, dim=dim)
    return kld

# TODO: Generalize structure for higher order expansions
def get_pseudoPN_expansion(model, z_abs=1., f_max=None, f_min=None, num_angles=360):
    f_max = f_max if f_max is not None else 1.8e-2
    f_min = f_min if f_min is not None else 4e-5 if model.network_type in [NetworkType.BNS, NetworkType.NSBH, NetworkType.CBC] else 4e-4
    z_theta_tensor = torch.Tensor(np.linspace(0, 2*np.pi, num_angles, endpoint=False)).reshape(num_angles, 1)
    z_abs_tensor = torch.full_like(z_theta_tensor, z_abs)
    z1_tensor = z_abs_tensor * torch.cos(z_theta_tensor)
    z2_tensor = z_abs_tensor * torch.sin(z_theta_tensor)
    z_tensor = torch.cat([z1_tensor, z2_tensor], dim=1)

    cond_tensor = torch.ones((num_angles, model.cond_dim))

    x1, x2 = model.grid_decoder(z_tensor, cond_tensor)

    # Rescale grid coefficients and exponents
    x1 = x1.detach().cpu().numpy()
    x2 = x2.detach().cpu().numpy()
    idx = np.argsort(x2, axis=1)
    x1 = np.take_along_axis(x1, idx, axis=1)
    x2 = np.take_along_axis(x2, idx, axis=1)
    x2 *= 3 / (np.log(f_max/f_min))
    x1 /= (np.pi*f_min) ** (x2/3)
    x2[:,1:] -= x2[:,[0]]
    x1[:,1:] /= x1[:,[0]]

    leading_PN_order = (x2[:,0]+5)/2
    subleading_PN_order = (x2[:,0] + x2[:,1]+5)/2

    coefficient_ratio = np.abs(x1[:,1])

    return leading_PN_order, subleading_PN_order, coefficient_ratio