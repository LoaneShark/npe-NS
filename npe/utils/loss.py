import torch

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
