from collections import deque, defaultdict
from matplotlib import pyplot as plt
import numpy as np
import torch

from .loss import (
    mean_squared_error as mse_loss,
    kl_div_diagonal_gaussian_to_standard_gaussian as kl_loss,
)


class VAEEvaluationAgent(object):
    """
    Wraps a VAE network `model` with a loss function `loss_fn`, 
    evaluates given or self-generated data,
    and reports recorded results at request.

    Like ordinary torch models, this can switch between training mode and evaluation mode.

    Results to report
    -----------------
    `loss`: float,
            by loss function, for optimization,
            evaluated and recorded through both training and validation.
    `metrics`: dict of floats,
            complementary to loss, with clearer meaning,
            evaluated and recorded through validation.
    `latent_distrib`: figure, or dict of array-like,
            where given data locate themselves in the latent space,
            evaluated and recorded through validation.
    `latent_sample`: figure, or dict of array-like, 
            generated sample in the latent space, 
            potentially filling the gaps between clusters of given data,
            recorded when the `generate` method is called

    Calling as a function
    ---------------------
    Evaluates a batch of data, 
    record certain types of results according to current mode:
      - training: `loss`,
      - validation: `loss`, `metrics` and `latent_distrib`,
    and returns `loss` with attachment to the graph (for optimization if needed).

    Other parameters
    ----------------
    `loss_fn_kwargs`: dict, default {},
        to be passed to the loss function.
    `recent_train_losses`: int, default 1, 
        number of training losses to keep track of,
        useful for log writing.
    `distrib_thin_fac`: float, default 1., 
        fraction of given data recorded for `latent_distrib`, 
        small number necessary for efficient memory usage and fast plot generation.
    """

    # keys to maintain and to report
    # if others are generated at evaluation, they will not be recorded
    # _metrics_keys = frozenset({'kl_div', 'recon_err_shape', 'recon_err_scale'})
    _distrib_keys = frozenset({'values', 'labels', 'theory', 'mu', 'logvar'})
    _sample_keys = frozenset({'values', 'z'})

    def __init__(self, model, 
                 loss_fn, diagnosis_fn, 
                 loss_fn_kwargs={}, diagnosis_fn_kwargs={},
                 # norm=1., 
                 distrib_thin_fac=1., sample_size=1,
                 metrics_keys={}, 
                 distrib_keys={}, 
                 sample_keys={}):
        self._model = model
        self._loss_fn = loss_fn
        self._loss_fn_kwargs = loss_fn_kwargs
        self._diagnosis_fn = diagnosis_fn
        self._diagnosis_fn_kwargs = diagnosis_fn_kwargs
        # self._norm = norm
        self._distrib_thin_fac = distrib_thin_fac
        self._sample_size = sample_size
        self._acc_recent_weight = 0.
        self._acc_recent_loss = 0.
        self._acc_train_weight = 0.
        self._acc_train_loss = 0.
        self._acc_val_weight = 0.
        self._acc_val_loss = 0.
        self._acc_metrics = defaultdict(float)
        self._acc_distrib = {k: [] for k in self._distrib_keys}

    @property
    def model(self):
        return self._model
    @property
    def loss_fn(self):
        return self._loss_fn
    @property
    def loss_fn_kwargs(self):
        return self._loss_fn_kwargs
    @property
    def diagnosis_fn(self):
        return self._diagnosis_fn
    @property
    def diagnosis_fn_kwargs(self):
        return self._diagnosis_fn_kwargs
    # @property
    # def norm(self):
    #     return self._norm
    @property
    def distrib_thin_fac(self):
        return self._distrib_thin_fac
    @property
    def sample_size(self):
        return self._sample_size

    def train(self, mode=True):
        self._model.train(mode)
        return self
    def eval(self):
        self._model.eval()
        return self
    @property
    def training(self):
        return self._model.training

    def __call__(self, data_batch):
        return self.add_batch(data_batch)

    def add_batch(self, data_batch):
        size = len(data_batch[0])
        if self._model.training:
            # size, loss = evaluate_batch_for_training(
            #                 self.model, data_batch, 
            #                 self.loss_fn, self.loss_fn_kwargs, self.norm)
            loss = self.loss_fn(self.model, *data_batch, **self.loss_fn_kwargs)
            self._acc_recent_weight += size
            self._acc_recent_loss += size * loss.item()
            self._acc_train_weight += size
            self._acc_train_loss += size * loss.item()
        else:
            with torch.no_grad():
                # size, loss, metrics, distrib \
                #         = evaluate_batch_for_validation(
                #             self.model, data_batch, 
                #             self.loss_fn, self.loss_fn_kwargs, self.norm)
                loss = self.loss_fn(
                            self.model, *data_batch, **self.loss_fn_kwargs)
                metrics, distrib = self.diagnosis_fn(
                            self.model, *data_batch, **self.diagnosis_fn_kwargs)
            self._acc_recent_weight += size
            self._acc_recent_loss += size * loss.item()
            self._acc_val_weight += size
            self._acc_val_loss += size * loss.item()
            # for k in self._metrics_keys:
            #     self._acc_metrics[k] += size * metrics[k]
            for k in metrics:
                 self._acc_metrics[k] += size * metrics[k]
            mask_thin = np.random.sample(size) < self._distrib_thin_fac
            if np.any(mask_thin):
                for k in self._distrib_keys:
                    self._acc_distrib[k].append(distrib[k][:size][mask_thin])
                    self._acc_distrib[k].append(distrib[k][size:])
        return loss

    def pop_recent_loss(self):
        recent_loss = self._acc_recent_loss / self._acc_recent_weight
        self._acc_recent_weight = 0.
        self._acc_recent_loss = 0.
        return recent_loss

    def report_loss(self, training=None):
        if training is None:
            training = self._model.training
        if training:
            return self._acc_train_loss / self._acc_train_weight
        else:
            return self._acc_val_loss / self._acc_val_weight

    def report_metrics(self):
        return {k: self._acc_metrics[k] / self._acc_val_weight for k in self._acc_metrics}

    def report_latent_distrib(self):
        distrib = {k: np.concatenate(self._acc_distrib[k], axis=0) for k in self._distrib_keys}
        fig = plot_latent_distrib(distrib)
        return fig

    def report_latent_sample(self):
        distrib = {k: np.concatenate(self._acc_distrib[k], axis=0) for k in self._distrib_keys}
        sample = get_latent_sample(self._model, distrib, self._sample_size)
        sample = {k: sample[k] for k in self._sample_keys}
        # fig = plot_latent_sample(sample, norm=self.norm)
        fig = plot_latent_sample(sample)
        return fig


# def evaluate_batch_for_training(model, data_batch, loss_fn, loss_fn_kwargs, norm=1.):
#     v, l, t = align_data_format_with_model(model, *data_batch)
#     norm = align_data_format_with_model(model, norm).reshape(1, 1, -1)
#     v_recon_var, mu, logvar = model(v, l)
#     loss = loss_fn(v*norm, v_recon_var*norm, mu, logvar, l, t, **loss_fn_kwargs)
#     return len(v), loss

# def evaluate_batch_for_validation(model, data_batch, loss_fn, loss_fn_kwargs, norm=1.):
#     v, l, t = align_data_format_with_model(model, *data_batch)
#     norm = align_data_format_with_model(model, norm).reshape(1, 1, -1)
#     v_recon_var, mu, logvar = model(v, l)
#     v_recon = model.decoder(mu, l)
#     loss = loss_fn(v*norm, v_recon_var*norm, mu, logvar, l, t, **loss_fn_kwargs)
#     kl_div = torch.mean(kl_loss(mu, logvar, dim=-1))
#     recon_err = torch.mean(mse_loss(v*norm, v_recon*norm, dim=-1))
#     metrics = dict(
#         kl_div=kl_div,
#         recon_err=recon_err,
#     )
#     distrib = dict(
#         values=v,
#         labels=l,
#         theory=t,
#         mu=mu,
#         logvar=logvar,
#         recon=v_recon,
#         recon_var=v_recon_var,
#     )
#     metrics = {k: v.item() for k, v in metrics.items()}
#     distrib = {k: v.detach().cpu().numpy() for k, v in distrib.items()}
#     return len(v), loss, metrics, distrib

def get_latent_sample(model, ref_distrib, size):
    mu_ref = ref_distrib['mu']
    l_ref = ref_distrib['labels']
    if mu_ref.shape[-1] != 2:
        # FIXME: implement other-than-2d behavior
        return dict()
    r_ref = np.sqrt(np.mean(np.sum(mu_ref**2, -1)))
#     r_ref = 1.
    angles = np.linspace(0, 2*np.pi, size, endpoint=False)
    z = r_ref * np.exp(1j*angles)
    z = np.vstack([np.real(z), np.imag(z)]).T
    l = l_ref[np.random.choice(len(l_ref), size, replace=False)]
    z, l = align_data_format_with_model(model, z, l)
    with torch.no_grad():
        v = model.decoder(z, l)
        mu, logvar = model.encoder(v, l)
    sample = dict(
        z=z,
        labels=l,
        values=v,
        mu=mu,
        logvar=logvar,
    )
    sample = {k: v.detach().cpu().numpy() for k, v in sample.items()}
    return sample

def get_graph_arguments(model, data):
    v, l, t = align_data_format_with_model(model, *data)
    mu, logvar = model.encoder(v, l)
    return model, (v, l)

def get_model_format(model):
    p = next(model.parameters())
    format_kwargs = dict(dtype=p.dtype, device=p.device)
    return format_kwargs

def align_data_format_with_model(model, *data):
    format_kwargs = get_model_format(model)
    data = tuple(torch.as_tensor(d, **format_kwargs) for d in data)
    if len(data) == 1: data = data[0]
    return data

def plot_latent_distrib(distrib, zoomin_fac=20., **kwargs):
    mu = distrib['mu']
    logvar = distrib['logvar']
    color_by = distrib['theory'][:,0]
    size_by = np.sqrt(np.mean(distrib['values']**2, axis=-1))
    if mu.shape[-1] != 2:
        # FIXME: implement other-than-2d behavior
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    mu_sq = np.sum(mu**2, axis=-1)
    mask = np.sqrt(mu_sq/np.max(mu_sq)) < 1/zoomin_fac
    plot_latent_distrib_2d(
        mu[:,0], mu[:,1], logvar[:,0], logvar[:,1], 
        color_by, size_by, axis=axes[0], **kwargs)
    if np.any(mask):  
        plot_latent_distrib_2d(
            mu[mask][:,0], mu[mask][:,1], 
            logvar[mask][:,0], logvar[mask][:,1],
            color_by[mask], size_by[mask], axis=axes[1], **kwargs)
    axes[0].set_xlabel(r'$z_0$')
    axes[0].set_ylabel(r'$z_1$')
    axes[1].set_xlabel(r'$z_0$')
    axes[1].set_ylabel(r'$z_1$')
    return fig

def plot_latent_distrib_2d(
        mu1, mu2, logvar1, logvar2, color_by, size_by, axis, 
        scatter_marker = '+', 
        scatter_size_base = 5.,
        scatter_size_coeff = 1e2,
        scatter_alpha = .75,
        ellipse_relsize_cutoff = 2.5e-2,
        ellipse_alpha = .5,
        **kwargs):
    from matplotlib.patches import Ellipse
    from matplotlib.lines import Line2D
    
    mu = np.vstack([mu1, mu2]).T
    logvar = np.vstack([logvar1, logvar2]).T
    xylim = np.max(np.abs(mu)) * 1.05
    sigma_scale = np.exp(0.5*np.max(logvar, axis=-1))
    mask_sigma_vis = sigma_scale/xylim > ellipse_relsize_cutoff
    
    legend_handles = []
    legend_labels = []

    for icb, cb in enumerate(np.unique(color_by)):
        mask_cb = color_by == cb
        size_max = np.max(size_by[mask_cb])
#         if size_max == 0:
#             size = 1.
#         else:
#             size = (size_by[mask_cb] / size_max) ** 2
        # FIXME
        size = 1.
        marker = scatter_marker
        color = f'C{icb}'
        label = f"{cb}"
        
        for idx in np.where(mask_cb & mask_sigma_vis)[0]:
            x, y = mu[idx,0], mu[idx,1]
            dx, dy = np.exp(0.5*logvar[idx,0]), np.exp(0.5*logvar[idx,1])
            patch = Ellipse((x,y), dx, dy, angle=0, facecolor=color, alpha=ellipse_alpha)
            axis.add_patch(patch)
            
        x, y = mu[mask_cb][:,0], mu[mask_cb][:,1]
        size = scatter_size_base + scatter_size_coeff * size
        axis.scatter(x, y, marker=marker, s=size, c=color, alpha=scatter_alpha)
        legend_handles.append(Line2D([0], [0], marker=marker, linestyle='none', color=color))
        legend_labels.append(label)
        
    axis.legend(legend_handles, legend_labels)
    axis.set_xlim(-xylim, xylim)
    axis.set_ylim(-xylim, xylim)
    axis.grid(True)

    return axis

def plot_latent_sample(sample, norm=1., **kwargs):
    v = sample['values'][:,0]
    z = sample['z']
    norm = np.asarray(norm).reshape(-1)
    ncol = 4
    nrow = (len(v) - 1) // ncol + 1
    fig, axes = plt.subplots(nrow, ncol, figsize=(14, 2*nrow))
    axes = axes.flatten()
    for i in range(len(v)):
        label = "(" + ", ".join("{:0.1e}".format(zz) for zz in z[i]) + ")"
        axes[i].plot(v[i]*norm, label=label, **kwargs)
        axes[i].legend()
    return fig

