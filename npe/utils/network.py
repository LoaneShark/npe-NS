import torch
from torch import nn

# ===================

def get_fc_layers(input_dim, output_dim, width, depth, 
                  activation=nn.ReLU, batchnorm=nn.Identity):
    layers = [
        nn.Linear(input_dim, width),
        batchnorm(width),
        activation(),
    ]
    for _ in range(depth):
        layers += [
            nn.Linear(width, width),
            batchnorm(width),
            activation(),
        ]
    layers.append(nn.Linear(width, output_dim))
    layers = nn.Sequential(*layers)
    return layers


class Encoder(nn.Module):

    def __init__(self, 
                 width=64, depth=2, 
                 latent_dim=2, data_dim=640, 
                 cond_dim=4, use_cond=True,
                 activation=nn.ReLU, batchnorm=nn.Identity):
        # Assume (N, data_dim) -> (N, latent_dim) + (N, latent_dim)
        super().__init__()
        self.latent_dim = latent_dim
        self.data_dim = data_dim
        self.cond_dim = cond_dim
        self.use_cond = use_cond
        input_dim = data_dim + use_cond * cond_dim
        output_dim = latent_dim * 2 - 1
        self.fc = get_fc_layers(input_dim, output_dim, width, depth, 
                                activation=activation, batchnorm=batchnorm)

    def forward(self, x, cond):
        if self.use_cond:
            x = torch.cat([x, cond], dim=-1)
        x = self.fc(x)
        mu, logvar = x[:,:self.latent_dim], x[:,self.latent_dim:]
        logvar = torch.cat([logvar, logvar], dim=-1)
        return mu, logvar

        
class Decoder(nn.Module):
    
    def __init__(self, 
                 width=64, depth=2, 
                 latent_dim=2, data_dim=640, 
                 cond_dim=4, use_cond=True,
                 activation=nn.ReLU, batchnorm=nn.Identity):
        # Assume mapping (N, latent_dim) + (N, cond_dim) -> (N, data_dim)
        super().__init__()
        self.data_dim = data_dim
        self.cond_dim = cond_dim
        self.latent_dim = latent_dim
        self.use_cond = use_cond
        input_dim = latent_dim + use_cond * cond_dim
        output_dim = data_dim
        self.fc = get_fc_layers(input_dim, output_dim, width, depth,
                                activation=activation, batchnorm=batchnorm)

    def forward(self, z, cond):
        if self.use_cond:
            z = torch.cat([z, cond], dim=-1)
        x = self.fc(z)
        return x


class VAE(nn.Module):
    
    def __init__(self, 
                 width=64, depth=2, 
                 width_scale=None, depth_scale=None, 
                 latent_dim=2, data_dim=640, cond_dim=4, grid_dim=None,
                 freeze_shape=False, freeze_scale=False,
                 activation=nn.ReLU, batchnorm=nn.Identity):
        super().__init__()
        assert latent_dim == 2
        if width_scale is None:
            width_scale = width
        if depth_scale is None:
            depth_scale = depth
        if grid_dim is None:
            grid_dim = data_dim
        self.data_dim = data_dim
        self.grid_dim = grid_dim
        self.freeze_shape = freeze_shape
        self.freeze_scale = freeze_scale
        self.raw_encoder = Encoder(width, depth, 
                                   latent_dim, data_dim, 
                                   cond_dim, use_cond=False,
                                   activation=activation, batchnorm=batchnorm)
        self.raw_decoder0 = Decoder(width_scale, depth_scale, 
                                    latent_dim, 1,
                                    cond_dim, use_cond=True,
                                    activation=activation, batchnorm=batchnorm)
        self.raw_decoder1 = Decoder(width, depth, 
                                    latent_dim, grid_dim, 
                                    cond_dim, use_cond=False,
                                    activation=activation, batchnorm=batchnorm)
        self.raw_decoder2 = Decoder(width, depth, 
                                    latent_dim, grid_dim, 
                                    cond_dim, use_cond=False,
                                    activation=activation, batchnorm=batchnorm)
        if freeze_shape:
            for m in [self.raw_encoder, self.raw_decoder1, self.raw_decoder2]:
                for p in m.parameters():
                    p.requires_grad = False
        if freeze_scale:
            for p in self.raw_decoder0.parameters():
                p.requires_grad = False
        
    def encoder(self, x, cond):
        batch_size = x.shape[0]
        x = x / torch.sqrt(torch.mean(x*x, dim=-1, keepdim=True))
        x = x.view(-1, self.raw_encoder.data_dim)
        x = torch.cat([x, -x], dim=0)
        cond = torch.cat([cond, cond], dim=0)
        mu, logvar = self.raw_encoder(x, cond)
        mu = (mu[:batch_size] - mu[batch_size:]) / 2
        mu = mu / torch.sqrt(torch.sum(mu*mu, dim=-1, keepdim=True))
        logvar = (logvar[:batch_size] + logvar[batch_size:]) / 2
        return mu, logvar
    
    def grid_decoder(self, z, cond):
        batch_size = z.shape[0]
        z = torch.cat([z, -z], dim=0)
        cond = torch.cat([cond, cond], dim=0)
        x0 = self.raw_decoder0(z, cond)
        x1 = self.raw_decoder1(z, cond)
        x2 = self.raw_decoder2(z, cond)
        x0 = (x0[:batch_size] + x0[batch_size:]) / 2
        x1 = (x1[:batch_size] - x1[batch_size:]) / 2
        x2 = (x2[:batch_size] + x2[batch_size:]) / 2
        x1 = torch.exp(x0) * x1
        return x1, x2
    
    def decoder(self, z, cond, xin=None):
        batch_size = z.shape[0]
        x1, x2 = self.grid_decoder(z, cond)
        if xin is None:
            xin = torch.linspace(0, 1, self.data_dim, device=x1.device)
            xin = xin.repeat(batch_size, 1)
        xout = torch.sum(x1[:,None,:] * torch.exp(x2[:,None,:] * xin[:,:,None]), dim=-1)
        xout = xout.view(batch_size, 1, -1)
        return xout
    
    def varier(self, mu, logvar):
        z = mu + torch.exp(0.5*logvar) * torch.randn_like(mu)
        z = z / torch.sqrt(torch.sum(z*z, dim=-1, keepdim=True))
        return z

    def forward(self, x, cond):
        mu, logvar = self.encoder(x, cond)
        z = self.varier(mu, logvar)
        x_recon = self.decoder(z, cond)
        return x_recon, mu, logvar