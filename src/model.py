
################################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dists
import numpy as np

class MLP(nn.Module):
    '''
    Implements an MLP with ReLU activations.
    '''
    def __init__(self, input_size, output_sizes):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        for i, size in enumerate(output_sizes):
            if i > 0:  self.layers.append(nn.ReLU())
            self.layers.append(
                nn.Linear(input_size if i == 0 else output_sizes[i-1], size))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class DeterministicEncoder(nn.Module):
    '''
    Encodes context pairs to an aggregated deterministic representation.

    input_size: int, dimension of concatenated x and y pair, d_x + d_y
    output_sizes: list[int], dimensions of hidden layers

    Forward: (B, n_c, d_x) x (B, n_c, d_y) -> (B, output_sizes[-1])
    '''
    def __init__(self, input_size, output_sizes):
        super(DeterministicEncoder, self).__init__()
        self.mlp = MLP(input_size=input_size, output_sizes=output_sizes)

    def forward(self, context_x, context_y):
        # Concatenate x and y along last dimension
        context_xy = torch.cat([context_x, context_y], dim=-1)
        # Pass through MLP
        r = self.mlp(context_xy)
        # Aggregate over samples with mean
        r = torch.mean(r, dim=1)
        return r

class LatentEncoder(nn.Module):
    '''
    Encodes context pairs to a code used to parameterize the latent distribution.

    input_size: int, dimension of concatenated x and y pair, d_x + d_y
    output_sizes: list[int], dimensions of hidden layers

    Forward: (B, n_c, d_x) x (B, n_c, d_y) -> N ~ (B, num_latents)
    '''
    def __init__(self, input_size, output_sizes, num_latents):
        super(LatentEncoder, self).__init__()
        self.mlp = MLP(input_size=input_size, output_sizes=output_sizes)
        intermediate_size = (output_sizes[-1] + num_latents) // 2
        self.penultimate_layer = nn.Linear(output_sizes[-1], intermediate_size)
        self.mean_layer = nn.Linear(intermediate_size, num_latents)
        self.std_layer = nn.Linear(intermediate_size, num_latents)
        self._num_latents = num_latents

    def forward(self, x, y):
        # Concatenate x and y along last dimension
        context_xy = torch.cat([x, y], dim=-1)
        # Pass through MLP
        hidden = self.mlp(context_xy)
        # Aggregate over samples with mean
        hidden = torch.mean(hidden, dim=1)
        # Pass through the penultimate layer
        hidden = F.relu(self.penultimate_layer(hidden))
        # Compute mu and sigma
        mu = self.mean_layer(hidden)
        log_sigma = self.std_layer(hidden)
        sigma = 0.1 + 0.9 * torch.sigmoid(log_sigma)
        return dists.Normal(loc=mu, scale=sigma)

class Decoder(nn.Module):
    '''
    Decode the representation, latent and target inputs to distribution over
    target outputs.

    input_size: int, hidden_size + num_latents + d_x
    output_sizes: list[int], dimensions of hidden layers, with last entry
        being twice dimension of d_y

    Forward: (B, hidden_size) x (B, num_latents) x (B, n_t, d_x) ->
        N ~ (B, n_t, d_y)
    '''
    def __init__(self, input_size, output_sizes):
        super(Decoder, self).__init__()
        self.mlp = MLP(input_size=input_size, output_sizes=output_sizes)

    def forward(self, r, z, target_x):
        # Concatenate deterministic r, latent z and target inputs
        rep = torch.cat([r,z], dim=-1)
        hidden = torch.cat([rep, target_x], dim=-1)
        # Pass through MLP
        hidden = self.mlp(hidden)
        # Split the output of the MLP into mu and log sigma
        mu, log_sigma = torch.chunk(hidden, 2, dim=-1)
        sigma = 0.1 + 0.9 * F.softplus(log_sigma)
        # Predictive distribution
        dist = dists.MultivariateNormal(
            loc=mu, scale_tril=torch.diag_embed(sigma))
        return dist, mu, sigma

class LatentNeuralProcess(nn.Module):
    '''
    Implements the latent neural process with deterministic and latent paths.

    input_dim: int, d_x
    output_dim: int, d_y
    determ_encoder_output_size: list[int], dimensions of hidden layers
    latent_encoder_output_size: list[int], dimensions of hidden layers
    num_latents: int, dimension of latent z
    decoder_output_size: list[int], dimensions of hidden layers, with last
        entry being twice dimension of d_y
    '''
    def __init__(self, input_dim, output_dim,
                 determ_encoder_output_size,
                 latent_encoder_output_size, num_latents,
                 decoder_output_size):
        super(LatentNeuralProcess, self).__init__()
        self.latent_encoder = LatentEncoder(
            input_size=input_dim + output_dim,
            output_sizes=latent_encoder_output_size,
            num_latents=num_latents)
        self.deterministic_encoder = DeterministicEncoder(
            input_size=input_dim + output_dim,
            output_sizes=determ_encoder_output_size)
        self.decoder = Decoder(
            input_size=determ_encoder_output_size[-1] + \
                num_latents + input_dim,
            output_sizes=decoder_output_size)

    def forward(self, context_x, context_y, target_x, target_y=None):
        num_targets = target_x.shape[1]

        # 1. Pass context pairs through latent encoder
        prior = self.latent_encoder(context_x, context_y)
        z = prior.sample()
        # Optionally use target_y for posterior sampling during training
        if target_y is not None:
            posterior = self.latent_encoder(target_x, target_y)
            z = posterior.sample()
        z = z.unsqueeze(1).repeat(1, num_targets, 1)

        # 2. Pass context pairs through deterministic encoder
        r = self.deterministic_encoder(context_x, context_y)
        r = r.unsqueeze(1).repeat(1, num_targets, 1)

        # 3. Decode to get the predictive distribution of the target outputs
        pred_dist, mu, sigma = self.decoder(r, z, target_x)

        # Compute losses
        if target_y is not None:
            log_p = pred_dist.log_prob(target_y)
            kl_div = dists.kl_divergence(
                posterior, prior).sum(axis=-1, keepdim=True)
            loss = -torch.mean(log_p - kl_div / num_targets)
        else:
            log_p, kl_div, loss = None, None, None

        return mu, sigma, log_p, kl_div, loss
        