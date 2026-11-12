"""
Code in this file is partially or fully based on, and in some cases adapted from, the implementation provided in the following repository:https://github.com/eeulig/ldct-invariances
"""

import numpy as np
import torch


class AbstractDistribution:
    def sample(self):
        """Return a draw from the distribution.

        Implementations should return a sampled value.
        """
        raise NotImplementedError()

    def mode(self):
        """Return the distribution mode """
        raise NotImplementedError()


class DiracDistribution(AbstractDistribution):
    def __init__(self, value):
        """A deterministic distribution concentrated at value

        store the fixed value returned by sample() and mode().
        """
        self.value = value

    def sample(self):
        """Return the deterministic value"""
        return self.value

    def mode(self):

        return self.value


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        """Diagonal Gaussian parameterized by a concatenated mean/logvar tensor.

        `parameters` is a tensor where channels are [mean, logvar].
        If `deterministic=True` the distribution behaves as zero-variance.
        """
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 10.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        """Draw a sample using the reparameterization trick.

        return a tensor of the same shape as mean
        """
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x

    def kl(self, other=None):
        """Compute KL divergence to a standard normal
        Return: per-batch KL values summed over spatial and channel dims.
        """
        if self.deterministic:
            return torch.Tensor([0.0]).to(device=self.parameters.device)
        else:
            if other is None:
                return 0.5 * torch.sum(torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar, dim=[1, 2, 3])
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=[1, 2, 3],
                )

    def nll(self, sample):
        """Negative log-likelihood of `sample` under this diagonal Gaussian.

        Return per-batch NLL summed over spatial and channel dims
        """
        if self.deterministic:
            return torch.Tensor([0.0]).to(device=self.parameters.device)
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var, dim=[1, 2, 3])

    def mode(self):
        return self.mean
