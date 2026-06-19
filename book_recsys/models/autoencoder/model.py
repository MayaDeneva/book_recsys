"""Mult-VAE (Liang et al., 2018) — variational autoencoder for collaborative filtering."""
import torch
import torch.nn.functional as F
from torch import nn


class MultVAE(nn.Module):
    """Encoder MLP -> Gaussian latent -> decoder MLP over the full item vocab.

    The multinomial likelihood (softmax over items) is what makes this less
    popularity-biased than pointwise models: it normalizes probability mass
    across the catalog.
    """

    def __init__(self,
                 n_items: int,
                 hidden: int = 600,
                 latent: int = 200,
                 dropout: float = 0.5) -> None:
        super().__init__()
        self.enc1 = nn.Linear(n_items, hidden)
        self.enc2 = nn.Linear(hidden, latent * 2)
        self.dec1 = nn.Linear(latent, hidden)
        self.dec2 = nn.Linear(hidden, n_items)
        self.drop = nn.Dropout(dropout)
        self.latent = latent

    def encode(self, x):
        h = F.normalize(x, dim=1)  # L2-normalize the user vector (paper convention)
        h = self.drop(h)  # denoising corruption
        h = torch.tanh(self.enc1(h))
        h = self.enc2(h)
        return h[:, :self.latent], h[:, self.latent:]

    def decode(self, z):
        return self.dec2(torch.tanh(self.dec1(z)))

    def forward(self, x):
        mu, logvar = self.encode(x)
        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std
        return self.decode(z), mu, logvar

    def predict(self, x):
        """Deterministic inference path: decode the latent mean (no sampling)."""
        mu, _ = self.encode(x)
        return self.decode(mu)

    def loss(self, x, logits, mu, logvar, beta):
        nll = -(F.log_softmax(logits, dim=1) * x).sum(dim=1).mean()
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()
        return nll + beta * kl
