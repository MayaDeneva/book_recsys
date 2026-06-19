import math

import torch

from book_recsys.models.autoencoder.model import MultVAE


def test_shapes():
    m = MultVAE(n_items=8, hidden=16, latent=4, dropout=0.0)
    x = torch.zeros(3, 8)
    x[:, 0] = 1.0
    logits, mu, logvar = m(x)
    assert logits.shape == (3, 8)
    assert mu.shape == (3, 4) and logvar.shape == (3, 4)
    assert m.predict(x).shape == (3, 8)


def test_predict_is_deterministic():
    m = MultVAE(n_items=8, hidden=16, latent=4, dropout=0.0).eval()
    x = torch.zeros(1, 8)
    x[0, :3] = 1.0
    a = m.predict(x)
    b = m.predict(x)
    assert torch.allclose(a, b)


def test_loss_multinomial_nll_value():
    m = MultVAE(n_items=2, hidden=4, latent=2)
    x = torch.tensor([[1.0, 0.0]])
    logits = torch.zeros(1, 2)  # log_softmax -> [-ln2, -ln2]
    mu = torch.zeros(1, 2)
    logvar = torch.zeros(1, 2)  # KL = 0
    loss = m.loss(x, logits, mu, logvar, beta=1.0)
    assert math.isclose(loss.item(), math.log(2), rel_tol=1e-5)


def test_loss_beta_scales_kl():
    m = MultVAE(n_items=2, hidden=4, latent=2)
    x = torch.tensor([[1.0, 0.0]])
    logits = torch.zeros(1, 2)
    mu = torch.ones(1, 2)  # nonzero -> positive KL
    logvar = torch.zeros(1, 2)
    lo = m.loss(x, logits, mu, logvar, beta=0.0)
    hi = m.loss(x, logits, mu, logvar, beta=1.0)
    assert hi.item() > lo.item()
