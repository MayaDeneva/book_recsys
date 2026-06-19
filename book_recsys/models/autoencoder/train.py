"""Training loop, β-annealing, and atomic checkpointing for Mult-VAE."""
import os
import tempfile

import numpy as np
import torch
from tqdm.auto import tqdm

from book_recsys.models.autoencoder.model import MultVAE


def anneal_beta(step: int, anneal_steps: int, beta_cap: float) -> float:
    """Linear KL warm-up: 0 -> beta_cap over `anneal_steps` gradient steps, then flat."""
    if anneal_steps <= 0:
        return beta_cap
    return beta_cap * min(1.0, step / anneal_steps)


def _device_type(device: str) -> str:
    s = str(device)
    if "cuda" in s:
        return "cuda"
    if "mps" in s:
        return "mps"
    return "cpu"


def _save_atomic(path: str, payload: dict) -> None:
    folder = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=folder, suffix=".tmp")
    os.close(fd)
    torch.save(payload, tmp)
    os.replace(tmp, path)  # atomic on POSIX: a mid-write kill can't corrupt path


def _config_of(model: MultVAE, n_items: int) -> dict:
    return {
        "n_items": n_items,
        "hidden": model.enc1.out_features,
        "latent": model.latent,
        "dropout": model.drop.p
    }


def save_checkpoint(path, model, optimizer, epoch, config, ids) -> None:
    _save_atomic(
        path, {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "epoch": epoch,
            "config": config,
            "ids": ids,
        })


def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = MultVAE(cfg["n_items"], cfg["hidden"], cfg["latent"], cfg["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    return model, ckpt


def train_multvae(model,
                  matrix,
                  *,
                  epochs,
                  batch_size=500,
                  lr=1e-3,
                  anneal_steps=10000,
                  beta_cap=0.2,
                  device="cpu",
                  amp=False,
                  seed=42,
                  ids=None,
                  ckpt_dir=None,
                  ckpt_prefix="multvae",
                  start_epoch=0,
                  optimizer=None,
                  progress=False):
    """Train `model` on a user×item CSR `matrix`. Checkpoints `<prefix>_last.pt` every epoch
    (atomic). Resume by passing `start_epoch` + a restored `optimizer`. `progress=True` shows a
    per-epoch tqdm bar with the running mean loss (leave it off in tests for pristine output).
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model.to(device).train()
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n = matrix.shape[0]
    config = _config_of(model, matrix.shape[1])
    dev_type = _device_type(device)
    steps_per_epoch = (n + batch_size - 1) // batch_size
    step = start_epoch * steps_per_epoch
    epoch_bar = tqdm(range(start_epoch, epochs), disable=not progress, desc="epoch")
    for epoch in epoch_bar:
        order = rng.permutation(n)
        running, n_batches = 0.0, 0
        for i in range(0, n, batch_size):
            idx = order[i:i + batch_size]
            x = torch.from_numpy(matrix[idx].toarray().astype("float32")).to(device)
            beta = anneal_beta(step, anneal_steps, beta_cap)
            with torch.autocast(device_type=dev_type, enabled=amp):
                logits, mu, logvar = model(x)
                loss = model.loss(x, logits, mu, logvar, beta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += float(loss)
            n_batches += 1
            step += 1
        epoch_bar.set_postfix(loss=running / n_batches)
        if ckpt_dir is not None:
            save_checkpoint(os.path.join(ckpt_dir, f"{ckpt_prefix}_last.pt"), model, optimizer,
                            epoch + 1, config, ids)
    return model
