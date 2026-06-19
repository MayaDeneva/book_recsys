import numpy as np
import scipy.sparse as sp
import torch

from book_recsys.models.autoencoder.model import MultVAE
from book_recsys.models.autoencoder.train import (_device_type, anneal_beta, load_checkpoint,
                                                  save_checkpoint, train_multvae)


def test_anneal_beta_linear_then_flat():
    assert anneal_beta(0, 10, 0.2) == 0.0
    assert anneal_beta(5, 10, 0.2) == 0.1
    assert anneal_beta(10, 10, 0.2) == 0.2
    assert anneal_beta(99, 10, 0.2) == 0.2  # clamps
    assert anneal_beta(3, 0, 0.2) == 0.2  # no warm-up


def test_device_type_mapping():
    assert _device_type("cuda:0") == "cuda"
    assert _device_type("mps") == "mps"
    assert _device_type("cpu") == "cpu"


def _block_matrix():
    # 8 users x 4 items, two co-occurring pairs: users 0-3 read {0,1}, 4-7 read {2,3}
    rows, cols = [], []
    for u in range(4):
        rows += [u, u]
        cols += [0, 1]
    for u in range(4, 8):
        rows += [u, u]
        cols += [2, 3]
    data = np.ones(len(rows), dtype=np.float32)
    return sp.csr_matrix((data, (rows, cols)), shape=(8, 4))


def test_training_reduces_loss():
    torch.manual_seed(0)
    matrix = _block_matrix()
    model = MultVAE(n_items=4, hidden=16, latent=4, dropout=0.0)
    x = torch.from_numpy(matrix.toarray())
    with torch.no_grad():
        logits, mu, logvar = model(x)
        before = model.loss(x, logits, mu, logvar, beta=0.2).item()
    train_multvae(model, matrix, epochs=50, batch_size=8, anneal_steps=10, device="cpu")
    with torch.no_grad():
        logits, mu, logvar = model(x)
        after = model.loss(x, logits, mu, logvar, beta=0.2).item()
    assert after < before


def test_amp_path_runs_on_cpu():
    matrix = _block_matrix()
    model = MultVAE(n_items=4, hidden=8, latent=2, dropout=0.0)
    train_multvae(model, matrix, epochs=1, batch_size=8, device="cpu", amp=True)


def test_checkpoint_save_load_roundtrip(tmp_path):
    matrix = _block_matrix()
    model = MultVAE(n_items=4, hidden=8, latent=2, dropout=0.0)
    train_multvae(model,
                  matrix,
                  epochs=2,
                  batch_size=8,
                  device="cpu",
                  ids=["b0", "b1", "b2", "b3"],
                  ckpt_dir=str(tmp_path))
    ckpt_file = tmp_path / "multvae_last.pt"
    assert ckpt_file.exists()
    loaded, ckpt = load_checkpoint(str(ckpt_file), device="cpu")
    assert ckpt["epoch"] == 2 and ckpt["ids"] == ["b0", "b1", "b2", "b3"]
    x = torch.from_numpy(matrix.toarray())
    assert torch.allclose(loaded.predict(x), model.predict(x))


def test_resume_from_checkpoint(tmp_path):
    matrix = _block_matrix()
    model = MultVAE(n_items=4, hidden=8, latent=2, dropout=0.0)
    train_multvae(model,
                  matrix,
                  epochs=1,
                  batch_size=8,
                  device="cpu",
                  ids=None,
                  ckpt_dir=str(tmp_path))
    loaded, ckpt = load_checkpoint(str(tmp_path / "multvae_last.pt"), device="cpu")
    opt = torch.optim.Adam(loaded.parameters())
    opt.load_state_dict(ckpt["optimizer"])
    train_multvae(loaded,
                  matrix,
                  epochs=3,
                  batch_size=8,
                  device="cpu",
                  start_epoch=ckpt["epoch"],
                  optimizer=opt,
                  ckpt_dir=str(tmp_path))
    _, ckpt2 = load_checkpoint(str(tmp_path / "multvae_last.pt"), device="cpu")
    assert ckpt2["epoch"] == 3


def test_progress_prints_per_epoch(capsys):
    matrix = _block_matrix()
    model = MultVAE(n_items=4, hidden=8, latent=2, dropout=0.0)
    train_multvae(model, matrix, epochs=2, batch_size=8, device="cpu", progress=True)
    out = capsys.readouterr().out
    assert "epoch 1/2" in out and "epoch 2/2" in out and "loss" in out
