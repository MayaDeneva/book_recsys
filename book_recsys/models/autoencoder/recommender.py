"""Mult-VAE wrapped as a Recommender (fit / recommend / score_items)."""
import numpy as np
import torch

from book_recsys.models.autoencoder.data import build_matrix
from book_recsys.models.autoencoder.model import MultVAE
from book_recsys.models.autoencoder.train import train_multvae


def _auto_device() -> str:
    if torch.cuda.is_available():  # pragma: no cover - hardware-dependent
        return "cuda"
    if torch.backends.mps.is_available():  # pragma: no cover - hardware-dependent
        return "mps"
    return "cpu"  # pragma: no cover - hardware-dependent


class MultVaeRecommender:
    """Fold-in autoencoder recommender: history -> multi-hot vector -> reconstruct scores.

    `pop_discount` (alpha) subtracts `alpha * log(item_count)` from every score at inference,
    demoting popularity-head items — sweep it to trade accuracy for serendipity.
    """

    def __init__(self,
                 hidden=600,
                 latent=200,
                 dropout=0.5,
                 beta_cap=0.2,
                 epochs=50,
                 batch_size=500,
                 lr=1e-3,
                 anneal_steps=10000,
                 min_item_count=1,
                 pop_discount=0.0,
                 device=None,
                 seed=42,
                 ckpt_dir=None,
                 progress=False) -> None:
        self.hidden, self.latent, self.dropout = hidden, latent, dropout
        self.beta_cap, self.epochs, self.batch_size = beta_cap, epochs, batch_size
        self.lr, self.anneal_steps = lr, anneal_steps
        self.min_item_count, self.pop_discount = min_item_count, pop_discount
        self.device = device or _auto_device()
        self.seed, self.ckpt_dir, self.progress = seed, ckpt_dir, progress
        self._ids: list = []
        self._pos: dict = {}
        self._model = None
        self._log_pop = None

    def fit(self, train_data):
        matrix, ids, pos, counts = build_matrix(train_data, self.min_item_count)
        self._ids, self._pos = ids, pos
        self._log_pop = np.log(counts)
        model = MultVAE(len(ids), self.hidden, self.latent, self.dropout)
        train_multvae(model,
                      matrix,
                      epochs=self.epochs,
                      batch_size=self.batch_size,
                      lr=self.lr,
                      anneal_steps=self.anneal_steps,
                      beta_cap=self.beta_cap,
                      device=self.device,
                      seed=self.seed,
                      ids=ids,
                      ckpt_dir=self.ckpt_dir,
                      progress=self.progress)
        self._model = model.eval()
        return self

    def attach(self, model, ids, pos, counts):
        """Use an already-trained model + vocab (e.g. from load_checkpoint)."""
        self._model = model.to(self.device).eval()
        self._ids, self._pos = ids, pos
        self._log_pop = np.log(np.asarray(counts, dtype=float))
        return self

    def _scores(self, query):
        x = torch.zeros(1, len(self._ids))
        idx = [self._pos[b] for b in query if b in self._pos]
        if idx:
            x[0, idx] = 1.0
        with torch.no_grad():
            logits = self._model.predict(x.to(self.device)).cpu().numpy()[0]
        return logits - self.pop_discount * self._log_pop

    def recommend(self, query, k):
        if self._model is None or not self._ids:
            return []
        scores = self._scores(query)
        seen = {self._pos[b] for b in query if b in self._pos}
        out = []
        for j in np.argsort(-scores):
            if j not in seen:
                out.append(self._ids[j])
                if len(out) == k:
                    break
        return out

    def score_items(self, query, item_ids):
        if self._model is None:
            return [float("-inf")] * len(item_ids)
        scores = self._scores(query)
        return [float(scores[self._pos[b]]) if b in self._pos else float("-inf") for b in item_ids]
