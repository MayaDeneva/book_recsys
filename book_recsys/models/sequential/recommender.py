"""SASRec wrapped as a `Recommender` (fit / recommend / score_items) for UI serving.

Loads the clean tensor-only state file produced by `scripts/export_sasrec_state.py` plus
the `item_id_token` vocab; no RecBole import on the serving path. `query` is an ordered
list of book_ids (oldest -> newest); SASRec predicts the next item after that sequence.
"""
import json

import numpy as np
import torch

from book_recsys.models.sequential.model import SASRec


def _auto_device() -> str:
    if torch.cuda.is_available():  # pragma: no cover - hardware-dependent
        return "cuda"
    if torch.backends.mps.is_available():  # pragma: no cover - hardware-dependent
        return "mps"
    return "cpu"  # pragma: no cover - hardware-dependent


def load_state_dict_clean(path) -> dict:
    """Load a tensor-only state_dict (no RecBole objects) from `path`."""
    return torch.load(path, map_location="cpu", weights_only=True)


class SasRecRecommender:
    """Serve a RecBole-trained SASRec on MPS/CPU behind the Recommender Protocol."""

    weight_aware = False  # SASRec encodes recency via sequence position, not input weights

    def __init__(self, model: SASRec, item_tokens: list, device=None) -> None:
        self.device = device or _auto_device()
        self._model = model.to(self.device).eval()
        self._tokens = list(item_tokens)
        self._pos = {tok: i for i, tok in enumerate(self._tokens)}  # book_id -> internal id

    @classmethod
    def from_checkpoint(cls, state_path, item_map_path, device=None, n_heads: int = 2):
        sd = load_state_dict_clean(state_path)
        n_items, hidden = sd["item_embedding.weight"].shape
        max_seq = sd["position_embedding.weight"].shape[0]
        inner = sd["trm_encoder.layer.0.feed_forward.dense_1.weight"].shape[0]
        n_layers = 1 + max(
            int(key.split(".")[2]) for key in sd if key.startswith("trm_encoder.layer."))
        model = SASRec(n_items, hidden, n_layers, n_heads, inner, max_seq)
        model.load_state_dict(sd, strict=True)
        with open(item_map_path) as fh:
            tokens = json.load(fh)["item_id_token"]
        return cls(model, tokens, device)

    def fit(self, train_data) -> "SasRecRecommender":
        return self  # pre-trained checkpoint; nothing to fit

    def _seq_tensor(self, query):
        """Map book_ids -> internal ids, drop unknowns, keep the most recent max_seq items."""
        ids = [self._pos[b] for b in query if b in self._pos]
        ids = ids[-self._model.max_seq_length:]
        return ids

    def _scores(self, query):
        ids = self._seq_tensor(query)
        length = len(ids)
        padded = ids + [0] * (self._model.max_seq_length - length)
        item_seq = torch.tensor([padded], dtype=torch.long, device=self.device)
        item_seq_len = torch.tensor([length], dtype=torch.long, device=self.device)
        with torch.no_grad():
            hidden = self._model(item_seq, item_seq_len)  # (1, H)
            logits = hidden @ self._model.item_embedding.weight.t()  # (1, n_items)
        return logits[0].cpu().numpy()

    def recommend(self, query, k) -> list:
        ids = self._seq_tensor(query)
        if not ids:
            return []
        scores = self._scores(query)
        seen = set(ids) | {0}  # exclude already-seen items and PAD
        out = []
        for j in np.argsort(-scores):
            if int(j) not in seen:
                out.append(self._tokens[int(j)])
                if len(out) == k:
                    break
        return out

    def score_items(self, query, item_ids) -> list:
        if not self._seq_tensor(query):
            return [float("-inf")] * len(item_ids)
        scores = self._scores(query)
        return [float(scores[self._pos[b]]) if b in self._pos else float("-inf") for b in item_ids]
