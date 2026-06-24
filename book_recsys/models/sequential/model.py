"""SASRec re-implemented in standalone torch so a RecBole-trained checkpoint serves
on MPS/CPU without importing RecBole.

Submodule and tensor names match RecBole's `SASRec` / `TransformerEncoder` exactly, so
its `state_dict` loads with `strict=True`. The forward mirrors RecBole: item + positional
embeddings, input LayerNorm, N post-LayerNorm transformer blocks under a causal +
padding mask, then the hidden state gathered at the last real position (`len - 1`).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

_MASK_FILL = -10000.0  # RecBole's additive mask constant for disallowed attention


class _MultiHeadAttention(nn.Module):

    def __init__(self, hidden_size: int, n_heads: int, eps: float) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=eps)

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        b, length, _ = x.shape
        return x.view(b, length, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        q, k, v = self._heads(self.query(x)), self._heads(self.key(x)), self._heads(self.value(x))
        scores = q @ k.transpose(-1, -2) / math.sqrt(self.head_dim)
        probs = F.softmax(scores + mask, dim=-1)
        ctx = (probs @ v).permute(0, 2, 1, 3).contiguous()
        ctx = ctx.view(ctx.size(0), ctx.size(1), -1)
        return self.LayerNorm(self.dense(ctx) + x)  # residual + post-LN


class _FeedForward(nn.Module):

    def __init__(self, hidden_size: int, inner_size: int, eps: float) -> None:
        super().__init__()
        self.dense_1 = nn.Linear(hidden_size, inner_size)
        self.dense_2 = nn.Linear(inner_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dense_2(F.gelu(self.dense_1(x)))
        return self.LayerNorm(h + x)  # residual + post-LN


class _TransformerLayer(nn.Module):

    def __init__(self, hidden_size: int, n_heads: int, inner_size: int, eps: float) -> None:
        super().__init__()
        self.multi_head_attention = _MultiHeadAttention(hidden_size, n_heads, eps)
        self.feed_forward = _FeedForward(hidden_size, inner_size, eps)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.feed_forward(self.multi_head_attention(x, mask))


class _TransformerEncoder(nn.Module):

    def __init__(self, n_layers: int, hidden_size: int, n_heads: int, inner_size: int,
                 eps: float) -> None:
        super().__init__()
        self.layer = nn.ModuleList(
            [_TransformerLayer(hidden_size, n_heads, inner_size, eps) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layer:
            x = layer(x, mask)
        return x


class SASRec(nn.Module):
    """Self-attentive sequential recommender; state_dict-compatible with RecBole's SASRec."""

    def __init__(self,
                 n_items: int,
                 hidden_size: int = 64,
                 n_layers: int = 2,
                 n_heads: int = 2,
                 inner_size: int = 256,
                 max_seq_length: int = 50,
                 layer_norm_eps: float = 1e-12) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.max_seq_length = max_seq_length
        self.item_embedding = nn.Embedding(n_items, hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_length, hidden_size)
        self.trm_encoder = _TransformerEncoder(n_layers, hidden_size, n_heads, inner_size,
                                               layer_norm_eps)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def _attention_mask(self, item_seq: torch.Tensor) -> torch.Tensor:
        length = item_seq.size(1)
        pad = (item_seq != 0).unsqueeze(1).unsqueeze(2)  # (B,1,1,L)
        causal = torch.tril(torch.ones((length, length), device=item_seq.device)).bool()
        allowed = pad & causal  # (B,1,L,L)
        return torch.where(allowed, 0.0, torch.tensor(_MASK_FILL, device=item_seq.device))

    def forward(self, item_seq: torch.Tensor, item_seq_len: torch.Tensor) -> torch.Tensor:
        length = item_seq.size(1)
        pos_ids = torch.arange(length, device=item_seq.device).unsqueeze(0).expand_as(item_seq)
        x = self.item_embedding(item_seq) + self.position_embedding(pos_ids)
        x = self.LayerNorm(x)
        x = self.trm_encoder(x, self._attention_mask(item_seq))
        idx = (item_seq_len - 1).view(-1, 1, 1).expand(-1, 1, self.hidden_size)
        return x.gather(1, idx).squeeze(1)
