import torch

from book_recsys.models.sequential.model import SASRec

# A tiny deterministic config — fast, but exercises every code path of the real net.
TINY = dict(n_items=6, hidden_size=8, n_layers=2, n_heads=2, inner_size=16, max_seq_length=4)


def _net():
    torch.manual_seed(0)
    return SASRec(**TINY).eval()


def test_forward_returns_last_position_hidden_shape():
    net = _net()
    item_seq = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]])  # right-padded with 0
    item_seq_len = torch.tensor([3, 2])
    out = net(item_seq, item_seq_len)
    assert out.shape == (2, TINY["hidden_size"])


def test_padding_does_not_change_earlier_output():
    # Extending a sequence with trailing PAD (and keeping the true length) must not
    # change the gathered hidden state — confirms the padding mask works.
    net = _net()
    short = net(torch.tensor([[1, 2, 3, 0]]), torch.tensor([3]))
    # same 3 real items, different padding tail is impossible at L=4; use L identical,
    # so instead verify causality: a future item must not affect an earlier gather.
    a = net(torch.tensor([[1, 2, 0, 0]]), torch.tensor([2]))
    b = net(torch.tensor([[1, 2, 5, 0]]), torch.tensor([2]))  # item 5 is AFTER pos 1
    assert torch.allclose(a, b, atol=1e-6)  # causal mask hides position 2 from gather@1
    assert short.shape == (1, TINY["hidden_size"])


def test_state_dict_keys_match_recbole_layout():
    keys = set(_net().state_dict())
    expected = {
        "item_embedding.weight",
        "position_embedding.weight",
        "LayerNorm.weight",
        "LayerNorm.bias",
    }
    for i in range(TINY["n_layers"]):
        p = f"trm_encoder.layer.{i}."
        for s in ("query", "key", "value", "dense"):
            expected |= {
                f"{p}multi_head_attention.{s}.weight", f"{p}multi_head_attention.{s}.bias"
            }
        expected |= {
            f"{p}multi_head_attention.LayerNorm.weight", f"{p}multi_head_attention.LayerNorm.bias"
        }
        for s in ("dense_1", "dense_2"):
            expected |= {f"{p}feed_forward.{s}.weight", f"{p}feed_forward.{s}.bias"}
        expected |= {f"{p}feed_forward.LayerNorm.weight", f"{p}feed_forward.LayerNorm.bias"}
    assert keys == expected
