from book_recsys.llm.rerank import rerank

DOCS = {"b0": "doc0", "b1": "doc1", "b2": "doc2"}


class _ScoringClient:
    """Returns the same scores JSON every call; rerank filters to each batch."""

    def __init__(self):
        self.calls = 0

    def complete(self, prompt):
        self.calls += 1
        return '[{"id": "b2", "score": 9}, {"id": "b0", "score": 5}, {"id": "b1", "score": 1}]'


def test_orders_candidates_by_score():
    client = _ScoringClient()
    out = rerank("ctx", ["b0", "b1", "b2"], DOCS, client, k=3)
    assert out == ["b2", "b0", "b1"]


def test_batches_when_more_than_batch_size():
    client = _ScoringClient()
    rerank("ctx", ["b0", "b1", "b2"], DOCS, client, k=3, batch_size=2)
    assert client.calls == 2  # ceil(3/2)


def test_respects_k():
    out = rerank("ctx", ["b0", "b1", "b2"], DOCS, _ScoringClient(), k=1)
    assert out == ["b2"]


def test_unscored_candidates_keep_original_order():
    class _Blank:
        def complete(self, prompt):
            return "no json here"

    out = rerank("ctx", ["b0", "b1", "b2"], DOCS, _Blank(), k=3)
    assert out == ["b0", "b1", "b2"]


def test_malformed_json_in_brackets_scores_zero():
    class _BadJson:
        def complete(self, prompt):
            return "[this is not, valid json]"  # has brackets but won't parse

    out = rerank("ctx", ["b0", "b1"], DOCS, _BadJson(), k=2)
    assert out == ["b0", "b1"]  # all scores default to 0, original order kept


import pytest


@pytest.mark.parametrize("response", [
    '[{"id": "b0", "score": null}]',        # null score
    '[{"id": "b0", "score": "high"}]',      # non-numeric score
    '["b0", 42]',                            # non-dict list items
    '[{"id": "b0", "score": 7}, "garbage"]',  # mixed: one good, one bad
])
def test_tolerates_malformed_score_items(response):
    class _Client:
        def complete(self, prompt):
            return response

    out = rerank("ctx", ["b0", "b1"], DOCS, _Client(), k=2)
    # must not crash; returns both candidates (k=2), order may vary
    assert set(out) == {"b0", "b1"}
    assert len(out) == 2
