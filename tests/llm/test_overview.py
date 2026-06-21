from book_recsys.llm.overview import OverviewGenerator, parse_overview


def test_parse_overview_keeps_allowed_ids_and_skips_junk():
    raw = ('{"intro": "x", "categories": ["junk", {"header": "H", "items": '
           '["bad", {"id": "b1", "reason": "r"}, {"id": "zz", "reason": "y"}]}]}')
    out = parse_overview(raw, ["b1", "b2"])
    assert out == {
        "intro": "x",
        "categories": [{
            "header": "H",
            "items": [{
                "book_id": "b1",
                "reason": "r"
            }]
        }]
    }


def test_parse_overview_drops_category_with_no_allowed_items():
    raw = '{"intro": "x", "categories": [{"header": "H", "items": [{"id": "zz", "reason": "r"}]}]}'
    assert parse_overview(raw, ["b1"]) == {"intro": "x", "categories": []}


def test_parse_overview_no_json_returns_empty():
    assert parse_overview("just prose", ["b1"]) == {"intro": "", "categories": []}


def test_parse_overview_invalid_json_returns_empty():
    assert parse_overview("{not valid}", ["b1"]) == {"intro": "", "categories": []}


class _FakeRetriever:

    def by_text(self, text, n):
        return ["b1", "b2", "b3"][:n]

    def by_history(self, history, n):
        return ["b2", "b4"][:n]


class _FakeClient:

    def __init__(self, raw):
        self.raw = raw
        self.prompt = None

    def complete(self, prompt):
        self.prompt = prompt
        return self.raw


def test_generate_grounds_overview_in_retrieved_candidates():
    client = _FakeClient('{"intro": "ov", "categories": [{"header": "Top", '
                         '"items": [{"id": "b1", "reason": "good"}]}]}')
    gen = OverviewGenerator(_FakeRetriever(), {"b1": "Doc1", "b2": "Doc2"}, client, n=10)
    out = gen.generate("war books", history=[], history_titles=[])
    assert out["intro"] == "ov"
    assert out["categories"][0]["items"][0]["book_id"] == "b1"
    assert "id=b1: Doc1" in client.prompt  # candidates were put in the prompt


def test_generate_fuses_history_and_query_candidates():
    client = _FakeClient('{"intro": "", "categories": [{"header": "H", '
                         '"items": [{"id": "b4", "reason": "r"}]}]}')
    gen = OverviewGenerator(_FakeRetriever(), {"b4": "D"}, client, n=10)
    out = gen.generate("darker", history=["seed"], history_titles=["A Liked Book"])
    assert out["categories"][0]["items"][0]["book_id"] == "b4"  # b4 came from by_history
    assert "A Liked Book" in client.prompt


def test_generate_no_candidates_returns_empty():

    class _Empty:

        def by_text(self, t, n):
            return []

        def by_history(self, h, n):
            return []

    gen = OverviewGenerator(_Empty(), {}, _FakeClient("{}"), n=10)
    assert gen.generate("x", history=[], history_titles=[]) == {"intro": "", "categories": []}


def test_generate_no_query_and_no_history_returns_empty():
    gen = OverviewGenerator(_FakeRetriever(), {}, _FakeClient("{}"), n=10)
    assert gen.generate("", history=[], history_titles=[]) == {"intro": "", "categories": []}
