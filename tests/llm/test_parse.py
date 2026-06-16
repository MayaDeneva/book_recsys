from book_recsys.llm.parse import ParsedQuery, parse_query


class _Client:
    def __init__(self, response):
        self._response = response

    def complete(self, prompt):
        return self._response


def test_parses_well_formed_json():
    client = _Client('{"mood": "cozy", "audience": "adult", "avoid": ["gore"]}')
    parsed = parse_query("a cozy read", client)
    assert parsed == ParsedQuery(mood="cozy", audience="adult", avoid=["gore"])


def test_extracts_json_embedded_in_prose():
    client = _Client('Sure! {"mood": "dark", "audience": null, "avoid": []} hope that helps')
    parsed = parse_query("something dark", client)
    assert parsed.mood == "dark"
    assert parsed.audience is None
    assert parsed.avoid == []


def test_returns_empty_when_no_json():
    parsed = parse_query("x", _Client("I cannot help with that."))
    assert parsed == ParsedQuery()


def test_returns_empty_on_malformed_json():
    parsed = parse_query("x", _Client("{mood: cozy,,,}"))
    assert parsed == ParsedQuery()


def test_missing_avoid_defaults_to_empty_list():
    parsed = parse_query("x", _Client('{"mood": "calm"}'))
    assert parsed.avoid == []
