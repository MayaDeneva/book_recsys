from dataclasses import replace

from book_recsys.llm.steer import Steerer, SteeringState, build_steer_prompt, parse_steering


def test_parse_full_state_overrides_prev():
    prev = SteeringState(history_weight=1.0)
    raw = ('{"history_weight": 0.6, "topic": "WWII submarines", "avoid": ["too dark"], '
           '"genre": "history", "anchor_book": "Das Boot", "reply": "Shifting toward WWII."}')
    out = parse_steering(raw, prev)
    assert out == SteeringState(history_weight=0.6,
                                topic="WWII submarines",
                                avoid=["too dark"],
                                genre="history",
                                anchor_book="Das Boot",
                                reply="Shifting toward WWII.")


def test_parse_absent_key_keeps_prev_present_null_clears():
    prev = SteeringState(history_weight=0.5, topic="sailing", genre="history")
    # topic absent -> kept; genre explicit null -> cleared.
    out = parse_steering('{"genre": null, "reply": "ok"}', prev)
    assert out.topic == "sailing"
    assert out.genre is None
    assert out.history_weight == 0.5
    assert out.reply == "ok"


def test_parse_clamps_history_weight():
    out = parse_steering('{"history_weight": 1.7}', SteeringState())
    assert out.history_weight == 1.0
    out = parse_steering('{"history_weight": -3}', SteeringState())
    assert out.history_weight == 0.0


def test_parse_bad_history_weight_keeps_prev():
    out = parse_steering('{"history_weight": "lots"}', SteeringState(history_weight=0.4))
    assert out.history_weight == 0.4


def test_parse_avoid_sanitized_to_str_list():
    out = parse_steering('{"avoid": ["dark", 5, "", "  romance "]}', SteeringState())
    assert out.avoid == ["dark", "romance"]


def test_parse_empty_topic_string_becomes_none():
    out = parse_steering('{"topic": "   "}', SteeringState(topic="old"))
    assert out.topic is None


def test_parse_no_json_returns_copy_of_prev():
    prev = SteeringState(history_weight=0.3, topic="x")
    out = parse_steering("just prose, no json", prev)
    assert out == prev and out is not prev


def test_parse_invalid_json_returns_copy_of_prev():
    prev = SteeringState(topic="x")
    out = parse_steering("{not valid}", prev)
    assert out == prev and out is not prev


def test_reply_defaults_empty_when_absent():
    out = parse_steering('{"topic": "y"}', SteeringState(reply="old reply"))
    assert out.reply == ""


def test_reply_null_becomes_empty_string():
    out = parse_steering('{"reply": null}', SteeringState(reply="old"))
    assert out.reply == ""


def test_parse_avoid_non_list_becomes_empty():
    out = parse_steering('{"avoid": "dark"}', SteeringState())
    assert out.avoid == []


def test_build_steer_prompt_includes_state_messages_titles_and_rules():
    prev = SteeringState(history_weight=0.7, topic="sailing", avoid=["gore"])
    prompt = build_steer_prompt([{
        "role": "user",
        "text": "but about WWII"
    }], prev, ["Moby-Dick", "The Old Man and the Sea"])
    assert "0.7" in prompt and "sailing" in prompt and "gore" in prompt
    assert "but about WWII" in prompt
    assert "Moby-Dick" in prompt
    assert "genre" in prompt.lower()  # the null-unless-explicit rule
    assert "gift" in prompt.lower()  # the gift / for-someone-else rule


class _FakeClient:

    def __init__(self, raw):
        self.raw = raw
        self.prompt = None

    def complete(self, prompt):
        self.prompt = prompt
        return self.raw


def test_steerer_update_parses_client_reply_onto_prev():
    client = _FakeClient('{"history_weight": 0.4, "topic": "WWII", "reply": "ok"}')
    out = Steerer(client).update([{
        "role": "user",
        "text": "WWII please"
    }], SteeringState(topic="sailing"), [])
    assert out.history_weight == 0.4
    assert out.topic == "WWII"
    assert out.reply == "ok"
    assert "WWII please" in client.prompt  # the message reached the prompt
