from book_recsys.llm.client import LLMClient
from book_recsys.llm.clients import LiteLLMClient


class _Fake:

    def complete(self, prompt):
        return "ok"


def test_fake_satisfies_protocol():
    assert isinstance(_Fake(), LLMClient)


def test_non_conforming_fails_protocol():
    assert not isinstance(object(), LLMClient)


def test_litellm_client_stores_config():
    c = LiteLLMClient(model="ollama/qwen2.5:7b", api_base="http://localhost:11434")
    assert c.model == "ollama/qwen2.5:7b"
    assert c.api_base == "http://localhost:11434"
    assert isinstance(c, LLMClient)


def test_litellm_client_defaults():
    c = LiteLLMClient(model="gpt-4o-mini")
    assert c.api_base is None
    assert c.api_key is None
    assert c.temperature == 0.2
    assert c.timeout == 120
