"""LLM backend via LiteLLM (provider-agnostic). Network call excluded from coverage."""


class LiteLLMClient:
    """Call any LiteLLM-supported model behind the one `complete(prompt) -> str` interface.

    `model` examples (provider is the prefix):
      - "ollama/qwen2.5:7b"        local; set api_base="http://localhost:11434"
      - "gpt-4o-mini"              needs OPENAI_API_KEY (or pass api_key)
      - "claude-3-5-haiku-latest"  needs ANTHROPIC_API_KEY
      - "gemini/gemini-1.5-flash"  needs GEMINI_API_KEY

    api_base / api_key are passed to LiteLLM only when set (so local models need no key).
    """

    def __init__(self, model: str, api_base: str | None = None,
                 api_key: str | None = None, temperature: float = 0.2,
                 timeout: int = 120) -> None:
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.temperature = temperature
        self.timeout = timeout

    def complete(self, prompt: str) -> str:  # pragma: no cover
        import litellm
        kwargs: dict = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "timeout": self.timeout,
        }
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if self.api_key:
            kwargs["api_key"] = self.api_key
        response = litellm.completion(**kwargs)
        return response["choices"][0]["message"]["content"]
