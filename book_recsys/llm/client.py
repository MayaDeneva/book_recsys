"""The interface every LLM backend implements."""
from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    """Completes a prompt to a text response."""

    def complete(self, prompt: str) -> str:
        ...
