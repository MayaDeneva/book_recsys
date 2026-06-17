"""Runtime configuration for models / encoders, overridable via environment variables.

Defaults match the shipped artifacts (the catalog was embedded with bge-small, 384-d, so
the query encoder MUST stay bge-small unless you re-embed). Override per deployment, e.g.:

    export BOOK_RECSYS_LLM_MODEL="ollama/llama3.1:8b"
    export BOOK_RECSYS_LLM_API_BASE="http://localhost:11434"
"""
import os

# Sentence-transformer encoder for text retrieval. Must match the cached catalog
# embeddings' model/dimension (bge-small = 384-d).
EMBED_MODEL = os.getenv("BOOK_RECSYS_EMBED_MODEL", "BAAI/bge-small-en-v1.5")

# Chat/overview LLM, addressed via LiteLLM (the provider prefix is part of the name).
LLM_MODEL = os.getenv("BOOK_RECSYS_LLM_MODEL", "ollama/qwen2.5:7b")
LLM_API_BASE = os.getenv("BOOK_RECSYS_LLM_API_BASE", "http://localhost:11434")

# Number of catalog books retrieved as grounding candidates for the RAG overview.
OVERVIEW_N = int(os.getenv("BOOK_RECSYS_OVERVIEW_N", "40"))
