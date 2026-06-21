"""Runtime configuration for models / encoders, overridable via environment variables.

Defaults match the shipped artifacts (the catalog is embedded with the multilingual MiniLM,
384-d, so the query encoder MUST match it unless you re-embed). Override per deployment, e.g.:

    export BOOK_RECSYS_LLM_MODEL="ollama/llama3.1:8b"
    export BOOK_RECSYS_LLM_API_BASE="http://localhost:11434"
"""
import os

# Sentence-transformer encoder for text retrieval. Must match the cached catalog embeddings'
# model/dimension. Migrated from bge-small-en (English-only, degenerate on other scripts) to
# the multilingual MiniLM (also 384-d) so non-English seeds embed by meaning — see
# notebooks/12_embedding_migration.ipynb.
EMBED_MODEL = os.getenv("BOOK_RECSYS_EMBED_MODEL",
                        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Chat/overview LLM, addressed via LiteLLM (the provider prefix is part of the name).
LLM_MODEL = os.getenv("BOOK_RECSYS_LLM_MODEL", "ollama/qwen2.5:7b")
LLM_API_BASE = os.getenv("BOOK_RECSYS_LLM_API_BASE", "http://localhost:11434")

# Number of catalog books retrieved as grounding candidates for the RAG overview.
OVERVIEW_N = int(os.getenv("BOOK_RECSYS_OVERVIEW_N", "40"))
