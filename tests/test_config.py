from book_recsys import config


def test_config_defaults_match_shipped_artifacts():
    # multilingual MiniLM (384-d) — matches the re-embedded catalog (notebook 12)
    assert config.EMBED_MODEL == "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    assert config.LLM_MODEL.startswith("ollama/")
    assert config.LLM_API_BASE.startswith("http")
    assert config.OVERVIEW_N > 0
