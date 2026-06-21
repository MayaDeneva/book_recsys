from book_recsys import config


def test_config_defaults_match_shipped_artifacts():
    assert config.EMBED_MODEL == "BAAI/bge-small-en-v1.5"  # 384-d, matches the catalog
    assert config.LLM_MODEL.startswith("ollama/")
    assert config.LLM_API_BASE.startswith("http")
    assert config.OVERVIEW_N > 0
