import pytest
from pipeline.config import load_config, Config, ConfigError


def test_defaults_when_file_missing(tmp_path):
    cfg = load_config(str(tmp_path / "nope.yaml"))
    assert cfg.llm.provider == "anthropic_api"
    assert cfg.llm.models["analyze"] == "claude-haiku-5-5"
    assert cfg.llm.models["merge"] == "claude-sonnet-5-5"
    assert cfg.llm.max_parallel == 4
    assert cfg.transcription.providers == ["youtube_captions", "ytdlp_subs", "whisper_local"]
    assert cfg.run.budget_usd == 5.0


def test_paths_derive_from_home():
    cfg = Config(home="X")
    assert cfg.db_path.replace("\\", "/") == "X/domains/general/state.sqlite"
    assert cfg.vault_dir.replace("\\", "/") == "X/domains/general/vault"
    assert cfg.transcript_cache_dir.replace("\\", "/") == "X/cache/transcripts"


def test_reads_yaml(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("llm:\n  provider: claude_cli\n  max_parallel: 2\nrun:\n  budget_usd: 1.5\n", encoding="utf-8")
    cfg = load_config(str(p))
    assert cfg.llm.provider == "claude_cli"
    assert cfg.llm.max_parallel == 2
    assert cfg.run.budget_usd == 1.5


def test_unknown_key_is_an_error(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("subagents:\n  max_parallel: 10\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="subagents"):
        load_config(str(p))


def test_unknown_provider_is_an_error(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("llm:\n  provider: openai\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="provider"):
        load_config(str(p))


def test_models_must_include_analyze_and_merge(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("llm:\n  models:\n    analyze: x\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="merge"):
        load_config(str(p))


def test_repo_config_loads():
    cfg = load_config("config/config.yaml")
    assert cfg.llm.provider in ("anthropic_api", "claude_cli")


def test_malformed_yaml_is_a_config_error(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("llm: [unclosed\n", encoding="utf-8")
    with pytest.raises(ConfigError, match=r"c\.yaml"):
        load_config(str(p))


def test_max_parallel_zero_is_an_error(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("llm:\n  max_parallel: 0\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="max_parallel"):
        load_config(str(p))


def test_budget_usd_negative_is_an_error(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("run:\n  budget_usd: -1\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="budget_usd"):
        load_config(str(p))


def test_rate_limits_unknown_provider_is_an_error(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("llm:\n  rate_limits:\n    openai:\n      rpm: 10\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="openai"):
        load_config(str(p))


def test_rate_limits_partial_override_fills_missing_provider(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("llm:\n  rate_limits:\n    claude_cli:\n      rpm: 3\n", encoding="utf-8")
    cfg = load_config(str(p))
    assert cfg.llm.rate_limits["claude_cli"].rpm == 3
    assert cfg.llm.rate_limits["anthropic_api"].rpm == 50
    assert cfg.llm.rate_limits["anthropic_api"].tpm == 400_000


def test_rate_limits_partial_override_merges_within_provider(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("llm:\n  rate_limits:\n    anthropic_api:\n      rpm: 10\n", encoding="utf-8")
    cfg = load_config(str(p))
    assert cfg.llm.rate_limits["anthropic_api"].rpm == 10
    assert cfg.llm.rate_limits["anthropic_api"].tpm == 400_000


def test_rate_limits_non_mapping_is_an_error(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("llm:\n  rate_limits: 5\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(str(p))


def test_duplicate_provider_is_an_error(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("transcription:\n  providers: [ytdlp_subs, ytdlp_subs]\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="repeat"):
        load_config(str(p))


def test_supadata_is_accepted_but_off_by_default(tmp_path):
    assert "supadata" not in Config().transcription.providers
    p = tmp_path / "c.yaml"
    p.write_text("transcription:\n  providers: [youtube_captions, ytdlp_subs, supadata, whisper_local]\n",
                 encoding="utf-8")
    cfg = load_config(str(p))
    assert cfg.transcription.providers == ["youtube_captions", "ytdlp_subs", "supadata", "whisper_local"]


def test_duplicate_supadata_is_an_error(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("transcription:\n  providers: [supadata, supadata]\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="repeat"):
        load_config(str(p))


def test_batch_defaults_off(tmp_path):
    b = load_config(str(tmp_path / "missing.yaml")).llm.batch
    assert (b.enabled, b.min_requests, b.max_wave, b.gather_seconds, b.poll_seconds, b.max_wait_hours) == \
        (False, 2, 100, 2.0, 30.0, 24.0)


def test_batch_max_wait_over_24h_is_an_error(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("llm:\n  batch:\n    max_wait_hours: 30\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="max_wait_hours"):
        load_config(str(p))


def test_batch_max_wave_is_bounded_by_the_pool_size(tmp_path):
    # the runner opens one LLM worker thread per wave slot, so 10,000 would mean 10,000 threads
    from pipeline.config import MAX_WAVE
    assert load_config_text(tmp_path, f"llm:\n  batch:\n    max_wave: {MAX_WAVE}\n").llm.batch.max_wave == MAX_WAVE
    with pytest.raises(ConfigError, match="max_wave"):
        load_config_text(tmp_path, f"llm:\n  batch:\n    max_wave: {MAX_WAVE + 1}\n")


def load_config_text(tmp_path, text):
    p = tmp_path / "c.yaml"
    p.write_text(text, encoding="utf-8")
    return load_config(str(p))


def test_batch_journal_lives_in_the_cache():
    assert Config(home="h").batch_journal_path.replace("\\", "/") == "h/cache/batches.json"


def test_output_formats_default_and_summary_always_first(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("output:\n  formats: [digest, digest, outline]\n", encoding="utf-8")
    assert load_config(str(p)).output.formats == ["summary", "digest", "outline"]
    assert load_config(str(tmp_path / "missing.yaml")).output.formats == ["summary", "diff"]


def test_output_formats_rejects_unknown(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("output:\n  formats: [pdf]\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(str(p))


def test_output_html_defaults_on():
    from pipeline.config import Config
    assert Config().output.html is True
    assert Config(output={"html": False}).output.html is False
