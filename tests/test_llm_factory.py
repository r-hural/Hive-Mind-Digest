import pytest
from pipeline.config import Config, ConfigError
from pipeline.llm.factory import make_provider, make_client
from pipeline.llm.replay import ReplayProvider
from pipeline.llm.claude_cli import ClaudeCliProvider
from pipeline.llm.anthropic_api import AnthropicApiProvider
from pipeline import db


def test_default_provider_is_from_config(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    cfg = Config()
    assert isinstance(make_provider(cfg), AnthropicApiProvider)


def test_explicit_claude_cli(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: "C:/x/claude.exe")
    assert isinstance(make_provider(Config(), "claude_cli"), ClaudeCliProvider)


def test_unknown_provider_is_config_error():
    with pytest.raises(ConfigError):
        make_provider(Config(), "openai")


def test_cassette_dir_wraps_in_replay(tmp_path, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: "C:/x/claude.exe")
    p = make_provider(Config(), "claude_cli", cassette_dir=str(tmp_path / "claude_cli"))
    assert isinstance(p, ReplayProvider) and p.inner is None and p.name == "claude_cli"
    p2 = make_provider(Config(), "claude_cli", cassette_dir=str(tmp_path / "claude_cli"), record=True)
    assert isinstance(p2.inner, ClaudeCliProvider) and p2.record


def test_make_client_uses_provider_rate_limit(tmp_path, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: "C:/x/claude.exe")
    path = str(tmp_path / "s.sqlite"); db.init_db(path)
    cfg = Config()
    p = make_provider(cfg, "claude_cli")
    c = make_client(cfg, p, db_path=path, run_id=None, notebook_id=None, budget_usd=1.0)
    assert c._rate_limiter.rpm == 6 and c._budget_usd == 1.0 and c._max_output_tokens == 64000


def test_replay_only_provider_gets_no_rate_limiter(tmp_path):
    path = str(tmp_path / "s.sqlite"); db.init_db(path)
    cfg = Config()
    p = make_provider(cfg, "anthropic_api", cassette_dir=str(tmp_path / "anthropic_api"))
    c = make_client(cfg, p, db_path=path, run_id=None, notebook_id=None, budget_usd=0)
    assert c._rate_limiter is None


def test_unknown_provider_with_cassette_dir_is_config_error(tmp_path):
    with pytest.raises(ConfigError):
        make_provider(Config(), "openai", cassette_dir=str(tmp_path / "openai"))


def test_log_is_forwarded_to_client(tmp_path, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: "C:/x/claude.exe")
    path = str(tmp_path / "s.sqlite"); db.init_db(path)
    cfg = Config()
    p = make_provider(cfg, "claude_cli")
    sentinel = lambda msg: None
    c = make_client(cfg, p, db_path=path, run_id=None, notebook_id=None, budget_usd=0, log=sentinel)
    assert c._emit is sentinel


def test_record_without_cassette_dir_is_config_error(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: "C:/x/claude.exe")
    with pytest.raises(ConfigError):
        make_provider(Config(), "claude_cli", record=True)


from pipeline.llm.anthropic_batch import AnthropicBatchProvider


def test_batch_builds_the_batch_provider_from_config(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    cfg = Config(llm={"batch": {"min_requests": 3, "max_wave": 7, "poll_seconds": 5, "max_wait_hours": 2}})
    p = make_provider(cfg, batch=True)
    assert isinstance(p, AnthropicBatchProvider) and p.name == "anthropic_api"
    assert (p._min_requests, p._max_wave, p._poll, p._max_wait) == (3, 7, 5.0, 7200.0)


def test_batch_with_claude_cli_or_cassettes_is_a_config_error(tmp_path, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: "C:/x/claude.exe")
    with pytest.raises(ConfigError, match="batch"):
        make_provider(Config(), "claude_cli", batch=True)
    with pytest.raises(ConfigError, match="batch"):
        make_provider(Config(), "anthropic_api", cassette_dir=str(tmp_path / "c"), batch=True)


def test_batch_provider_gets_no_rate_limiter(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    path = str(tmp_path / "s.sqlite"); db.init_db(path)
    cfg = Config()
    c = make_client(cfg, make_provider(cfg, batch=True), db_path=path, run_id=None, notebook_id=None, budget_usd=0)
    assert c._rate_limiter is None


def test_batch_provider_journals_in_the_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    p = make_provider(Config(home=str(tmp_path)), batch=True)
    assert p._journal is not None and p._journal.path == Config(home=str(tmp_path)).batch_journal_path
