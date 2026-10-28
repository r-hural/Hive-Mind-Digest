"""Configuration. The only module that reads config/config.yaml.

Every key in the file is honoured; unknown keys are an error so the file can never
become decorative again (an ignored key silently drifts from what the code does).

Which config file is used (resolve_config_path, IDEA-310), first match wins:
1. `--config <path>`;
2. the `HMD_CONFIG` environment variable;
3. plugin mode (this hmd runs from the venv /hmd:init created, pipeline/bootstrap.py): the
   `config.yaml` next to that venv, i.e. `${CLAUDE_PLUGIN_DATA}/config.yaml`;
4. `config/config.yaml` relative to the working directory (a development checkout).
Sources 1-3 are explicit: a missing file is an error (except for `hmd init`, which writes it, and
`hmd doctor`, which reports it). Source 4 silently falls back to the built-in defaults.

Source 4 is trusted like the CWD's `.env` (pipeline/envfile.py): a downloaded folder must not lift a
security check, so unless the file is this checkout's own `config/config.yaml`, each key in
UNTRUSTED_KEYS (data folder, budget, LLM provider, transcribers, private hosts) is honoured only within
its rule and otherwise falls back to the default (a warning names each one); every other key is honoured.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from pipeline import bootstrap

CONFIG_PATH = "config/config.yaml"
CONFIG_ENV = "HMD_CONFIG"
# This checkout's root (pipeline/config.py -> root); its config/config.yaml is trusted in full.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROVIDERS = ("anthropic_api", "claude_cli")
# Every known provider name. `supadata` (DECISION-026, a third-party API) is off by default: only
# DEFAULT_TRANSCRIBERS run unless config lists it (intended order: ytdlp_subs, supadata, whisper_local).
TRANSCRIBERS = ("youtube_captions", "ytdlp_subs", "supadata", "whisper_local")
DEFAULT_TRANSCRIBERS = ("youtube_captions", "ytdlp_subs", "whisper_local")
# Keys an implicit config from a foreign folder may set only within a rule: (section or None for the
# top level, key) -> allowed(value, default). A value outside its rule falls back to the default.
# Tightening is allowed (a lower budget, fewer default transcribers); lifting a safeguard is not.
UNTRUSTED_KEYS = {
    (None, "home"): lambda v, d: v == d,                                   # where the DB, cache and reports go
    ("run", "budget_usd"): lambda v, d: 0 < v <= d,                        # 0 would switch the cap off
    ("llm", "provider"): lambda v, d: v == d,                              # claude_cli bills the subscription
    ("transcription", "providers"): lambda v, d: set(v) <= set(DEFAULT_TRANSCRIBERS),   # no supadata
    ("transcription", "allow_private_hosts"): lambda v, d: v == d,
}
DEFAULT_RATE_LIMITS: dict[str, dict[str, int]] = {
    "anthropic_api": {"rpm": 50, "tpm": 400_000},
    "claude_cli": {"rpm": 6, "tpm": 0},
}


class ConfigError(Exception):
    pass


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RateLimitConfig(_Strict):
    rpm: int = Field(default=50, ge=0)   # requests per minute; 0 = unlimited
    tpm: int = Field(default=0, ge=0)    # input tokens per minute (estimated); 0 = unlimited


# The runner opens one LLM worker thread per wave slot (a wave gathers the calls in flight together), so
# max_wave is also the LLM pool's thread count; a playlist brings at most 200 videos.
MAX_WAVE = 500


class BatchConfig(_Strict):
    """Message Batches for anthropic_api runs (IDEA-304): half price, results in minutes to hours."""
    enabled: bool = False
    min_requests: int = Field(default=2, ge=1)          # a smaller wave goes to the direct API
    max_wave: int = Field(default=100, ge=1, le=MAX_WAVE)
    gather_seconds: float = Field(default=2.0, ge=0)    # quiet time that closes a wave
    poll_seconds: float = Field(default=30.0, gt=0)
    max_wait_hours: float = Field(default=24.0, gt=0, le=24)   # the API expires a batch after 24 h


class LLMConfig(_Strict):
    provider: str = "anthropic_api"
    models: dict[str, str] = Field(default_factory=lambda: {
        "analyze": "claude-haiku-5-5",
        "merge": "claude-sonnet-5-5",
    })
    max_output_tokens: int = Field(default=64000, gt=0)
    max_parallel: int = Field(default=4, gt=0)
    retry_attempts: int = Field(default=5, ge=0)
    retry_base_delay_seconds: float = Field(default=2.0, ge=0)
    rate_limits: dict[str, RateLimitConfig] = Field(default_factory=lambda: {
        p: RateLimitConfig(**vals) for p, vals in DEFAULT_RATE_LIMITS.items()
    })
    batch: BatchConfig = Field(default_factory=BatchConfig)

    @field_validator("provider")
    @classmethod
    def _provider_known(cls, v: str) -> str:
        if v not in PROVIDERS:
            raise ValueError(f"provider must be one of {PROVIDERS}, got {v!r}")
        return v

    @field_validator("models")
    @classmethod
    def _models_have_stages(cls, v: dict[str, str]) -> dict[str, str]:
        for stage in ("analyze", "merge"):
            if stage not in v:
                raise ValueError(f"models.{stage} is required")
        return v

    @field_validator("rate_limits", mode="before")
    @classmethod
    def _rate_limits_known_and_filled(cls, v):
        if v is None or not isinstance(v, dict):
            return v
        bad = [k for k in v if k not in PROVIDERS]
        if bad:
            raise ValueError(f"unknown llm.rate_limits provider(s) {bad}; known: {PROVIDERS}")
        merged = {p: dict(vals) for p, vals in DEFAULT_RATE_LIMITS.items()}
        for p, vals in v.items():
            merged[p] = {**DEFAULT_RATE_LIMITS.get(p, {}), **dict(vals)}
        return merged


class WhisperConfig(_Strict):
    model_size: str = "base"
    timeout_seconds: int = 1800


class TranscriptionConfig(_Strict):
    providers: list[str] = Field(default_factory=lambda: list(DEFAULT_TRANSCRIBERS))
    languages: list[str] = Field(default_factory=lambda: ["en"])
    whisper: WhisperConfig = Field(default_factory=WhisperConfig)
    allow_private_hosts: bool = False      # fetch web pages / PDFs from loopback, private and link-local addresses

    @field_validator("providers")
    @classmethod
    def _providers_known(cls, v: list[str]) -> list[str]:
        bad = [p for p in v if p not in TRANSCRIBERS]
        if bad:
            raise ValueError(f"unknown transcription providers {bad}; known: {TRANSCRIBERS}")
        if not v:
            raise ValueError("transcription.providers must not be empty")
        if len(set(v)) != len(v):
            raise ValueError("transcription.providers must not repeat a provider")
        return v


class RunConfig(_Strict):
    budget_usd: float = Field(default=5.0, ge=0)   # per run; 0 disables the cap


FORMATS = ("summary", "digest", "hybrid", "outline", "quotes", "diff")


def normalize_formats(requested: list[str]) -> list[str]:
    """Always start with "summary"; drop duplicates, keep order. Callers validate names first."""
    out = ["summary"]
    for f in requested:
        if f not in out:
            out.append(f)
    return out


class OutputConfig(_Strict):
    show_anchors: bool = False
    html: bool = True       # also write a self-contained .html twin of every report
    # Report files written by the output stage (DECISION-031). "summary" is always written (REPORT=
    # names it); "diff" is skipped silently when the notebook has no previous merge.
    formats: list[str] = Field(default_factory=lambda: ["summary", "diff"])

    @field_validator("formats")
    @classmethod
    def _formats(cls, v: list[str]) -> list[str]:
        bad = [f for f in v if f not in FORMATS]
        if bad:
            raise ValueError(f"unknown output formats {bad}; choose from {list(FORMATS)}")
        return normalize_formats(v)


class Config(_Strict):
    home: str = "data"

    @field_validator("home")
    @classmethod
    def _home_expands_user(cls, v: str) -> str:
        # "~/HiveMindDigest" works on every platform; a relative home stays relative to the cwd.
        return os.path.normpath(os.path.expanduser(v)) if v.startswith("~") else v
    llm: LLMConfig = Field(default_factory=LLMConfig)
    transcription: TranscriptionConfig = Field(default_factory=TranscriptionConfig)
    run: RunConfig = Field(default_factory=RunConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @property
    def domain_dir(self) -> Path:
        return Path(self.home) / "domains" / "general"

    @property
    def db_path(self) -> str:
        return str(self.domain_dir / "state.sqlite")

    @property
    def vault_dir(self) -> str:
        return str(self.domain_dir / "vault")

    @property
    def transcript_cache_dir(self) -> str:
        return str(Path(self.home) / "cache" / "transcripts")

    @property
    def batch_journal_path(self) -> str:
        return str(Path(self.home) / "cache" / "batches.json")


def plugin_data_dir(prefix: str | None = None) -> str | None:
    """`${CLAUDE_PLUGIN_DATA}` when this interpreter is the plugin venv /hmd:init created (its
    install marker sits in sys.prefix), else None."""
    prefix = prefix if prefix is not None else sys.prefix
    if bootstrap.read_marker(prefix) is None:
        return None
    return bootstrap.data_dir_of(prefix)


def run_mode(prefix: str | None = None) -> str:
    """"plugin" (the /hmd:init venv), "development" (this package runs from a checkout with its
    pyproject.toml) or "installed" (a plain pip install anywhere else)."""
    if plugin_data_dir(prefix) is not None:
        return "plugin"
    return "development" if (PROJECT_ROOT / "pyproject.toml").is_file() else "installed"


def resolve_config_path(cli_value: str | None, environ=None,
                        prefix: str | None = None) -> tuple[str, str]:
    """(path, source) of the config file to use; source is "--config", "HMD_CONFIG", "plugin" or
    "default". See the module docstring for the order."""
    environ = os.environ if environ is None else environ
    if cli_value is not None:
        return cli_value, "--config"
    env = (environ.get(CONFIG_ENV) or "").strip()
    if env:
        return os.path.expanduser(env), CONFIG_ENV
    data_dir = plugin_data_dir(prefix)
    if data_dir is not None:
        return os.path.join(data_dir, bootstrap.CONFIG_NAME), "plugin"
    return CONFIG_PATH, "default"


def plugin_stale_reason(environ=None, prefix: str | None = None) -> str | None:
    """Why the plugin venv this hmd runs from is out of date, or None (IDEA-310). Only judged in
    plugin mode and only when the skill passed the current plugin root as HMD_PLUGIN_ROOT (the
    root changes on every plugin update; the venv under the plugin data dir does not)."""
    environ = os.environ if environ is None else environ
    root = (environ.get(bootstrap.PLUGIN_ROOT_ENV) or "").strip()
    if not root or not os.path.isfile(os.path.join(root, "pyproject.toml")):
        return None
    prefix = prefix if prefix is not None else sys.prefix
    if plugin_data_dir(prefix) is None:
        return None
    return bootstrap.stale_reason(bootstrap.read_marker(prefix), bootstrap.desired_state(root))


def _own_default_config(path: str) -> bool:
    """True when `path` is this checkout's own config/config.yaml."""
    own = PROJECT_ROOT / CONFIG_PATH          # realpath on both sides: a junction or subst drive is the same file
    return os.path.normcase(os.path.realpath(path)) == os.path.normcase(os.path.realpath(own))


def load_config(path: str = CONFIG_PATH, explicit: bool = True) -> Config:
    """`explicit` is False for the implicit default (no --config, HMD_CONFIG or plugin mode): such a
    file from a folder other than this checkout is read without UNTRUSTED_KEYS."""
    if not os.path.exists(path):
        return Config()
    try:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        cfg = Config.model_validate(raw)
        if not explicit and not _own_default_config(path):
            defaults = Config()
            for (section, key), allowed in UNTRUSTED_KEYS.items():
                part = getattr(cfg, section) if section else cfg
                default = getattr(getattr(defaults, section) if section else defaults, key)
                if not allowed(getattr(part, key), default):
                    name = f"{section}.{key}" if section else key
                    print(f"WARNING: ignoring {name} in {path}: the implicit config in the current "
                          f"folder may not set it so; pass --config {path} to trust that file", file=sys.stderr)
                    setattr(part, key, default)
        return cfg
    except ValidationError as e:
        msg = "\n".join(
            f"{path}: {'.'.join(str(p) for p in err['loc'])}: {err['msg']}"
            for err in e.errors()
        )
        raise ConfigError(msg) from e
    except (yaml.YAMLError, OSError) as e:
        raise ConfigError(f"{path}: {e}") from e
