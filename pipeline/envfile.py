"""Where `.env` comes from: next to the config file in use, and this checkout's own project root;
never an ancestor directory.

Shared by the anthropic_api provider (which loads it) and `hmd doctor` (which reports it), so the
path doctor names is the file the provider would load. No LLM/SDK imports here.

Two places, in this order (IDEA-310), each loaded without overriding a variable that is already
set, so the shell environment wins over both and the first file wins over the second:
1. the `.env` in the directory of the config file the CLI resolved (`set_config_path`); in plugin
   mode that is `${CLAUDE_PLUGIN_DATA}/.env`, next to `${CLAUDE_PLUGIN_DATA}/config.yaml`;
2. the `.env` in this checkout's root (a development checkout; an installed package has none).

The config-adjacent `.env` is read only for an explicitly chosen config (`--config`, `HMD_CONFIG`,
plugin mode), never for the implicit `config/config.yaml` under the CWD: a downloaded folder must not
inject variables. A `.env` sets only the keys in DOTENV_KEYS; anything else in it (ANTHROPIC_BASE_URL,
proxy or certificate variables, ...) is ignored, so a `.env` cannot redirect the API key.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import dotenv_values

# The repo root of the checkout this module runs from (pipeline/envfile.py -> root).
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# The only variables a `.env` may set (everything hmd reads from the environment as a credential).
DOTENV_KEYS = ("ANTHROPIC_API_KEY", "SUPADATA_API_KEY")

# Absolute path of the config file the CLI resolved; None until pipeline.cli.main sets it.
_config_path: str | None = None
# False for the implicit default config: its directory is then not trusted for a `.env`.
_config_explicit = True


def set_config_path(path: str | None, explicit: bool = True) -> None:
    """Record the config file in use (pipeline.cli.main), so its directory's `.env` is found.
    `explicit` is False when the path is the implicit default (no --config, HMD_CONFIG or plugin)."""
    global _config_path, _config_explicit
    _config_path = os.path.abspath(path) if path else None
    _config_explicit = explicit


def config_dotenv_path() -> str:
    """The `.env` next to an explicitly chosen config file, or ""."""
    if not _config_path or not _config_explicit:
        return ""
    env = Path(_config_path).parent / ".env"
    return str(env) if env.is_file() else ""


def project_dotenv_path() -> str:
    """The `.env` in this checkout's own root (the directory holding its pyproject.toml), or "".

    Never searches ancestor directories or the CWD: a git worktree nested inside the main checkout
    must not pick up the parent checkout's `.env` (and its billing key).
    """
    if not (PROJECT_ROOT / "pyproject.toml").is_file():
        return ""
    env = PROJECT_ROOT / ".env"
    return str(env) if env.is_file() else ""


def dotenv_paths(config_path: str | None = None, project_path: str | None = None) -> list[str]:
    """The existing `.env` files in load order, without duplicates. Callers that already looked
    the two paths up (doctor) pass them in."""
    first = config_dotenv_path() if config_path is None else config_path
    second = project_dotenv_path() if project_path is None else project_path
    out: list[str] = []
    for p in (first, second):
        if p and all(not _same(p, q) for q in out):
            out.append(p)
    return out


def _same(a: str, b: str) -> bool:
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def load_dotenv_files(paths: list[str]) -> None:
    """Set the DOTENV_KEYS found in these files, never overriding a variable already set."""
    for path in paths:
        for key, value in dotenv_values(path).items():
            if key in DOTENV_KEYS and value is not None and key not in os.environ:
                os.environ[key] = value


def load_project_dotenv() -> str:
    """Load every `.env` of dotenv_paths() (allowlisted keys only, without overriding variables
    already set); return the first path loaded or ""."""
    paths = dotenv_paths()
    load_dotenv_files(paths)
    return paths[0] if paths else ""
