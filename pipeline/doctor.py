"""`hmd doctor` (IDEA-041): environment checks, effective configuration, egress per module,
Whisper GPU report (IDEA-271), Whisper model pre-warm. Never makes an LLM call: no provider is
instantiated and no LLM request is sent (see test_run_checks_never_makes_an_llm_call).

Every check becomes a `Check` dataclass entry; a missing tool never raises here, it becomes
a failed or warned check instead. `format_report` renders the list and decides the process
exit code: 2 if any *required* check failed, 0 otherwise.

Requiredness rule for ffmpeg/whisper/whisper_model: `required=True` means `hmd run` cannot work
at all without it, which is only the case when `whisper_local` is the *sole* configured
transcription provider. When it is one of several fallback providers (the default), captions or
ytdlp_subs can still carry the run, so a missing ffmpeg/whisper is only a `[warn]`. The same
rule applies to SUPADATA_API_KEY when `supadata` is configured (DECISION-026); doctor never calls
that API.
"""
from __future__ import annotations

import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from dataclasses import dataclass

from dotenv import dotenv_values, load_dotenv

from pipeline import db, models_monitor
from pipeline.config import CONFIG_ENV, Config, plugin_data_dir, plugin_stale_reason, run_mode
from pipeline.envfile import config_dotenv_path, dotenv_paths, load_dotenv_files
from pipeline.envfile import project_dotenv_path
from pipeline.exit_codes import EXIT_ERROR, EXIT_OK
from pipeline.llm.base import LLMError
from pipeline.llm.claude_cli import resolve_claude_binary, which_on_path
from pipeline.pricing import price_for


@dataclass
class Check:
    name: str
    ok: bool
    required: bool
    detail: str


def gpu_report() -> tuple[bool, str]:
    """ok is False only when `import whisper` itself fails -- every branch reached after a
    successful import returns True, so callers can read gpu_report()[0] as "whisper is
    importable" without importing it a second time (see _check_transcription)."""
    try:
        import whisper  # noqa: F401
    except ImportError:
        return False, "openai-whisper not installed (pip install -r requirements-whisper.txt -c constraints-whisper.txt); Whisper fallback unavailable"
    try:
        import torch
        if torch.cuda.is_available():
            return True, f"GPU {torch.cuda.get_device_name(0)}: Whisper base runs ~10x faster than real time"
        return True, "CPU only: Whisper base runs near real time; a 1 h video takes ~1 h (timeout applies)"
    except Exception as e:  # torch import can fail in odd ways
        return True, f"whisper importable; GPU status unknown ({e})"


def _whisper_cache_dir() -> str:
    """Mirrors openai-whisper's own default cache location, including its os.getenv() quirk: an
    XDG_CACHE_HOME set to an empty string is honoured verbatim rather than treated as unset. This
    is so the model-file check looks in the same place `whisper.load_model()` would actually use."""
    default = os.path.join(os.path.expanduser("~"), ".cache")
    base = os.getenv("XDG_CACHE_HOME", default)
    return os.path.join(base, "whisper")


def _whisper_model_filename(size: str) -> str:
    """The cache filename whisper.load_model(size) would use. openai-whisper's own _MODELS dict
    maps a size like "large" to a versioned URL (.../large-v3.pt), so a plain f"{size}.pt" guess
    is wrong for "large"/"turbo"; read the real mapping when whisper is importable and fall back
    to the naive guess only if that mapping is missing or shaped differently than expected."""
    try:
        import whisper
        return os.path.basename(whisper._MODELS[size])
    except Exception:
        return f"{size}.pt"


def prewarm_whisper(cfg: Config, log=print) -> bool:
    """Download (or confirm cached) the configured Whisper model. Does not transcribe anything
    and never allocates a GPU (device="cpu") -- this call exists purely to fetch the model file."""
    try:
        import whisper
    except ImportError:
        log("openai-whisper not installed; run: pip install -r requirements-whisper.txt -c constraints-whisper.txt")
        return False
    size = cfg.transcription.whisper.model_size
    cache_dir = _whisper_cache_dir()
    log(f"downloading Whisper model {size!r} into {cache_dir} (this can take a while on first run)...")
    try:
        whisper.load_model(size, device="cpu", download_root=cache_dir)
    except Exception as e:
        log(f"failed to prewarm Whisper model {size!r}: {e}")
        return False
    log(f"Whisper model {size!r} is cached and ready.")
    return True


def _nearest_existing_ancestor(path: str) -> str:
    """Walk up from `path` until an existing directory is found. Used to writability-probe a
    directory that `hmd run` will create later, without run_checks itself creating anything."""
    path = os.path.abspath(path)
    while not os.path.isdir(path):
        parent = os.path.dirname(path)
        if parent == path:  # reached the filesystem root without finding an existing directory
            break
        path = parent
    return path


def _probe_writable(dir_path: str) -> None:
    """Raises OSError if dir_path is not writable. A real write-and-delete probe, not
    os.access(W_OK) -- os.access(W_OK) is unreliable on Windows (it can report a directory
    writable when the actual create/write later fails, e.g. under some ACL configurations)."""
    with tempfile.NamedTemporaryFile(dir=dir_path, delete=True):
        pass


_SOURCE_LABELS = {"--config": "from --config", CONFIG_ENV: f"from {CONFIG_ENV}",
                  "plugin": "plugin data dir", "default": "default"}


def _check_location(cfg: Config, ctx: dict) -> list[Check]:
    """Printed first by cmd_doctor; ctx holds mode, config_path, config_source, config_missing and
    stale from the CLI. IDEA-310: which install runs (development checkout or the /hmd:init plugin venv), which config
    file is in use and where home (state, caches, reports) is."""
    mode = ctx.get("mode") or run_mode()
    if mode == "plugin":
        mode_detail = f"plugin venv ({sys.prefix})"
    elif mode == "development":
        mode_detail = f"development checkout ({os.path.dirname(os.path.dirname(os.path.abspath(__file__)))})"
    else:
        mode_detail = f"installed package ({sys.prefix})"
    checks = [Check("mode", True, False, mode_detail)]

    stale = ctx.get("stale")
    if stale:
        checks.append(Check("plugin_env", False, True, f"out of date ({stale}); run /hmd:init to update it"))

    path, source = ctx.get("config_path"), ctx.get("config_source")
    if path:
        abs_path = os.path.abspath(path)
        label = _SOURCE_LABELS.get(source, source or "default")
        if ctx.get("config_missing"):
            checks.append(Check("config_file", False, True,
                                f"{abs_path} ({label}) not found; run hmd init (/hmd:init) first"))
        elif os.path.exists(path):
            checks.append(Check("config_file", True, True, f"{abs_path} ({label})"))
        else:
            checks.append(Check("config_file", True, False,
                                f"{abs_path} not found; using built-in defaults"))

    home = os.path.abspath(cfg.home)
    data_dir = plugin_data_dir() if mode == "plugin" else None
    if ctx.get("config_missing"):
        checks.append(Check("home", True, False, "not set yet (no config file; hmd init writes it)"))
    elif data_dir and (os.path.normcase(home) + os.sep).startswith(os.path.normcase(data_dir) + os.sep):
        checks.append(Check("home", False, False,
                            f"{home} is inside the plugin data dir, which uninstalling the plugin "
                            f"deletes; set home in the config to a folder of your own"))
    else:
        checks.append(Check("home", True, False, home))
    return checks


def _check_basics(cfg: Config) -> list[Check]:
    provider = cfg.llm.provider
    return [
        Check("python", sys.version_info >= (3, 12), True, sys.version.split()[0]),
        Check("config", True, True,
             f"provider={provider} analyze={cfg.llm.models['analyze']} merge={cfg.llm.models['merge']} "
             f"budget=${cfg.run.budget_usd} max_parallel={cfg.llm.max_parallel}"),
    ]


def _check_llm(cfg: Config, dotenv_files: list[str], had_key: bool) -> list[Check]:
    checks: list[Check] = []
    provider = cfg.llm.provider

    key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    if key:
        if had_key:
            # The key was already in os.environ before run_checks ever called load_dotenv, which
            # never overrides an existing environment value -- crediting a .env file here would
            # be misleading even if that file happens to name the same key.
            detail = "set (from environment)"
        else:
            # Only credit a .env file loaded by run_checks's own load_dotenv() calls -- naming a
            # path that was never actually loaded (e.g. a stale cwd-based guess) would be
            # misleading.
            detail = _key_source("ANTHROPIC_API_KEY", dotenv_files, had_key)
    else:
        detail = "ANTHROPIC_API_KEY missing (.env)"
    checks.append(Check("anthropic_api_key", key, provider == "anthropic_api", detail))

    checks.append(_check_claude_binary(provider))

    missing = [m for m in ("analyze", "merge") if price_for(cfg.llm.models[m]) is None]
    checks.append(Check(
        "model_prices", not missing, provider == "anthropic_api" and cfg.run.budget_usd > 0,
        "known" if not missing else "no price for " + ", ".join(cfg.llm.models[m] for m in missing)))
    return checks


def _check_claude_binary(provider: str) -> Check:
    required = provider == "claude_cli"
    found = which_on_path("claude")
    if not found:
        return Check("claude_binary", False, required,
                     "claude binary not on PATH (needed for llm.provider: claude_cli)")
    try:
        path = resolve_claude_binary("claude")
    except LLMError as e:
        return Check("claude_binary", False, required, str(e))
    try:
        out = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=30,
                             stdin=subprocess.DEVNULL)
        detail = f"{path}: {out.stdout.strip() or out.stderr.strip()}"
        ok = out.returncode == 0
    except Exception as e:
        ok, detail = False, f"{path}: {e}"
    return Check("claude_binary", ok, required, detail)


def _check_transcription(cfg: Config) -> list[Check]:
    checks: list[Check] = []

    try:
        import yt_dlp
        checks.append(Check("yt_dlp", True, True, yt_dlp.version.__version__))
    except ImportError:
        checks.append(Check("yt_dlp", False, True, "not installed"))

    providers = cfg.transcription.providers
    whisper_configured = "whisper_local" in providers
    whisper_sole = whisper_configured and len(providers) == 1

    ff = shutil.which("ffmpeg")
    if ff:
        checks.append(Check("ffmpeg", True, whisper_sole, ff))
    elif whisper_sole:
        checks.append(Check("ffmpeg", False, True, "not on PATH (Whisper fallback needs it)"))
    elif whisper_configured:
        checks.append(Check("ffmpeg", False, False,
                            "not on PATH; videos without captions will fail transcription"))
    else:
        checks.append(Check("ffmpeg", False, False, "not on PATH (not needed: whisper_local not configured)"))

    whisper_ok, whisper_detail = gpu_report()
    whisper_importable = whisper_ok  # see gpu_report's docstring
    if whisper_ok:
        checks.append(Check("whisper", True, whisper_sole, whisper_detail))
    elif whisper_sole:
        checks.append(Check("whisper", False, True, whisper_detail))
    elif whisper_configured:
        checks.append(Check("whisper", False, False,
                            whisper_detail + "; videos without captions will fail transcription"))
    else:
        checks.append(Check("whisper", False, False, whisper_detail))

    if not whisper_configured:
        checks.append(Check("local_media", False, False, "local audio/video needs whisper_local and ffmpeg"))

    if whisper_importable:
        model_size = cfg.transcription.whisper.model_size
        model_path = os.path.join(_whisper_cache_dir(), _whisper_model_filename(model_size))
        cached = os.path.exists(model_path)
        checks.append(Check(
            "whisper_model", cached, whisper_sole and not cached,
            f"{model_path} cached" if cached
            else f"{model_path} not cached yet; run: hmd doctor --prewarm-whisper"))

    return checks


def _key_source(name: str, dotenv_files: list[str], had_key: bool) -> str:
    """Where a set key came from, never its value (see _check_llm for the provenance rules): the
    first loaded .env that names it, since load_dotenv never overrides an earlier value."""
    if not had_key:
        for path in dotenv_files:
            if name in dotenv_values(path):
                return f"set (from {path})"
    return "set (from environment)"


def _check_supadata(cfg: Config, dotenv_files: list[str], had_key: bool) -> list[Check]:
    """SUPADATA_API_KEY, only when `supadata` is configured (DECISION-026). Same requiredness rule
    as Whisper: a [FAIL] only when supadata is the sole provider, otherwise a [warn] -- the chain
    just falls through to the next provider. Never calls the API."""
    providers = cfg.transcription.providers
    if "supadata" not in providers:
        return []
    sole = len(providers) == 1
    if os.environ.get("SUPADATA_API_KEY", "").strip():
        return [Check("supadata_api_key", True, sole, _key_source("SUPADATA_API_KEY", dotenv_files, had_key))]
    detail = "SUPADATA_API_KEY missing (.env)"
    if not sole:
        detail += "; the supadata provider will be skipped"
    return [Check("supadata_api_key", False, sole, detail)]


def _check_state(cfg: Config) -> list[Check]:
    return [_probe_state_db(cfg.db_path), _probe_vault_dir(cfg.vault_dir)]


def _probe_state_db(db_path: str) -> Check:
    try:
        if os.path.exists(db_path):
            conn = sqlite3.connect(db_path)
            try:
                reason = db.legacy_reason(conn)
                if reason is not None:
                    return Check("state_db", False, True, f"{db_path} {reason}")
                pending = db.pending_migration(conn)
                tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            finally:
                conn.close()
            if pending is not None:
                return Check("state_db", True, True, f"migrates {pending} on next run (backup kept)")
            return Check("state_db", True, True, f"{db_path} ({len(tables)} tables)")
        parent = _nearest_existing_ancestor(os.path.dirname(db_path) or ".")
        _probe_writable(parent)
        return Check("state_db", True, True, f"{db_path} will be created")
    except (OSError, sqlite3.Error) as e:
        return Check("state_db", False, True, f"{db_path}: {e}")


def _probe_vault_dir(vault_dir: str) -> Check:
    try:
        if os.path.isdir(vault_dir):
            _probe_writable(vault_dir)
            return Check("vault_dir", True, True, vault_dir)
        parent = _nearest_existing_ancestor(os.path.dirname(vault_dir) or ".")
        _probe_writable(parent)
        return Check("vault_dir", True, True, f"{vault_dir} will be created")
    except OSError as e:
        return Check("vault_dir", False, True, f"{vault_dir}: {e}")


def _check_egress(cfg: Config) -> list[Check]:
    # informational summary of every network endpoint a run might touch.
    egress_lines = ["fetch: youtube.com (yt-dlp metadata)"]
    egress_lines.append("transcribe: web_text -> the page's own host (web sources only)")
    for p in cfg.transcription.providers:
        if p == "youtube_captions":
            egress_lines.append("transcribe: youtube_captions -> youtube.com (captions)")
        elif p == "ytdlp_subs":
            egress_lines.append("transcribe: ytdlp_subs -> youtube.com (yt-dlp subtitles)")
        elif p == "supadata":
            egress_lines.append("transcribe: supadata -> api.supadata.ai (video ids sent to a third party)")
        elif p == "whisper_local":
            egress_lines.append("transcribe: whisper_local -> youtube.com (audio download), openaipublic.azureedge.net "
                                "(model download, first use); whisper runs locally")
    if cfg.llm.provider == "anthropic_api":
        egress_lines.append("analyze/merge: anthropic_api -> api.anthropic.com")
    else:
        egress_lines.append("analyze/merge: claude_cli -> claude binary (Anthropic)")
    egress_lines.append("doctor model check: platform.claude.com (public docs pages)"
                        + ("; api.anthropic.com Models API (free)" if cfg.llm.provider == "anthropic_api" else ""))
    return [Check("egress", True, False, "; ".join(egress_lines))]


def run_checks(cfg: Config) -> list[Check]:
    had_key = "ANTHROPIC_API_KEY" in os.environ
    had_supadata_key = "SUPADATA_API_KEY" in os.environ
    # The same .env files, in the same order, that the anthropic_api/supadata providers load.
    dotenv_files = dotenv_paths(config_path=config_dotenv_path(), project_path=project_dotenv_path())
    load_dotenv_files(dotenv_files)
    checks: list[Check] = []
    checks += _check_basics(cfg)
    checks += _check_llm(cfg, dotenv_files, had_key)
    checks += _check_transcription(cfg)
    checks += _check_supadata(cfg, dotenv_files, had_supadata_key)
    checks += _check_state(cfg)
    checks += _check_egress(cfg)
    return checks


def format_report(checks: list[Check]) -> tuple[str, int]:
    lines = []
    exit_code = EXIT_OK
    for c in checks:
        if c.ok:
            tag = "[ok]  "
        elif c.required:
            tag = "[FAIL]"
            exit_code = EXIT_ERROR
        else:
            tag = "[warn]"
        lines.append(f"{tag} {c.name:<18} {c.detail}")
    return "\n".join(lines), exit_code


def _model_section(cfg: Config, args) -> list[str]:
    """The model monitor (IDEA-302): never changes the exit code, never bills. Refreshes the cache that
    run commands read. run_checks has already loaded .env, so the key is in os.environ when set."""
    if args is not None and getattr(args, "no_model_check", False):
        return ["Models: not checked (--no-model-check)"]
    key = os.environ.get("ANTHROPIC_API_KEY") if cfg.llm.provider == "anthropic_api" else None
    findings = models_monitor.check_models(cfg, api_key=key or None)
    try:
        models_monitor.write_cache(cfg, findings)
    except OSError:
        pass
    return ["Models:"] + ["  " + line for line in models_monitor.format_findings(findings)]


def cmd_doctor(cfg: Config, args=None) -> int:
    if args is not None and getattr(args, "prewarm_whisper", False):
        if not prewarm_whisper(cfg):
            print("prewarm failed; see the whisper_model check below for the current cache state")
    ctx = {k: getattr(args, k, None) for k in ("mode", "config_path", "config_source", "config_missing")}
    if args is not None and getattr(args, "config_path", None):
        ctx["stale"] = plugin_stale_reason()
    text, code = format_report(_check_location(cfg, ctx) + run_checks(cfg))
    print(text)
    for line in _model_section(cfg, args):
        print(line)
    print("DOCTOR=" + ("ok" if code == 0 else "fail"))
    return code
