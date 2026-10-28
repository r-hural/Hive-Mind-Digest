"""`hmd init` (IDEA-040): write config.yaml and .env.example from flags. The interactive questions
live in the /hmd:init skill; this command is the non-interactive writer it calls.

Never touches `.env` -- only `.env.example`, and only when it does not already exist. The rendered
config.yaml is written atomically (temp file + os.replace) and round-tripped through load_config
before it replaces any existing file, so a bad render never leaves a half-written or unloadable
config on disk.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

from pydantic import ValidationError

from pipeline.config import Config, ConfigError, load_config
from pipeline.bootstrap import SUGGESTED_HOME as PLUGIN_HOME
from pipeline.exit_codes import EXIT_OK, EXIT_USAGE

DEV_HOME = "data"

ENV_EXAMPLE = """# Copy to .env and fill in. Only needed for llm.provider: anthropic_api
ANTHROPIC_API_KEY=your-anthropic-api-key

# Optional: server-side YouTube captions via Supadata, used only if "supadata" is in transcription.providers
# SUPADATA_API_KEY=
"""


def _q(value) -> str:
    """A YAML double-quoted scalar (json.dumps' escaping is a valid subset for YAML double
    quotes), so backslashes, spaces and colons in Windows home paths or model ids round-trip.
    ensure_ascii=False keeps non-BMP characters (e.g. emoji) as literal UTF-8 instead of \\uXXXX
    surrogate-pair escapes -- PyYAML does not recombine a split surrogate pair back into one
    code point, so the ensure_ascii default would silently corrupt the value on load."""
    return json.dumps(str(value), ensure_ascii=False)


def resolve_home(home: str | None, plugin: bool) -> str:
    """The `home` to write. A development checkout keeps today's default (`data`, relative to the
    working directory). In plugin mode (IDEA-310) hmd runs from any cwd, so home is stored as an
    absolute path, and defaults to ~/HiveMindDigest -- never the plugin data dir, which uninstalling
    the plugin deletes along with the reports."""
    if not plugin:
        return DEV_HOME if home is None else home
    return os.path.abspath(os.path.expanduser(PLUGIN_HOME if home is None or not str(home).strip() else home))


def render_config(args) -> str:
    languages = [x.strip() for x in str(args.languages).split(",") if x.strip()] or ["en"]
    try:
        budget = float(args.budget)
        cfg = Config.model_validate({
            "home": args.home,
            "llm": {"provider": args.provider, "models": {"analyze": args.analyze_model, "merge": args.merge_model}},
            "transcription": {"languages": languages},
            "run": {"budget_usd": budget},
        })
    except ValidationError as e:
        msg = "\n".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}"
            for err in e.errors()
        )
        raise ConfigError(msg) from e
    except ValueError as e:
        raise ConfigError(f"--budget: {e}") from e

    providers = ", ".join(_q(p) for p in cfg.transcription.providers)
    langs = ", ".join(_q(x) for x in cfg.transcription.languages)
    anthropic_rl = cfg.llm.rate_limits["anthropic_api"]
    claude_cli_rl = cfg.llm.rate_limits["claude_cli"]
    show_anchors = "true" if cfg.output.show_anchors else "false"
    formats = ", ".join(_q(f) for f in cfg.output.formats)

    lines = [
        "# Hive Mind Digest configuration. Every key here is read by pipeline/config.py;",
        "# unknown keys are rejected at startup.",
        "#",
        "# Written by `hmd init`; unknown keys are rejected; re-run `hmd init --force` to regenerate.",
        "",
        f"home: {_q(cfg.home)}                      # hmd_home: state, cache and vault live under here",
        "",
        "llm:",
        f"  provider: {_q(cfg.llm.provider)}       "
        "# anthropic_api (API key, per token) | claude_cli (Claude subscription)",
        "  models:",
        f"    analyze: {_q(cfg.llm.models['analyze'])}",
        f"    merge: {_q(cfg.llm.models['merge'])}",
        f"  max_output_tokens: {cfg.llm.max_output_tokens}      "
        "# per call; responses cut at this limit are an error, never silently truncated",
        f"  max_parallel: {cfg.llm.max_parallel}               "
        "# worker threads for transcribe / analyze / merge sections",
        f"  retry_attempts: {cfg.llm.retry_attempts}             "
        "# on 429 / 5xx / overloaded / network errors, exponential backoff",
        f"  retry_base_delay_seconds: {cfg.llm.retry_base_delay_seconds}",
        "  rate_limits:",
        f"    anthropic_api: {{rpm: {anthropic_rl.rpm}, tpm: {anthropic_rl.tpm}}}",
        f"    claude_cli: {{rpm: {claude_cli_rl.rpm}, tpm: {claude_cli_rl.tpm}}}",
        "  batch:                        # Message Batches, anthropic_api only (IDEA-304): half price, slower",
        f"    enabled: {'true' if cfg.llm.batch.enabled else 'false'}              # or per run: --batch / --no-batch",
        f"    min_requests: {cfg.llm.batch.min_requests}             "
        "# a smaller wave (merge outline, verification) goes to the direct API",
        f"    max_wave: {cfg.llm.batch.max_wave}               # requests per batch",
        f"    gather_seconds: {cfg.llm.batch.gather_seconds:g}           # quiet time that closes a wave",
        f"    poll_seconds: {cfg.llm.batch.poll_seconds:g}",
        f"    max_wait_hours: {cfg.llm.batch.max_wait_hours:g}          "
        "# cancel a batch still running after this (the API expires it at 24 h)",
        "",
        "transcription:",
        f"  providers: [{providers}]   # tried in this order",
        "  # optional: \"supadata\" before whisper_local = captions via a third-party API (needs SUPADATA_API_KEY)",
        f"  languages: [{langs}]                                            "
        "# caption languages in preference order",
        "  whisper:",
        f"    model_size: {_q(cfg.transcription.whisper.model_size)}",
        f"    timeout_seconds: {cfg.transcription.whisper.timeout_seconds}       "
        "# per source; a timeout marks the source `error`",
        f"  allow_private_hosts: {str(cfg.transcription.allow_private_hosts).lower()}   "
        "# true lets web/PDF sources come from loopback, private and link-local addresses",
        "",
        "run:",
        f"  budget_usd: {cfg.run.budget_usd}               "
        "# per run; the run pauses when the usage log crosses it; 0 disables",
        "",
        "output:",
        f"  show_anchors: {show_anchors}           # render the verbatim quote under each point",
        f"  formats: [{formats}]  # report files; summary is always written; diff needs a previous merge",
        f"  html: {str(cfg.output.html).lower()}                  # also write an .html twin of every report",
        "",
    ]
    return "\n".join(lines)


def env_example_path_for(config_path: str) -> str:
    """Where `.env.example` belongs for a given config path: next to the config's
    grandparent directory when the config lives in a `config/` subdirectory -- the standard
    `<root>/config/config.yaml` layout -- so `.env.example` lands at the project root next to
    `config/`, not wherever the process happened to be started from; otherwise (no `config/`
    parent) next to the config file itself. The basename comparison is case-insensitive: Windows
    paths are case-preserving but case-insensitive, so `Config/config.yaml` is the same layout."""
    parent = os.path.dirname(os.path.abspath(config_path))
    root = os.path.dirname(parent) if os.path.basename(parent).lower() == "config" else parent
    return os.path.join(root, ".env.example")


def cmd_init(args) -> int:
    path = args.config
    args.home = resolve_home(getattr(args, "home", None), getattr(args, "mode", None) == "plugin")
    if os.path.exists(path) and not args.force:
        print(f"hmd: {path} exists; re-run with --force to overwrite", file=sys.stderr)
        return EXIT_USAGE

    env_example_path = env_example_path_for(path)
    try:
        text = render_config(args)
        parent = os.path.dirname(path) or "."
        os.makedirs(parent, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=parent, prefix=".hmd-init-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(text)
            load_config(tmp_path)  # round-trip check: never replace path with an unloadable file
            os.chmod(tmp_path, 0o644)  # no-op on Windows; keeps the file's POSIX mode conventional
            os.replace(tmp_path, path)
        except BaseException:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            raise

        if not os.path.exists(env_example_path):
            with open(env_example_path, "w", encoding="utf-8") as f:
                f.write(ENV_EXAMPLE)
    except ConfigError as e:
        print(f"hmd: {e}", file=sys.stderr)
        return EXIT_USAGE
    except OSError as e:
        print(f"hmd: {path}: {e}", file=sys.stderr)
        return EXIT_USAGE

    print(f"CONFIG={path}")
    print(f"HOME={os.path.abspath(load_config(path).home)}")
    env_marker = os.path.join(os.path.dirname(env_example_path), ".env")
    if os.path.exists(env_marker):
        print(f"ENV={env_example_path}")
    else:
        print(f"ENV={env_example_path} (copy to .env and add your key if using anthropic_api)")
    return EXIT_OK
