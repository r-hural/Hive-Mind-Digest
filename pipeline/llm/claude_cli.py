"""LLMProvider over headless `claude -p`, billed to the Claude subscription.

Sandbox (DECISION-003 amendment, verified against `claude --help` 2.1.263 on 2026-09-06).
What each flag actually promises (do not overstate these):
- `--tools ""` removes every built-in tool.
- `--restricted` ignores user/project/local settings (`~/.claude/settings.json`,
  `.claude/settings.json`, `.claude/settings.local.json`) — it does NOT ignore
  *managed* (enterprise/organization-policy) settings, which still apply.
- `--safe-mode` disables CLAUDE.md discovery, skills, plugins, hooks, MCP servers, and
  custom slash commands/subagents for the session.
- `--disable-slash-commands` additionally turns off built-in slash command handling.
- `--permission-prompts none` never prompts for permission (nothing to approve since
  tools are off; belt-and-suspenders against a future tool being re-enabled by mistake).
- `--strict-mcp-config` with no `--mcp-config` loads no MCP server.
- `--no-session-persistence` means the session/transcript is not saved to disk.
- the working directory is a fresh empty temp dir, so no CLAUDE.md is discovered there either.
- the prompt travels on stdin (Windows caps argv at 32 KB), argv holds only fixed flags,
  a validated model id and the (validated, length-capped) system prompt.
- every `ANTHROPIC_*`/`CLAUDE_*` env var (except `CLAUDE_CONFIG_DIR`, which holds the OAuth
  credential location and must survive) and `CLAUDECODE` are stripped from the child
  environment — see `_child_env()` — so the subscription's stored OAuth credentials are used
  and no ambient API key or session-identifying variable reaches the child (`SUPADATA_API_KEY`
  is dropped too). `--bare` is NOT used: it disables OAuth.
- the `claude` binary is resolved from PATH's absolute entries, never the current directory
  (`which_on_path()`), so a planted `claude.exe` in the folder hmd runs in is not run.
- Not enforced here: `max_tokens` truncation is delegated entirely to the `claude` CLI, which
  manages its own output limits and continuations — the client's `max_tokens` guard/truncation
  check does not apply to this provider.
- `total_cost_usd` in the CLI's JSON result is the API-equivalent dollar cost of the call, for
  cost accounting/comparison purposes; it is not money actually charged, since usage here is
  billed against the Claude subscription, not metered per token.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time

from pipeline.llm.base import LLMError, LLMFatalError, LLMResult, LLMTransientError

MODEL_RE = re.compile(r"^[a-z0-9][a-z0-9.\-]{0,63}$")
# Other services' keys the claude child never needs (ANTHROPIC_*/CLAUDE_* are dropped by prefix).
OTHER_KEYS = frozenset({"SUPADATA_API_KEY"})

FIXED_ARGS = ["-p", "--output-format", "json", "--tools", "", "--restricted", "--safe-mode",
              "--disable-slash-commands", "--permission-prompts", "none", "--strict-mcp-config",
              "--no-session-persistence"]
TRANSIENT_MARKERS = ("rate limit", "rate_limit", "overloaded", "http 429",
                     "status 429", "529", "capacity")
# Messages of a failed call that will fail every other call of the run too (LLMFatalError): the
# CLI is not logged in or its credential is invalid/expired, or the subscription's usage limit is
# reached (it resets hours later, far beyond the client's backoff). Checked before
# TRANSIENT_MARKERS and only on a failed call (non-zero exit or is_error), never on a model answer.
# Until 2026-10-09 "usage limit" was a transient marker; retrying it only burned five attempts.
FATAL_MARKERS = ("not logged in", "please run /login", "invalid api key", "oauth token has expired",
                 "oauth token revoked", "authentication_error", "credit balance is too low",
                 "usage limit", "hit your limit")
MAX_SYSTEM_PROMPT_CHARS = 8000


def build_argv(binary: str, model: str, system: str | None = None) -> list[str]:
    if not MODEL_RE.match(model):
        raise LLMError(f"refusing to put model id {model!r} on argv")
    argv = [binary, *FIXED_ARGS, "--model", model]
    if system:
        if system.startswith("-"):
            raise LLMError("system prompt must not start with '-' (would be parsed as a flag)")
        if len(system) > MAX_SYSTEM_PROMPT_CHARS:
            raise LLMError(f"system prompt too long for argv ({len(system)} chars > {MAX_SYSTEM_PROMPT_CHARS})")
        argv += ["--system-prompt", system]
    return argv


def _child_env() -> dict[str, str]:
    """The child environment: drop every ANTHROPIC_*/CLAUDE_* var and CLAUDECODE, except
    CLAUDE_CONFIG_DIR (where the OAuth credentials the subscription needs actually live)."""
    env = {}
    for k, v in os.environ.items():
        if k == "CLAUDE_CONFIG_DIR":
            env[k] = v
            continue
        if k == "CLAUDECODE" or k.startswith("ANTHROPIC_") or k.startswith("CLAUDE_") or k in OTHER_KEYS:
            continue
        env[k] = v
    return env


def _looks_transient(text: str) -> bool:
    t = text.lower()
    return any(m in t for m in TRANSIENT_MARKERS)


def _looks_fatal(text: str) -> bool:
    t = text.lower()
    return any(m in t for m in FATAL_MARKERS)


def which_on_path(name: str) -> str | None:
    """shutil.which without the Windows current-directory lookup. Unless
    NoDefaultCurrentDirectoryInExePath is set, Windows (and shutil.which) look in the current
    directory before PATH, so a `claude.exe` planted in the folder hmd runs in would win; that
    hit comes back relative (`.\\claude.EXE`). Then only PATH's absolute entries are searched."""
    found = shutil.which(name)
    if found is None or os.path.isabs(found):
        return found
    if os.path.dirname(name):
        return None
    exts = [""]
    if os.name == "nt":
        pathext = [e for e in os.environ.get("PATHEXT", ".COM;.EXE;.BAT;.CMD").split(os.pathsep) if e]
        if os.path.splitext(name)[1].upper() not in {e.upper() for e in pathext}:
            exts = pathext
    for d in os.environ.get("PATH", "").split(os.pathsep):
        d = d.strip().strip('"')
        if not d or not os.path.isabs(d):
            continue
        for ext in exts:
            p = os.path.join(d, name + ext)
            if os.path.isfile(p) and os.access(p, os.X_OK):
                return p
    return None


def resolve_claude_binary(binary: str = "claude") -> str:
    """Resolve `binary` via PATH and reject an npm/yarn cmd or bat shim (Windows): a shim
    re-parses argv through cmd.exe, which breaks the sandboxed argv contract build_argv()
    constructs. Raises LLMError on a shim or when `binary` is not on PATH; otherwise returns the
    resolved absolute path.
    Shared by ClaudeCliProvider (below) and `hmd doctor` (pipeline/doctor.py) so the guard
    lives in exactly one place."""
    resolved = which_on_path(binary)
    if resolved is None:
        # never fall back to the bare name: Windows' CreateProcess would look in the current
        # directory first, where a planted claude.exe could sit
        raise LLMError(f"{binary} not found on PATH; install Claude Code or use llm.provider: anthropic_api")
    if resolved.lower().endswith((".cmd", ".bat")):
        raise LLMError(
            f"claude resolved to a cmd/bat shim ({resolved}); install the native claude "
            "binary -- a shim re-parses argv through cmd.exe, which breaks the sandboxed argv"
        )
    return resolved


class ClaudeCliProvider:
    name = "claude_cli"

    def __init__(self, binary: str = "claude", timeout_seconds: int = 1800):
        self.binary = resolve_claude_binary(binary)
        self.timeout_seconds = timeout_seconds

    def complete(self, prompt: str, *, model: str, system: str | None = None,
                 max_tokens: int = 16000) -> LLMResult:
        # max_tokens is unused: Claude Code manages its own output limits and continuations.
        argv = build_argv(self.binary, model, system)
        env = _child_env()
        t0 = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="hmd-cli-", ignore_cleanup_errors=True) as cwd:
            try:
                proc = subprocess.run(argv, input=prompt, capture_output=True, text=True, encoding="utf-8",
                                      errors="replace", cwd=cwd, env=env, shell=False,
                                      timeout=self.timeout_seconds)
            except FileNotFoundError as e:
                raise LLMError(f"claude binary not found: {self.binary}; run `hmd doctor`") from e
            except subprocess.TimeoutExpired as e:
                raise LLMTransientError(f"claude -p exceeded {self.timeout_seconds}s") from e
        duration_ms = int((time.monotonic() - t0) * 1000)

        if proc.returncode != 0:
            msg = (proc.stderr or proc.stdout or "").strip()[:500]
            if _looks_fatal(msg):
                raise LLMFatalError(f"claude exited {proc.returncode}: {msg}")
            if _looks_transient(msg):
                raise LLMTransientError(f"claude exited {proc.returncode}: {msg}")
            raise LLMError(f"claude exited {proc.returncode}: {msg}")
        try:
            data = json.loads(proc.stdout)
        except json.JSONDecodeError as e:
            stderr_snippet = (proc.stderr or "")[:200]
            raise LLMError(
                f"claude returned non-JSON output: {proc.stdout[:200]!r}; stderr: {stderr_snippet!r}"
            ) from e
        if not isinstance(data, dict):
            raise LLMError(
                f"claude returned a JSON {type(data).__name__}, expected an object: {json.dumps(data)[:200]}"
            )
        result_field = data.get("result", "")
        result_text = result_field if isinstance(result_field, str) else json.dumps(result_field)
        if data.get("is_error"):
            if _looks_fatal(result_text):
                raise LLMFatalError(f"claude error: {result_text[:500]}")
            if _looks_transient(result_text):
                raise LLMTransientError(result_text[:500])
            raise LLMError(f"claude error: {result_text[:500]}")
        subtype = data.get("subtype")
        if subtype is not None and subtype != "success":
            raise LLMError(f"claude result subtype={subtype!r}: {result_text[:500]}")
        usage = data.get("usage") or {}
        cost = data.get("total_cost_usd")
        return LLMResult(
            text=result_text, model=model, provider=self.name,
            input_tokens=int(usage.get("input_tokens", 0) or 0),
            output_tokens=int(usage.get("output_tokens", 0) or 0),
            cache_read_tokens=int(usage.get("cache_read_input_tokens", 0) or 0),
            cache_write_tokens=int(usage.get("cache_creation_input_tokens", 0) or 0),
            cost_usd=float(cost) if cost is not None else None,
            stop_reason="end_turn", duration_ms=duration_ms,
        )
