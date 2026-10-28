"""Model monitor (IDEA-302, cost only): tells the owner when a configured model has a newer model on the
same line (cheaper or equal -> suggest, dearer -> warn), when it is deprecated, retiring or unpriced, and
when the bundled price table (`pipeline/pricing.py`) differs from Anthropic's published price list.

Sources, none of which needs an API key (the hybrid price source, owner decision 2026-10-15): three public
markdown pages on platform.claude.com -- the models overview (current and legacy model ids), the price
list and the model status table. With `ANTHROPIC_API_KEY` set, the Models API (free, no LLM call) also
says whether the account can use a candidate. On `claude_cli` (subscription) prices do not apply: only
newer models and retirements are reported. Nothing here bills, switches a model or edits the price table;
a person applies changes (`hmd models use <stage> <model>`, an edit of `pipeline/pricing.py`).

Every network or page-format failure becomes a "skipped" finding, never an exception. `hmd doctor`
refreshes the cache (`<hmd_home>/cache/model-monitor.json`); run commands only read it.
"""
from __future__ import annotations

import contextlib
import http.client
import json
import re
import sqlite3
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable
from urllib.parse import urlsplit

from pipeline.pricing import (
    PRICES, TOKENIZER_FACTOR, Price, model_line, model_version, new_tokenizer, price_for,
    prices_age_days,
)

PRICING_URL = "https://platform.claude.com/docs/en/about-claude/pricing.md"
DEPRECATIONS_URL = "https://platform.claude.com/docs/en/about-claude/model-deprecations.md"
OVERVIEW_URL = "https://platform.claude.com/docs/en/models/overview.md"
FETCH_TIMEOUT = 10
RETIRING_SOON_DAYS = 90
PRICE_TABLE_MAX_AGE_DAYS = 60
CACHE_NAME = "model-monitor.json"


class PageFormatError(ValueError):
    """A published page no longer has the table this module reads."""


@dataclass(frozen=True)
class ModelStatus:
    state: str                     # Active | Legacy | Deprecated | Retired
    deprecated_on: date | None
    retires_on: date | None
    retire_not_before: bool        # "Not sooner than <date>": a commitment, not a scheduled retirement


@dataclass(frozen=True)
class Finding:
    level: str                     # ok | suggest | warn | note | skipped
    stage: str | None
    text: str


# --- parsers -------------------------------------------------------------------------------------

_MONEY = re.compile(r"\$\s*([0-9]+(?:\.[0-9]+)?)\s*/\s*MTok")
_NAME = re.compile(r"Claude ([A-Z][a-z]+) (\d+)(?:\.(\d+))?\b")
_TIER = re.compile(r"over ([0-9,]+) tokens")


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _section(md: str, heading: str) -> str:
    start = md.find(heading)
    if start < 0:
        return md
    end = md.find("\n## ", start + len(heading))
    return md[start:end if end >= 0 else len(md)]


def _money(cell: str) -> float | None:
    m = _MONEY.search(cell)
    return float(m.group(1)) if m else None


def _name_to_id(name: str) -> str | None:
    m = _NAME.search(name)
    if not m:
        return None
    return "claude-" + m.group(1).lower() + "-" + m.group(2) + (f"-{m.group(3)}" if m.group(3) else "")


def parse_price_list(md: str) -> dict[str, Price]:
    """The "Model pricing" table: model id -> Price (base input, output, cache-hit multiplier). A row for
    prompts over a token threshold (Claude Haiku 5.5) becomes the base row's `long_prompt` tier. Retired rows
    are skipped."""
    base: dict[str, Price] = {}
    tiers: dict[str, tuple[int, Price]] = {}
    for line in _section(md, "## Model pricing").splitlines():
        if not line.startswith("| Claude"):
            continue
        cells = _cells(line)
        if len(cells) < 6 or "retired" in cells[0].lower():
            continue
        model = _name_to_id(cells[0])
        inp, hit, out = _money(cells[1]), _money(cells[4]), _money(cells[5])
        if model is None or inp is None or out is None or hit is None or inp == 0:
            continue
        price = Price(inp, out, cache_read_x=round(hit / inp, 4))
        over = _TIER.search(cells[0])
        if over:
            tiers[model] = (int(over.group(1).replace(",", "")), price)
        else:
            base[model] = price
    for model, (threshold, price) in tiers.items():
        if model in base:
            b = base[model]
            base[model] = Price(b.input, b.output, b.cache_read_x, long_prompt_tokens=threshold, long_prompt=price)
    if not base:
        raise PageFormatError("no model price rows found")
    return base


def _date(cell: str) -> date | None:
    m = re.search(r"([A-Z][a-z]+ \d{1,2}, \d{4})", cell)
    return datetime.strptime(m.group(1), "%B %d, %Y").date() if m else None


def parse_deprecations(md: str) -> dict[str, ModelStatus]:
    """The "Model status" table: API model id -> ModelStatus."""
    out: dict[str, ModelStatus] = {}
    for line in md.splitlines():
        if not line.startswith("| claude-"):
            continue
        cells = _cells(line)
        if len(cells) < 4:
            continue
        out[cells[0]] = ModelStatus(cells[1], _date(cells[2]), _date(cells[3]),
                                    "not sooner than" in cells[3].lower())
    if not out:
        raise PageFormatError("no model status rows found")
    return out


def parse_overview(md: str) -> list[str]:
    """Current model ids (the "Claude API ID" row), then the legacy models still available."""
    ids: list[str] = []
    for line in md.splitlines():
        if line.startswith("| Claude API ID"):
            ids += re.findall(r"`(claude-[a-z0-9-]+)`", line)
        elif line.startswith("Legacy models"):
            ids += ["claude-" + slug for slug in re.findall(r"/models/([a-z0-9-]+)/overview", line)]
    if not ids:
        raise PageFormatError("no model id row found")
    return list(dict.fromkeys(ids))


# --- findings ------------------------------------------------------------------------------------

def _lookup(table: dict, model: str):
    """Exact key, else a key the model extends by a date snapshot or that extends the model (a dated
    status row for a dateless configured id)."""
    if model in table:
        return table[model]
    for key, value in table.items():
        rest = model[len(key) + 1:] if model.startswith(key + "-") else None
        if rest is not None and re.fullmatch(r"\d{8}", rest):
            return value
        if key.startswith(model + "-") and re.fullmatch(r"\d{8}", key[len(model) + 1:]):
            return value
    return None


def _status_findings(stage: str, model: str, st: ModelStatus | None, today: date) -> list[Finding]:
    if st is None:
        return []
    if st.state.lower() == "retired":
        return [Finding("warn", stage, f"{model} is retired; requests to it fail. Switch this stage to a current model.")]
    if st.state.lower() == "deprecated":
        when = f", retires on {st.retires_on.isoformat()}" if st.retires_on else ""
        return [Finding("warn", stage, f"{model} is deprecated{when}. Switch this stage to a current model.")]
    if st.retires_on and not st.retire_not_before and (st.retires_on - today).days <= RETIRING_SOON_DAYS:
        return [Finding("warn", stage, f"{model} retires on {st.retires_on.isoformat()}.")]
    return []


def _candidate(model: str, listed: list[str]) -> str | None:
    line, version = model_line(model), model_version(model)
    if line is None or version is None:
        return None
    newer = [m for m in listed if model_line(m) == line and (model_version(m) or (0, 0)) > version]
    return max(newer, key=lambda m: model_version(m)) if newer else None


def _estimate(p: Price, input_tokens: float, output_tokens: float, calls: float) -> float:
    """Cost of a notebook's tokens; a prompt-length tier is chosen by the mean prompt per request."""
    if p.long_prompt and input_tokens / max(calls, 1.0) > (p.long_prompt_tokens or 0):
        p = p.long_prompt
    return (input_tokens * p.input + output_tokens * p.output) / 1_000_000


def _cost_finding(stage: str, model: str, cand: str, usage: tuple[float, float] | None,
                  prices: dict[str, Price]) -> Finding:
    try_it = f" Test it on one golden notebook first (--{stage}-model {cand}), then `hmd models use {stage} {cand}`."
    cur, new = _lookup(prices, model), _lookup(prices, cand)
    if new is None:
        return Finding("warn", stage, f"newer model {cand} has no price in pipeline/pricing.py; add it from the "
                                      f"published price list before a budgeted run.")
    if cur is None:
        return Finding("suggest", stage, f"newer model {cand} available (${new.input:g}/${new.output:g} per MTok); "
                                         f"{model} has no price to compare." + try_it)
    scale = TOKENIZER_FACTOR if new_tokenizer(cand) and not new_tokenizer(model) else 1.0
    note = f", counting ~{round((TOKENIZER_FACTOR - 1) * 100)}% more tokens on the newer tokenizer" if scale > 1 else ""
    if usage is None:
        cheaper = new.input * scale <= cur.input and new.output * scale <= cur.output
        return Finding("suggest" if cheaper else "warn", stage,
                       f"newer model {cand}: ${new.input:g}/${new.output:g} per MTok vs ${cur.input:g}/${cur.output:g} "
                       f"for {model} (no usage history for a per-notebook estimate{note})." + try_it)
    calls = usage[2] if len(usage) > 2 else 1.0
    now_cost = _estimate(cur, usage[0], usage[1], calls)
    new_cost = _estimate(new, usage[0] * scale, usage[1] * scale, calls)
    pct = (new_cost / now_cost - 1) * 100 if now_cost else 0.0
    level, word = ("suggest", "less") if new_cost <= now_cost else ("warn", "more")
    return Finding(level, stage,
                   f"newer model {cand} would cost about ${new_cost:.2f} per notebook vs ${now_cost:.2f} now with "
                   f"{model} ({abs(pct):.0f}% {word}{note}; thinking tokens not included)." + try_it)


def evaluate(models: dict[str, str], provider: str, prices: dict[str, Price],
             published: dict[str, Price] | None, statuses: dict[str, ModelStatus] | None,
             listed: list[str] | None, account: set[str] | None,
             usage: dict[str, tuple[float, float] | None], today: date) -> list[Finding]:
    api = provider == "anthropic_api"
    findings: list[Finding] = []
    candidates: dict[str, str] = {}
    for stage, model in models.items():
        if statuses is not None:
            findings += _status_findings(stage, model, _lookup(statuses, model), today)
        if api and price_for(model) is None and _lookup(prices, model) is None:
            findings.append(Finding("warn", stage, f"{model} has no price in pipeline/pricing.py; a budgeted run refuses it."))
        cand = _candidate(model, listed) if listed is not None else None
        if cand is None:
            continue
        candidates[stage] = cand
        if account is not None and cand not in account and not any(a.startswith(cand + "-") for a in account):
            findings.append(Finding("warn", stage, f"newer model {cand} is not available to this account."))
        elif api:
            findings.append(_cost_finding(stage, model, cand, usage.get(stage), prices))
        else:
            findings.append(Finding("suggest", stage,
                                    f"newer model available: {cand} (now {model}). Test it on one golden notebook "
                                    f"first (--{stage}-model {cand}), then `hmd models use {stage} {cand}`."))
    if api:
        if published is not None:
            for model in dict.fromkeys(list(models.values()) + list(candidates.values())):
                pub, table = _lookup(published, model), _lookup(prices, model)
                if pub is not None and table is not None and pub != table:
                    findings.append(Finding("warn", None, f"price list differs for {model}: published "
                                            f"${pub.input:g}/${pub.output:g} per MTok (cache x{pub.cache_read_x:g}), "
                                            f"table ${table.input:g}/${table.output:g} (cache x{table.cache_read_x:g}); "
                                            f"update pipeline/pricing.py."))
                elif pub is None:
                    findings.append(Finding("note", None, f"{model} is not in the published price list; its "
                                            f"table price could not be compared."))
                elif table is None:
                    findings.append(Finding("warn", None, f"price list has {model} (${pub.input:g}/${pub.output:g} "
                                            f"per MTok), the table does not; add it to pipeline/pricing.py."))
        age = prices_age_days(today)
        if age > PRICE_TABLE_MAX_AGE_DAYS:
            findings.append(Finding("warn", None, f"the price table is {age} days old (checked on "
                                    f"{(today - timedelta(days=age)).isoformat()}); compare it with {PRICING_URL}."))
    unread = [name for name, value in (("price list", published if api else {}), ("model status", statuses),
                                       ("models overview", listed)) if value is None]
    if unread:
        findings.append(Finding("skipped", None, "could not read: " + ", ".join(unread) + "."))
    elif not any(f.level in ("suggest", "warn") for f in findings):
        unmatched = any("not in the published price list" in f.text for f in findings)
        findings.append(Finding("ok", None, "configured models are current; prices match the published list."
                                if api and not unmatched else "configured models are current."))
    return findings


# --- inputs: pages, account, usage ---------------------------------------------------------------

MAX_PAGE_BYTES = 8 * 1024 * 1024     # a docs page is far smaller; more is refused, not read


# The docs pages may move between Anthropic's docs hosts; a redirect anywhere else is refused.
DOCS_HOSTS = frozenset({"platform.claude.com", "docs.claude.com", "docs.anthropic.com"})


class _SameHostRedirect(urllib.request.HTTPRedirectHandler):
    """Follow a redirect only to https on the same host or another of DOCS_HOSTS."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        old, new = urlsplit(req.full_url), urlsplit(newurl)
        host = (new.hostname or "").lower()
        if new.scheme != "https" or (host != (old.hostname or "").lower() and host not in DOCS_HOSTS):
            raise urllib.error.HTTPError(newurl, code, f"refusing a redirect off {old.hostname}: {newurl}",
                                         headers, fp)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_opener = urllib.request.build_opener(_SameHostRedirect)


def _open(req, timeout):
    return _opener.open(req, timeout=timeout)


def fetch_text(url: str, timeout: float = FETCH_TIMEOUT) -> str:
    """One public docs page as text: https only, redirects within the same host, at most
    MAX_PAGE_BYTES (release security review, 2026-10-28). OSError on any refusal."""
    if urlsplit(url).scheme != "https":
        raise OSError(f"models page must be https: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "hmd-model-monitor"})
    with _open(req, timeout) as resp:
        body = resp.read(MAX_PAGE_BYTES + 1)
    if len(body) > MAX_PAGE_BYTES:
        raise OSError(f"models page is larger than {MAX_PAGE_BYTES} bytes: {url}")
    return body.decode("utf-8")


def list_account_models(api_key: str) -> set[str]:
    """Model ids the account can use (Models API: free, no LLM call)."""
    import anthropic
    client = anthropic.Anthropic(api_key=api_key, timeout=FETCH_TIMEOUT, max_retries=0)
    return {m.id for m in client.models.list(limit=100)}


def stage_usage(db_path: str, stage: str, runs: int = 5) -> tuple[float, float, float] | None:
    """Mean input (incl. cache), output tokens and calls per API run for a stage ("analyze", or "merge" =
    every merge:* step) over the last `runs` runs that used it; None without history. claude_cli runs are
    left out: their token counts are unreliable (IDEA-292). Batched calls (anthropic_batch) count: the same
    tokens, at half the price."""
    pattern = "merge:%" if stage == "merge" else stage
    try:
        with contextlib.closing(sqlite3.connect(db_path)) as conn:
            rows = conn.execute(
                "SELECT SUM(input_tokens + cache_read_tokens + cache_write_tokens), SUM(output_tokens), COUNT(*) "
                "FROM usage_events WHERE stage LIKE ? AND run_id IS NOT NULL AND provider IN ('anthropic_api', 'anthropic_batch') "
                "GROUP BY run_id ORDER BY run_id DESC LIMIT ?", (pattern, runs)).fetchall()
    except sqlite3.Error:
        return None
    if not rows:
        return None
    n = len(rows)
    return (sum(r[0] or 0 for r in rows) / n, sum(r[1] or 0 for r in rows) / n, sum(r[2] or 0 for r in rows) / n)


def check_models(cfg, *, api_key: str | None, fetch: Callable[..., str] | None = None,
                 list_account: Callable[[str], set[str]] | None = None,
                 today: date | None = None) -> list[Finding]:
    """Fetches the pages (and the account's models when `api_key` is set) and evaluates the configured
    models. Never raises for a network or page problem. The fetchers default to this module's
    `fetch_text` / `list_account_models`, looked up at call time (tests replace them)."""
    fetch = fetch or fetch_text
    list_account = list_account or list_account_models
    def read(url: str, parser):
        try:
            return parser(fetch(url, timeout=FETCH_TIMEOUT))
        except (OSError, ValueError, UnicodeDecodeError, http.client.HTTPException):
            return None

    provider = cfg.llm.provider
    models = {s: cfg.llm.models[s] for s in ("analyze", "merge") if s in cfg.llm.models}
    published = read(PRICING_URL, parse_price_list) if provider == "anthropic_api" else None
    statuses = read(DEPRECATIONS_URL, parse_deprecations)
    listed = read(OVERVIEW_URL, parse_overview)
    account = None
    notes: list[Finding] = []
    if api_key:
        try:
            account = list_account(api_key)
        except Exception as e:     # any SDK or network failure: the account check is optional
            notes.append(Finding("note", None, f"account check failed ({type(e).__name__}); availability "
                                               f"is from the public models page only."))
    else:
        notes.append(Finding("note", None, "no API key: availability is from the public models page; "
                                           "account access was not checked."))
    usage = {s: stage_usage(cfg.db_path, s) for s in models} if Path(cfg.db_path).exists() else {}
    findings = evaluate(models, provider, PRICES, published, statuses, listed, account, usage,
                        today or date.today())
    return findings + notes


# --- cache ---------------------------------------------------------------------------------------

def _cache_path(cfg) -> Path:
    return Path(cfg.home) / "cache" / CACHE_NAME


def _cache_key(cfg) -> dict:
    """What the findings were computed for: a provider or model change makes them stale."""
    return {"provider": cfg.llm.provider, "models": dict(sorted(cfg.llm.models.items()))}


def write_cache(cfg, findings: list[Finding], *, now: datetime | None = None) -> None:
    path = _cache_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"checked_at": (now or datetime.now()).isoformat(), "checked_for": _cache_key(cfg),
               "findings": [asdict(f) for f in findings]}
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")


def summary_line(findings: list[Finding]) -> str | None:
    """The one line run commands print: None when nothing needs attention."""
    n_suggest = sum(1 for f in findings if f.level == "suggest")
    n_warn = sum(1 for f in findings if f.level == "warn")
    if not n_suggest and not n_warn:
        return None
    return f"models: {n_suggest} suggestion(s), {n_warn} warning(s) -- see hmd doctor"


def format_findings(findings: list[Finding]) -> list[str]:
    return [f"[{f.level}] " + (f"{f.stage}: " if f.stage else "") + " ".join(f.text.split()) for f in findings]


def cached_findings(cfg, *, max_age_hours: float = 24, now: datetime | None = None) -> list[Finding] | None:
    try:
        payload = json.loads(_cache_path(cfg).read_text(encoding="utf-8"))
        checked = datetime.fromisoformat(payload["checked_at"])
        if payload.get("checked_for") != _cache_key(cfg):
            return None
        if (now or datetime.now()) - checked > timedelta(hours=max_age_hours):
            return None
        return [Finding(f["level"], f.get("stage"), f["text"]) for f in payload["findings"]]
    except (OSError, ValueError, KeyError, TypeError):
        return None


# --- hmd models use ------------------------------------------------------------------------------

STAGES = ("analyze", "merge")
_MODEL_ARG = re.compile(r"claude-[a-z0-9]+(?:-[a-z0-9]+)*")


def set_stage_model(config_path: str, stage: str, model: str) -> str:
    """Rewrites the `llm.models.<stage>` line of the config file in place and returns the old id. Only that
    line's value changes: indentation, a trailing comment, line endings and every other byte stay. Raises
    ConfigError for an unknown stage, a malformed id, a missing file or a config without that line."""
    from pipeline.config import ConfigError, load_config
    if stage not in STAGES:
        raise ConfigError(f"unknown stage {stage!r}; choose from {', '.join(STAGES)}")
    if not _MODEL_ARG.fullmatch(model):
        raise ConfigError(f"not a Claude model id: {model!r}")
    path = Path(config_path)
    if not path.exists():
        raise ConfigError(f"config file not found: {config_path} (run hmd init first)")
    raw = path.read_bytes().decode("utf-8")
    bom = "\ufeff" if raw.startswith("\ufeff") else ""
    lines = raw[len(bom):].splitlines(keepends=True)
    in_llm = in_models = False
    for i, line in enumerate(lines):
        body = line.rstrip("\r\n")
        indent = len(body) - len(body.lstrip(" "))
        if not body.strip() or body.lstrip().startswith("#"):
            continue
        if indent == 0:
            in_llm, in_models = body.startswith("llm:"), False
            continue
        if in_llm and body.strip() == "models:":
            in_models, models_indent = True, indent
            continue
        if in_models and indent <= models_indent:
            in_models = False
        m = re.match(rf"(\s+{stage}:\s*)(['\"]?)([^\s'\"#]+)\2(.*)", body) if in_models else None
        if m:
            old = m.group(3)
            lines[i] = m.group(1) + model + m.group(4) + line[len(body):]
            path.write_bytes((bom + "".join(lines)).encode("utf-8"))
            try:
                load_config(str(path))
            except Exception:
                path.write_bytes(raw.encode("utf-8"))
                raise
            return old
    raise ConfigError(f"{config_path} has no llm.models.{stage} line; add it (see hmd init) and retry")
