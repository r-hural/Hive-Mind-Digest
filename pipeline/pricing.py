"""USD prices per million tokens, from the Anthropic price list
(https://platform.claude.com/docs/en/about-claude/pricing), checked on PRICES_CHECKED.

Add a row when a new model is configured; `hmd run` refuses to start a budgeted run
with the anthropic_api provider when a configured model has no price.

Batch API calls (IDEA-304) pay BATCH_DISCOUNT times every token price: estimate_cost(..., batch=True).
Not modelled: 1-hour cache writes (2x), fast mode, data residency (1.1x). Claude 4.7 and later models use
a newer tokenizer that produces about 30% more tokens for the same text, so a per-token price is not
comparable across that line without that factor.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

_DATE_SUFFIX = re.compile(r"\d{8}")

PRICES_CHECKED = "2026-10-15"
# Claude 4.7 and later (and every Fable / Mythos model) count about this many more tokens for the same text.
TOKENIZER_FACTOR = 1.3
# Message Batches API: every token price (input, output, cache read and write) is halved.
BATCH_DISCOUNT = 0.5
_MODEL_ID = re.compile(r"claude-([a-z]+)-(\d{1,2})(?:-(\d{1,2}))?(?:-\d{8})?")


@dataclass(frozen=True)
class Price:
    input: float
    output: float
    cache_read_x: float = 0.1               # cache hits and refreshes, times the base input price
    # Prompt-length tiers (Claude Haiku 5.5): a request whose prompt -- input plus cache reads and
    # writes -- is over long_prompt_tokens pays `long_prompt` prices instead.
    long_prompt_tokens: int | None = None
    long_prompt: "Price | None" = None

    @property
    def cache_read(self) -> float:
        return self.input * self.cache_read_x

    @property
    def cache_write(self) -> float:
        return self.input * 1.25            # 5-minute cache write


PRICES: dict[str, Price] = {
    "claude-haiku-4-5": Price(1.00, 5.00),
    "claude-haiku-5-5": Price(0.10, 0.50, long_prompt_tokens=100_000, long_prompt=Price(0.50, 2.50)),
    "claude-sonnet-4-5": Price(3.00, 15.00),
    "claude-sonnet-4-6": Price(3.00, 15.00),
    "claude-sonnet-5": Price(2.00, 10.00),  # launch price made standard (no Sep 2026 increase)
    "claude-sonnet-5-5": Price(2.00, 10.00, cache_read_x=0.05),
    "claude-opus-4-5": Price(5.00, 25.00),
    "claude-opus-4-6": Price(5.00, 25.00),
    "claude-opus-4-7": Price(5.00, 25.00),
    "claude-opus-4-8": Price(5.00, 25.00),
    "claude-opus-5": Price(5.00, 25.00),
    "claude-opus-5-5": Price(4.00, 20.00, cache_read_x=0.05),
    "claude-fable-5": Price(10.00, 50.00),
    "claude-fable-5-1": Price(10.00, 50.00, cache_read_x=0.025),
}


def price_for(model: str) -> Price | None:
    """Longest-prefix match so dated ids like claude-haiku-4-5-20251001 resolve.

    Only an 8-digit date snapshot suffix is accepted after the dash, so a distinct
    successor model like claude-sonnet-5-1 does not silently inherit claude-sonnet-5's price.
    """
    best_key = None
    for key in PRICES:
        if model == key:
            matches = True
        elif model.startswith(key + "-"):
            remainder = model[len(key) + 1:]
            matches = bool(_DATE_SUFFIX.fullmatch(remainder))
        else:
            matches = False
        if matches and (best_key is None or len(key) > len(best_key)):
            best_key = key
    return PRICES[best_key] if best_key is not None else None


def estimate_cost(model: str, input_tokens: int, output_tokens: int,
                  cache_read_tokens: int = 0, cache_write_tokens: int = 0, *, batch: bool = False) -> float | None:
    p = price_for(model)
    if p is None:
        return None
    if p.long_prompt and input_tokens + cache_read_tokens + cache_write_tokens > p.long_prompt_tokens:
        p = p.long_prompt
    cost = (input_tokens * p.input + output_tokens * p.output
            + cache_read_tokens * p.cache_read + cache_write_tokens * p.cache_write) / 1_000_000
    return cost * BATCH_DISCOUNT if batch else cost


def _parse_id(model: str) -> re.Match | None:
    m = _MODEL_ID.fullmatch(model)
    return m if m and not m.group(1).isdigit() else None


def model_line(model: str) -> str | None:
    """The model line ("haiku", "sonnet", "opus", "fable", ...) named in a current-style id, else None."""
    m = _parse_id(model)
    return m.group(1) if m else None


def model_version(model: str) -> tuple[int, int] | None:
    """(major, minor) from a current-style id: claude-opus-4-5-20251101 -> (4, 5), claude-sonnet-5 -> (5, 0)."""
    m = _parse_id(model)
    return (int(m.group(2)), int(m.group(3) or 0)) if m else None


def new_tokenizer(model: str) -> bool:
    """True for models on the newer tokenizer (Claude 4.7 and later, every Fable and Mythos model)."""
    line, version = model_line(model), model_version(model)
    if line is None or version is None:
        return False
    return line in ("fable", "mythos") or version >= (4, 7)


def prices_age_days(today: date) -> int:
    return (today - date.fromisoformat(PRICES_CHECKED)).days
