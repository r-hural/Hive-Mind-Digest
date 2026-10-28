import pytest

from pipeline.pricing import BATCH_DISCOUNT, PRICES, Price, estimate_cost, price_for


def test_dated_model_id_resolves_by_prefix():
    assert price_for("claude-haiku-4-5-20251001") is not None
    assert price_for("claude-haiku-4-5") is not None


def test_successor_model_with_dash_suffix_is_not_confused_with_dated_snapshot():
    assert price_for("claude-sonnet-5-1") is None
    assert price_for("claude-opus-5-") is None
    assert price_for("claude-haiku-4-5-20251001") is not None


def test_longest_prefix_wins_between_genuine_prefix_pair(monkeypatch):
    monkeypatch.setitem(PRICES, "claude-sonnet-5-1", Price(4.0, 20.0))
    assert price_for("claude-sonnet-5-1-20260401").input == 4.0
    assert price_for("claude-sonnet-5-20260401").input == 2.0


def test_sonnet_46_and_sonnet_5_are_distinct():
    assert price_for("claude-sonnet-4-6").input == 3.00
    assert price_for("claude-sonnet-5").input == 2.00


def test_unknown_model_is_none():
    assert price_for("gpt-9") is None
    assert estimate_cost("gpt-9", 1000, 1000) is None


def test_estimate_cost_haiku():
    # 1M input at $1 + 1M output at $5
    assert estimate_cost("claude-haiku-4-5-20251001", 1_000_000, 1_000_000) == 6.0


def test_cache_tokens_priced():
    p = price_for("claude-sonnet-4-6")
    assert abs(p.cache_read - 0.30) < 1e-9
    assert abs(p.cache_write - 3.75) < 1e-9
    cost = estimate_cost("claude-sonnet-4-6", 0, 0, cache_read_tokens=1_000_000)
    assert abs(cost - 0.30) < 1e-9


def test_claude_5_5_family_is_priced():
    # Anthropic price list, checked 2026-10-15
    assert (price_for("claude-haiku-5-5").input, price_for("claude-haiku-5-5").output) == (0.10, 0.50)
    assert (price_for("claude-sonnet-5-5").input, price_for("claude-sonnet-5-5").output) == (2.00, 10.00)
    assert (price_for("claude-opus-5-5").input, price_for("claude-opus-5-5").output) == (4.00, 20.00)
    assert (price_for("claude-fable-5-1").input, price_for("claude-fable-5-1").output) == (10.00, 50.00)


def test_cache_read_multiplier_is_per_model():
    assert abs(price_for("claude-sonnet-5-5").cache_read - 0.10) < 1e-9    # 0.05x
    assert abs(price_for("claude-opus-5-5").cache_read - 0.20) < 1e-9      # 0.05x
    assert abs(price_for("claude-fable-5-1").cache_read - 0.25) < 1e-9     # 0.025x
    assert abs(price_for("claude-haiku-4-5").cache_read - 0.10) < 1e-9     # 0.1x
    assert abs(price_for("claude-sonnet-5-5").cache_write - 2.50) < 1e-9   # 1.25x (5-minute write)


def test_haiku_5_5_long_prompt_pays_the_higher_price():
    # a request whose prompt (input + cache reads + cache writes) is over 100,000 tokens
    assert abs(estimate_cost("claude-haiku-5-5", 100_000, 1_000_000) - 0.51) < 1e-9
    assert abs(estimate_cost("claude-haiku-5-5", 100_001, 0) - 100_001 * 0.50 / 1e6) < 1e-12
    over = estimate_cost("claude-haiku-5-5", 60_000, 1_000_000, cache_read_tokens=50_000)
    assert abs(over - (60_000 * 0.50 + 1_000_000 * 2.50 + 50_000 * 0.05) / 1e6) < 1e-12


def test_price_list_date_is_recorded():
    from pipeline.pricing import PRICES_CHECKED
    assert PRICES_CHECKED == "2026-10-15"


def test_model_line_and_version_from_the_id():
    from pipeline.pricing import model_line, model_version
    assert model_line("claude-haiku-4-5-20251001") == "haiku"
    assert model_line("claude-sonnet-5-5") == "sonnet"
    assert model_line("gpt-9") is None and model_line("claude-mythos-preview") is None
    assert model_version("claude-opus-4-5-20251101") == (4, 5)
    assert model_version("claude-sonnet-5") == (5, 0)
    assert model_version("claude-sonnet-5-20260401") == (5, 0)
    assert model_version("claude-3-5-haiku-20241022") is None


def test_new_tokenizer_from_claude_4_7_on():
    from pipeline.pricing import TOKENIZER_FACTOR, new_tokenizer
    assert TOKENIZER_FACTOR == 1.3
    for m in ("claude-opus-4-7", "claude-sonnet-5-5", "claude-haiku-5-5", "claude-fable-5-1", "claude-sonnet-5"):
        assert new_tokenizer(m), m
    for m in ("claude-haiku-4-5-20251001", "claude-sonnet-4-6", "claude-opus-4-6", "gpt-9"):
        assert not new_tokenizer(m), m


def test_price_table_age():
    from datetime import date
    from pipeline.pricing import prices_age_days
    assert prices_age_days(date(2026, 12, 14)) == 60


def test_batch_halves_every_token_price():
    full = estimate_cost("claude-sonnet-5-5", 1_000_000, 1_000_000, 1_000_000, 1_000_000)
    half = estimate_cost("claude-sonnet-5-5", 1_000_000, 1_000_000, 1_000_000, 1_000_000, batch=True)
    assert BATCH_DISCOUNT == 0.5
    assert half == pytest.approx(full * 0.5)


def test_batch_keeps_the_long_prompt_tier():
    # Haiku 5.5 over 100k prompt tokens pays $0.50/MTok input; the batch halves that.
    assert estimate_cost("claude-haiku-5-5", 200_000, 0, batch=True) == pytest.approx(200_000 * 0.50 / 1e6 / 2)


def test_batch_price_of_unknown_model_is_none():
    assert estimate_cost("no-such-model", 1, 1, batch=True) is None
