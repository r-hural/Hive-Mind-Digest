"""Remote text (an error body, a page title) must not drive the terminal: ESC and other control
characters are replaced before hmd prints it (release security review, 2026-10-28)."""
from pipeline.runner import safe_log
from pipeline.termtext import terminal_safe


def test_escape_sequences_and_controls_are_replaced():
    assert terminal_safe("bad\x1b[2J\x1b]0;title\x07end") == "bad?[2J?]0;title?end"
    assert terminal_safe("a\rb\x00c\x7fd\x9be") == "a?b?c?d?e"


def test_newlines_tabs_and_ordinary_unicode_are_kept():
    assert terminal_safe("line 1\n\tline 2 × 中") == "line 1\n\tline 2 × 中"


def test_bidi_overrides_are_replaced():
    assert terminal_safe("abc‮dcba⁦x⁩") == "abc?dcba?x?"


def test_safe_log_strips_controls():
    lines = []
    safe_log(lines.append)("  [web-1] metadata ERROR: \x1b[31mred\x1b[0m")
    assert lines == ["  [web-1] metadata ERROR: ?[31mred?[0m"]
