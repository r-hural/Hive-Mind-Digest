from pipeline.transcription.base import Segment
from pipeline.transcription.scrub import scrub_segments, scrub_text

# Fake keys are assembled at run time, so no key-shaped literal sits in the repository for the
# release grep or a secret scanner to flag.
ANT_PREFIX = "sk" + "-ant-"
ANT_KEY = ANT_PREFIX + "abcdefghijklmnopqrstuvwxyz0123"
PLAIN_KEY = "sk" + "-" + "abcdefghijklmnopqrstuvwxyz"
AWS_PREFIX = "AK" + "IA"
AWS_KEY = AWS_PREFIX + "ABCDEFGHIJKLMNOP"
PEM_LABEL = "PRIVATE" + " KEY"


def _pem(kind=""):
    label = f"{kind} {PEM_LABEL}".strip()
    body = "\n".join(["MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC7", "q1w2e3r4t5y6u7i8o9p0"])
    return f"-----BEGIN {label}-----\n{body}\n-----END {label}-----"


def test_known_key_shapes_are_redacted():
    text, n = scrub_text(f"use {ANT_KEY} and {AWS_KEY} now")
    assert ANT_PREFIX not in text and AWS_PREFIX not in text and n == 2 and text.count("[REDACTED]") == 2


def test_private_key_block_is_redacted_with_its_body():
    for kind in ("", "RSA", "EC", "OPENSSH"):
        text, n = scrub_text(f"before\n{_pem(kind)}\nafter")
        assert text == "before\n[REDACTED]\nafter" and n == 1, kind


def test_private_key_block_without_an_end_line_is_redacted_to_the_end():
    pem = _pem().rsplit("\n", 1)[0]               # cut off before the END line
    text, n = scrub_text(f"see {pem}")
    assert text == "see [REDACTED]" and n == 1


def test_text_after_a_private_key_block_is_kept():
    text, n = scrub_text(f"{_pem()} first\n\n{_pem('RSA')} second")
    assert text == "[REDACTED] first\n\n[REDACTED] second" and n == 2


def test_labelled_secret_keeps_the_label():
    # The fake key is assembled at run time so the release secret scan (gitleaks) sees no literal.
    text, n = scrub_text("password: hunter2hunter2 and api_key=" + "0123456789" + "abcdef")
    assert text == "password: [REDACTED] and api_key=[REDACTED]" and n == 2


def test_ordinary_text_is_untouched():
    text, n = scrub_text("The token economy and a secret recipe are not credentials.")
    assert n == 0 and text.startswith("The token economy")


def test_ordinary_prose_after_a_label_is_untouched():
    for sentence in ("The secret: consistency beats intensity.",
                     "Her password: rememberable sentences work best.",
                     "One token: everything-in-moderation is the rule."):
        text, n = scrub_text(sentence)
        assert (text, n) == (sentence, 0)


def test_segments_scrubbed_in_place():
    segs = [Segment(0, 1, f"key {PLAIN_KEY}"), Segment(1, 2, "fine")]
    assert scrub_segments(segs) == 1 and segs[0].text == "key [REDACTED]"


def test_hyphenated_words_and_url_slugs_are_untouched():
    for sentence in ("a task-oriented-productivity-system works",
                     "risk-adjusted-return-on-investment",
                     "https://example.com/disk-usage-analyzer-for-linux",
                     "ask-me-anything-about-everything"):
        assert scrub_text(sentence) == (sentence, 0)


def test_key_shape_inside_a_longer_word_is_not_redacted():
    assert scrub_text("X" + AWS_KEY) == ("X" + AWS_KEY, 0)
    assert scrub_text(AWS_KEY) == ("[REDACTED]", 1)


def test_real_key_is_redacted_after_space_equals_quote_and_at_line_start():
    key = ANT_KEY
    for text in (f"use {key}", f"KEY={key}", f'"{key}"', key, "first line" + chr(10) + key):
        out, n = scrub_text(text)
        assert key not in out and n == 1, text


def test_labelled_scrub_is_linear_on_a_run_of_labels():
    import time
    t0 = time.perf_counter()
    scrub_text("token:" * 40_000)
    assert time.perf_counter() - t0 < 2.0


def test_a_long_labelled_value_is_redacted_whole_and_short_prose_is_unchanged():
    text, n = scrub_text("password: " + "a1" * 500 + " after")
    assert n == 1 and text == "password: [REDACTED] after"
    assert scrub_text("The secret: consistency beats intensity") == ("The secret: consistency beats intensity", 0)
