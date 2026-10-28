import json, os
import pytest
from pipeline.transcription.base import Segment, Transcript, TranscriptionError
from pipeline.transcription.cache import TranscriptCache


def _t(method="youtube_captions"):
    return Transcript("abcdefghijk", method, "en", [Segment(0, 1, "hi"), Segment(1, 2, "there")],
                      extra={"is_generated": True})


def test_save_and_load_roundtrip(tmp_path):
    c = TranscriptCache(str(tmp_path))
    path = c.save(_t())
    assert path.replace("\\", "/").endswith("abcdefghijk/youtube_captions.json")
    assert c.has("abcdefghijk", "youtube_captions")
    tr = c.load("abcdefghijk", "youtube_captions")
    assert tr.segments[1].text == "there" and tr.language == "en" and tr.extra == {"is_generated": True}
    assert tr.created_at           # stamped on save
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    assert data["segments"][0] == {"start": 0.0, "end": 1.0, "text": "hi"}


def test_methods_do_not_overwrite_each_other(tmp_path):
    c = TranscriptCache(str(tmp_path))
    c.save(_t("youtube_captions")); c.save(_t("whisper_local"))
    assert c.available("abcdefghijk") == ["whisper_local", "youtube_captions"]


def test_load_best_follows_preference_order(tmp_path):
    c = TranscriptCache(str(tmp_path))
    c.save(_t("whisper_local"))
    assert c.load_best("abcdefghijk", ["youtube_captions", "ytdlp_subs", "whisper_local"]).method == "whisper_local"
    c.save(_t("ytdlp_subs"))
    assert c.load_best("abcdefghijk", ["youtube_captions", "ytdlp_subs", "whisper_local"]).method == "ytdlp_subs"
    assert c.load_best("zzzzzzzzzzz", ["youtube_captions"]) is None


def test_rejects_unsafe_ids(tmp_path):
    c = TranscriptCache(str(tmp_path))
    with pytest.raises(ValueError):
        c.path("../etc", "youtube_captions")
    with pytest.raises(ValueError):
        c.path("abcdefghijk", "../x")


def test_rejects_trailing_newline_and_wrong_length_ids(tmp_path):
    c = TranscriptCache(str(tmp_path))
    with pytest.raises(ValueError):
        c.path("abcdefghijk\n", "youtube_captions")
    with pytest.raises(ValueError):
        c.path("abcdefghij", "youtube_captions")   # 10 chars
    with pytest.raises(ValueError):
        c.path("abcdefghijkl", "youtube_captions")  # 12 chars
    with pytest.raises(ValueError):
        c.path("abcdefghijk", "YouTube_Captions")   # uppercase method


def test_corrupt_cache_file_is_a_miss_not_a_crash(tmp_path):
    c = TranscriptCache(str(tmp_path))
    path = c.path("abcdefghijk", "youtube_captions")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("{not json")
    with pytest.raises(TranscriptionError):
        c.load("abcdefghijk", "youtube_captions")
    assert c.load_best("abcdefghijk", ["youtube_captions"]) is None


def test_load_best_skips_corrupt_method_and_uses_next(tmp_path):
    c = TranscriptCache(str(tmp_path))
    bad_path = c.path("abcdefghijk", "youtube_captions")
    os.makedirs(os.path.dirname(bad_path), exist_ok=True)
    with open(bad_path, "w", encoding="utf-8") as f:
        f.write("{not json")
    c.save(_t("whisper_local"))
    tr = c.load_best("abcdefghijk", ["youtube_captions", "whisper_local"])
    assert tr is not None and tr.method == "whisper_local"


def test_available_ignores_stray_tmp_and_unrelated_files(tmp_path):
    c = TranscriptCache(str(tmp_path))
    c.save(_t("youtube_captions"))
    d = os.path.join(str(tmp_path), "abcdefghijk")
    with open(os.path.join(d, "whisper_local.json.tmp"), "w", encoding="utf-8") as f:
        f.write("partial")
    with open(os.path.join(d, "notes.txt"), "w", encoding="utf-8") as f:
        f.write("hi")
    assert c.available("abcdefghijk") == ["youtube_captions"]


def test_save_failure_leaves_no_tmp_file_behind(tmp_path):
    c = TranscriptCache(str(tmp_path))
    bad = Transcript("abcdefghijk", "youtube_captions", "en", [Segment(0, 1, "hi")], extra={"x": {1, 2}})
    with pytest.raises(TypeError):
        c.save(bad)
    d = os.path.join(str(tmp_path), "abcdefghijk")
    leftovers = os.listdir(d) if os.path.isdir(d) else []
    assert not any(f.endswith(".tmp") for f in leftovers)


def test_available_rejects_unsafe_id(tmp_path):
    c = TranscriptCache(str(tmp_path))
    with pytest.raises(ValueError):
        c.available("../etc")


def test_load_uses_args_not_payload_for_id_and_method(tmp_path):
    c = TranscriptCache(str(tmp_path))
    path = c.save(_t("youtube_captions"))
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    data["video_id"] = "zzzzzzzzzzz"
    data["method"] = "whisper_local"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    tr = c.load("abcdefghijk", "youtube_captions")
    assert tr.video_id == "abcdefghijk" and tr.method == "youtube_captions"


def test_load_coerces_segment_times_to_float(tmp_path):
    c = TranscriptCache(str(tmp_path))
    path = c.save(_t("youtube_captions"))
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    data["segments"][0]["start"] = 0
    data["segments"][0]["end"] = 1
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    tr = c.load("abcdefghijk", "youtube_captions")
    assert isinstance(tr.segments[0].start, float) and isinstance(tr.segments[0].end, float)


def test_saved_json_has_schema_key(tmp_path):
    c = TranscriptCache(str(tmp_path))
    path = c.save(_t())
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    assert data["schema"] == 1


def test_resave_overwrites_and_preserves_preset_created_at(tmp_path):
    c = TranscriptCache(str(tmp_path))
    t1 = _t()
    c.save(t1)
    stamped = t1.created_at
    t2 = _t()
    t2.created_at = stamped
    t2.segments = [Segment(0, 1, "updated")]
    c.save(t2)
    assert t2.created_at == stamped  # preserved, not overwritten
    tr = c.load("abcdefghijk", "youtube_captions")
    assert tr.segments[0].text == "updated"
    assert tr.created_at == stamped


def test_null_segment_text_is_unreadable(tmp_path):
    c = TranscriptCache(str(tmp_path))
    path = c.save(_t())
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    data["segments"][0]["text"] = None
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    with pytest.raises(TranscriptionError):
        c.load("abcdefghijk", "youtube_captions")
    assert c.load_best("abcdefghijk", ["youtube_captions"]) is None


def test_load_best_returns_none_when_cached_method_not_in_preference(tmp_path):
    c = TranscriptCache(str(tmp_path))
    c.save(_t("whisper_local"))
    assert c.load_best("abcdefghijk", ["youtube_captions"]) is None


def test_concurrent_save_race_treated_as_done(tmp_path, monkeypatch):
    # Simulate the Windows os.replace(tmp, path) PermissionError when another writer already
    # holds the destination: the destination existing (someone else finished first) makes the
    # patched os.replace raise, while a non-existent destination behaves normally.
    real_replace = os.replace

    def flaky_replace(src, dst):
        if os.path.exists(dst):
            raise PermissionError("simulated race: destination locked by other writer")
        real_replace(src, dst)

    monkeypatch.setattr("pipeline.transcription.cache.os.replace", flaky_replace)

    c = TranscriptCache(str(tmp_path))
    path1 = c.save(_t())
    assert os.path.exists(path1)

    # Second save of the same (video_id, method): destination now exists, so the patched
    # os.replace raises. save() should treat this as "the other writer won" and succeed.
    path2 = c.save(_t())
    assert path2 == path1

    d = os.path.dirname(path1)
    assert not any(f.endswith(".tmp") for f in os.listdir(d))
    tr = c.load("abcdefghijk", "youtube_captions")
    assert tr.segments[1].text == "there"


def test_concurrent_save_race_with_no_valid_destination_raises(tmp_path, monkeypatch):
    real_replace = os.replace

    def flaky_replace(src, dst):
        if os.path.exists(dst):
            os.remove(dst)  # the destination never ends up valid after the race
            raise PermissionError("simulated race: destination locked by other writer")
        real_replace(src, dst)

    monkeypatch.setattr("pipeline.transcription.cache.os.replace", flaky_replace)

    c = TranscriptCache(str(tmp_path))
    c.save(_t())  # first save succeeds normally, destination now exists

    with pytest.raises(TranscriptionError):
        c.save(_t())


def test_youtube_cache_entry_is_unchanged_and_pages_round_trip(tmp_path):
    c = TranscriptCache(str(tmp_path))
    t = Transcript("dQw4w9WgXcQ", "youtube_captions", "en", [Segment(0, 1, "hi")], created_at="2026-01-01T00:00:00+00:00")
    data = json.load(open(c.save(t), encoding="utf-8"))
    assert "kind" not in data and data["segments"] == [{"start": 0.0, "end": 1.0, "text": "hi"}]
    t2 = Transcript("pdf-0123456789abcdef", "pdf_text", "und", [Segment(0, 0, "p1", page=1), Segment(0, 0, "p2", page=2)],
                    extra={"kind": "pdf"})
    path2 = c.save(t2)
    assert json.load(open(path2, encoding="utf-8"))["kind"] == "pdf"
    back = c.load("pdf-0123456789abcdef", "pdf_text")
    assert [s.page for s in back.segments] == [1, 2] and back.flatten().seg_pages == [1, 2]


def test_youtube_kind_is_not_written(tmp_path):
    c = TranscriptCache(str(tmp_path))
    t = Transcript("dQw4w9WgXcQ", "youtube_captions", "en", [Segment(0, 1, "hi")], extra={"kind": "youtube"})
    assert "kind" not in json.load(open(c.save(t), encoding="utf-8"))


def test_cache_save_redacts_and_counts(tmp_path):
    c = TranscriptCache(str(tmp_path))
    fake_key = "AK" + "IA" + "B" * 16            # built at run time: no key-shaped literal in the repo
    t = Transcript("web-0123456789abcdef", "web_text", "und", [Segment(0, 0, fake_key)])
    c.save(t)
    back = c.load("web-0123456789abcdef", "web_text")
    assert back.segments[0].text == "[REDACTED]" and back.extra["redacted"] == 1
