from pipeline.sources import KINDS
from pipeline.transcription.base import Segment, Transcript, flatten, watch_url, Transcriber


def _t(segs):
    return Transcript(video_id="abcdefghijk", method="youtube_captions", language="en", segments=segs)


def test_flatten_joins_with_single_spaces_and_records_offsets():
    tr = _t([Segment(0.0, 1.5, "Hello  world"), Segment(1.5, 3.0, " how are\nyou "), Segment(3.0, 4.0, "")])
    flat = tr.flatten()
    assert flat.text == "Hello world how are you"
    assert flat.seg_starts == [0, 12]
    assert flat.seg_times == [0.0, 1.5]


def test_time_at_maps_offsets_to_segment_start():
    flat = _t([Segment(0.0, 1.0, "aaa"), Segment(10.0, 11.0, "bbb"), Segment(20.0, 21.0, "ccc")]).flatten()
    assert flat.text == "aaa bbb ccc"
    assert flat.time_at(0) == 0.0
    assert flat.time_at(3) == 0.0        # the joining space belongs to the previous segment
    assert flat.time_at(4) == 10.0
    assert flat.time_at(9) == 20.0
    assert flat.time_at(999) == 20.0


def test_empty_transcript():
    flat = _t([]).flatten()
    assert flat.text == "" and flat.time_at(0) == 0.0
    assert _t([]).duration == 0.0


def test_word_count_and_duration():
    tr = _t([Segment(0, 2, "one two"), Segment(2, 5.5, "three")])
    assert tr.word_count() == 3 and tr.duration == 5.5


def test_time_at_clamps_negative_offsets():
    flat = _t([Segment(0.0, 1.0, "aaa"), Segment(10.0, 11.0, "bbb")]).flatten()
    assert flat.time_at(-1) == 0.0


def test_duration_uses_latest_end_not_last_segment():
    assert _t([Segment(0, 50, "a"), Segment(5, 7, "b")]).duration == 50.0


def test_module_level_flatten_matches_method():
    segs = [Segment(0.0, 1.0, "aaa"), Segment(10.0, 11.0, "bbb")]
    assert flatten(segs) == _t(segs).flatten()


def test_watch_url():
    assert watch_url("abcdefghijk") == "https://www.youtube.com/watch?v=abcdefghijk"


def test_protocol_is_runtime_checkable():
    class P:
        name = "x"
        kinds = frozenset(KINDS)

        def transcribe(self, ref, languages): ...
    assert isinstance(P(), Transcriber)
