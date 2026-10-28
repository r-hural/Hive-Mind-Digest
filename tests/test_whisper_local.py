from tests.helpers import yt_ref
import json, os, subprocess, sys, tempfile
from types import SimpleNamespace
from unittest.mock import patch
import pytest
from pipeline.sources import SourceRef
from pipeline.transcription.base import Transcriber, TranscriptionError
from pipeline.transcription import whisper_local
from pipeline.transcription.whisper_local import MAX_AUDIO_BYTES, _REPO_ROOT, WhisperLocalProvider, _worker

_NO_OUT = object()  # sentinel: the fake worker writes nothing to --out


def make_fake_ydl(write_files=None):
    """Build a fresh fake yt_dlp.YoutubeDL class per test (no shared state across tests).

    write_files: list of filenames to create under outtmpl's dir on download(), or a
    callable(dir) for full control. Defaults to writing a single "audio.webm".
    """
    calls = []

    class FakeYDL:
        def __init__(self, opts):
            self.opts = opts
            calls.append(opts)

        def __enter__(self):
            return self

        def add_post_processor(self, pp, when):   # media_guard.install
            pass

        def __exit__(self, *a):
            return False

        def download(self, urls):
            d = os.path.dirname(self.opts["outtmpl"])
            if write_files is None:
                open(os.path.join(d, "audio.webm"), "wb").write(b"\x00")
            elif callable(write_files):
                write_files(d)
            else:
                for name in write_files:
                    open(os.path.join(d, name), "wb").write(b"\x00")

    FakeYDL.calls = calls
    return FakeYDL


def make_fake_run(stdout="", rc=0, stderr="", write_out=_NO_OUT):
    """Build a fresh fake subprocess.run per test. Records each (argv, kwargs) on `.calls`.

    If write_out is given (including None or []), it is JSON-dumped to the path following
    --out in argv -- mimicking what the real worker's _worker() does.
    """
    calls = []

    def run(argv, **kw):
        calls.append((argv, kw))
        if write_out is not _NO_OUT:
            out_path = argv[argv.index("--out") + 1]
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(write_out, f)
        return subprocess.CompletedProcess(argv, rc, stdout=stdout, stderr=stderr)

    run.calls = calls
    return run


def test_downloads_bestaudio_without_transcode_and_parses_worker_output():
    FakeYDL = make_fake_ydl()
    fake_run = make_fake_run(write_out={"language": "en", "segments": [{"start": 0.0, "end": 2.0, "text": " Hi "}]})
    with patch("pipeline.transcription.whisper_local.yt_dlp.YoutubeDL", FakeYDL), \
         patch("pipeline.transcription.whisper_local.subprocess.run", fake_run):
        tr = WhisperLocalProvider(model_size="base", timeout_seconds=99).transcribe(yt_ref("abcdefghijk"), ["en"])
    opts = FakeYDL.calls[0]
    assert opts["format"].startswith("bestaudio") and opts.get("postprocessors", []) == []
    assert "audio-format" not in json.dumps(opts)
    argv, kw = fake_run.calls[0]
    assert argv[0] == sys.executable and argv[1:4] == ["-P", "-m", "pipeline.transcription.whisper_local"]
    assert argv[argv.index("--audio") + 1].endswith("audio.webm")
    assert argv[argv.index("--model") + 1] == "base"
    assert argv[argv.index("--out") + 1].endswith("whisper.json")
    assert kw["timeout"] == 99
    assert kw["shell"] is False
    assert kw["env"]["PYTHONPATH"].startswith(_REPO_ROOT)
    assert tr.method == "whisper_local" and tr.language == "en" and tr.segments[0].text == " Hi "
    assert tr.extra == {"model": "base", "is_generated": True}


def test_worker_runs_in_its_temp_dir_without_api_keys(monkeypatch):
    # the user's working directory must not be the child's cwd (Windows looks there first for
    # ffmpeg) nor on its sys.path (-P), and the child never needs an API key
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")
    monkeypatch.setenv("SUPADATA_API_KEY", "sd-test-key")
    FakeYDL = make_fake_ydl()
    fake_run = make_fake_run(write_out={"language": "en", "segments": [{"start": 0.0, "end": 2.0, "text": "Hi"}]})
    with patch("pipeline.transcription.whisper_local.yt_dlp.YoutubeDL", FakeYDL), \
         patch("pipeline.transcription.whisper_local.subprocess.run", fake_run):
        WhisperLocalProvider().transcribe(yt_ref("abcdefghijk"), ["en"])
    argv, kw = fake_run.calls[0]
    assert kw["cwd"] == os.path.dirname(argv[argv.index("--audio") + 1])
    assert os.path.basename(kw["cwd"]).startswith("hmd-whisper-")
    assert "ANTHROPIC_API_KEY" not in kw["env"] and "SUPADATA_API_KEY" not in kw["env"]


def test_timeout_is_transcription_error():
    def run(argv, **kw):
        raise subprocess.TimeoutExpired(argv, kw["timeout"])
    FakeYDL = make_fake_ydl()
    with patch("pipeline.transcription.whisper_local.yt_dlp.YoutubeDL", FakeYDL), \
         patch("pipeline.transcription.whisper_local.subprocess.run", run):
        with pytest.raises(TranscriptionError, match="exceeded 5s"):
            WhisperLocalProvider(timeout_seconds=5).transcribe(yt_ref("abcdefghijk"), ["en"])


def test_worker_failure_is_transcription_error():
    FakeYDL = make_fake_ydl()
    fake_run = make_fake_run(rc=1, stderr="No module named whisper")
    with patch("pipeline.transcription.whisper_local.yt_dlp.YoutubeDL", FakeYDL), \
         patch("pipeline.transcription.whisper_local.subprocess.run", fake_run):
        with pytest.raises(TranscriptionError, match="whisper"):
            WhisperLocalProvider().transcribe(yt_ref("abcdefghijk"), ["en"])


def test_missing_audio_file_is_error():
    class NoFile(make_fake_ydl()):
        def download(self, urls):
            pass
    with patch("pipeline.transcription.whisper_local.yt_dlp.YoutubeDL", NoFile):
        with pytest.raises(TranscriptionError, match="no file"):
            WhisperLocalProvider().transcribe(yt_ref("abcdefghijk"), ["en"])


def test_worker_output_missing_segment_field_is_transcription_error():
    # A segment missing "end"/"text" must surface as TranscriptionError, not a raw KeyError.
    FakeYDL = make_fake_ydl()
    fake_run = make_fake_run(write_out={"language": "en", "segments": [{"start": 0}]})
    with patch("pipeline.transcription.whisper_local.yt_dlp.YoutubeDL", FakeYDL), \
         patch("pipeline.transcription.whisper_local.subprocess.run", fake_run):
        with pytest.raises(TranscriptionError):
            WhisperLocalProvider().transcribe(yt_ref("abcdefghijk"), ["en"])


def test_worker_output_null_is_transcription_error():
    FakeYDL = make_fake_ydl()
    fake_run = make_fake_run(write_out=None)
    with patch("pipeline.transcription.whisper_local.yt_dlp.YoutubeDL", FakeYDL), \
         patch("pipeline.transcription.whisper_local.subprocess.run", fake_run):
        with pytest.raises(TranscriptionError):
            WhisperLocalProvider().transcribe(yt_ref("abcdefghijk"), ["en"])


def test_worker_output_list_is_transcription_error():
    FakeYDL = make_fake_ydl()
    fake_run = make_fake_run(write_out=[])
    with patch("pipeline.transcription.whisper_local.yt_dlp.YoutubeDL", FakeYDL), \
         patch("pipeline.transcription.whisper_local.subprocess.run", fake_run):
        with pytest.raises(TranscriptionError):
            WhisperLocalProvider().transcribe(yt_ref("abcdefghijk"), ["en"])


def test_worker_writes_nothing_is_transcription_error_mentioning_stderr():
    FakeYDL = make_fake_ydl()
    fake_run = make_fake_run(rc=0, stderr="disk full", write_out=_NO_OUT)
    with patch("pipeline.transcription.whisper_local.yt_dlp.YoutubeDL", FakeYDL), \
         patch("pipeline.transcription.whisper_local.subprocess.run", fake_run):
        with pytest.raises(TranscriptionError, match="disk full"):
            WhisperLocalProvider().transcribe(yt_ref("abcdefghijk"), ["en"])


def test_download_audio_skips_part_and_ytdl_siblings():
    FakeYDL = make_fake_ydl(write_files=["audio.webm.part", "audio.webm", "audio.webm.ytdl"])
    with patch("pipeline.transcription.whisper_local.yt_dlp.YoutubeDL", FakeYDL):
        with tempfile.TemporaryDirectory() as tmp:
            path = WhisperLocalProvider().download_audio("https://example.com/v", tmp)
    assert path.endswith("audio.webm")
    assert not path.endswith((".part", ".ytdl"))


def test_download_audio_opts_include_max_filesize():
    FakeYDL = make_fake_ydl()
    with patch("pipeline.transcription.whisper_local.yt_dlp.YoutubeDL", FakeYDL):
        with tempfile.TemporaryDirectory() as tmp:
            WhisperLocalProvider().download_audio("https://example.com/v", tmp)
    assert FakeYDL.calls[0]["max_filesize"] == MAX_AUDIO_BYTES


def test_provider_satisfies_transcriber_protocol():
    assert isinstance(WhisperLocalProvider(), Transcriber)


def test_whitespace_only_segments_is_empty_transcript_error():
    FakeYDL = make_fake_ydl()
    fake_run = make_fake_run(write_out={"language": "en", "segments": [{"start": 0, "end": 1, "text": "   "}]})
    with patch("pipeline.transcription.whisper_local.yt_dlp.YoutubeDL", FakeYDL), \
         patch("pipeline.transcription.whisper_local.subprocess.run", fake_run):
        with pytest.raises(TranscriptionError, match="empty"):
            WhisperLocalProvider().transcribe(yt_ref("abcdefghijk"), ["en"])


def test_temp_dir_removed_after_success():
    FakeYDL = make_fake_ydl()
    fake_run = make_fake_run(write_out={"language": "en", "segments": [{"start": 0, "end": 1, "text": "hi"}]})
    with patch("pipeline.transcription.whisper_local.yt_dlp.YoutubeDL", FakeYDL), \
         patch("pipeline.transcription.whisper_local.subprocess.run", fake_run):
        WhisperLocalProvider().transcribe(yt_ref("abcdefghijk"), ["en"])
    audio_argv = fake_run.calls[0][0]
    tmp_dir = os.path.dirname(audio_argv[audio_argv.index("--audio") + 1])
    assert not os.path.exists(tmp_dir)


def test_worker_writes_out_file(tmp_path, monkeypatch):
    class StubModel:
        def transcribe(self, path):
            return {"language": "en",
                     "segments": [{"start": 0, "end": 1, "text": "hi", "no_speech_prob": 0.1}]}

    class StubWhisper:
        @staticmethod
        def load_model(name):
            return StubModel()

    monkeypatch.setitem(sys.modules, "whisper", StubWhisper)
    audio_path = str(tmp_path / "audio.webm")
    open(audio_path, "wb").write(b"\x00")
    out_path = str(tmp_path / "whisper.json")

    _worker(["--audio", audio_path, "--model", "base", "--out", out_path])

    with open(out_path, encoding="utf-8") as f:
        data = json.load(f)
    assert data == {"language": "en", "segments": [{"start": 0, "end": 1, "text": "hi"}]}


def test_video_kind_downloads_the_ref_url():
    FakeYDL = make_fake_ydl()
    fake_run = make_fake_run(write_out={"language": "en", "segments": [{"start": 0, "end": 1, "text": "hi"}]})
    ref = SourceRef("video-0123456789abcdef", "video", "https://vimeo.com/1")
    downloaded = []
    orig = FakeYDL.download
    FakeYDL.download = lambda self, urls: (downloaded.extend(urls), orig(self, urls))[1]
    with patch("pipeline.transcription.whisper_local.yt_dlp.YoutubeDL", FakeYDL),          patch("pipeline.transcription.whisper_local.subprocess.run", fake_run):
        tr = WhisperLocalProvider().transcribe(ref, ["en"])
    assert downloaded == ["https://vimeo.com/1"] and tr.video_id == ref.key
    assert FakeYDL.calls[0]["outtmpl"].endswith("audio.%(ext)s")


def _media_ref(path):
    return SourceRef("media-0123456789abcdef", "media", path.resolve().as_uri())


def test_media_kind_copies_the_file_and_never_writes_beside_it(tmp_path):
    src_dir = tmp_path / "user"
    src_dir.mkdir()
    audio = src_dir / "talk.mp3"
    audio.write_bytes(b"\x00\x01")
    fake_run = make_fake_run(write_out={"language": "en", "segments": [{"start": 0, "end": 1, "text": "hi"}]})
    FakeYDL = make_fake_ydl()
    with patch("pipeline.transcription.whisper_local.yt_dlp.YoutubeDL", FakeYDL),          patch("pipeline.transcription.whisper_local.subprocess.run", fake_run):
        tr = WhisperLocalProvider().transcribe(_media_ref(audio), ["en"])
    argv = fake_run.calls[0][0]
    staged = argv[argv.index("--audio") + 1]
    assert os.path.dirname(staged) != str(src_dir) and staged.endswith("audio.mp3")
    assert argv[argv.index("--out") + 1].startswith(os.path.dirname(staged))
    assert FakeYDL.calls == []                                   # no download for a local file
    assert sorted(os.listdir(src_dir)) == ["talk.mp3"]           # nothing landed beside the user's file
    assert tr.video_id == "media-0123456789abcdef" and tr.method == "whisper_local"


def test_media_kind_over_the_cap_is_rejected_before_copying(tmp_path, monkeypatch):
    audio = tmp_path / "big.mp4"
    audio.write_bytes(b"\x00")
    monkeypatch.setattr(whisper_local, "MAX_AUDIO_BYTES", 0)
    fake_run = make_fake_run(write_out={"language": "en", "segments": []})
    with patch("pipeline.transcription.whisper_local.subprocess.run", fake_run):
        with pytest.raises(TranscriptionError, match="over the 0 MB cap"):
            WhisperLocalProvider().transcribe(_media_ref(audio), ["en"])
    assert fake_run.calls == []
    with tempfile.TemporaryDirectory() as stage:
        with pytest.raises(TranscriptionError, match="cap"):
            WhisperLocalProvider._stage_local_file(_media_ref(audio), stage)
        assert os.listdir(stage) == []


def test_media_kind_missing_file_is_transcription_error(tmp_path):
    with patch("pipeline.transcription.whisper_local.subprocess.run", make_fake_run()):
        with pytest.raises(TranscriptionError, match="not a regular file"):
            WhisperLocalProvider().transcribe(_media_ref(tmp_path / "gone.mp3"), ["en"])


def test_download_options_never_expand_playlists():
    FakeYDL = make_fake_ydl()
    with patch("pipeline.transcription.whisper_local.yt_dlp.YoutubeDL", FakeYDL):
        with tempfile.TemporaryDirectory() as tmp:
            WhisperLocalProvider().download_audio("https://vimeo.com/channels/x?list=1", tmp)
    opts = FakeYDL.calls[0]
    assert opts["noplaylist"] is True and opts["playlist_items"] == "1"


def test_download_audio_ignores_fragments_and_picks_the_largest(tmp_path):
    def write(d):
        for name, size in [("audio.f137.mp4", 99), ("audio.webm.part-Frag3", 50), ("audio.m4a", 5),
                           ("audio.webm", 10), ("audio.f251.webm", 80)]:
            with open(os.path.join(d, name), "wb") as f:
                f.write(b"x" * size)
    FakeYDL = make_fake_ydl(write_files=write)
    with patch("pipeline.transcription.whisper_local.yt_dlp.YoutubeDL", FakeYDL):
        path = WhisperLocalProvider().download_audio("https://example.com/v", str(tmp_path))
    assert os.path.basename(path) == "audio.webm"


def test_media_directory_is_refused(tmp_path):
    d = tmp_path / "folder.mp3"
    d.mkdir()
    with pytest.raises(TranscriptionError, match="not a regular file"):
        WhisperLocalProvider._stage_local_file(_media_ref(d), str(tmp_path))


def test_media_symlink_is_refused(tmp_path):
    target = tmp_path / "real.mp3"
    target.write_bytes(b"x")
    link = tmp_path / "link.mp3"
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not permitted here")
    # SourceRef.path resolves a link the user names, so hand the guard the raw link path: it
    # covers a link swapped in after the source was registered (as test_local_text does).
    ref = SimpleNamespace(path=str(link), url=link.as_uri())
    with pytest.raises(TranscriptionError, match="not a regular file"):
        WhisperLocalProvider._stage_local_file(ref, str(tmp_path))


def test_media_copy_that_grew_past_the_cap_is_rejected(tmp_path, monkeypatch):
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"12345")
    monkeypatch.setattr(whisper_local, "MAX_AUDIO_BYTES", 8)
    real = whisper_local.shutil.copyfile
    def grow(src, dst):
        real(src, dst)
        with open(dst, "ab") as f:
            f.write(b"x" * 20)
        return dst
    monkeypatch.setattr(whisper_local.shutil, "copyfile", grow)
    stage = tmp_path / "stage"
    stage.mkdir()
    with pytest.raises(TranscriptionError, match="cap"):
        WhisperLocalProvider._stage_local_file(_media_ref(audio), str(stage))
    assert os.listdir(stage) == []
