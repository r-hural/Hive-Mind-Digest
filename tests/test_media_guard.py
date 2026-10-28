"""yt-dlp's resolved media, manifest and subtitle URLs are checked against private and local
addresses before anything is downloaded (release security review, 2026-10-28). The input host is
already one of the known video sites; this covers where its formats point."""
import socket

import pytest
import yt_dlp

from pipeline.transcription import media_guard, web_text


def _resolver(table):
    def resolve(host, port, type=0):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (table[host], port))]
    return resolve


@pytest.fixture
def dns(monkeypatch):
    monkeypatch.setattr(web_text, "_resolver", _resolver({"cdn.example.org": "93.184.216.34",
                                                          "evil.example.org": "127.0.0.1",
                                                          "meta.example.org": "169.254.169.254"}))
    monkeypatch.setattr(web_text, "_allow_private_hosts", False)


def _info(fmt_url, subs_url="https://cdn.example.org/subs.json3", **fmt):
    return {"id": "v1", "title": "t", "extractor": "generic", "extractor_key": "Generic",
            "webpage_url": "https://vimeo.com/1",
            "formats": [{"format_id": "a", "url": fmt_url, "ext": "m4a", "acodec": "mp4a", "vcodec": "none", **fmt}],
            "subtitles": {"en": [{"ext": "json3", "url": subs_url}]}}


def _process(tmp_path, info, **opts):
    params = {"skip_download": True, "quiet": True, "no_warnings": True, "writesubtitles": True,
              "subtitleslangs": ["en"], "subtitlesformat": "json3",
              "outtmpl": str(tmp_path / "%(id)s.%(ext)s"), **opts}
    with yt_dlp.YoutubeDL(params) as ydl:
        media_guard.install(ydl)
        ydl.process_ie_result(info, download=True)


def test_media_urls_collects_formats_manifests_fragments_and_subtitles():
    info = {"url": "https://a.example/x", "manifest_url": "https://b.example/m.m3u8",
            "requested_formats": [{"url": "https://c.example/v", "fragment_base_url": "https://d.example/",
                                   "fragments": [{"url": "https://e.example/f1"}, {"path": "f2"}]}],
            "requested_subtitles": {"en": {"url": "https://f.example/s"}}}
    hosts = {u.split("/")[2] for u in media_guard.media_urls(info)}
    assert hosts == {"a.example", "b.example", "c.example", "d.example", "e.example", "f.example"}


def test_a_format_on_a_private_address_is_refused_before_download(tmp_path, dns):
    with pytest.raises(media_guard.PrivateMediaHost, match="evil.example.org"):
        _process(tmp_path, _info("http://evil.example.org/a.m4a"))
    assert list(tmp_path.iterdir()) == []                 # the subtitle was not written either


def test_a_manifest_on_a_link_local_address_is_refused(tmp_path, dns):
    with pytest.raises(media_guard.PrivateMediaHost, match="meta.example.org"):
        _process(tmp_path, _info("https://cdn.example.org/a.m4a",
                                 manifest_url="http://meta.example.org/latest/meta-data"))


def test_a_subtitle_on_a_private_address_is_refused(tmp_path, dns):
    with pytest.raises(media_guard.PrivateMediaHost):
        _process(tmp_path, _info("https://cdn.example.org/a.m4a", subs_url="http://evil.example.org/s.json3"))


def test_public_media_passes_and_allow_private_hosts_lifts_the_check(tmp_path, dns, monkeypatch):
    seen = []
    monkeypatch.setattr(yt_dlp.YoutubeDL, "dl", lambda self, name, info, **kw: seen.append(info["url"]) or True)
    _process(tmp_path, _info("https://cdn.example.org/a.m4a"), writesubtitles=False)
    monkeypatch.setattr(web_text, "_allow_private_hosts", True)
    _process(tmp_path, _info("http://evil.example.org/a.m4a"), writesubtitles=False)


def test_subtitle_and_audio_downloads_install_the_guard(monkeypatch, tmp_path):
    from pipeline.sources import SourceRef
    from pipeline.transcription import whisper_local, ytdlp_subs
    from pipeline.transcription.base import TranscriptionError
    installed = []

    class _YDL:
        def __init__(self, opts):
            self.params = opts

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def add_post_processor(self, pp, when):
            installed.append((type(pp).__name__, when))

        def extract_info(self, *a, **k):
            return {"subtitles": {"en": [{"ext": "json3"}]}}

        def process_ie_result(self, *a, **k):
            raise media_guard.PrivateMediaHost("refusing")

        def download(self, urls):
            raise media_guard.PrivateMediaHost("refusing")

    monkeypatch.setattr(ytdlp_subs.yt_dlp, "YoutubeDL", _YDL)
    monkeypatch.setattr(whisper_local.yt_dlp, "YoutubeDL", _YDL)
    with pytest.raises(TranscriptionError, match="refusing"):
        ytdlp_subs.YtDlpSubsProvider().transcribe(SourceRef("video-0123456789abcdef", "video", "https://vimeo.com/1"), ["en"])
    with pytest.raises(TranscriptionError, match="refusing"):
        whisper_local.WhisperLocalProvider().download_audio("https://vimeo.com/1", str(tmp_path))
    assert installed == [("PrivateHostGuard", "video"), ("PrivateHostGuard", "video")]
