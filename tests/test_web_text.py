import http.client

import pytest
from pipeline import db
from pipeline.sources import SourceRef
from pipeline.stages import fetch
from pipeline.transcription.base import TranscriptionError
from pipeline.transcription import web_text
from pipeline.transcription.textseg import MAX_SEGMENT_WORDS, split_paragraphs

ARTICLE = """<html><head><title>Habit loops explained</title><meta property="og:site_name" content="Example Mag">
<meta name="date" content="2026-03-14"></head><body><nav>Home | About</nav><article><h1>Habit loops explained</h1>
<p>{p1}</p><p>{p2}</p></article><footer>Cookie settings</footer></body></html>""".format(
    p1=" ".join(["Cue routine reward is the basic loop."] * 12), p2=" ".join(["Small changes compound over months."] * 12))


def test_article_becomes_paragraph_segments(monkeypatch):
    monkeypatch.setattr(web_text, "fetch_html", lambda url, **k: (url, ARTICLE))
    t = web_text.WebTextExtractor().transcribe(SourceRef("web-0123456789abcdef", "web", "https://e.org/a"), ["en"])
    assert t.method == "web_text" and len(t.segments) >= 2
    assert all(len(s.text.split()) <= MAX_SEGMENT_WORDS for s in t.segments)
    assert "Cookie settings" not in " ".join(s.text for s in t.segments)
    assert t.extra["title"] == "Habit loops explained" and t.extra["site"] == "Example Mag"


def test_page_without_main_text_is_an_error_not_an_empty_source(monkeypatch):
    monkeypatch.setattr(web_text, "fetch_html", lambda url, **k: (url, "<html><body><p>Please log in.</p></body></html>"))
    with pytest.raises(TranscriptionError, match="no main text"):
        web_text.WebTextExtractor().transcribe(SourceRef("web-0123456789abcdef", "web", "https://e.org/a"), ["en"])


class _Resp:
    """A streaming response: read(n) returns the next n bytes, like http.client's."""

    def __init__(self, ctype, body, url="https://e.org/a"):
        self.headers, self._b, self.url, self._pos = {"Content-Type": ctype}, body, url, 0

    def read(self, n=-1):
        end = len(self._b) if n < 0 else self._pos + n
        chunk, self._pos = self._b[self._pos:end], min(end, len(self._b))
        return chunk

    def geturl(self):
        return self.url

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_fetch_html_refuses_binary_and_oversized(monkeypatch):
    monkeypatch.setattr(web_text, "_open", lambda req, timeout: _Resp("application/zip", b"PK"))
    with pytest.raises(TranscriptionError, match="not an HTML page"):
        web_text.fetch_html("https://e.org/a")
    monkeypatch.setattr(web_text, "_open", lambda req, timeout: _Resp("text/html", b"x" * 20))
    with pytest.raises(TranscriptionError, match="larger than"):
        web_text.fetch_html("https://e.org/a", max_bytes=10)


def test_fetch_html_decodes_with_response_charset(monkeypatch):
    body = "<p>café</p>".encode("latin-1")
    monkeypatch.setattr(web_text, "_open", lambda req, timeout: _Resp("text/html; charset=ISO-8859-1", body))
    assert web_text.fetch_html("https://e.org/a") == ("https://e.org/a", "<p>café</p>")


@pytest.mark.parametrize("target", ["file:///etc/passwd", "ftp://e.org/a.pdf"])
def test_redirect_to_non_http_is_refused(target):
    import urllib.error
    import urllib.request
    h = web_text._HttpOnlyRedirect()
    req = urllib.request.Request("https://e.org/a")
    with pytest.raises(urllib.error.URLError, match="non-http"):
        h.redirect_request(req, None, 302, "Found", {}, target)


class _BrokenResp(_Resp):
    def read(self, n=-1):
        raise http.client.IncompleteRead(b"partial")


def test_broken_download_is_a_transcription_error(monkeypatch):
    monkeypatch.setattr(web_text, "_open", lambda req, timeout: _BrokenResp("text/html", b""))
    with pytest.raises(TranscriptionError, match="could not fetch"):
        web_text.fetch_html("https://e.org/a")

    def bad_status(req, timeout):
        raise http.client.BadStatusLine("garbage")
    monkeypatch.setattr(web_text, "_open", bad_status)
    with pytest.raises(TranscriptionError, match="could not fetch"):
        web_text.fetch_html("https://e.org/a")


@pytest.mark.parametrize("charset", ["undefined", "idna"])
def test_unusable_charset_falls_back_to_utf8(monkeypatch, charset):
    body = "<p>café ÿ</p>".encode("utf-8")
    monkeypatch.setattr(web_text, "_open", lambda req, timeout: _Resp(f"text/html; charset={charset}", body))
    assert web_text.fetch_html("https://e.org/a") == ("https://e.org/a", "<p>café ÿ</p>")


def test_slow_download_stops_at_the_overall_deadline(monkeypatch):
    clock = iter(range(0, 1000, 20))           # every reading of the clock is 20 s later
    monkeypatch.setattr(web_text.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(web_text, "_FETCH_CHUNK", 4)
    monkeypatch.setattr(web_text, "_open", lambda req, timeout: _Resp("text/html", b"x" * 100))
    with pytest.raises(TranscriptionError, match="took longer than 30 s"):
        web_text.fetch_html("https://e.org/a")


def test_download_within_the_deadline_is_read_in_chunks(monkeypatch):
    monkeypatch.setattr(web_text, "_FETCH_CHUNK", 4)
    monkeypatch.setattr(web_text, "_open", lambda req, timeout: _Resp("text/html", b"<p>" + b"x" * 30 + b"</p>"))
    assert web_text.fetch_html("https://e.org/a")[1] == "<p>" + "x" * 30 + "</p>"


def test_non_http_url_is_refused():
    with pytest.raises(TranscriptionError, match="http"):
        web_text.fetch_html("file:///etc/passwd")


def test_long_paragraph_is_split_at_sentence_ends():
    para = " ".join(["This is a sentence of eight words only."] * 40)
    parts = split_paragraphs(para, 120)
    assert len(parts) >= 3 and all(len(p.split()) <= 120 for p in parts)
    assert " ".join(parts) == para


def test_segments_are_sentence_groups_of_at_most_40_words():
    s15 = " ".join(["word"] * 14) + " end."                       # one 15-word sentence
    assert MAX_SEGMENT_WORDS == 40
    parts = split_paragraphs(" ".join([s15] * 3))                  # default cap: 40 words
    assert [len(p.split()) for p in parts] == [30, 15]
    long_sentence = " ".join(["w"] * 50) + "."
    assert [len(p.split()) for p in split_paragraphs(long_sentence)] == [40, 10]
    assert split_paragraphs("Short. Also short?") == ["Short. Also short?"]


def test_web_paragraphs_are_never_merged_into_one_segment(monkeypatch):
    html = ("<html><body><article><p>" + "Alpha beta gamma delta. " * 5 + "</p><p>" + "Epsilon zeta eta theta. " * 5
            + "</p><p>" + "Iota kappa lambda mu. " * 5 + "</p></article></body></html>")
    segments, _meta = web_text.html_to_segments(html, "https://e.org/a")
    assert [s.text.split()[0] for s in segments] == ["Alpha", "Epsilon", "Iota"]


def test_configuration_free_transcribers_are_registered_last():
    from pipeline.config import Config
    from pipeline.transcription.chain import build_transcribers
    ts = build_transcribers(Config())
    assert [t.name for t in ts[-3:]] == ["web_text", "pdf_text", "local_text"] and ts[-3].kinds == frozenset({"web"})


def test_web_metadata_fills_title_site_and_date(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(web_text, "fetch_html", lambda url, **k: calls.append(url) or (url, ARTICLE))
    info = web_text.page_metadata("https://e.org/a")
    assert info == {"title": "Habit loops explained", "channel": "Example Mag", "upload_date": "20260314"}
    monkeypatch.setattr(web_text, "fetch_html", lambda url, **k: (url, "<html><body><p>hi</p></body></html>"))
    assert web_text.page_metadata("https://news.e.org/a")["channel"] == "news.e.org"


@pytest.fixture
def state(tmp_path):
    path = str(tmp_path / "s.sqlite")
    db.init_db(path)
    return path, db.create_notebook(path, "T", "t")


def test_fetch_source_metadata_dispatches_web_and_is_not_refetched(monkeypatch, state):
    path, nid = state
    sid = db.add_source(path, nid, "web-0123456789abcdef", "https://e.org/a", kind="web")
    calls = []
    monkeypatch.setattr(web_text, "fetch_html", lambda url, **k: calls.append(url) or (url, ARTICLE))
    assert fetch.fetch_source_metadata(path, sid, log=lambda m: None) == "done"
    row = db.get_source(path, sid)
    assert (row["title"], row["channel"], row["upload_date"]) == ("Habit loops explained", "Example Mag", "20260314")
    assert fetch.fetch_source_metadata(path, sid, log=lambda m: None) == "skipped"
    assert calls == ["https://e.org/a"]


def test_dateless_page_is_not_refetched(monkeypatch, state):
    path, nid = state
    sid = db.add_source(path, nid, "web-0123456789abcdef", "https://e.org/a", kind="web")
    calls = []
    page = ARTICLE.replace('<meta name="date" content="2026-03-14">', "")
    monkeypatch.setattr(web_text, "fetch_html", lambda url, **k: calls.append(url) or (url, page))
    assert fetch.fetch_source_metadata(path, sid, log=lambda m: None) == "done"
    assert fetch.fetch_source_metadata(path, sid, log=lambda m: None) == "skipped"
    assert len(calls) == 1


def test_web_metadata_failure_is_an_error_status(monkeypatch, state):
    path, nid = state
    sid = db.add_source(path, nid, "web-0123456789abcdef", "https://e.org/a", kind="web")

    def boom(url, **k):
        raise TranscriptionError("web_text: HTTP 404")
    monkeypatch.setattr(web_text, "fetch_html", boom)
    assert fetch.fetch_source_metadata(path, sid, log=lambda m: None) == "error"
    assert "404" in db.get_source(path, sid)["error_message"]


def test_local_kinds_are_marked_done_with_basename_title(state):
    path, nid = state
    sid = db.add_source(path, nid, "pdf-0123456789abcdef", "file:///tmp/dir/Paper.pdf", kind="pdf")
    assert fetch.fetch_source_metadata(path, sid, log=lambda m: None) == "done"
    assert db.get_source(path, sid)["title"] == "Paper.pdf"
    assert fetch.fetch_source_metadata(path, sid, log=lambda m: None) == "skipped"


def test_video_kind_uses_its_own_url_with_ytdlp(monkeypatch, state):
    path, nid = state
    sid = db.add_source(path, nid, "video-0123456789abcdef", "https://vimeo.com/1", kind="video")
    seen = []

    class YDL:
        def __init__(self, opts): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def extract_info(self, url, download=False):
            seen.append(url)
            return {"title": "V", "uploader": "U", "duration": 10, "upload_date": "20260101"}
    monkeypatch.setattr(fetch.yt_dlp, "YoutubeDL", YDL)
    assert fetch.fetch_source_metadata(path, sid, log=lambda m: None) == "done"
    assert seen == ["https://vimeo.com/1"]


def test_extracted_text_over_the_word_cap_is_refused(monkeypatch):
    monkeypatch.setattr(web_text, "MAX_SOURCE_WORDS", 20)
    with pytest.raises(TranscriptionError, match="more than 20 words"):
        web_text.html_to_segments(ARTICLE, "https://e.org/a")


@pytest.mark.parametrize("charset", ["punycode", "idna", "unicode_escape", "raw_unicode_escape", "utf-7"])
def test_non_text_codecs_decode_as_utf8(monkeypatch, charset):
    body = "<p>café ÿ</p>".encode("utf-8")
    monkeypatch.setattr(web_text, "_open", lambda req, timeout: _Resp(f"text/html; charset={charset}", body))
    assert web_text.fetch_html("https://e.org/a") == ("https://e.org/a", "<p>café ÿ</p>")


@pytest.mark.parametrize("charset,text", [("iso-8859-1", "café"), ("windows-1251", "привет"),
                                          ("utf-16", "café ÿ"), ("shift_jis", "日本語"),
                                          ("euc-kr", "한국어"), ("gb2312", "中文"), ("koi8-r", "привет")])
def test_ordinary_charsets_still_decode_with_their_codec(monkeypatch, charset, text):
    body = f"<p>{text}</p>".encode(charset)
    monkeypatch.setattr(web_text, "_open", lambda req, timeout: _Resp(f"text/html; charset={charset}", body))
    assert web_text.fetch_html("https://e.org/a")[1].replace("\ufeff", "") == f"<p>{text}</p>"


# --- private and local fetch targets (F8) ---------------------------------------------------------
import io
import socket
import urllib.request


class _FakeSock:
    def __init__(self, response: bytes):
        self.response, self.sent = response, b""

    def sendall(self, data):
        self.sent += data

    def makefile(self, *a, **k):
        return io.BytesIO(self.response)

    def settimeout(self, t):
        pass

    def close(self):
        pass


def _ok(body=b"<p>hello</p>"):
    return (b"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: %d\r\nConnection: close\r\n\r\n"
            % len(body)) + body


def _net(monkeypatch, hosts, responses):
    """Fake DNS (name -> address; IP literals resolve to themselves) and a fake dial that records
    which address was connected to and answers with the next canned response."""
    dialed, socks = [], []

    def resolver(host, port, type=0):
        ip = hosts.get(host, host)
        fam = socket.AF_INET6 if ":" in ip else socket.AF_INET
        return [(fam, socket.SOCK_STREAM, 6, "", (ip, port))]

    def dial(family, stype, proto, addr, timeout):
        dialed.append(addr[0])
        s = _FakeSock(responses.pop(0))
        socks.append(s)
        return s

    monkeypatch.setattr(web_text, "_resolver", resolver)
    monkeypatch.setattr(web_text, "_dial", dial)
    monkeypatch.setattr(web_text, "_allow_private_hosts", False)
    return dialed, socks


@pytest.mark.parametrize("url", ["http://127.0.0.1/", "http://[::1]/", "http://10.0.0.5/",
                                 "http://169.254.169.254/latest", "http://0.0.0.0/", "http://[::ffff:127.0.0.1]/",
                                 "http://nas.lan/", "http://224.0.0.1/", "http://[64:ff9b::7f00:1]/",
                                 "http://[2002:7f00:1::]/", "http://[2002:a00:5::1]/"])
def test_private_and_local_targets_are_refused(monkeypatch, url):
    dialed, _ = _net(monkeypatch, {"nas.lan": "192.168.1.1"}, [_ok()])
    with pytest.raises(TranscriptionError, match="refusing to fetch a private or local address"):
        web_text.fetch_html(url)
    assert dialed == []


def test_public_host_is_fetched_from_the_checked_address(monkeypatch):
    dialed, socks = _net(monkeypatch, {"example.org": "93.184.216.34"}, [_ok()])
    assert web_text.fetch_html("http://example.org/a") == ("http://example.org/a", "<p>hello</p>")
    assert dialed == ["93.184.216.34"] and b"Host: example.org" in socks[0].sent


def test_redirect_to_loopback_is_refused(monkeypatch):
    redirect = b"HTTP/1.1 302 Found\r\nLocation: http://127.0.0.1:8080/\r\nContent-Length: 0\r\n\r\n"
    dialed, _ = _net(monkeypatch, {"example.org": "93.184.216.34"}, [redirect, _ok()])
    with pytest.raises(TranscriptionError, match="refusing to fetch a private or local address"):
        web_text.fetch_html("http://example.org/a")
    assert dialed == ["93.184.216.34"]


def test_allow_private_hosts_lifts_the_check(monkeypatch):
    dialed, _ = _net(monkeypatch, {"nas.lan": "192.168.1.1"}, [_ok(), _ok()])
    web_text.set_allow_private_hosts(True)
    try:
        assert web_text.fetch_html("http://nas.lan/")[1] == "<p>hello</p>"
        assert web_text.fetch_html("http://127.0.0.1/")[1] == "<p>hello</p>"
    finally:
        web_text.set_allow_private_hosts(False)
    assert dialed == ["192.168.1.1", "127.0.0.1"]


def test_config_key_allow_private_hosts_defaults_to_false():
    from pipeline.config import Config
    assert Config().transcription.allow_private_hosts is False
    assert Config.model_validate({"transcription": {"allow_private_hosts": True}}).transcription.allow_private_hosts
