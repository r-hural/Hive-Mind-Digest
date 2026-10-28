"""Web pages as sources (IDEA-004): fetch one page, extract the main text with trafilatura and
split each paragraph into sentence-group segments of at most 40 words (textseg.py; start = end = 0.0:
a page has no timeline).

Fetched pages are untrusted: only http(s) URLs and redirects, HTML content types only, a hard body
cap, a per-socket timeout and an overall deadline for the whole download; the content is parsed, never executed. All network access is in `fetch_html`
(through `_open`), which tests replace."""
from __future__ import annotations

import codecs
import http.client
import ipaddress
import re
import socket
import time
import urllib.error
import urllib.request
from typing import Callable
from urllib.parse import urlsplit

import trafilatura

from pipeline.sources import SourceRef
from pipeline.transcription.base import Segment, Transcript, TranscriptionError
from pipeline.transcription.textseg import MAX_SEGMENT_WORDS, MAX_SOURCE_WORDS, split_paragraphs

MAX_HTML_BYTES = 5 * 1024 * 1024
FETCH_TIMEOUT = 30.0
MIN_WORDS = 50
_USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
               "Chrome/126.0 Safari/537.36")
_HTML_TYPES = ("text/html", "application/xhtml+xml")
_FETCH_CHUNK = 64 * 1024


class _HttpOnlyRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urlsplit(newurl).scheme.lower() not in ("http", "https"):
            raise urllib.error.URLError(f"redirect to a non-http(s) URL refused: {newurl!r}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


# Private and local targets (SSRF): every connection, including each redirect hop, resolves its host,
# refuses loopback / private / link-local / multicast / reserved / unspecified addresses and dials the
# address it checked, so a second lookup (DNS rebinding) cannot swap it. `transcription.allow_private_hosts`
# lifts the check for intranet sources.
_allow_private_hosts = False
_resolver = socket.getaddrinfo      # tests replace


def set_allow_private_hosts(allow: bool) -> None:
    global _allow_private_hosts
    _allow_private_hosts = bool(allow)


_NAT64_PREFIX = ipaddress.ip_network("64:ff9b::/96")


def _is_private(ip: str) -> bool:
    addr = ipaddress.ip_address(ip.split("%")[0])
    if isinstance(addr, ipaddress.IPv6Address):
        # An IPv4 address wrapped in IPv6 (mapped, 6to4, Teredo, NAT64) is judged as the IPv4 address.
        if addr.ipv4_mapped is not None:
            return _is_private(str(addr.ipv4_mapped))
        if addr.sixtofour is not None:
            return _is_private(str(addr.sixtofour))
        if addr.teredo is not None:
            return any(_is_private(str(a)) for a in addr.teredo)
        if addr in _NAT64_PREFIX:
            return _is_private(str(ipaddress.IPv4Address(int(addr) & 0xFFFFFFFF)))
    return (not addr.is_global or addr.is_loopback or addr.is_private or addr.is_link_local
            or addr.is_multicast or addr.is_reserved or addr.is_unspecified)


def _vet(host: str, port: int) -> list[tuple]:
    """The resolved (family, type, proto, canonname, sockaddr) entries of host, all public."""
    try:
        infos = _resolver(host, port, type=socket.SOCK_STREAM)
    except OSError as e:
        raise urllib.error.URLError(f"could not resolve {host}: {e}") from e
    if not infos:
        raise urllib.error.URLError(f"could not resolve {host}")
    if not _allow_private_hosts and any(_is_private(i[4][0]) for i in infos):
        raise urllib.error.URLError(f"refusing to fetch a private or local address: {host}")
    return infos


def _dial(family: int, stype: int, proto: int, addr: tuple, timeout) -> socket.socket:
    sock = socket.socket(family, stype, proto)
    try:
        if timeout is not socket._GLOBAL_DEFAULT_TIMEOUT:
            sock.settimeout(timeout)
        sock.connect(addr)
    except OSError:
        sock.close()
        raise
    return sock


def _connect_vetted(host: str, port: int, timeout):
    err: OSError | None = None
    for family, stype, proto, _canon, addr in _vet(host, port):
        try:
            return _dial(family, stype, proto, addr, timeout)
        except OSError as e:
            err = e
    raise err or OSError(f"could not connect to {host}")


class _VettedHTTPConnection(http.client.HTTPConnection):
    def connect(self):
        self.sock = _connect_vetted(self.host, self.port, self.timeout)


class _VettedHTTPSConnection(http.client.HTTPSConnection):
    def connect(self):
        sock = _connect_vetted(self.host, self.port, self.timeout)
        self.sock = self._context.wrap_socket(sock, server_hostname=self.host)   # SNI and certificate: the name


def _open_checked(handler, req, pinned, plain, **kw):
    if req.has_proxy() or getattr(req, "_tunnel_host", None):
        # The proxy resolves and connects, so only a name check is possible here (it cannot be pinned);
        # with a proxy configured, the proxy's own egress policy is the control against rebinding.
        if not _allow_private_hosts:
            parts = urlsplit(req.full_url)
            _vet(parts.hostname or "", parts.port or (443 if parts.scheme == "https" else 80))
        return handler.do_open(plain, req, **kw)
    return handler.do_open(pinned, req, **kw)    # _vet lets private targets through when allowed


class _VettedHTTPHandler(urllib.request.HTTPHandler):
    def http_open(self, req):
        return _open_checked(self, req, _VettedHTTPConnection, http.client.HTTPConnection)


class _VettedHTTPSHandler(urllib.request.HTTPSHandler):
    def https_open(self, req):
        return _open_checked(self, req, _VettedHTTPSConnection, http.client.HTTPSConnection,
                             context=self._context)


def _open(req: urllib.request.Request, timeout: float):
    opener = urllib.request.build_opener(_HttpOnlyRedirect, _VettedHTTPHandler, _VettedHTTPSHandler)
    return opener.open(req, timeout=timeout)


def fetch_bytes(url: str, *, accept: str, type_ok: Callable[[str], bool], what: str, max_bytes: int,
                timeout: float = FETCH_TIMEOUT, label: str = "web_text") -> tuple[str, bytes, str, str | None]:
    """Guarded download shared by the web and PDF extractors: only http(s) URLs and redirects, a
    content-type check, a hard body cap (max+1 bytes read) and two time limits: `timeout` per
    socket operation, and the same number of seconds for the whole download (a server trickling
    bytes would never trip the socket timeout). Every network or HTTP-protocol failure becomes a
    TranscriptionError. Returns (final URL, body, content type, charset)."""
    if urlsplit(url).scheme.lower() not in ("http", "https"):
        raise TranscriptionError(f"{label}: only http(s) URLs are fetched: {url!r}")
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT, "Accept": accept})
    deadline = time.monotonic() + timeout
    try:
        with _open(req, timeout) as resp:
            ctype = (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            if not type_ok(ctype):
                raise TranscriptionError(f"{label}: not {what} (content type {ctype or 'unknown'})")
            chunks: list[bytes] = []
            size = 0
            while size <= max_bytes:
                chunk = resp.read(min(_FETCH_CHUNK, max_bytes + 1 - size))
                if not chunk:
                    break
                chunks.append(chunk)
                size += len(chunk)
                if time.monotonic() > deadline:
                    raise TranscriptionError(f"{label}: download from {url} took longer than {timeout:g} s")
            body = b"".join(chunks)
            final_url = resp.geturl()
            charset = resp.headers.get_content_charset() if hasattr(resp.headers, "get_content_charset") else None
            if charset is None:
                m = re.search(r"charset=([\w-]+)", resp.headers.get("Content-Type") or "", re.I)
                charset = m.group(1) if m else None
    except TranscriptionError:
        raise
    except urllib.error.HTTPError as e:
        raise TranscriptionError(f"{label}: HTTP {e.code} for {url}") from e
    except (urllib.error.URLError, http.client.HTTPException, OSError, ValueError) as e:
        raise TranscriptionError(f"{label}: could not fetch {url}: {e}") from e
    if len(body) > max_bytes:
        raise TranscriptionError(f"{label}: download is larger than {max_bytes} bytes")
    return final_url, body, ctype, charset


# Ordinary text encodings only. The server picks the charset, and codecs such as punycode, idna,
# unicode_escape or utf-7 decode a large body in quadratic time or are not text encodings at all.
_TEXT_CODEC_RE = re.compile(
    r"^(utf[-_]?(8|16|32)([-_](le|be|sig))?|ascii|iso8859-\d+|cp125\d|cp932|cp949|cp950|cp874|cp866|"
    r"koi8-[ru]|mac[-_]\w+|shift_jis\w*|euc[-_]\w+|gb2312|gbk|gb18030|big5\w*|iso2022[-_]\w+)$")


def _decode_html(body: bytes, charset: str | None) -> str:
    """Decode with the response charset when it is an ordinary text encoding, else as UTF-8."""
    try:
        name = codecs.lookup(charset).name if charset else "utf-8"
        if _TEXT_CODEC_RE.match(name):
            return body.decode(name, errors="replace")
    except (LookupError, UnicodeError, ValueError):
        pass
    return body.decode("utf-8", errors="replace")


def fetch_html(url: str, *, max_bytes: int = MAX_HTML_BYTES, timeout: float = FETCH_TIMEOUT) -> tuple[str, str]:
    """Return (final URL after redirects, decoded HTML)."""
    final_url, body, _ctype, charset = fetch_bytes(
        url, accept="text/html,application/xhtml+xml;q=0.9", type_ok=lambda c: c in _HTML_TYPES,
        what="an HTML page", max_bytes=max_bytes, timeout=timeout)
    return final_url, _decode_html(body, charset)


def _date_yyyymmdd(value: str | None) -> str | None:
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})", value or "")
    return "".join(m.groups()) if m else None


def _metadata(html: str, url: str) -> dict:
    meta = trafilatura.extract_metadata(html, default_url=url)
    host = urlsplit(url).hostname or ""
    return {"title": (getattr(meta, "title", None) or None),
            "site": (getattr(meta, "sitename", None) or host or None),
            "date": _date_yyyymmdd(getattr(meta, "date", None))}


def html_to_segments(html: str, url: str) -> tuple[list[Segment], dict]:
    text = trafilatura.extract(html, url=url, output_format="txt", include_comments=False,
                               include_tables=True, favor_precision=True) or ""
    words = len(text.split())
    if words < MIN_WORDS:
        raise TranscriptionError(f"web_text: no main text found at {url} ({words} words extracted, "
                                 f"need {MIN_WORDS}); the page may need a login or JavaScript")
    if words > MAX_SOURCE_WORDS:
        raise TranscriptionError(f"web_text: page text has more than {MAX_SOURCE_WORDS} words")
    segments: list[Segment] = []
    # trafilatura's txt output puts each paragraph (or list item, heading, table row) on its own
    # line, with no blank line between them: split on every line break, or paragraphs merge.
    for paragraph in text.splitlines():
        paragraph = " ".join(paragraph.split())
        if not paragraph:
            continue
        pieces = split_paragraphs(paragraph, MAX_SEGMENT_WORDS)
        segments.extend(Segment(0.0, 0.0, p) for p in pieces)
    return segments, _metadata(html, url)


def page_metadata(url: str) -> dict:
    """title / channel (= site name, else host) / upload_date (YYYYMMDD or None) for a source row."""
    final_url, html = fetch_html(url)
    meta = _metadata(html, final_url)
    return {"title": meta["title"], "channel": meta["site"], "upload_date": meta["date"]}


class WebTextExtractor:
    name = "web_text"
    kinds = frozenset({"web"})

    def transcribe(self, ref: SourceRef, languages: list[str]) -> Transcript:
        if ref.kind not in self.kinds:
            raise TranscriptionError(f"web_text: cannot handle {ref.kind} sources")
        final_url, html = fetch_html(ref.url)
        segments, meta = html_to_segments(html, final_url)
        return Transcript(video_id=ref.key, method=self.name, language=languages[0] if languages else "",
                          segments=segments, created_at="",
                          extra={"kind": "web", "title": meta["title"], "site": meta["site"], "date": meta["date"]})
