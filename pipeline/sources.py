"""Source identity (MILESTONE-2 Track B, DECISION-031): what kind a source is, its key, its URL.

The key is what `sources.video_id` stores (the column keeps its historical name): the YouTube id
for YouTube, `<kind>-<16 hex>` for every other kind -- a hash of the normalised URL for remote
sources, of the file content for local files -- so the same page or file given twice is one
source, and the transcript cache path `<key>/<method>.json` is always filesystem-safe."""
from __future__ import annotations

import hashlib
import ntpath
import os
import pathlib
import re
from dataclasses import dataclass
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.request import url2pathname

KINDS = ("youtube", "video", "web", "pdf", "text", "media")
TEXT_KINDS = frozenset({"web", "pdf", "text"})
MEDIA_KINDS = frozenset({"youtube", "video", "media"})


def kind_of(src: dict) -> str:
    """A source row's kind; rows stored before kinds existed have none and are YouTube."""
    return src.get("kind") or "youtube"

YOUTUBE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}\Z")
SOURCE_KEY_RE = re.compile(r"^(?:[A-Za-z0-9_-]{11}|(?:video|web|pdf|text|media)-[0-9a-f]{16})\Z")

VIDEO_HOSTS = frozenset({"vimeo.com", "player.vimeo.com", "ted.com", "www.ted.com", "loom.com",
                         "www.loom.com", "dailymotion.com", "www.dailymotion.com"})
MEDIA_EXTENSIONS = frozenset({".mp3", ".m4a", ".wav", ".flac", ".ogg", ".opus", ".mp4", ".mkv",
                              ".mov", ".webm", ".avi"})
TEXT_EXTENSIONS = frozenset({".txt", ".md", ".markdown"})
# Size caps for local files, per kind: classify_input refuses a larger file before hashing it, and
# the extractors (pdf_text, local_text, whisper_local) read at most this many bytes.
MAX_FILE_BYTES = {"pdf": 50 * 1024 * 1024, "text": 10 * 1024 * 1024, "media": 500 * 1024 * 1024}
_TRACKING_PARAMS = re.compile(r"^(?:utm_.*|fbclid|gclid)\Z", re.I)


def is_source_key(key: str) -> bool:
    return bool(SOURCE_KEY_RE.match(key or ""))


@dataclass(frozen=True)
class SourceRef:
    key: str
    kind: str
    url: str

    @property
    def path(self) -> str | None:
        if not self.url.startswith("file:"):
            return None
        parts = urlsplit(self.url)
        if parts.netloc and parts.netloc.lower() != "localhost":
            return None     # a network share: never opened (Windows would authenticate to the server)
        local = url2pathname(parts.path)
        if is_network_path(local):  # file:////host/share/x carries the server in its path
            return None
        return str(pathlib.Path(local).resolve())

    @classmethod
    def from_row(cls, row: dict) -> "SourceRef":
        return cls(row["video_id"], row.get("kind") or "youtube", row["url"])


def is_network_path(path: str, *, windows: bool = os.name == "nt") -> bool:
    """True unless `path` is plainly local. Opening a share (\\\\host\\share\\..., //host/...,
    \\\\host@SSL\\... WebDAV), even to check that it exists, makes Windows connect and authenticate
    to that host with the user's credentials, outside the web fetcher's address checks. On Windows
    this fails closed: only a path on a drive letter (C:\\...) is local; device and NT paths
    (\\\\?\\..., \\\\.\\..., \\??\\...) and rooted paths without a drive are refused, because Windows
    can turn them into a share (\\\\.\\C:\\..\\UNC\\host\\share). Elsewhere a leading // is refused."""
    if windows:
        return not re.fullmatch(r"[A-Za-z]:", ntpath.splitdrive(path)[0])
    return path.startswith("//")


def normalize_url(url: str) -> str:
    parts = urlsplit(url.strip())
    host = (parts.hostname or "").lower()
    if ":" in host:
        host = f"[{host}]"
    try:
        port = parts.port
    except ValueError:
        port = None
    if port and port != {"http": 80, "https": 443}.get(parts.scheme.lower()):
        host = f"{host}:{port}"
    query = urlencode([(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
                       if not _TRACKING_PARAMS.match(k)])
    path = parts.path
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")
    path = re.sub(r"%[0-9a-fA-F]{2}", lambda m: m.group(0).upper(), path)
    return urlunsplit((parts.scheme.lower(), host, path, query, ""))


def hashed_key(kind: str, identity: str) -> str:
    return f"{kind}-{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:16]}"


def file_identity(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


class OutsideListError(ValueError):
    """A local path in an --input list that resolves outside the list's folder. The whole list is
    refused, not the line skipped: a downloaded list must not pull in any local file."""


def _inside(path: str, folder: str) -> bool:
    real, base = os.path.normcase(os.path.realpath(path)), os.path.normcase(os.path.realpath(folder))
    try:
        return os.path.commonpath([real, base]) == base
    except ValueError:                  # another drive
        return False


def _classify_url(line: str) -> tuple[str, SourceRef | None]:
    """(unquoted text, the ref of a URL line or None for a local path); ValueError for a URL hmd
    does not take."""
    from pipeline.stages.fetch import YOUTUBE_HOSTS, parse_youtube_url   # local import: fetch imports this module
    from pipeline.transcription.base import watch_url
    text = line.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":     # a line copied from a shell
        text = text[1:-1].strip()
    try:
        vid = parse_youtube_url(text)
        return text, SourceRef(vid, "youtube", watch_url(vid))
    except ValueError:
        pass
    scheme = urlsplit(text).scheme.lower()
    if scheme in ("http", "https") and (urlsplit(text).hostname or "").lower() in YOUTUBE_HOSTS:
        # a playlist, channel or malformed watch URL: never a web page to scrape
        raise ValueError(f"not a single YouTube video; use --playlist for playlists: {text!r}")
    if scheme in ("http", "https"):
        url = normalize_url(text)
        parts = urlsplit(url)
        if not parts.netloc:
            raise ValueError(f"not a URL: {text!r}")
        if parts.hostname in VIDEO_HOSTS:
            kind = "video"
        elif parts.path.lower().endswith(".pdf"):
            kind = "pdf"
        else:
            kind = "web"
        return text, SourceRef(hashed_key(kind, url), kind, url)
    if scheme and len(scheme) > 1:                                # "C:" is a drive letter, not a scheme
        raise ValueError(f"unsupported URL scheme {scheme!r} in {text!r}")
    return text, None


def _local_path(text: str, base_dir: str | None, confine: bool) -> str:
    """The path a local-file line names, checked before any file access."""
    path = text if os.path.isabs(text) else os.path.join(base_dir or os.getcwd(), text)
    if is_network_path(path):     # checked before any file access: opening it already connects
        raise ValueError(f"network share paths are not supported (copy the file locally): {text!r}")
    if confine and base_dir and not _inside(path, base_dir):
        raise OutsideListError(f"local file outside the input list's folder {base_dir}: {text!r}; "
                               f"move it next to the list, or add it with --file")
    return path


def check_list_paths(lines: list[str], base_dir: str) -> None:
    """Raise OutsideListError for the first local path in `lines` (an --input list) outside
    `base_dir`. Reads no file, so a caller can refuse the list before anything else runs; any other
    bad line is left to classify_input (which skips it with a warning)."""
    for line in lines:
        try:
            text, ref = _classify_url(line)
            if ref is None:
                _local_path(text, base_dir, confine=True)
        except OutsideListError:
            raise
        except ValueError:
            pass


def classify_input(line: str, *, base_dir: str | None = None, confine: bool = False) -> SourceRef:
    """`base_dir` resolves a relative local path; with `confine` (the lines of an --input list) a
    local path must also resolve inside `base_dir`, symlinks followed, else OutsideListError."""
    text, ref = _classify_url(line)
    if ref is not None:
        return ref
    path = _local_path(text, base_dir, confine)
    if not os.path.isfile(path):
        raise ValueError(f"not a URL and not an existing file: {text!r}")
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        kind = "pdf"
    elif ext in TEXT_EXTENSIONS:
        kind = "text"
    elif ext in MEDIA_EXTENSIONS:
        kind = "media"
    else:
        raise ValueError(f"unsupported file type {ext or '(none)'!r}: {text!r}")
    size = os.path.getsize(path)
    if size > MAX_FILE_BYTES[kind]:
        raise ValueError(f"file is larger than {MAX_FILE_BYTES[kind]} bytes ({size} bytes): {text!r}")
    return SourceRef(hashed_key(kind, file_identity(path)), kind, pathlib.Path(path).resolve().as_uri())
