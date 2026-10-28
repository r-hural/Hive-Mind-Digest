"""Fetch stage: inputs -> source rows with canonical URLs (IDEA-206), flat playlist metadata and a
lazy per-source metadata worker that stores upload_date (IDEA-207).

Two steps: `fetch_stage` resolves the input to video ids (one yt-dlp call for a playlist, none
for URLs) and registers rows; `fetch_source_metadata` fills title/channel/duration/upload_date
per row and runs in the runner's thread pool.
"""
from __future__ import annotations

import os
import re
from typing import Callable
from urllib.parse import parse_qs, unquote, urlparse

import yt_dlp

from pipeline import db
from pipeline.sources import SourceRef, classify_input
from pipeline.transcription.base import watch_url

YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com", "youtu.be",
                 "www.youtu.be", "youtube-nocookie.com", "www.youtube-nocookie.com"}
from pipeline.sources import YOUTUBE_ID_RE as VIDEO_ID_RE
PLAYLIST_ID_RE = re.compile(r"^[A-Za-z0-9_-]{10,64}\Z")
PLAYLIST_LIMIT = 200

# Mirrors pipeline/transcription/ytdlp_subs.py's robustness options for the metadata-only calls
# this module makes (no subtitle download here, so ignore_no_formats_error only applies to the
# per-source metadata call, where a video with no playable formats still has metadata).
_YDL_BASE_OPTS = {"quiet": True, "no_warnings": True, "socket_timeout": 30, "retries": 3, "extractor_retries": 3}


class _YdlLogger:
    """yt-dlp's own report_error() writes to stderr regardless of quiet/no_warnings unless a
    logger object is supplied in opts (its debug/warning/info hooks stay silent by design here;
    only error is worth surfacing). `log=None` makes error() a no-op: fetch_source_metadata
    already logs its own line with the video id on failure, so it passes None here to avoid
    logging the same failure twice; resolve_playlist has no such line of its own, so it passes
    its real log callable through."""

    def __init__(self, log: Callable[[str], None] | None):
        self._log = log

    def debug(self, msg: str) -> None:
        pass

    def info(self, msg: str) -> None:
        pass

    def warning(self, msg: str) -> None:
        pass

    def error(self, msg: str) -> None:
        if self._log is not None:
            self._log(f"  yt-dlp: {msg}")


def _ydl_opts(log: Callable[[str], None] | None, **extra) -> dict:
    return {**_YDL_BASE_OPTS, "logger": _YdlLogger(log), **extra}


def _youtube_host(u, url: str, kind: str) -> str:
    """Scheme check, non-ASCII netloc guard and host-allowlist lookup shared by
    parse_youtube_url and parse_playlist_url. `kind` ("video" or "playlist") only shapes the
    error message.

    The ASCII check runs against the raw netloc, not u.hostname: urlparse's `.hostname` property
    already lowercases with str.lower(), which would turn a non-ASCII codepoint (e.g. the Kelvin
    sign U+212A) into a plain ASCII "k" *before* we ever see it -- letting it slip into the
    allowlist if we checked after.
    """
    if not u.netloc.isascii():
        raise ValueError(f"not a YouTube {kind} URL: {url!r}")
    host = (u.hostname or "").lower()
    if u.scheme not in ("http", "https") or host not in YOUTUBE_HOSTS:
        raise ValueError(f"not a YouTube {kind} URL: {url!r}")
    return host


def parse_youtube_url(url: str) -> str:
    """Return the 11-char video id or raise ValueError. Host must be YouTube; no substring matching."""
    u = urlparse(url.strip())
    host = _youtube_host(u, url, "video")
    path = u.path.rstrip("/")
    parts = [p for p in path.split("/") if p]
    if host in ("youtu.be", "www.youtu.be"):
        vid = parts[0] if parts else ""
    elif path == "/watch":
        vid = parse_qs(u.query).get("v", [""])[0]
    elif parts and parts[0] in ("embed", "v", "shorts", "live") and len(parts) > 1:
        vid = parts[1]
    else:
        raise ValueError(f"not a YouTube video URL: {url!r}")
    if not VIDEO_ID_RE.match(vid):
        raise ValueError(f"no video id in {url!r}")
    return vid


def canonical_url(video_id: str) -> str:
    """The single choke point for IDEA-206: build the watch URL yt-dlp is given, and validate
    the id shape here so every caller in this module (and any future one) is provably safe
    without re-checking. Raises ValueError on anything that isn't an 11-char video id."""
    if not VIDEO_ID_RE.match(video_id):
        raise ValueError(f"not a YouTube video id: {video_id!r}")
    return watch_url(video_id)


def parse_playlist_url(url: str) -> str:
    """Return the playlist id or raise ValueError. Host must be YouTube; the id must appear in
    a `list=` query parameter (either a bare playlist URL or a video URL with a playlist)."""
    u = urlparse(url.strip())
    _youtube_host(u, url, "playlist")
    pid = parse_qs(u.query).get("list", [""])[0]
    if not PLAYLIST_ID_RE.match(pid):
        raise ValueError(f"no playlist id in {url!r}")
    return pid


def read_url_file(path: str) -> list[str]:
    """One URL per line; '#' comments and blank lines skipped. utf-8-sig tolerates a Notepad BOM;
    str.strip() also drops the trailing \\r from Windows line endings."""
    with open(path, encoding="utf-8-sig") as f:
        return [line.strip() for line in f if line.strip() and not line.lstrip().startswith("#")]


def resolve_playlist(playlist_url: str, limit: int = PLAYLIST_LIMIT,
                     log: Callable[[str], None] = print) -> list[dict]:
    """Flat extraction: one round trip for the whole playlist; entries carry title, uploader,
    duration. Any yt-dlp failure (network, private/deleted playlist) propagates unchanged.
    A successful call that yields zero usable video entries -- an empty playlist, or one where
    every entry lacks a valid id -- raises ValueError naming the playlist id rather than
    returning [].

    The cap uses yt-dlp's `playlist_items` ("1:N", 1-based, inclusive); the old `playlistend`
    option is deprecated. yt-dlp applies it in its playlist processing even with
    extract_flat=True; the slice below keeps the cap if a future yt-dlp ever stops honouring it."""
    pid = parse_playlist_url(playlist_url)
    opts = _ydl_opts(log, extract_flat=True, playlist_items=f"1:{limit}")
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(f"https://www.youtube.com/playlist?list={pid}", download=False)
    if not info:
        raise ValueError(f"playlist {pid} returned no entries")
    entries = []
    for e in list(info.get("entries") or [])[:limit]:
        if not isinstance(e, dict):
            # yt-dlp can leave a failed entry as None (or similar) when ignoreerrors swallows it.
            continue
        vid = e.get("id")
        if not vid or not VIDEO_ID_RE.match(vid):
            continue
        duration = e.get("duration")
        entries.append({"video_id": vid, "title": e.get("title"), "channel": e.get("uploader") or e.get("channel"),
                        # 0 is falsy and means "unknown here", not a zero-length video.
                        "duration_seconds": int(duration) if duration else None})
    if not entries:
        raise ValueError(f"playlist {pid} had no usable video entries")
    return entries


def register_refs(db_path: str, notebook_id: int, refs: list[SourceRef],
                  metadata: list[dict] | None = None) -> list[int]:
    """Insert each ref as a source row and return the resulting ids, one per ref and in the same
    order (a duplicate ref produces a duplicate id). `db.add_source`'s ON CONFLICT DO NOTHING means
    an existing row's title/channel/duration are never overwritten by a later call."""
    ids = []
    for i, ref in enumerate(refs):
        meta = metadata[i] if metadata else {}
        ids.append(db.add_source(db_path, notebook_id, ref.key, ref.url, kind=ref.kind,
                                 title=meta.get("title"), channel=meta.get("channel"),
                                 duration_seconds=meta.get("duration_seconds"),
                                 upload_date=meta.get("upload_date")))
    return ids


def register_sources(db_path: str, notebook_id: int, entries: list[dict]) -> list[int]:
    """YouTube entries (`video_id` plus optional flat metadata) -> source rows; a thin wrapper over
    register_refs for callers that already hold video ids (evals)."""
    refs = [SourceRef(e["video_id"], "youtube", canonical_url(e["video_id"])) for e in entries]
    return register_refs(db_path, notebook_id, refs, entries)


def fetch_stage(db_path: str, notebook_id: int, *, input_file: str | None = None, url: str | None = None,
                playlist_url: str | None = None, file_path: str | None = None,
                log: Callable[[str], None] = print) -> list[int]:
    """Resolve exactly one of input_file, url, playlist_url or file_path to source rows and register them.

    Lines of input_file (relative paths resolve against the list file's directory) and the single
    url/file_path go through classify_input; a line that cannot be classified is skipped with a
    warning. A playlist stays YouTube-only. Giving more than one input is rejected rather than
    silently picking a precedence order. Returns the source ids in first-seen order, deduped. Any
    yt-dlp failure from the playlist path propagates unchanged; raises ValueError if nothing
    usable was registered at all.
    """
    given = [x for x in (input_file, url, playlist_url, file_path) if x]
    if not given:
        raise ValueError("no input: give --input, --url, --playlist or --file")
    if len(given) > 1:
        raise ValueError("give exactly one of --input, --url, --playlist or --file")
    refs: list[SourceRef] = []
    metadata: list[dict] | None = None
    if playlist_url:
        entries = resolve_playlist(playlist_url, log=log)
        refs = [SourceRef(e["video_id"], "youtube", canonical_url(e["video_id"])) for e in entries]
        metadata = entries
    else:
        if input_file:
            lines = read_url_file(input_file)
            base_dir = os.path.dirname(os.path.abspath(input_file))
        else:
            lines = [url or file_path]
            base_dir = None
        for line in lines:
            try:
                refs.append(classify_input(line, base_dir=base_dir))
            except ValueError as e:
                log(f"  WARNING: Skipping {line!r}: {e}")
    ids = register_refs(db_path, notebook_id, refs, metadata)
    seen: set[int] = set()
    deduped = [i for i in ids if not (i in seen or seen.add(i))]
    if not deduped:
        source_desc = playlist_url or input_file or url or file_path
        raise ValueError(f"no usable sources in input {source_desc!r}")
    log(f"Registered {len(deduped)} sources for notebook {notebook_id}")
    return deduped


def _fetch_web_metadata(db_path: str, src: dict, log: Callable[[str], None]) -> str:
    from pipeline.transcription import web_text      # lazy: keeps trafilatura off non-web start paths
    try:
        meta = web_text.page_metadata(src["url"])
    except Exception as e:
        db.update_source_metadata(db_path, src["id"], status="error", error=f"metadata: {e}")
        log(f"  [{src['video_id']}] metadata ERROR: {e}")
        return "error"
    db.update_source_metadata(db_path, src["id"], title=meta["title"] or src["url"], channel=meta["channel"],
                              upload_date=meta["upload_date"])
    return "done"


def _fetch_document_metadata(db_path: str, src: dict, kind: str, log: Callable[[str], None]) -> str:
    """PDF and local text files: no network or parsing failure is fatal here (the transcribe step
    reports the real error); the title falls back to the PDF's file/URL basename or the text
    file's stem (the same name text_title gives a file without a heading)."""
    from pipeline.transcription import local_text, pdf_text      # lazy: keeps pypdf off other start paths
    ref = SourceRef.from_row(src)
    basename = os.path.basename(ref.path) if ref.path else pdf_text.url_basename(ref.url)
    fallback = basename if kind == "pdf" else (os.path.splitext(basename)[0] or basename)
    channel = urlparse(ref.url).hostname if kind == "pdf" and ref.path is None else None
    try:
        if kind == "pdf":
            title = pdf_text.pdf_title(ref) or fallback
        else:
            title = local_text.text_title(ref.path) if ref.path else fallback
    except Exception as e:  # any failure: the fallback; the transcribe step reports the real error
        log(f"  [{src['video_id']}] metadata fallback to {fallback!r}: {e}")
        title = fallback
    db.update_source_metadata(db_path, src["id"], title=title, channel=channel, status="done")
    return "done"


def fetch_source_metadata(db_path: str, source_id: int, log: Callable[[str], None] = print) -> str:
    """Fill title/channel/duration/upload_date from yt-dlp using the canonical URL. 'done' | 'skipped' | 'error'."""
    src = db.get_source(db_path, source_id)
    if not src:
        raise ValueError(f"source {source_id} not found")
    kind = src.get("kind") or "youtube"
    # Only YouTube and other yt-dlp videos are expected to have an upload date; a page or file
    # may have none, and must not be re-fetched on every run for that.
    needs_date = kind in ("youtube", "video")
    if src["fetch_status"] == "done" and src["title"] and (src["upload_date"] or not needs_date):
        return "skipped"
    if kind == "web":
        return _fetch_web_metadata(db_path, src, log)
    if kind in ("pdf", "text"):
        return _fetch_document_metadata(db_path, src, kind, log)
    if kind == "media":
        # No network: the title is the file stem. The duration is unknown until Whisper has run,
        # so duration_seconds stays None and the transcript coverage rule is skipped.
        name = unquote(os.path.basename(urlparse(src["url"]).path.rstrip("/")))
        title = os.path.splitext(name)[0] or name or src["url"]
        db.update_source_metadata(db_path, source_id, title=title, status="done")
        return "done"
    # log=None: the except block below already logs this failure with the video id, so the
    # adapter's error() stays a no-op here to avoid logging the same failure twice.
    opts = _ydl_opts(None, skip_download=True, ignore_no_formats_error=True,
                     noplaylist=True, playlist_items="1")
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(canonical_url(src["video_id"]) if kind == "youtube" else src["url"],
                                    download=False)
        if not info:
            raise RuntimeError("no metadata returned")
        if info.get("_type") == "playlist" or "entries" in info:
            raise RuntimeError("URL is a playlist, not a single video")
    except Exception as e:
        db.update_source_metadata(db_path, source_id, title=None, channel=None, duration_seconds=None,
                                  upload_date=None, status="error", error=f"metadata: {e}")
        log(f"  [{src['video_id']}] metadata ERROR: {e}")
        return "error"
    duration = info.get("duration")
    db.update_source_metadata(db_path, source_id, title=info.get("title"),
                              channel=info.get("uploader") or info.get("channel"),
                              # 0 is falsy and means "unknown here", not a zero-length video.
                              duration_seconds=int(duration) if duration else None,
                              upload_date=info.get("upload_date"))
    return "done"
