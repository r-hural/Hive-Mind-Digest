"""The private-address check for yt-dlp downloads (release security review, 2026-10-28).

A `video` source's page is on one of the known video sites (sources.VIDEO_HOSTS), but the formats,
manifests and subtitle tracks its extractor resolves can point at any host. PrivateHostGuard runs in
yt-dlp's "video" stage (after format and subtitle selection, before any subtitle is written or media
downloaded) and refuses a URL whose host resolves to a loopback, private, link-local or otherwise
non-public address, with web_text's rule (`transcription.allow_private_hosts` lifts it).

What it does not cover, by design (documented in the README, transcription.allow_private_hosts): HTTP redirects and HLS
segment URLs that yt-dlp's downloader and ffmpeg follow later, a second DNS lookup by yt-dlp
(rebinding), and the extractor's own requests to the video site.
"""
from __future__ import annotations

from urllib.parse import urlsplit

from yt_dlp.postprocessor.common import PostProcessor


class PrivateMediaHost(Exception):
    """Not a yt-dlp PostProcessingError: yt-dlp logs those in the "video" stage and downloads anyway."""


def media_urls(info: dict) -> list[str]:
    """Every absolute URL yt-dlp would fetch for `info`: the chosen format(s), their manifests,
    fragment bases and fragments, and the chosen subtitle tracks."""
    urls: list[str] = []

    def add(d: dict) -> None:
        for key in ("url", "manifest_url", "fragment_base_url"):
            if isinstance(d.get(key), str):
                urls.append(d[key])
        for frag in d.get("fragments") or []:
            if isinstance(frag, dict) and isinstance(frag.get("url"), str):
                urls.append(frag["url"])

    add(info)
    for fmt in info.get("requested_formats") or []:
        if isinstance(fmt, dict):
            add(fmt)
    for sub in (info.get("requested_subtitles") or {}).values():
        if isinstance(sub, dict):
            add(sub)
    return [u for u in urls if urlsplit(u).scheme in ("http", "https")]


def check_media_urls(info: dict) -> None:
    from pipeline.transcription import web_text      # lazy: keeps trafilatura off the video start path
    seen: set[tuple[str, int]] = set()
    for url in media_urls(info):
        parts = urlsplit(url)
        target = (parts.hostname or "", parts.port or (443 if parts.scheme == "https" else 80))
        if target in seen:
            continue
        seen.add(target)
        try:
            web_text._vet(*target)
        except OSError as e:          # urllib.error.URLError: private, or not resolvable
            raise PrivateMediaHost(f"refusing a media URL on {target[0]}: {getattr(e, 'reason', e)}") from e


class PrivateHostGuard(PostProcessor):
    def run(self, info):
        check_media_urls(info)
        return [], info


def install(ydl) -> None:
    """Add the guard to a YoutubeDL before it downloads anything."""
    ydl.add_post_processor(PrivateHostGuard(ydl), when="video")
