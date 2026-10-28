"""Per-kind links and locator labels for a cited claim (IDEA-202).

A source row's `kind` (default "youtube" for rows without one) decides how a claim's anchor is
turned into a link: a timestamp for YouTube/Vimeo, a text fragment for web pages, a page for PDFs,
the bare URL otherwise. Untrusted text is escaped by the callers (`_md_link_text`); URLs here are
stored by our own code, but everything outside URL syntax (spaces, `(`, `)`, `|`, `<`, `>`,
non-ASCII) is still percent-encoded so a URL can never end the Markdown link or table cell it sits in.
"""
from __future__ import annotations

import urllib.parse
from urllib.parse import urlsplit

from pipeline.schemas import Anchor
from pipeline.sources import MEDIA_KINDS, kind_of

_VIMEO_HOSTS = ("vimeo.com", "www.vimeo.com", "player.vimeo.com")
_NOUNS = {"youtube": ("video", "videos"), "video": ("video", "videos"), "media": ("recording", "recordings"),
          "web": ("article", "articles"), "pdf": ("PDF", "PDFs"), "text": ("document", "documents")}


_URL_SAFE = ":/?#[]@!$&'*+,;=%~"       # URL syntax and existing %-escapes stay as they are


def safe_url(url: str) -> str:
    """The URL percent-encoded outside URL syntax (spaces, `(`, `)`, `|`, `<`, `>`, non-ASCII), safe
    inside Markdown link syntax and table cells."""
    return urllib.parse.quote(url, safe=_URL_SAFE)


_LIST_MARKERS = frozenset({"-", "*", "+", "•"})


def text_fragment(quote: str) -> str:
    """The first 8 words of the quote as a text fragment. Page text is extracted with list
    bullets and table pipes that the browser never shows, and a fragment cannot match across
    table cells: a leading bullet is skipped and the words stop at the first cell boundary."""
    words: list[str] = []
    for w in quote.replace("|", " | ").split():
        if w == "|":
            if words:
                break
            continue
        if not words and w in _LIST_MARKERS:
            continue
        words.append(w)
        if len(words) == 8:
            break
    return urllib.parse.quote(" ".join(words), safe="").replace("-", "%2D")


def source_link(src: dict, anchor: Anchor, quote: str) -> str:
    kind = kind_of(src)
    if kind == "youtube":
        from pipeline.stages.output import deep_link   # lazy: output imports this module
        return deep_link(src["video_id"], anchor.t_start)
    url = safe_url(src["url"])
    if kind == "video":
        host = (urlsplit(src["url"]).hostname or "").lower()
        return f"{url}#t={int(anchor.t_start)}s" if host in _VIMEO_HOSTS else url
    if kind == "web":
        frag = text_fragment(quote)
        return f"{url}#:~:text={frag}" if frag else url
    if kind == "pdf" and anchor.page is not None:
        return f"{url}#page={anchor.page}"
    return url


def locator_label(src: dict, anchor: Anchor) -> str:
    kind = kind_of(src)
    if kind in MEDIA_KINDS:
        from pipeline.stages.output import fmt_time
        return fmt_time(anchor.t_start)
    if kind == "pdf":
        return f"p. {anchor.page}" if anchor.page is not None else "PDF"
    if anchor.segment is not None:
        return f"¶ {anchor.segment + 1}"
    return "text"


def kind_noun(kind: str, n: int) -> str:
    one, many = _NOUNS.get(kind, ("source", "sources"))
    return one if n == 1 else many
