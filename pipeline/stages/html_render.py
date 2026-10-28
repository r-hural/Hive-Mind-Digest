"""HTML twins of the Markdown reports (IDEA-312).

Every report the output stage writes as `reports/<date>-<format>.md` can get a self-contained
`.html` twin: one file, inline CSS, no script, font, image or stylesheet fetched from anywhere, so it
opens offline in any browser and can be sent on its own. The pages are rendered from the same
RenderContext as the Markdown, never converted from the Markdown text.

Every untrusted or model-written string (source titles, channels, quotes, the TL;DR, section titles,
points, contradiction text) goes through `esc`; the only `href` values are built by `link` from the
URLs the pipeline itself stores, and only for http, https and file. Layout B: a contents sidebar
beside the report. The whole look is one theme block (`<style id="hmd-theme">` between the
THEME_START and THEME_END comments): the markup carries classes only, never a `style` attribute, so
swapping that block restyles a page. The hooks a theme styles are listed at the top of BASE_CSS.
"""
from __future__ import annotations

import html
import re
from dataclasses import dataclass
from urllib.parse import urlsplit

_LINK_SCHEMES = frozenset({"http", "https", "file"})
_SLUG_DROP_RE = re.compile(r"[^a-z0-9]+")


def esc(text: str) -> str:
    """Whitespace collapsed (as in the Markdown reports), then escaped for text and attributes."""
    return html.escape(" ".join(str(text).split()), quote=True)


def link(url: str, label: str) -> str:
    """An `<a>` to `url` when its scheme is http, https or file; otherwise the label as plain text."""
    try:
        scheme = urlsplit(url or "").scheme.lower()
    except ValueError:
        scheme = ""
    if scheme not in _LINK_SCHEMES:
        return esc(label)
    return f'<a class="cite" href="{html.escape(url, quote=True)}">{esc(label)}</a>'


class IdMaker:
    """Unique, ASCII `id` values for one page: a slug of the text, `-2`, `-3`, ... on repeats."""

    def __init__(self) -> None:
        self._seen: dict[str, int] = {}

    def __call__(self, text: str) -> str:
        base = _SLUG_DROP_RE.sub("-", str(text).lower()).strip("-") or "section"
        n = self._seen.get(base, 0) + 1
        self._seen[base] = n
        if n == 1:
            return base
        candidate = f"{base}-{n}"
        while candidate in self._seen:          # "setup-2" may already be a real heading's slug
            n += 1
            candidate = f"{base}-{n}"
        self._seen[base] = n
        self._seen[candidate] = 1
        return candidate


@dataclass
class HtmlPage:
    """One report page before it is wrapped: `header` lines are HTML already escaped by the renderer;
    `sidebar` entries are (level 1 or 2, id, plain-text label); `body` is HTML; `kind` is the report
    format (summary, digest, ...), named on `<body>` as `report-<kind>`."""
    title: str
    header: list[str]
    sidebar: list[tuple[int, str, str]]
    body: str
    kind: str = ""


THEME_START, THEME_END = "<!-- hmd:theme:start -->", "<!-- hmd:theme:end -->"
_NUM_CELL_RE = re.compile(r"^(\d+|\d+ of \d+|\d+(:\d\d){1,2}|★|)$")


BASE_CSS = """
/* Hive Mind Digest report theme "paper" (v1).
   This block is the whole look of the page: replace everything between the hmd:theme markers to
   restyle it. The markup carries no inline style; it uses only these hooks:
   body.report-{summary,digest,hybrid,outline,quotes,diff}, .skip-link, nav.sidebar (#contents, li.level-1/2),
   main#content, .sample-banner, .report-header (h1 .report-kind), nav.report-links (span[aria-current], .contents-link),
   h2-h4 with ids, ul.points, .badge + .badge-{confirmed,sources,partly,disputed,new}, .cites, a.cite,
   details.quotes, blockquote, q.quote, cite.source, p.contradiction, .legend, .note, .sources-line,
   .table-wrap, table, th.num/td.num. */
:root {
  color-scheme: light dark;
  --paper: #fbfaf6; --ink: #1d1c19; --ink-2: #3f3c35; --muted: #5f5b52;
  --rule: #e4e0d5; --rule-strong: #cbc5b6; --panel: #f3f0e7; --row-hover: #f7f5ef;
  --link: #1f4b99; --link-hover: #12326b; --accent: #a4500a; --accent-soft: #f5e4cf;
  --ok: #1d6436; --ok-bg: #e2f0e4; --warn: #744800; --warn-bg: #f8ecc8;
  --bad: #9e2428; --bad-bg: #f8e4e1; --new: #1f4b99; --new-bg: #e3eaf6;
  --serif: Charter, "Bitstream Charter", "Sitka Text", Cambria, "Iowan Old Style", Georgia, serif;
  --sans: system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  --measure: 42rem;
}
@media (prefers-color-scheme: dark) {
  :root {
    --paper: #15140f; --ink: #ebe7dc; --ink-2: #d3cdbf; --muted: #a39e91;
    --rule: #2b2923; --rule-strong: #403c33; --panel: #1d1c16; --row-hover: #1a1914;
    --link: #a3bdf2; --link-hover: #cbdafa; --accent: #eaa45e; --accent-soft: #3d2c17;
    --ok: #86d29b; --ok-bg: #16291d; --warn: #e9c374; --warn-bg: #2e2512;
    --bad: #f39a94; --bad-bg: #351a1a; --new: #a3bdf2; --new-bg: #19243a;
  }
}

*, *::before, *::after { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; scroll-padding-top: 1.5rem; scrollbar-color: var(--rule-strong) var(--paper); }
@media (prefers-reduced-motion: no-preference) { html { scroll-behavior: smooth; } }
body { margin: 0; background: var(--paper); color: var(--ink);
  font: 1.0625rem/1.65 var(--serif); font-kerning: normal;
  display: grid; grid-template-columns: minmax(15rem, 18rem) minmax(0, 1fr); }
::selection { background: var(--accent-soft); color: var(--ink); }
a { color: var(--link); text-decoration-thickness: 1px; text-underline-offset: 0.18em;
  text-decoration-color: color-mix(in srgb, currentColor 40%, transparent); }
a:hover { color: var(--link-hover); text-decoration-color: currentColor; }
:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; border-radius: 2px; }

.skip-link { position: absolute; left: 1rem; top: -4rem; z-index: 2; padding: 0.6rem 0.9rem;
  background: var(--ink); color: var(--paper); font: 600 0.875rem/1 var(--sans); border-radius: 4px; }
.skip-link:focus { top: 1rem; }
.skip-link:hover { color: var(--paper); }

/* contents */
nav.sidebar { grid-column: 1; grid-row: 1; position: sticky; top: 0; height: 100vh; overflow-y: auto;
  padding: 2.5rem 1.25rem 3rem 1.5rem; border-right: 1px solid var(--rule);
  font: 0.8125rem/1.4 var(--sans); scrollbar-width: thin; }
nav.sidebar strong { display: block; margin-bottom: 0.9rem; color: var(--muted);
  font-size: 0.75rem; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; }
nav.sidebar ol { list-style: none; margin: 0; padding: 0; }
nav.sidebar a { display: block; padding: 0.22rem 0; text-decoration: none; color: var(--ink-2); }
nav.sidebar a:hover { color: var(--link); }
nav.sidebar li.level-1 { margin-top: 0.55rem; }
nav.sidebar li.level-1:first-child { margin-top: 0; }
nav.sidebar li.level-1 > a { color: var(--ink); font-weight: 600; }
nav.sidebar li.level-2 { margin-left: 0.15rem; padding-left: 0.7rem; border-left: 1px solid var(--rule); }
nav.sidebar li.level-2 > a { color: var(--muted); }
nav.sidebar li.level-2 > a:hover { color: var(--link); }

/* page */
main { grid-column: 2; min-width: 0; padding: 1rem clamp(1.5rem, 4vw, 3rem) 6rem clamp(2rem, 7vw, 7rem); }
main > * { max-width: var(--measure); }
main > .table-wrap { max-width: 56rem; }
.sample-banner { margin: 0 0 2.25rem; padding: 0.6rem 0.85rem; background: var(--panel);
  border: 1px solid var(--rule); border-radius: 6px; color: var(--muted); font: 0.8125rem/1.5 var(--sans); }
.report-header h1 { margin: 0 0 0.6rem; font-weight: 600; letter-spacing: -0.015em; line-height: 1.12;
  font-size: clamp(1.625rem, 1.2rem + 1.4vw, 2.25rem); text-wrap: balance; }
.report-kind { display: inline-block; color: var(--muted); font-weight: 400; }
.report-header { margin-bottom: 2.75rem; }
.report-header p { margin: 0; color: var(--muted); font: 0.875rem/1.5 var(--sans); font-variant-numeric: tabular-nums; }
nav.report-links { display: flex; flex-wrap: wrap; column-gap: 0.25rem; margin: 0 0 2.5rem;
  border-bottom: 1px solid var(--rule); font: 0.875rem/1 var(--sans); }
nav.report-links a, nav.report-links span { display: inline-flex; align-items: center; min-height: 2.75rem;
  padding: 0 0.7rem; margin-bottom: -1px; border-bottom: 2px solid transparent;
  color: var(--muted); text-decoration: none; }
nav.report-links a:first-child, nav.report-links span:first-child { padding-left: 0; }
nav.report-links a:hover { color: var(--ink); border-bottom-color: var(--rule-strong); }
nav.report-links span[aria-current] { color: var(--ink); font-weight: 600; border-bottom-color: var(--accent); }
nav.report-links .contents-link { display: none; margin-left: auto; }

h2, h3, h4 { line-height: 1.25; text-wrap: balance; font-weight: 600; }
h2 { margin: 3.5rem 0 1rem; padding-top: 1.5rem; border-top: 1px solid var(--rule);
  font-size: 1.625rem; letter-spacing: -0.01em; }
.report-header + h2 { margin-top: 0; padding-top: 0; border-top: 0; }
h3 { margin: 2.25rem 0 0.6rem; font-size: 1.1875rem; }
h4 { margin: 1.75rem 0 0.4rem; color: var(--muted); font: 600 0.75rem/1.3 var(--sans);
  letter-spacing: 0.06em; text-transform: uppercase; }
.report-digest h3:not([id]) { color: var(--muted); font: 600 0.75rem/1.3 var(--sans);
  letter-spacing: 0.06em; text-transform: uppercase; margin-top: 1.75rem; }
:is(h2, h3):target { text-decoration: underline 2px var(--accent); text-underline-offset: 0.3em; }
p { margin: 0 0 1rem; }
ul { margin: 0 0 1.25rem; padding-left: 1.25rem; }
li { margin: 0.35rem 0; }
li::marker { color: var(--muted); }

/* points */
ul.points { list-style: none; padding: 0; }
ul.points > li { margin: 0; padding: 0.85rem 0; border-bottom: 1px solid var(--rule); }
ul.points > li:first-child { padding-top: 0.25rem; }
ul.points > li:last-child { border-bottom: 0; }
.badge { display: inline-block; padding: 0 0.5rem; border-radius: 999px; white-space: nowrap;
  font: 600 0.6875rem/1.65 var(--sans); letter-spacing: 0.02em; vertical-align: 0.12em;
  font-variant-numeric: tabular-nums; }
.badge-confirmed { color: var(--ok); background: var(--ok-bg); }
.badge-sources { color: var(--muted); background: var(--panel); box-shadow: inset 0 0 0 1px var(--rule); }
.badge-partly { color: var(--warn); background: var(--warn-bg); }
.badge-disputed { color: var(--bad); background: var(--bad-bg); }
.badge-new { color: var(--new); background: var(--new-bg); }
.cites { display: block; margin-top: 0.3rem; color: var(--muted); font: 0.8125rem/1.5 var(--sans); }
.cites a.cite { font-variant-numeric: tabular-nums; }

/* quotes */
details.quotes { margin-top: 0.4rem; }
details.quotes summary { width: max-content; cursor: pointer; color: var(--muted); font: 0.8125rem/1.8 var(--sans); }
details.quotes summary:hover { color: var(--ink); }
blockquote { margin: 0.7rem 0 0.2rem; padding: 0.05rem 0 0.05rem 1rem; border-left: 1px solid var(--rule-strong); }
q.quote { color: var(--ink-2); font-style: italic; }
cite.source { display: block; margin-top: 0.2rem; color: var(--muted); font: normal 0.8125rem/1.5 var(--sans); }
.report-quotes main ul { list-style: none; padding: 0; }
.report-quotes main ul > li { margin: 0; padding: 1.1rem 0 1.25rem; border-bottom: 1px solid var(--rule); }
.report-quotes main ul > li:last-child { border-bottom: 0; }
.report-quotes main li > strong { display: block; font-weight: 600; line-height: 1.5; }
.report-quotes blockquote { margin-top: 0.85rem; }
.report-quotes q.quote { font-size: 1.0625rem; }

/* notes, contradictions */
.legend, .note, .sources-line { color: var(--muted); font: 0.8125rem/1.55 var(--sans); }
.legend { margin: 1.5rem 0 0; padding: 0.75rem 1rem; background: var(--panel); border-radius: 6px; }
p.contradiction { margin: 1.25rem 0; padding: 0.9rem 1.1rem; background: var(--bad-bg); border-radius: 6px;
  color: var(--ink); font-size: 1rem; }
p.contradiction strong { color: var(--bad); font: 600 0.875rem/1.5 var(--sans); }

/* tables */
.table-wrap { margin: 1rem 0 1.5rem; overflow-x: auto; border: 1px solid var(--rule); border-radius: 6px; }
table { width: 100%; border-collapse: collapse; font: 0.8125rem/1.45 var(--sans); font-variant-numeric: tabular-nums; }
th, td { padding: 0.55rem 0.75rem; text-align: left; vertical-align: top; border-bottom: 1px solid var(--rule); }
th { background: var(--panel); color: var(--muted); font-weight: 600; }
tbody tr:last-child td { border-bottom: 0; }
tbody tr:hover td { background: var(--row-hover); }
th.num, td.num { text-align: right; white-space: nowrap; }
td a { overflow-wrap: anywhere; }
td > a.cite:only-child { display: inline-block; min-width: 15rem; }

@media (max-width: 900px) {
  body { display: flex; flex-direction: column; font-size: 1rem; }
  main { order: 1; padding: 1.25rem 1rem 3rem; }
  nav.sidebar { order: 2; position: static; height: auto; overflow: visible; padding: 2rem 1rem 3rem;
    border-right: 0; border-top: 1px solid var(--rule); font-size: 0.875rem; }
  nav.sidebar a { padding: 0.45rem 0; }
  nav.report-links .contents-link { display: inline-flex; }
  main { padding-top: 0.5rem; }
  nav.report-links { margin-bottom: 1.75rem; }
  .report-header { margin-bottom: 2rem; }
  table { min-width: 34rem; }
  h2 { font-size: 1.375rem; margin-top: 2.75rem; }
}
@media print {
  body { display: block; background: #fff; color: #000; font-size: 10.5pt; }
  nav.sidebar, nav.report-links, .skip-link { display: none; }
  main { padding: 0; }
  main > *, main > .table-wrap { max-width: none; }
  a { color: inherit; text-decoration: none; }
  details.quotes::details-content { content-visibility: visible; display: block; }
  h2, h3 { break-after: avoid; }
  ul.points > li, blockquote, tr { break-inside: avoid; }
}
"""


def _sidebar(entries: list[tuple[int, str, str]]) -> str:
    if not entries:
        return ""
    items = "".join(f'<li class="level-{1 if level <= 1 else 2}"><a href="#{html.escape(ident, quote=True)}">'
                    f"{esc(label)}</a></li>" for level, ident, label in entries)
    return (f'<nav class="sidebar" id="contents" aria-label="Contents"><strong>Contents</strong>'
            f"<ol>{items}</ol></nav>\n")


def _report_links(links: list[tuple[str, str, bool]], *, contents: bool) -> str:
    """The row of sibling reports; `contents` adds a jump to the sidebar (themes show it on narrow screens,
    where the contents come after the report)."""
    if not links:
        return ""
    parts = [f'<span aria-current="page">{esc(label)}</span>' if current
             else f'<a href="{html.escape(filename, quote=True)}">{esc(label)}</a>'
             for label, filename, current in links]
    if contents:
        parts.append('<a class="contents-link" href="#contents">Contents</a>')
    return f'<nav class="report-links" aria-label="Reports">{"".join(parts)}</nav>\n'


def render_page(page: HtmlPage, report_links: list[tuple[str, str, bool]]) -> str:
    """The complete, self-contained document for one report page."""
    header = "\n".join(page.header)
    kind = _SLUG_DROP_RE.sub("-", page.kind.lower()).strip("-")
    body = f'<body class="report-{kind}">' if kind else "<body>"
    return (
        "<!doctype html>\n"
        '<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{esc(page.title)}</title>\n"
        f'{THEME_START}\n<style id="hmd-theme">{BASE_CSS}</style>\n{THEME_END}\n</head>\n{body}\n'
        '<a class="skip-link" href="#content">Skip to the report</a>\n'
        f"{_sidebar(page.sidebar)}"
        '<main id="content">\n'
        f"{_report_links(report_links, contents=bool(page.sidebar))}"
        f'<header class="report-header">\n{header}\n</header>\n'
        f"{page.body}\n"
        "</main>\n</body>\n</html>\n"
    )


# --- report renderers ------------------------------------------------------------------------------
# Imported late: pipeline.stages.output imports this module to write the twins.

from pipeline.diff import point_key  # noqa: E402
from pipeline.links import locator_label, safe_url, source_link  # noqa: E402
from pipeline.sources import MEDIA_KINDS, kind_of  # noqa: E402
from pipeline.stages import output as out  # noqa: E402
from pipeline.stages.stats import WORTH_WATCHING_MIN_UNIQUE, coverage, worth_watching  # noqa: E402

_BADGE_CLASS = {out.PARTLY_SUPPORTED: "badge-partly", out.DISPUTED: "badge-disputed", out.NEW_BADGE: "badge-new"}
LEGEND = out.BADGE_LEGEND.strip("_")


def _badge(cls: str, text: str) -> str:
    return f'<span class="badge {cls}">{esc(text)}</span>'


def cap_words(quote: str) -> str:
    """The quote's first MAX_QUOTE_WORDS words, an ellipsis glued to the last when cut; escaped."""
    words = str(quote).split()
    if len(words) > out.MAX_QUOTE_WORDS:
        return esc(" ".join(words[:out.MAX_QUOTE_WORDS]) + "…")
    return esc(" ".join(words))


def title_of(src: dict) -> str:
    return src.get("title") or src.get("video_id") or ""


def cite(src: dict, anchor, quote: str, *, with_title: bool = True) -> str:
    label = f"{title_of(src)} {locator_label(src, anchor)}" if with_title else locator_label(src, anchor)
    return link(source_link(src, anchor, quote), label)


def quotes_block(refs) -> str:
    """The folded quotes of a point: one blockquote per cited source, as the Markdown shows them."""
    items = "".join(f'<blockquote><q class="quote">{cap_words(quote)}</q><cite class="source">'
                    f"{esc(title_of(src))} {esc(locator_label(src, anchor))}</cite></blockquote>"
                    for src, anchor, quote in refs)
    return f'<details class="quotes"><summary>Quotes ({len(refs)})</summary>{items}</details>'


def point_item(p, prep, *, disputed: bool, new: bool, show_anchors: bool) -> str:
    refs = prep.res.anchored_refs(p)
    badges = []
    count = out.count_badge(p, len(refs), disputed=disputed)
    if count:
        badges.append(_badge("badge-confirmed" if count[1] else "badge-sources", count[0]))
    badges += [_badge(_BADGE_CLASS[b], b.strip("()")) for b in out.flag_badges(p, disputed=disputed, new=new)]
    links = ", ".join(cite(src, a, q) for src, a, q in refs)
    quotes = quotes_block(refs) if show_anchors else ""
    return (f"<li>{esc(prep.label(p.text))} {' '.join(badges)}"
            f' <span class="cites">{links}</span>{quotes}</li>')


def contradiction_par(c, prep) -> str:
    a, b = prep.res.source(c.source_a), prep.res.source(c.source_b)
    return (f'<p class="contradiction"><strong>Contradiction — {esc(prep.label(c.topic))}'
            f"{esc(out._contradiction_note(c))}:</strong> {link(safe_url(a['url']), title_of(a))}"
            f"{esc(out._dated(a))} claims “{esc(prep.label(c.position_a))}” while "
            f"{link(safe_url(b['url']), title_of(b))}{esc(out._dated(b))} claims "
            f"“{esc(prep.label(c.position_b))}”{esc(out._date_gap_note(a, b))}</p>")


def table(headers: list[str], rows: list[list[str]]) -> str:
    """Headers are plain text; cells are HTML already escaped by the caller. A column whose cells are all
    counts, "n of m", durations, ★ or empty (and not all empty) is marked `num`; the table sits in a
    `table-wrap` so a wide one scrolls on its own, not the page."""
    num = [bool(rows) and all(_NUM_CELL_RE.match(r[i]) for r in rows) and any(r[i] for r in rows)
           for i in range(len(headers))]
    cls = [' class="num"' if n else "" for n in num]
    head = "".join(f"<th{c}>{esc(h)}</th>" for c, h in zip(cls, headers))
    body = "".join("<tr>" + "".join(f"<td{c}>{cell}</td>" for c, cell in zip(cls, row)) + "</tr>" for row in rows)
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def heading(ids: IdMaker, sidebar: list, level: int, label: str, *, html_label: str | None = None) -> str:
    """An h2 (level 1) or h3 (level 2) with a unique id, recorded in the sidebar."""
    ident = ids(label)
    sidebar.append((level, ident, " ".join(label.split())))
    tag = "h2" if level == 1 else "h3"
    return f'<{tag} id="{ident}">{html_label if html_label is not None else esc(label)}</{tag}>'


def summary_parts(ctx, ids: IdMaker, *, before_sources=None):
    """(header lines, sidebar, body) of the summary. `before_sources(ids, sidebar) -> html` adds a block
    before the Sources section (the hybrid's per-source chapters)."""
    doc, prep = ctx.doc, out.prepare_summary(ctx.doc, ctx.sources_by_id, ctx.slug)
    new_ids = set(ctx.diff.new_points) if ctx.diff else set()
    sidebar: list[tuple[int, str, str]] = []
    body: list[str] = [heading(ids, sidebar, 1, "TL;DR")]
    if doc.tldr:
        body.append("<ul>" + "".join(f"<li>{esc(prep.label(b))}</li>" for b in doc.tldr) + "</ul>")
    else:
        body.append('<p class="note">No TL;DR: no point survived claim validation; only contradictions are '
                    "reported below.</p>")
    body.append(f'<p class="legend">{esc(LEGEND)}</p>')

    flat_i = -1
    for i, section in enumerate(doc.sections, 1):
        body.append(heading(ids, sidebar, 1, f"{i}. {section.title}"))
        srcs = prep.stats.section_sources[i - 1]
        if srcs:
            names = ", ".join(f"{esc(prep.label.labels.get(prep.tok_of[s], prep.tok_of[s]))} ({n})" for s, n in srcs)
            body.append(f'<p class="sources-line">Sources: {names}</p>')
        for j, sub in enumerate(section.subsections, 1):
            body.append(heading(ids, sidebar, 2, f"{i}.{j} {sub.title}"))
            items = []
            for p in sub.points:
                flat_i += 1
                items.append(point_item(p, prep, disputed=id(p) in prep.disputed,
                                        new=point_key(p, flat_i) in new_ids, show_anchors=ctx.show_anchors))
            if items:
                body.append('<ul class="points">' + "".join(items) + "</ul>")
        body += [contradiction_par(c, prep) for c in section.contradictions]

    if before_sources:
        body.append(before_sources(ids, sidebar))

    body.append(heading(ids, sidebar, 1, "Sources"))
    rows = []
    for token in doc.source_map:
        s = prep.res.source(token)
        dur = out.fmt_time(s["duration_seconds"]) if s.get("duration_seconds") and kind_of(s) in MEDIA_KINDS else ""
        url = safe_url(s["url"])
        rows.append([esc(token), esc(kind_of(s)), esc(s.get("title") or ""), esc(s.get("channel") or ""),
                     esc(out._fmt_date(s.get("upload_date"))), esc(dur), link(url, url)])
    body.append(table(["#", "Kind", "Title", "Channel", "Published", "Length", "URL"], rows))

    if prep.missing:
        body.append(heading(ids, sidebar, 1, "Sources not analyzed"))
        items = []
        for s in prep.missing:
            reason = s.get("error_message") or f"{s.get('transcribe_status')}/{s.get('analyze_status')}"
            items.append(f"<li>{link(safe_url(s['url']), title_of(s))} — {esc(reason)}</li>")
        body.append("<ul>" + "".join(items) + "</ul>")

    stars = worth_watching(prep.stats)
    body.append(heading(ids, sidebar, 1, "Source scorecard"))
    rows = []
    for token, sid in doc.source_map.items():
        st, row = prep.stats.sources[sid], prep.res.source(token)
        rows.append([esc(title_of(row)), str(st.points), str(st.unique_points), str(st.shared_points),
                     f"{st.claims_cited} of {st.claims}", str(st.contradictions), "★" if sid in stars else ""])
    body.append(table(["Source", "Points", "Unique", "Shared", "Claims used", "Contradictions", "Worth watching"],
                      rows))
    body.append('<p class="note">Unique: points no other source makes. Worth watching: at least '
                f"{WORTH_WATCHING_MIN_UNIQUE} unique points and in the top third of sources by them.</p>")

    cov = coverage(prep.stats)
    pct = round(100 * cov.single_source / cov.points) if cov.points else 0
    body.append(heading(ids, sidebar, 1, "Coverage"))
    items = [f"<li>Single-source points: {cov.single_source} of {cov.points} ({pct}%)</li>"]
    thin = [f"{i}. {' '.join(sec.title.split())}" for i, (sec, (n_src, _, _)) in
            enumerate(zip(doc.sections, cov.sections), 1) if n_src == 1]
    if thin:
        items.append(f"<li>Sections drawn from one source: {esc('; '.join(thin))}</li>")
    body.append("<ul>" + "".join(items) + "</ul>")
    body.append(table(["Section", "Sources", "Points", "Single-source"],
                      [[f"{i}. {esc(sec.title)}", str(a), str(b), str(c)]
                       for i, (sec, (a, b, c)) in enumerate(zip(doc.sections, cov.sections), 1)]))

    all_c = [c for sec in doc.sections for c in sec.contradictions]
    if all_c:
        body.append(heading(ids, sidebar, 1, "Contradictions"))
        rows = []
        for c in all_c:
            a, b = prep.res.source(c.source_a), prep.res.source(c.source_b)
            rows.append([esc(prep.label(c.topic) + out._contradiction_note(c)), esc(prep.label(c.position_a)),
                         esc(a.get("title") or ""), esc(prep.label(c.position_b)), esc(b.get("title") or ""),
                         esc(f"{out._iso_date(a) or '?'} / {out._iso_date(b) or '?'}")])
        body.append(table(["Topic", "Position A", "Source", "Position B", "Source", "Dates"], rows))

    header = [f"<p>Generated: {ctx.today.isoformat()} | Sources: {esc(prep.counts_text or '0 videos')}, "
              f"{esc(out._plural(len(prep.channels), 'channel'))}</p>"]
    return header, sidebar, "\n".join(body)


def page_title(ctx, kind: str) -> str:
    return f"{' '.join(str(ctx.topic).split())} — {kind}"


def page_h1(ctx, kind: str) -> str:
    """The page title as the h1: the topic, then the report kind in its own span (it never wraps alone)."""
    return f'<h1>{esc(ctx.topic)} <span class="report-kind">— {esc(kind)}</span></h1>'


def render_summary_html(ctx) -> HtmlPage:
    header, sidebar, body = summary_parts(ctx, IdMaker())
    return HtmlPage(title=page_title(ctx, "Research Summary"), header=[page_h1(ctx, "Research Summary")] + header,
                    sidebar=sidebar, body=body, kind="summary")


def _format_page(ctx, fmt: str, kind: str, sidebar: list, body: list[str],
                 header_line: str | None = None) -> HtmlPage:
    """A page with the header the extra Markdown formats use (pipeline/stages/formats.py::_header)."""
    line = header_line or (f"Generated: {ctx.today.isoformat()} | Sources: "
                           f"{out._plural(len(ctx.doc.source_map), 'video')}")
    return HtmlPage(title=page_title(ctx, kind), header=[page_h1(ctx, kind), f"<p>{esc(line)}</p>"],
                    sidebar=sidebar, body="\n".join(body), kind=fmt)


def digest_chapters(ctx, ids: IdMaker, sidebar: list, *, level: int) -> str:
    """One chapter per analyzed source in source order: meta, summary, linked key claims (quotes with
    anchors), and its share of the summary. `level` 1 = h2 chapters (digest), 2 = h3 (hybrid)."""
    res = out._Resolver(ctx.doc, ctx.sources_by_id, ctx.slug)
    stats = out.notebook_stats(ctx.doc, {t: len(res.analysis(t).key_claims) for t in ctx.doc.source_map})
    sub_tag = "h3" if level == 1 else "h4"
    parts: list[str] = []
    for n, (tok, sid) in enumerate(sorted(ctx.doc.source_map.items(), key=lambda kv: kv[1]), 1):
        src, an = res.source(tok), res.analysis(tok)
        day = out._parse_upload(src.get("upload_date"))
        meta = [esc(src.get("channel") or ""), day.isoformat() if day else "",
                out.fmt_time(src["duration_seconds"])
                if src.get("duration_seconds") and kind_of(src) in MEDIA_KINDS else "",
                link(safe_url(src["url"]), "watch" if kind_of(src) in MEDIA_KINDS else "open")]
        parts.append(heading(ids, sidebar, level, f"{n}. {title_of(src)}"))
        parts.append(f'<p class="note">{" · ".join(x for x in meta if x)}</p>')
        parts.append(f"<p>{esc(an.summary)}</p>")
        parts.append(f"<{sub_tag}>Key claims</{sub_tag}>")
        items = []
        for c in sorted((c for c in an.key_claims if c.anchor),
                        key=lambda c: (c.anchor.t_start, c.anchor.segment or 0)):
            quote = (f'<blockquote><q class="quote">{cap_words(c.quote)}</q></blockquote>'
                     if ctx.show_anchors else "")
            items.append(f'<li>{esc(c.claim)} <span class="cites">{cite(src, c.anchor, c.quote, with_title=False)}'
                         f"</span>{quote}</li>")
        parts.append("<ul>" + "".join(items) + "</ul>")
        s = stats.sources[sid]
        parts.append(f'<p class="note">In the summary: {s.points} points, {s.unique_points} only from this '
                     "source.</p>")
    return "\n".join(parts)


def render_digest_html(ctx) -> HtmlPage:
    ids, sidebar = IdMaker(), []
    return _format_page(ctx, "digest", "Digest", sidebar, [digest_chapters(ctx, ids, sidebar, level=1)])


def render_hybrid_html(ctx) -> HtmlPage:
    def chapters(ids, sidebar):
        return heading(ids, sidebar, 1, "Per-source chapters") + "\n" + digest_chapters(ctx, ids, sidebar, level=2)
    header, sidebar, body = summary_parts(ctx, IdMaker(), before_sources=chapters)
    return HtmlPage(title=page_title(ctx, "Hybrid"), header=[page_h1(ctx, "Hybrid")] + header,
                    sidebar=sidebar, body=body, kind="hybrid")


def _source_names(label, refs) -> str:
    names: list[str] = []
    for ref in refs:
        name = label.labels.get(ref.source, ref.source)
        if name not in names:
            names.append(name)
    return ", ".join(names)


def render_outline_html(ctx) -> HtmlPage:
    label = out.SourceLabeler(ctx.doc.source_map, ctx.sources_by_id)
    ids, sidebar = IdMaker(), []
    body = [heading(ids, sidebar, 1, "TL;DR"),
            "<ul>" + "".join(f"<li>{esc(label(b))}</li>" for b in ctx.doc.tldr) + "</ul>"]
    for i, sec in enumerate(ctx.doc.sections, 1):
        body.append(heading(ids, sidebar, 1, f"{i}. {sec.title}"))
        for j, sub in enumerate(sec.subsections, 1):
            body.append(heading(ids, sidebar, 2, f"{i}.{j} {sub.title}"))
            body.append("<ul>" + "".join(f"<li>{esc(label(p.text))} [{esc(_source_names(label, p.claim_refs))}]</li>"
                                         for p in sub.points) + "</ul>")
    return _format_page(ctx, "outline", "Outline", sidebar, body)


def render_quotes_html(ctx) -> HtmlPage:
    res = out._Resolver(ctx.doc, ctx.sources_by_id, ctx.slug)
    label = out.SourceLabeler(ctx.doc.source_map, ctx.sources_by_id)
    ids, sidebar = IdMaker(), []
    body: list[str] = []
    for i, sec in enumerate(ctx.doc.sections, 1):
        body.append(heading(ids, sidebar, 1, f"{i}. {sec.title}"))
        for j, sub in enumerate(sec.subsections, 1):
            body.append(heading(ids, sidebar, 2, f"{i}.{j} {sub.title}"))
            items = []
            for p in sub.points:
                quotes, seen = [], set()
                for ref in p.claim_refs:
                    if (ref.source, ref.claim) in seen:
                        continue
                    seen.add((ref.source, ref.claim))
                    src, claims = res.source(ref.source), res.analysis(ref.source).key_claims
                    if not (0 <= ref.claim < len(claims)) or claims[ref.claim].anchor is None:
                        raise out.RenderError(
                            f"point {p.id or ''} cites {ref.source} claim {ref.claim} which is missing or not anchored")
                    claim = claims[ref.claim]
                    quotes.append(f'<blockquote><q class="quote">{cap_words(claim.quote)}</q><cite class="source">'
                                  f"{cite(src, claim.anchor, claim.quote)}</cite></blockquote>")
                items.append(f"<li><strong>{esc(label(p.text))}</strong>{''.join(quotes)}</li>")
            body.append("<ul>" + "".join(items) + "</ul>")
    return _format_page(ctx, "quotes", "Quote bank", sidebar, body)


def render_diff_html(ctx) -> HtmlPage | None:
    """What changed against the previous merge; None when there is no diff (as formats.render_diff)."""
    from pipeline.schemas import MergedDocument
    d, prev = ctx.diff, ctx.previous
    if d is None or prev is None:
        return None
    old = MergedDocument.model_validate_json(prev["merged_json"])
    prep = out.prepare_summary(ctx.doc, ctx.sources_by_id, ctx.slug)
    old_label = out.SourceLabeler(old.source_map, ctx.sources_by_id)
    flat = [p for sec in ctx.doc.sections for sub in sec.subsections for p in sub.points]
    pts = {point_key(p, i): p for i, p in enumerate(flat)}
    ids, sidebar = IdMaker(), []
    body: list[str] = []

    def src_title(sid: int) -> str:
        src = ctx.sources_by_id.get(sid)
        return esc((src and title_of(src)) or f"source {sid}")

    def confirms(sid: int) -> int:
        return sum(1 for gained in d.confirmed.values() if sid in gained)

    def cited(p) -> set[int]:
        return {ctx.doc.source_map[r.source] for r in p.claim_refs if r.source in ctx.doc.source_map}

    n_re = len(d.restructured)
    regrouped = (f'<p class="note">{out._plural(n_re, "point")} {"was" if n_re == 1 else "were"} regrouped '
                 "from claims already in the previous report.</p>" if d.restructured else None)
    if d.new_sources:
        body.append(heading(ids, sidebar, 1, "New sources"))
        body.append(table(["Source", "New points", "Points it now confirms"],
                          [[src_title(sid), str(sum(1 for k in d.new_points if sid in cited(pts[k]))),
                            str(confirms(sid))] for sid in d.new_sources]))
        if d.silent_sources:
            body.append("<ul>" + "".join(
                f"<li>{src_title(sid)}: added nothing new; it confirms "
                f"{out._plural(confirms(sid), 'point')} already in the report.</li>" for sid in d.silent_sources)
                + "</ul>")
    if d.new_points:
        body.append(heading(ids, sidebar, 1, "New points"))
        body.append('<ul class="points">' + "".join(
            point_item(pts[k], prep, disputed=id(pts[k]) in prep.disputed, new=True, show_anchors=False)
            for k in d.new_points) + "</ul>")
    if regrouped:
        body.append(regrouped)
    if d.confirmed:
        body.append(heading(ids, sidebar, 1, "Newly confirmed"))
        items = []
        for k, gained in d.confirmed.items():
            item = point_item(pts[k], prep, disputed=id(pts[k]) in prep.disputed, new=False, show_anchors=False)
            also = ", ".join(src_title(sid) for sid in gained)
            items.append(item[:-len("</li>")] + f' <span class="cites">Now also: {also}</span></li>')
        body.append('<ul class="points">' + "".join(items) + "</ul>")
    if d.new_contradictions:
        body.append(heading(ids, sidebar, 1, "New contradictions"))
        body += [contradiction_par(c, prep) for c in d.new_contradictions]
    if d.removed:
        body.append(heading(ids, sidebar, 1, "No longer in the report"))
        body.append("<ul>" + "".join(f"<li>{esc(old_label(p.text))} (from {esc(_source_names(old_label, p.claim_refs))})"
                                     "</li>" for p in d.removed) + "</ul>")
    if d.dropped_sources:
        body.append(heading(ids, sidebar, 1, "Dropped sources"))
        body.append("<ul>" + "".join(f"<li>{src_title(sid)}</li>" for sid in d.dropped_sources) + "</ul>")
    if d.not_compared:
        body.append(heading(ids, sidebar, 1, "Not compared"))
        why = ("the previous report predates claim hashes; sources were not compared" if not old.claim_hashes
               else "re-analyzed since the previous report; its points were not compared")
        body.append("<ul>" + "".join(f"<li>{src_title(sid)}: {why}.</li>" for sid in d.not_compared) + "</ul>")
        body.append(f'<p class="note">{out._plural(len(d.not_compared_points), "point")} not compared.</p>')
    if not (d.new_points or d.confirmed or d.removed or d.new_contradictions or d.new_sources
            or d.dropped_sources or d.not_compared):
        body = ["<p>No changes: the same points, sources and contradictions as the previous report.</p>"]
        if regrouped:
            body.append(regrouped)
        sidebar = []
    day = str(prev.get("created_at") or "")[:10] or "an earlier date"
    return _format_page(ctx, "diff", "What changed", sidebar, body,
                        header_line=f"Compared with the report of {day} (merge {prev['id']} -> {ctx.merge_id}).")


HTML_RENDERERS = {"digest": render_digest_html, "hybrid": render_hybrid_html, "outline": render_outline_html,
                  "quotes": render_quotes_html, "diff": render_diff_html}
