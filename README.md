# Hive Mind Digest

Multi-source research synthesis with verifiable claims. Give it a topic and a set of sources --
YouTube and other videos, web articles, PDFs, local text, audio and video files; it transcribes or
extracts them, extracts claims, merges what the sources say into one Markdown report, and links
every point to the place that supports it: the moment in the video, the passage on the web page,
the page of the PDF.

> **Public Preview 0.1.** Markdown output only. Expect rough edges; see
> [Known limitations](#known-limitations).

## What you get

- One report per notebook (a topic plus its sources): a TL;DR, themed sections and a source list.
- Every point cites its sources and links to the supporting place: a timestamp
  (`youtube.com/watch?v=...&t=...`) that plays the moment, a text fragment that highlights the
  passage on a web page, or a PDF page (see [Inputs](#inputs)). Quotes are copied from the
  transcript or document by code; a point whose quote cannot be found is dropped, never guessed.
- A post-merge citation check: a point the check finds only partly supported is marked
  "(partly supported)"; one it rejects is dropped.
- Contradictions between sources are listed with both positions and their sources.
- A per-run budget cap and a cost line for every run (`COST=`), with per-stage token usage.

See [`examples/samples/`](examples/samples/) for real reports in every format before spending
anything:

- `claude-code-best-practices` -- 19 sources from 2025 and 2026 (18 videos and the official
  best-practices page), merged once and again after adding six newer sources: the `diff` report and
  `(new)` badges, and older advice shown with its date.
- `claude-code-fundamentals` -- the six talks of the bundled example list (`/hmd:init` offers it).
- `productivity` -- 42 videos, about 10 hours.
- `micro-habits` and `micro-habits-mixed` -- contradictions and a `(disputed)` point; the mixed one
  combines videos, web articles and a PDF (`#:~:text=` and `#page=` links).

Each sample starts with a banner naming the models and the cost of the run that made it.

## Requirements

- Python 3.12 or newer (CI runs 3.12 and 3.14), Windows, macOS or Linux.
- One LLM provider:
  - `anthropic_api` (default): an Anthropic API key, billed per token.
  - `claude_cli` (opt-in): a signed-in Claude Code install; see
    [Using your Claude subscription](#using-your-claude-subscription-claude_cli-provider-opt-in).
- Optional: `ffmpeg` and `requirements-whisper.txt` for local Whisper transcription of videos
  without captions and of local audio/video files (required for those). Caption-based
  transcription, web pages, PDFs and text files need neither.

## Install

### As a Claude Code plugin

In Claude Code:

```
/plugin marketplace add r-hural/Hive-Mind-Digest
/plugin install hmd@hive-mind-digest
```

(or, from a terminal, `claude plugin marketplace add r-hural/Hive-Mind-Digest` and
`claude plugin install hmd@hive-mind-digest`). Then run `/hmd:init` in any project. It needs a
Python 3.12 or newer on PATH (`python3`, `python` or, on Windows, `py`), and on first run it:

1. creates the plugin's own Python environment in the plugin data folder
   (`~/.claude/plugins/data/<plugin id>/venv/<digest>`) and installs `hmd` and its pinned requirements
   into it -- a minute or two; after a plugin update `/hmd:init` rebuilds it, and the other skills
   ask you to run `/hmd:init` when it is missing or out of date;
2. asks for the provider, the per-run budget, the caption languages and where reports and data
   should live (suggested: a `HiveMindDigest` folder in your home directory);
3. writes `config.yaml` and `.env.example` into the plugin data folder; copy `.env.example` to
   `.env` in the same folder and set `ANTHROPIC_API_KEY` if you use `anthropic_api`;
4. checks the environment with `hmd doctor` and offers the bundled example.

Uninstalling the plugin deletes the plugin data folder (the environment, `config.yaml` and
`.env`), but not your reports: they live in the folder you chose in step 2.

### From a clone (development)

```
git clone <this repository>
cd Hive-Mind-Digest
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements-dev.txt -c constraints.txt -e .   # Windows
# .venv/bin/python -m pip install -r requirements-dev.txt -c constraints.txt -e .     # macOS / Linux
```

Then use the CLI from the repository root:

```
hmd init              # writes config/config.yaml and .env.example
hmd doctor            # checks the environment, the API key and the configured models
```

Copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY` if you use `anthropic_api`.

## Quick start

```
hmd run --topic "Claude Code fundamentals" --input examples/claude-code-fundamentals.txt
```

The last lines of the output name the report and the cost:

```
REPORT=data/domains/general/vault/claude-code-fundamentals/reports/<date>-summary.md
COST=0.98
STATUS=done
```

## Commands

```
hmd run --topic "..." --input sources.txt       # one URL or local file path per line, kinds mixed
hmd run --topic "..." --url "<url>"             # one video, web page or PDF URL
hmd run --topic "..." --file "<path>"           # one local .pdf, .txt/.md, audio or video file
hmd run --topic "..." --playlist "<url>"        # YouTube playlist (first 200 videos)
hmd resume <slug> [--stages transcribe,analyze,merge,output]
hmd analyze <slug> [--force]                    # re-run analyze -> merge -> output
hmd status [<slug>] | hmd notebook list
hmd notebook add <slug> --url "<url>"           # add sources (or --input / --file / --playlist); then hmd resume <slug>
hmd dry-run <slug> [--batch] [--force]          # cost and time estimate before a run; never bills
hmd init | hmd doctor
hmd models use <analyze|merge> <model-id>
```

Run options (`run`, `resume`, `analyze`): `--provider`, `--analyze-model`, `--merge-model`,
`--budget` (USD per run; 0 disables the cap), `--max-parallel`, `--show-anchors` (print the
verbatim quote under each point), `--formats` (which report files to write, see Reports),
`--batch` (`anthropic_api` only: Message Batches at half the token price, slower), `--html/--no-html`
(write the HTML twins of the reports; default `output.html: true`).

In Claude Code the skills `/hmd:init`, `/hmd:run`, `/hmd:analyze` and `/hmd:doctor` wrap the same
commands.

Exit codes: 0 done, 2 stage error, 3 paused at the budget cap (`hmd resume` continues), 4 usage or
config error, 130 interrupted.

**What costs money:** `run`, `resume` and `analyze` call the LLM provider. `init`, `doctor`,
`status`, `notebook list`, `notebook add` and `dry-run` never do.

## Inputs

A notebook can mix any of these; `--input` takes a text file with one URL or local file path per
line (blank lines and `#` comments are skipped; relative paths are relative to the list file). A
local path in a list must lie inside the list file's folder (or below it), so a list you downloaded
cannot send your other files to the model; `hmd` refuses a list that names one outside it, and
`--file` takes a single file from anywhere:

```
# sources.txt
https://www.youtube.com/watch?v=dQw4w9WgXcQ
https://vimeo.com/123456789
https://example.org/blog/a-long-article
https://example.org/papers/study.pdf
notes/interview.md
recordings/talk.mp3
```

| Kind | What | Text from | Link in the report |
|---|---|---|---|
| `youtube` | a YouTube video (playlists with `--playlist`) | captions, yt-dlp subtitles, Supadata or Whisper | timestamp (`&t=...s`) |
| `video` | Vimeo, TED, Loom, Dailymotion | yt-dlp subtitles or Whisper | timestamp on Vimeo, the page elsewhere |
| `web` | any other web page | its main text (`trafilatura`) | a text fragment (`#:~:text=...`) that highlights the quoted passage, labelled with its paragraph number |
| `pdf` | a PDF URL or a local `.pdf` | the text layer of each page (`pypdf`) | the page (`#page=n`) |
| `text` | a local `.txt`, `.md` or `.markdown` file | the file, split into paragraphs | the file, labelled with the paragraph number |
| `media` | a local audio or video file (`.mp3`, `.m4a`, `.wav`, `.flac`, `.ogg`, `.opus`, `.mp4`, `.mkv`, `.mov`, `.webm`, `.avi`) | Whisper (needs `whisper_local`, `ffmpeg`) | the file, labelled with the timestamp |

The same page given twice (with or without a trailing slash, a `#fragment` or `utm_` tracking
parameters) is one source; so is the same file under two names. A line that is neither a URL nor
an existing file of a supported type is skipped with a warning, and the rest of the list runs.
Network share paths (`\\server\share\...`) are refused: copy such a file to a local folder first. A
source that cannot be read (a page with no main text, a scanned PDF) ends with a one-line error in
`hmd status <slug>` and is listed under "Sources not analyzed"; the report covers the others.

A local file is linked by its full `file:///` path, so you can open it from the report; that path
shows your folder names (on Windows usually your user name). Check those links before you share a
report built from local files.
Secret-shaped strings (API keys, tokens, private keys) are replaced with `[REDACTED]` before any
text is cached or sent to the model.

## Reports

Every run writes `reports/<date>-summary.md`: a TL;DR, a contents list, themed sections with a
`_Sources:_` line each, a source scorecard (points, unique points, shared points, contradictions
per source, and which videos are worth watching in full), a coverage section (single-source
points, thin sections) and contradictions with the sources' upload dates (`(may differ by edition or date)` when the check
finds the two positions may describe different times or editions). Badges: `(confirmed by
N sources)`, `(disputed)`, `(new)` (new since the previous report), `(partly supported)`.

Every report is also written as a self-contained HTML twin beside it (`<date>-summary.html`, ...):
one file with its styles inline and nothing loaded from the web, so it opens offline in any browser
and can be sent on its own. It shows the same content with a contents sidebar, real links and
foldable quotes. `output.html: false` in the config, or `--no-html` for one run, turns the twins
off; the run prints `REPORT_HTML=` with the summary twin's path.

More files from the same stored result, at no LLM cost, chosen with `--formats` or
`output.formats` (default `summary,diff`; the summary is always written):

- `digest`: a chapter per source with its summary and timestamped key claims.
- `hybrid`: the summary plus the per-source chapters.
- `outline`: the section tree with one line per point and its sources.
- `quotes`: a quote bank with timestamped links, grouped by point.
- `diff`: what changed since the previous report of the notebook.

Add videos to an existing notebook with `hmd notebook add`, then `hmd resume <slug>`: only the
new sources are analyzed and the diff report lists the new sources, new and newly confirmed
points, new contradictions and what is no longer in the report, and says when a source added
nothing new. `hmd resume <slug> --stages output --formats ...` re-writes the reports for free.

## Cost and privacy

With the default models (`claude-haiku-5-5` for analysis, `claude-sonnet-5-5` for the merge), a
notebook of 13-14 videos costs about $0.50-1.00 with `--batch` and roughly twice that without it.
A 42-video notebook (about 10 hours of video) cost $1.03 with `--batch`; the samples' banners give
the cost of each. `hmd dry-run <slug>` estimates a run before it bills. The budget cap
(`run.budget_usd` in `config/config.yaml`, default $5) pauses a run that reaches it; `hmd resume`
continues it.

What leaves your machine, and to whom:

- **Anthropic API** (`anthropic_api`, the default): the transcript and document text of your
  sources and the prompts, billed to your API key under your agreement with Anthropic.
- **Your Claude subscription** (`claude_cli`, opt-in): the same text, sent by your own signed-in
  `claude -p`; see [below](#using-your-claude-subscription-claude_cli-provider-opt-in).
- **YouTube and the other video sites** (`youtube-transcript-api`, `yt-dlp`): the video ids or URLs
  you list, to fetch metadata and captions.
- **Supadata** (opt-in, `supadata` in `transcription.providers`, needs `SUPADATA_API_KEY`): the
  video ids whose captions YouTube would not serve.
- **The web pages and PDF URLs you list**: fetched directly from their sites.
- **`openaipublic.azureedge.net`** (OpenAI's model host): the Whisper model download, once per model
  size, only when you use `whisper_local` (`hmd doctor --prewarm-whisper` fetches it ahead).
- **PyPI**: the pinned packages, when you install.
- **`hmd doctor`**: Anthropic's public docs pages on `platform.claude.com` (model prices and
  deprecations, to check the configured models) and, with an API key, the free Models API list.

Nothing else is sent anywhere: no telemetry, no account. State, transcripts, extracted text and
reports stay in your `home` folder.

## Content and copyright

Reports hold short quotes (40 words at most) of the sources you choose, with a link to each.
You are responsible for how you use and share them. Transcripts and extracted text are cached
locally and never shipped with a report; this repository ships only URL lists and the sample
reports.

## Transcription

Providers are tried in the order set in `transcription.providers`:

- `youtube_captions` and `ytdlp_subs`: the video's captions, free.
- `supadata` (off by default): a server-side transcript API, for networks where YouTube blocks
  caption requests. Needs `SUPADATA_API_KEY` in `.env`.
- `whisper_local`: local speech-to-text with Whisper for videos without captions and for local
  audio/video files. Needs `ffmpeg` and `pip install -r requirements-whisper.txt -c constraints-whisper.txt`
  (every transitive pin; for a CUDA build of torch, install it first from PyTorch's own index); slow on CPU.

Web pages, PDFs and text files need no provider setting: they are always read by the built-in
extractors. Transcripts and extracted text are cached locally and never leave your machine except
as LLM input.

## Configuration

`config/config.yaml` ships with defaults; `/hmd:init` (or `hmd init --provider ... --budget ...
--languages ... --force`) rewrites it from a few answers. Every key is read by `pipeline/config.py`
and unknown keys are rejected.

Which config file `hmd` uses, first match wins: `--config <path>`; the `HMD_CONFIG` environment
variable; when `hmd` runs from the plugin's own environment, `config.yaml` in the plugin data
folder; otherwise `config/config.yaml` relative to the current directory. That last, implicit file
may not turn off a safety check: outside this repository's own checkout, `hmd` ignores, with a
warning, a `home` other than `data`, a `run.budget_usd` of 0 or above the default, an
`llm.provider` other than `anthropic_api`, `supadata` in `transcription.providers` and
`transcription.allow_private_hosts: true` (pass `--config` to trust it). `hmd doctor` prints the
mode (development checkout or plugin), the config file in use and the home folder.

- `home` -- where state, caches and the reports live (default `data/`, relative to the current
  directory; the plugin stores an absolute path, `~` is allowed).
- `llm.provider` -- `anthropic_api` (API key, per token) or `claude_cli` (Claude subscription).
- `llm.models.analyze` / `llm.models.merge` -- model ids per stage (`hmd models use` edits them).
- `llm.max_parallel`, `llm.retry_attempts`, `llm.rate_limits`, `llm.batch.*` -- concurrency,
  retries, per-provider rate limits and Message Batches settings.
- `transcription.providers` -- tried in order; `transcription.languages` -- caption languages in
  preference order.
- `transcription.allow_private_hosts` -- `false` (default): web pages and PDFs are not fetched from
  loopback, private, link-local or other non-public addresses (also not through a redirect), and a
  video's media, manifest and subtitle URLs are checked the same way before yt-dlp downloads them
  (a redirect inside that download is not); `true` allows intranet sources.
- `run.budget_usd` -- per-run cap; the run pauses when its spend crosses it; 0 disables it.
- `output.show_anchors` -- render the verbatim quote under each point.
- `output.html` -- also write a self-contained `.html` twin of every report (default `true`).

`.env` is read from the folder of the config file in use when you chose it (`--config`, `HMD_CONFIG`
or the plugin data folder; not for the implicit `config/config.yaml`) and then from the repository
root of a clone; never from a parent directory, and never overriding a variable that is already
set. A `.env` sets only `ANTHROPIC_API_KEY` and `SUPADATA_API_KEY`; any other line is ignored.

## Troubleshooting

- **Dropped quotes and points are normal, not bugs:** a claim whose quote cannot be located, a
  point with no valid claim reference, or a point the citation check rejects is dropped, never
  guessed.
- **The budget is per run:** `hmd resume` starts a new run with a fresh cap. A merge error is
  all-or-nothing; the spend is still recorded.
- **Fatal LLM errors stop the run** (authentication, credit balance, usage limit): one `ERROR:`
  line, exit code 2, and untouched sources stay pending. Fix the account and run
  `hmd resume <slug>`.
- **YouTube may block your IP** (`IpBlocked`, yt-dlp HTTP 429): add `supadata` to
  `transcription.providers`. A transcript covering under half the video is rejected as truncated.
- **`claude_cli` runs sandboxed:** tools off, an empty temporary working directory, the prompt on
  stdin, and subscription environment variables stripped except `CLAUDE_CONFIG_DIR`.
- **Prices are maintained by hand** in `pipeline/pricing.py`; `hmd doctor` reports a difference from
  the published price list or a table over 60 days old. Nothing switches models automatically.
- **hmd writes UTF-8** even when its output is redirected; read its log files as UTF-8.

## Development

```
.venv/Scripts/python.exe -m pytest tests -q       # all mocked, no network, never bills
```

```
pipeline/cli.py, runner.py           CLI and orchestrator (thread pool, run rows, budget)
pipeline/db.py, schemas.py           SQLite state; pydantic contracts for every LLM response
pipeline/llm/                        providers: anthropic_api, claude_cli, replay; rate limits
pipeline/sources.py, links.py        source kinds and keys; per-kind citation links
pipeline/transcription/              captions, yt-dlp subtitles, Supadata, local Whisper, web/PDF/
                                     text extractors, secret scrubbing; cache
pipeline/stages/, workers/           fetch, transcribe, analyze, merge and output stages
pipeline/prompts/                    prompts as Python functions; PROMPT_VERSION per file
skills/                              plugin skills (/hmd:<name>), thin wrappers over the CLI
.claude-plugin/                      plugin manifest and marketplace entry
examples/                            bundled URL list for /hmd:init and a sample digest
```

- Change prompts only in `pipeline/prompts/` and the report layout only in
  `pipeline/stages/output.py`.
- **A prompt edit needs a `PROMPT_VERSION` bump:** `tests/test_prompt_versions.py` fingerprints the
  rendered prompts per version. Bump it in `pipeline/prompts/<file>.py`, then run
  `.venv/Scripts/python.exe -m pytest tests/test_prompt_versions.py --update-fingerprints` and
  commit the updated fixture.
- Skills never inline Python and never pass a topic through a shell-interpolated string.
- This README and the skills must stay ASCII-only: a legacy Windows console code page raises
  `UnicodeEncodeError` on print (`tests/test_repo_hygiene.py` enforces it).
- Windows: the interpreter is `.venv/Scripts/python.exe` and the console script
  `.venv/Scripts/hmd.exe` after `pip install -e .`; stage and worker scripts assume the repository
  root as the working directory.
- Never commit `data/`, `.env` or `credentials/`.

## Using your Claude subscription (claude_cli provider, opt-in)

By default Hive Mind Digest calls the Claude API with your own `ANTHROPIC_API_KEY`. If you choose
the `claude_cli` provider instead, `hmd` runs your locally installed, signed-in Claude Code
(`claude -p`, tools disabled) once per analysis or merge step. Those calls count against your
Claude plan's usage limits (5-hour and weekly windows, shared with Claude and Claude Code) and are
governed by Anthropic's Consumer Terms, Usage Policy and the Claude Code legal and compliance page:
https://code.claude.com/docs/en/legal-and-compliance

- Use it only with your own Claude account, on your own machine, for your own work. Do not run it
  on a shared server or offer it to other people.
- `hmd` never reads, stores or transmits your Claude login; Claude Code handles sign-in itself.
- Anthropic's limits assume ordinary, individual use. Large notebooks can use up your plan's
  limits; for large, scheduled or team use, use an API key (`anthropic_api`), which is the route
  Anthropic recommends for tools built on Claude.
- Anthropic may change these terms; if they no longer allow this use, switch to `anthropic_api`
  (`hmd init --provider anthropic_api --force`).

`claude_cli` is also slower (6 calls a minute by default) and its `COST=` is a nominal figure
computed from token counts, not a charge.

## Known limitations

This preview covers the first milestones of a larger plan: videos, web pages, PDFs and local
files in; summary, digest, hybrid, outline, quotes and diff reports out. Not available yet:

- Podcast and RSS feeds, and video sites other than YouTube, Vimeo, TED, Loom and Dailymotion
  (any other URL is read as a web page).
- OCR: a scanned PDF (no text layer) or a password-protected PDF fails with a clear error.
- Pages behind a login, a paywall or a cookie wall, and pages that need JavaScript to show their
  text: there are no cookies or browser rendering, so such a page fails ("no main text found").
- A knowledge vault: each run writes one Markdown report, not linked notes; there is no
  Obsidian vault layout, no search index and no `hmd reindex`.
- Editing a report and re-running: a re-run overwrites the report.
- Feeds and scheduled runs.
- An MCP server or any integration beyond the Claude Code skills.

Known quality limits:

- Some points are marked "(partly supported)": 9-26% of points in the samples, typically a point
  that names a detail its quote does not contain.
- The same point can appear in more than one section, and "(confirmed by N sources)" can join
  sources that give different figures.
- The diff compares a notebook's two latest reports only, matched by the claims the points cite: a
  re-analyzed source is listed as "not compared", and a point reworded over the same claims is
  counted as regrouped, not new. There is no "minority position" badge yet.
- `hmd dry-run` is an estimate calibrated on your earlier runs on the same models; on our four test
  notebooks and a 42-video one it was within 23% of the real cost (-25% to +20% when each run was
  left out of the calibration), so read its range rather than the single figure.
- A contradiction between figures from sources months apart (the report shows their dates but does
  not say which is outdated), or about a product sold in several
  editions, is reported as a plain contradiction.
- Video transcription depends on captions; some networks are blocked by YouTube, and local
  Whisper (the only option for local audio/video files) is slow.
- A web page's text is fetched once, when the source is first transcribed, and cached; later runs
  use the cached text even if the page has changed. `hmd resume <slug> --retranscribe` fetches
  every source again (and re-analyzes them, which bills).
- Local files are identified by their content: the same file under another name is the same
  source, and an edited file is a new source (add it again; the old version stays in the
  notebook). Local files are read where they are at transcribe time, so do not move them before
  the run has transcribed them.
- A web link finds its passage by the quoted words: the text fragment highlights it in
  Chromium-based browsers and Safari, and other browsers open the top of the page. Links to local
  text and media files open the file, not the passage or the moment.
- `--batch` waits for each Message Batch to finish (usually minutes, up to 24 hours).
- The HTML reports have a plain, functional design: a contents sidebar, light and dark themes and
  print styles, but no search or filtering.
- The `claude_cli` provider depends on Claude Code's `claude -p` behaviour, which may change.
- The first run takes longer than 15 minutes from install: `/hmd:init` builds the environment
  (a minute or two), and the 6-video example took 33 minutes with `--batch` (each Message Batch
  waits in a queue; without `--batch` it is faster and costs about twice as much).
- Advice that changed over time often comes out as dated points ("As of 2025-02-25 ...") rather
  than as a contradiction: the Claude Code samples, built from 2025 and 2026 sources, list no
  contradictions.

## License

MIT; see [LICENSE](LICENSE).

Hive Mind Digest is an independent project and is not made or endorsed by Anthropic.
