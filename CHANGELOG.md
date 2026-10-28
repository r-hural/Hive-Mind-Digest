# Changelog

## 0.1.1

Security and install hardening.

- Skills treat report text as untrusted data: they show it but never follow instructions in it.
- An `--input` list may name local files only inside its own folder; a path outside it stops the
  run before anything is fetched (use `--file` for a single file elsewhere).
- A `config/config.yaml` picked up from the current folder (not passed with `--config`) can no
  longer switch off a safeguard: it cannot turn off or raise the spending cap, move the data
  folder, change the LLM provider, add the Supadata transcriber or allow private network
  addresses. It may still tighten them (a lower budget, fewer transcribers).
- Video downloads check the resolved media, manifest and subtitle addresses against private
  networks before downloading.
- PDFs are parsed in a separate process with a two-minute limit, so a malformed file cannot hang
  a run.
- Text from sources is stripped of terminal control characters before it is printed.
- The plugin installs from a hashed lock file (`requirements.lock`, `pip --require-hashes`), with
  the build tool pinned; Whisper has its own full pin set (`constraints-whisper.txt`).
- Test tools moved to `requirements-dev.txt`.
- The price and model check reads only Anthropic's documentation pages, over https, with a size
  cap.
- Docs: the Whisper model download host is `openaipublic.azureedge.net`; reports link local files
  by their full `file:///` path (shared reports show local folder names).

## 0.1.0

Public Preview.

- `hmd` command line tool and Claude Code plugin (`/hmd:init`, `/hmd:run`, `/hmd:analyze`,
  `/hmd:doctor`): fetch, transcribe or extract, analyze, merge and report for a notebook of
  sources.
- Inputs: YouTube videos and playlists, Vimeo, TED, Loom and Dailymotion videos, web pages, PDFs,
  local text, audio and video files, mixed in one list.
- Reports: summary (always), digest, hybrid, outline, quotes and diff, each with a self-contained
  HTML twin. Quotes are copied from the source text by code and every point links to the
  supporting moment, passage or page; a post-merge citation check marks or drops weak points.
- Providers: the Anthropic API with your own key (default, optional Message Batches at half
  price) or your signed-in Claude Code (`claude_cli`, opt-in).
- A per-run budget cap, a cost line for every run and `hmd dry-run` estimates.
- Sample reports in `examples/samples/`.
