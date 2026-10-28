# Changelog

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
