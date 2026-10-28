# Examples

- `claude-code-fundamentals.txt` -- six public talks from Anthropic's YouTube channels about
  Claude Code, used by `/hmd:init`.
  Run: `.venv/Scripts/hmd.exe run --topic "Claude Code fundamentals" --input examples/claude-code-fundamentals.txt --show-anchors`
  Cost with the default models is on the order of $1 with `anthropic_api` (the exact spend
  prints as `COST=`).
- `samples/<notebook>/<format>.md` and `<format>.html` -- sample reports in every format
  (`summary`, `digest`, `hybrid`, `outline`, `quotes`, and `diff` where a notebook was re-merged
  after adding sources), each as Markdown and as a self-contained HTML page that opens offline in
  any browser, copied from real notebooks. Each file starts with a banner naming
  the date, the analyze and merge models and the cost. They are model output: quotes are short
  excerpts of the cited public sources, capped at 40 words; check them at the linked timestamp or
  page. `tests/test_samples.py` checks the banner, the quote cap, and that together the samples show
  every report element.

Only URLs and rendered reports are shipped here, never transcripts.
