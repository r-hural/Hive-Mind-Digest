---
name: init
description: First-time setup for Hive Mind Digest; asks provider, budget and languages, writes config.yaml and .env template via hmd init, runs hmd doctor, offers the bundled example.
---

# /hmd:init

Goal: a new user reaches a first report in under 15 minutes.
Run every command with the Bash tool (any working directory); never write inline Python. Every
`hmd` command below is the plugin's own one, under `${CLAUDE_PLUGIN_DATA}`, which step 0 sets up.

## 0. Set up the plugin's Python environment

Run this first, every time (it returns at once when the environment is already up to date, and
rebuilds it after a plugin update). Set the Bash tool timeout to 600000: the first install takes a
minute or two.
```bash
python3 "${CLAUDE_PLUGIN_ROOT}/pipeline/bootstrap.py" "${CLAUDE_PLUGIN_DATA}"
```
If `python3` is not found (exit 127, or Windows prints "Python was not found"), run the same line
with `python` in place of `python3`; on Windows, then with `py -3`. If none exists, tell the user
to install Python 3.10 or newer (python.org) and run `/hmd:init` again, and stop.

- `BOOTSTRAP=ok` or `BOOTSTRAP=installed` (exit 0): continue. Note the `SUGGESTED_HOME=` line, and
  whether the `CONFIG=` line ends in "(not written yet)".
- `BOOTSTRAP=fail` with exit 4: the Python found is too old; show the message and stop.
- `BOOTSTRAP=fail` with exit 2: installing failed; show the last 30 lines of output and stop.

If `CONFIG=` names an existing file (no "(not written yet)"), the plugin is already configured:
ask with AskUserQuestion whether to keep that configuration (then skip to step 4) or set it up
again (continue with step 1).

## 1. Detect what is available (no questions yet)

```bash
HMD_PLUGIN_ROOT="${CLAUDE_PLUGIN_ROOT}" "${CLAUDE_PLUGIN_DATA}/hmd" doctor
```
`doctor` exits 0 or 2 -- either is fine here (before step 3 it reports `config_file` as `[FAIL]`),
this is only reconnaissance before asking questions; do not stop on a non-zero exit at this step.
Note whether `claude_binary` is `[ok]` and whether `anthropic_api_key` is `[ok]`.

## 2. Ask, with AskUserQuestion, at most these questions

1. **Provider** -- options: `claude_cli` "use my Claude subscription (no API key)" (offer first if
   the binary is ok), `anthropic_api` "use an Anthropic API key (per-token billing, more control)".
2. **Per-run budget in USD** -- options: 2, 5 (recommended), 20, 0 = no cap. Explain: the run pauses
   at the cap and `hmd resume <slug>` continues it.
3. **Caption languages** -- default `en`; accept a comma list.
4. **Where reports and data live** -- suggest the absolute path from the `SUGGESTED_HOME=` line
   (a `HiveMindDigest` folder in the user's home directory) and show it in full; accept another
   absolute path. Explain: the reports, the transcript cache and the notebook database go there;
   it must not be inside `${CLAUDE_PLUGIN_DATA}`, which uninstalling the plugin deletes.

Do not ask about models or parallelism; defaults are fine and editable in
`${CLAUDE_PLUGIN_DATA}/config.yaml`.

## 3. Write the config

```bash
HMD_PLUGIN_ROOT="${CLAUDE_PLUGIN_ROOT}" "${CLAUDE_PLUGIN_DATA}/hmd" init --provider '<provider>' --budget '<usd>' --languages '<langs>' --home '<home>'
```
Do not add `--force` up front. If this exits 0, the command printed `CONFIG=<path>` and an
`ENV=<path> ...` line naming the actual `.env.example` location it just wrote (next to the
config's project root, not necessarily the current directory) and a `HOME=<path>` line naming
where reports and data will live; move on to step 4. Parse KEY=value only from lines that START
with the key (`^(NOTEBOOK|RUN_ID|REPORT|COST|STATUS|CONFIG|ENV|HOME|DOCTOR)=`); never by substring.

If it exits 4 with a message like "${CLAUDE_PLUGIN_DATA}/config.yaml exists; re-run with --force to overwrite",
an existing config is already there -- show that message to the user and ask with
AskUserQuestion whether to overwrite it. Only if they confirm, re-run the same command with
`--force` appended. If they decline, stop here and leave their existing `${CLAUDE_PLUGIN_DATA}/config.yaml` as is.

If the provider is `anthropic_api` and `anthropic_api_key` was not `[ok]` in step 1: tell the user
to copy the file named on the `ENV=` line to `${CLAUDE_PLUGIN_DATA}/.env` next to it and paste their key, then wait for
them to confirm before moving on. Never ask for the key in the chat, and never read or print `${CLAUDE_PLUGIN_DATA}/.env`.

## 4. Verify

Run `HMD_PLUGIN_ROOT="${CLAUDE_PLUGIN_ROOT}" "${CLAUDE_PLUGIN_DATA}/hmd" doctor` again. If a `[FAIL]` remains, show it with the fix from
`/hmd:doctor` and stop. `[warn]` lines (usually ffmpeg/whisper) do not block captioned videos;
mention them once.

## 5. Offer the example

Ask: "Run the bundled example (6 public Anthropic videos about Claude Code, on the order of $1
with anthropic_api; the exact spend prints as COST=) now?" If yes:
```bash
HMD_PLUGIN_ROOT="${CLAUDE_PLUGIN_ROOT}" "${CLAUDE_PLUGIN_DATA}/hmd" run --topic 'Claude Code fundamentals' --input "${CLAUDE_PLUGIN_ROOT}/examples/claude-code-fundamentals.txt"
```
Set the Bash tool timeout to 600000; this can take several minutes.

Report the result the same way `/hmd:run` does:
- Exit code 0 (done): read the report with the Read tool and show the path and its `## TL;DR`
  section.
- Exit code 3 (paused at the budget): show the printed cost summary and the `Resume with: hmd
  resume <slug>` line; tell the user to continue with that command.
- Exit code 2 or 4: show the last 30 lines of output and stop; do not retry automatically.

If no, point to `${CLAUDE_PLUGIN_ROOT}/examples/samples/claude-code-fundamentals/summary.md` (the report of this example)
and the other samples under `${CLAUDE_PLUGIN_ROOT}/examples/samples/` so they can see what a report looks like before
spending anything.
