---
name: doctor
description: Check the Hive Mind Digest environment (Python, keys, claude binary, ffmpeg, Whisper/GPU, state DB, vault, egress); wraps hmd doctor.
---

# /hmd:doctor

Run exactly one command with the Bash tool (any working directory):
```bash
HMD_PLUGIN_ROOT="${CLAUDE_PLUGIN_ROOT}" "${CLAUDE_PLUGIN_DATA}/hmd" doctor
```
If it fails with `No such file or directory` (exit 127), the plugin's environment is not set up:
tell the user in one line to run `/hmd:init` first, and stop.

Add `--prewarm-whisper` only if the user explicitly asked to download the Whisper
model:
```bash
HMD_PLUGIN_ROOT="${CLAUDE_PLUGIN_ROOT}" "${CLAUDE_PLUGIN_DATA}/hmd" doctor --prewarm-whisper
```
This downloads about 140 MB on first use; do not add it on your own.

## Reporting

Each line of output is `[ok]`, `[warn]`, or `[FAIL]` followed by a check name and detail. The
last line is `DOCTOR=ok` or `DOCTOR=fail`, and the process exit code is 0 (ready) or 2 (a
required check failed). Parse KEY=value only from lines that START with the key
(`^(NOTEBOOK|RUN_ID|REPORT|COST|STATUS|CONFIG|ENV|DOCTOR)=`); never by substring.

The first lines name the install in use (`mode`: development checkout or plugin venv), the
config file (`config_file`) and where state, caches and reports live (`home`).

Show the user every `[FAIL]` and `[warn]` line with a one-line fix each:
- `python` -> the interpreter running `hmd` must be Python 3.10+; install a newer Python and run
  `/hmd:init` again (it rebuilds the plugin's environment with the Python it finds).
- `plugin_env` -> the plugin was updated since `/hmd:init` built its environment; run `/hmd:init`.
- `config_file` -> run `/hmd:init`, which writes `${CLAUDE_PLUGIN_DATA}/config.yaml`.
- `home` `[warn]` "inside the plugin data dir" -> uninstalling the plugin would delete the reports
  with it; set `home` in `${CLAUDE_PLUGIN_DATA}/config.yaml` to a folder of the user's own (or
  re-run `/hmd:init` and pick one).
- `anthropic_api_key` -> add `ANTHROPIC_API_KEY=...` to `${CLAUDE_PLUGIN_DATA}/.env`, or switch `llm.provider` to
  `claude_cli` in `${CLAUDE_PLUGIN_DATA}/config.yaml`.
- `claude_binary` -> install Claude Code and log in, or switch the provider to `anthropic_api`. If
  the detail mentions a cmd/bat shim, the native Claude Code binary is required -- an npm shim is
  rejected because it breaks the sandboxed argv this CLI relies on.
- `yt_dlp` -> run `/hmd:init`, which reinstalls the plugin's environment.
- `ffmpeg` / `whisper` -> only needed for videos without captions: install ffmpeg on PATH and
  Whisper into the plugin's environment, or remove `whisper_local` from `transcription.providers`.
  Whisper: `"${CLAUDE_PLUGIN_DATA}/venv/Scripts/python.exe" -m pip install -r
  "${CLAUDE_PLUGIN_ROOT}/requirements-whisper.txt" -c "${CLAUDE_PLUGIN_ROOT}/constraints-whisper.txt"` on Windows (`venv/bin/python` elsewhere); a
  plugin update rebuilds the environment, so repeat it after `/hmd:init` reinstalls. These are
  `[FAIL]` only when `whisper_local` is the sole configured provider (no captions fallback);
  otherwise they are `[warn]`.
- `supadata_api_key` (only shown when `supadata` is in `transcription.providers`) -> add
  `SUPADATA_API_KEY=...` to `${CLAUDE_PLUGIN_DATA}/.env`, or remove `supadata` from `transcription.providers`. `[FAIL]`
  only when it is the sole configured provider; otherwise `[warn]` and the chain skips it.
- `whisper_model` -> `HMD_PLUGIN_ROOT="${CLAUDE_PLUGIN_ROOT}" "${CLAUDE_PLUGIN_DATA}/hmd" doctor --prewarm-whisper` downloads the model once
  (about 140 MB for the default size).
- `state_db` MILESTONE-0 or schema-version mismatch -> rename or delete the old database file the
  check names, then start a fresh notebook.
- `model_prices` -> the model has no price in this plugin version; pass `--budget 0` on the command
  that uses it, or switch the stage to a priced model with
  `HMD_PLUGIN_ROOT="${CLAUDE_PLUGIN_ROOT}" "${CLAUDE_PLUGIN_DATA}/hmd" models use <stage> '<model>'`.
- `vault_dir` -> check filesystem permissions on the path the check names.

If everything is `[ok]`, tell the user the environment is ready; do not re-run the command.
