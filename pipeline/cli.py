"""`hmd` command line (DECISION-003). argparse only; all behaviour lives in runner/stages.

Exit codes: 0 done, 2 stage error, 3 paused on budget, 4 configuration / usage error, 130 interrupted.
Every line the skills parse is printed as KEY=value.

argparse itself exits the process directly on a parse failure or `--help` (its own SystemExit,
code 2 or 0 respectively) before `main()`'s try/except ever runs -- those two codes are argparse's,
not this module's EXIT_* constants, and are not caught here.

stdout vs stderr is deliberate and skill-relevant: `KEY=value` lines, stage progress and pause
messages go to stdout (skills parse them there); usage/configuration errors go to stderr (skills
key off the process exit code, not stderr text, to decide what happened).

A caller must parse KEY=value only from lines that START with the key
(`^(NOTEBOOK|RUN_ID|REPORT|COST|STATUS|CONFIG|ENV|DOCTOR|GOLDEN|PROVIDER|HOME)=`), never by
substring: user-supplied text (a notebook topic, a video title) can legitimately contain e.g.
"REPORT=" and is never placed at the start of a printed line for exactly this reason.

Console encoding: `main()` first switches stdout and stderr to UTF-8 with errors="replace"
(`configure_console`). On Windows a redirected stdout defaults to the legacy code page (cp1252),
and model or YouTube text in a log line (a point quoted by merge:check, a topic in `hmd status`)
used to raise UnicodeEncodeError and abort a billed run at the very end (2026-10-09).
"""
from __future__ import annotations

import argparse
import importlib
import os
import sqlite3
import sys

from pipeline import bootstrap, db, envfile
from pipeline.config import (CONFIG_ENV, CONFIG_PATH, Config, ConfigError, FORMATS, PROVIDERS, load_config,
                             normalize_formats,
                             plugin_data_dir, plugin_stale_reason, resolve_config_path, run_mode)
from pipeline.exit_codes import EXIT_ERROR, EXIT_INTERRUPTED, EXIT_OK, EXIT_PAUSED, EXIT_USAGE
from pipeline.llm.base import LLMError
from pipeline.llm.factory import LLM_STAGES, OfflineProvider, make_provider
from pipeline.quality import quality_report
from pipeline.runner import Runner, STAGES, safe_log, timing_report, usage_report
from pipeline.termtext import terminal_safe
from pipeline.stages.fetch import fetch_stage
from pipeline.stages.output import html_twin_path


def _add_run_options(p: argparse.ArgumentParser) -> None:
    p.add_argument("--stages", default=None,
                   help="comma-separated subset of: " + ",".join(STAGES) +
                        " (order is normalized to pipeline order)")
    p.add_argument("--provider", choices=PROVIDERS, default=None, help="LLM provider to use")
    p.add_argument("--analyze-model", dest="analyze_model", default=None, help="model id for the analyze stage")
    p.add_argument("--merge-model", dest="merge_model", default=None, help="model id for the merge stage")
    p.add_argument("--budget", type=float, default=None, help="per-run cap in USD; 0 disables it")
    p.add_argument("--max-parallel", dest="max_parallel", type=int, default=None,
                   help="max sources processed concurrently")
    p.add_argument("--show-anchors", dest="show_anchors", action="store_true", default=None,
                   help="render the verbatim quote under each point")
    p.add_argument("--no-show-anchors", dest="show_anchors", action="store_false", default=None,
                   help="do not render the verbatim quote under each point "
                        "(overrides a config with output.show_anchors: true)")
    p.add_argument("--html", dest="html", action="store_true", default=None,
                   help="also write an .html twin of every report (default output.html)")
    p.add_argument("--no-html", dest="html", action="store_false", default=None,
                   help="write no .html twins (overrides a config with output.html: true)")
    p.add_argument("--formats", default=None,
                   help="comma-separated report formats: " + ",".join(FORMATS) +
                        " (summary is always written); default output.formats")
    p.add_argument("--batch", dest="batch", action=argparse.BooleanOptionalAction, default=None,
                   help="anthropic_api only: send each step's calls as a Message Batch (half price, "
                        "results in minutes to hours); default llm.batch.enabled")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hmd", description="Hive Mind Digest pipeline CLI")
    parser.add_argument("--config", default=None,
                        help=f"path to config.yaml (default: ${CONFIG_ENV}, else the plugin data "
                             f"dir's config.yaml in plugin mode, else {CONFIG_PATH}; "
                             "an explicit path that does not exist is an error, "
                             f"a missing {CONFIG_PATH} silently falls back to built-in defaults)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run the pipeline for a new notebook")
    p_run.add_argument("--topic", help="research topic; becomes the notebook slug and its header")
    p_run.add_argument("--topic-file", help="read the topic from this UTF-8 text file instead of --topic")
    p_run.add_argument("--input", dest="input_file", help="text file, one URL or local file path per line")
    p_run.add_argument("--url", help="a single URL (YouTube, another video site or a web page)")
    p_run.add_argument("--playlist", help="a YouTube playlist URL (first 200 videos)")
    p_run.add_argument("--file", dest="file_path", help="a single local file (.pdf, .txt, .md, audio or video)")
    _add_run_options(p_run)

    p_resume = sub.add_parser("resume", help="continue an existing notebook")
    p_resume.add_argument("notebook", help="notebook slug or id")
    p_resume.add_argument("--retranscribe", action="store_true",
                          help="redo transcription for every source; implies transcribe,analyze,merge,output "
                               "-- --stages must include them")
    _add_run_options(p_resume)

    p_analyze = sub.add_parser("analyze", help="re-run analyze -> merge -> output")
    p_analyze.add_argument("notebook", help="notebook slug or id")
    p_analyze.add_argument("--force", action="store_true",
                           help="re-analyze every source even if already analyzed; "
                                "implies analyze,merge,output -- --stages must include them")
    _add_run_options(p_analyze)

    p_status = sub.add_parser("status", help="show notebook or pipeline status")
    p_status.add_argument("notebook", nargs="?", help="notebook slug or id; omit to list all notebooks")

    p_dry = sub.add_parser("dry-run", help="estimate cost and time for the next run of a notebook; never bills")
    p_dry.add_argument("notebook", help="notebook slug or id")
    p_dry.add_argument("--provider", choices=PROVIDERS, default=None)
    p_dry.add_argument("--analyze-model", dest="analyze_model", default=None)
    p_dry.add_argument("--merge-model", dest="merge_model", default=None)
    p_dry.add_argument("--batch", dest="batch", action=argparse.BooleanOptionalAction, default=None)
    p_dry.add_argument("--force", action="store_true",
                       help="estimate a full re-analysis of every source, as hmd analyze --force")

    p_notebook = sub.add_parser("notebook", help="notebook management")
    nb_sub = p_notebook.add_subparsers(dest="notebook_command", required=True)
    nb_sub.add_parser("list", help="list notebooks")
    p_add = nb_sub.add_parser("add", help="add sources to an existing notebook (then: hmd resume <notebook>)")
    p_add.add_argument("notebook", help="notebook slug or id")
    p_add.add_argument("--input", dest="input_file", help="text file, one URL or local file path per line")
    p_add.add_argument("--url", help="a single URL (YouTube, another video site or a web page)")
    p_add.add_argument("--playlist", help="a YouTube playlist URL (first 200 videos)")
    p_add.add_argument("--file", dest="file_path", help="a single local file (.pdf, .txt, .md, audio or video)")

    p_models = sub.add_parser("models", help="model settings")
    models_sub = p_models.add_subparsers(dest="models_command", required=True)
    p_use = models_sub.add_parser("use", help="set the model of one stage in the config file")
    p_use.add_argument("stage", help="analyze or merge")
    p_use.add_argument("model", help="Claude model id, e.g. claude-sonnet-5-5")

    p_doctor = sub.add_parser("doctor", help="check environment")
    p_doctor.add_argument("--no-model-check", dest="no_model_check", action="store_true",
                          help="skip the model monitor (no network for it)")
    p_doctor.add_argument("--prewarm-whisper", dest="prewarm_whisper", action="store_true",
                          help="download the configured Whisper model before running checks")

    p_init = sub.add_parser("init", help="write config.yaml and .env.example")
    p_init.add_argument("--provider", default="anthropic_api")
    p_init.add_argument("--budget", type=float, default=5.0)
    p_init.add_argument("--home", default=None,
                        help="where state, caches and reports live (default: data, relative to the "
                             "working directory; in plugin mode ~/HiveMindDigest, stored absolute)")
    p_init.add_argument("--languages", default="en")
    p_init.add_argument("--analyze-model", dest="analyze_model", default="claude-haiku-5-5")
    p_init.add_argument("--merge-model", dest="merge_model", default="claude-sonnet-5-5")
    p_init.add_argument("--force", action="store_true")

    p_eval = sub.add_parser(
        "eval", help="golden notebook evals: replay cassettes (never bills); --live records and bills")
    p_eval.add_argument("name", nargs="?", help="golden name under evals/golden; omit for all")
    p_eval.add_argument("--live", action="store_true",
                        help="record cassettes with the real provider; bills once; "
                             "needs --budget or run.budget_usd > 0")
    p_eval.add_argument("--provider", choices=PROVIDERS, default=None,
                        help="provider to replay/record; default both")
    p_eval.add_argument("--budget", type=float, default=None,
                        help="per-run cap in USD for --live only; 0 = uncapped on purpose")
    p_eval.add_argument("--keep-home", dest="keep_home", default=None,
                        help="replay only, one golden and one --provider: run in this folder and keep "
                             "it, so the notebook can be re-rendered (hmd --config ... resume <slug> "
                             "--stages output)")
    p_eval.add_argument("--export-from", dest="export_from",
                        help="notebook slug/id to turn into a golden dir named NAME")

    return parser


def _formats(arg: str | None, default: list[str]) -> list[str]:
    if arg is None or not arg.strip():
        return list(default)
    requested = [f.strip() for f in arg.split(",") if f.strip()]
    bad = [f for f in requested if f not in FORMATS]
    if bad:
        raise ConfigError(f"unknown formats {bad}; choose from {list(FORMATS)}")
    return normalize_formats(requested)


def _stages(arg: str | None, default: list[str]) -> list[str]:
    if arg is None or not arg.strip():
        return default
    requested = [s.strip() for s in arg.split(",") if s.strip()]
    unknown = [s for s in requested if s not in STAGES]
    if unknown:
        raise ConfigError(f"unknown stages {unknown}; choose from {STAGES}")
    result = [s for s in STAGES if s in requested]
    if not result:
        raise ConfigError("--stages must not be empty")
    return result


def _check_batch(name: str, batch: bool | None) -> None:
    if batch and name != "anthropic_api":
        raise ConfigError(f"--batch needs the anthropic_api provider ({name} has no batch API)")


def _runner(cfg: Config, args, stages: list[str] | None = None) -> Runner:
    """`stages` None means the run may call the LLM. A run with no LLM stage (e.g. `hmd resume <slug>
    --stages output`, which re-renders from the stored merge) is free: it gets an OfflineProvider, so
    it needs no API key, and neither --batch nor the price gate applies to it."""
    name = args.provider or cfg.llm.provider
    if stages is not None and not any(s in LLM_STAGES for s in stages):
        provider = OfflineProvider(name)
    else:
        _check_batch(name, args.batch)
        batch = args.batch if args.batch is not None else (cfg.llm.batch.enabled and name == "anthropic_api")
        provider = make_provider(cfg, args.provider, batch=batch)
    budget = cfg.run.budget_usd if args.budget is None else args.budget
    if budget < 0:
        raise ConfigError("--budget must be >= 0")
    analyze_model = args.analyze_model or cfg.llm.models["analyze"]
    merge_model = args.merge_model or cfg.llm.models["merge"]
    # The unpriced-model price gate lives only in Runner.__init__ now (a cleanup removed the
    # duplicate copy that used to live here). It still refuses a bad model before any notebook
    # row exists for `hmd run`, because cmd_run constructs the Runner (right here) before it calls
    # create_notebook.
    return Runner(cfg, provider=provider, analyze_model=analyze_model, merge_model=merge_model,
                 budget_usd=budget, max_parallel=args.max_parallel, show_anchors=args.show_anchors, html=args.html,
                 formats=_formats(args.formats, cfg.output.formats))


def _finish(result, slug: str) -> int:
    print(f"RUN_ID={result.run_id}")
    if result.report_path:
        print(f"REPORT={result.report_path}")
    if getattr(result, "report_html_path", None):
        print(f"REPORT_HTML={result.report_html_path}")
    print(f"COST={result.cost_usd:.4f}")
    if result.status == "paused":
        print("STATUS=paused")
        print(terminal_safe(result.message))
        print(f"Resume with: hmd resume {slug}")
        return EXIT_PAUSED
    if result.status == "error":
        print("STATUS=error")
        if getattr(result, "fatal", False):
            # One line: the run stopped on an error every call would hit (auth, billing, credit
            # balance, subscription limit); the runner already made it single-line.
            print(f"ERROR: {terminal_safe(result.message)} -- the run stopped; fix the account/key and run hmd resume {slug}")
        else:
            print(terminal_safe(result.message))
        return EXIT_ERROR
    print("STATUS=done")
    return EXIT_OK


def _notebook(cfg: Config, ref: str) -> dict:
    notebook = db.get_notebook(cfg.db_path, ref)
    if not notebook:
        raise ConfigError(f"notebook {ref!r} not found; `hmd status` lists notebooks")
    return notebook


def _optional_command(module_name: str, label: str):
    """Import module_name for an as-yet-unbuilt command (doctor/init/eval). Returns the module,
    or None when module_name itself is what's missing -- the normal "not built yet" case. Any
    other ModuleNotFoundError (module_name exists but one of *its* imports is broken) is
    re-raised so a real bug surfaces instead of a misleading "not available yet"."""
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        if e.name != module_name:
            raise
        print(f"hmd: {label} is not available yet", file=sys.stderr)
        return None


def cmd_models_use(cfg: Config, config_path: str, args) -> int:
    """`hmd models use <stage> <model>` (IDEA-302): edits one line of the config file; runs nothing."""
    from pipeline import models_monitor
    from pipeline.pricing import price_for
    old = models_monitor.set_stage_model(config_path, args.stage, args.model)
    print(f"{args.stage}: {old} -> {args.model} ({config_path})")
    if cfg.llm.provider == "anthropic_api" and price_for(args.model) is None:
        print(f"warning: {args.model} has no price in pipeline/pricing.py; a budgeted run refuses it "
              f"(add the price, or run with --budget 0)")
    return EXIT_OK


def _model_notice(cfg: Config) -> None:
    """One line from the last `hmd doctor` model check (IDEA-302), only when fresh and only when something
    needs attention. Never touches the network."""
    from pipeline import models_monitor
    findings = models_monitor.cached_findings(cfg)
    line = models_monitor.summary_line(findings) if findings else None
    if line:
        print(line)


def cmd_run(cfg: Config, args) -> int:
    inputs = [args.input_file, args.url, args.playlist, args.file_path]
    if sum(1 for x in inputs if x) != 1:
        raise ConfigError("exactly one of --input, --url, --playlist, --file is required")
    topic = None
    if args.topic_file:
        with open(args.topic_file, encoding="utf-8") as f:
            topic = f.read().strip()
    elif args.topic:
        topic = args.topic.strip()
    if not topic:
        raise ConfigError("--topic or --topic-file is required")
    # Collapse embedded newlines/whitespace runs: the topic is echoed back verbatim after
    # NOTEBOOK= by `status` and `notebook list`, so a topic containing a raw "\n" could otherwise
    # forge an arbitrary KEY=value line (e.g. a fake REPORT=) in output the skills parse.
    topic = " ".join(topic.split())
    stages = _stages(args.stages, STAGES)
    runner = _runner(cfg, args, stages)
    _model_notice(cfg)
    # The notebook is created before validate() -- unlike resume/analyze below -- because
    # validate() needs a notebook id and this is a brand-new notebook. The validate() call below
    # is defensive rather than live: `cmd_run` has no --force/--retranscribe flags, _stages()
    # above already rejects an unknown stage, and the earlier "exactly one of --input/--url/
    # --playlist" check guarantees has_input, so every branch inside validate() that could raise
    # for a `run` call is currently unreachable. If that ever changes (a new flag, a relaxed
    # stages check), validate() failing here would still leave the just-created notebook row
    # behind: this is a usage error, not a run, so nothing is printed and the ValueError/
    # ConfigError propagates to main()'s exit-4 mapping exactly as before. A stray empty
    # notebook row from a usage error is harmless clutter, not a correctness problem.
    notebook = runner.create_notebook(topic)
    runner.validate(notebook["id"], stages, input_file=args.input_file, url=args.url, playlist_url=args.playlist,
                    file_path=args.file_path)
    print(f"NOTEBOOK={notebook['slug']}")
    result = runner.run(notebook["id"], stages, input_file=args.input_file, url=args.url,
                        playlist_url=args.playlist, file_path=args.file_path)
    return _finish(result, notebook["slug"])


def cmd_resume(cfg: Config, args) -> int:
    notebook = _notebook(cfg, args.notebook)
    stages = _stages(args.stages, ["transcribe", "analyze", "merge", "output"])
    runner = _runner(cfg, args, stages)
    _model_notice(cfg)
    runner.validate(notebook["id"], stages, retranscribe=args.retranscribe)
    print(f"NOTEBOOK={notebook['slug']}")
    result = runner.run(notebook["id"], stages, retranscribe=args.retranscribe)
    return _finish(result, notebook["slug"])


def cmd_analyze(cfg: Config, args) -> int:
    notebook = _notebook(cfg, args.notebook)
    stages = _stages(args.stages, ["analyze", "merge", "output"])
    runner = _runner(cfg, args, stages)
    _model_notice(cfg)
    runner.validate(notebook["id"], stages, force_analyze=args.force)
    print(f"NOTEBOOK={notebook['slug']}")
    result = runner.run(notebook["id"], stages, force_analyze=args.force)
    return _finish(result, notebook["slug"])


def cmd_notebook_list(cfg: Config, args) -> int:
    notebooks = db.list_notebooks(cfg.db_path)
    if not notebooks:
        print("No notebooks yet. Start with: hmd run --topic '...' --input urls.txt")
        return EXIT_OK
    for nb in notebooks:
        run = db.latest_run(cfg.db_path, nb["id"])
        state = run["status"] if run else "new"
        print(f"{nb['slug']:<24}[{state}]  {terminal_safe(nb['topic'])}")
    return EXIT_OK


def cmd_notebook_add(cfg: Config, args) -> int:
    """IDEA-311: register new sources on an existing notebook; never transcribes or bills. The next
    `hmd resume` fetches metadata, transcribes and analyzes only the new sources and re-merges."""
    if sum(1 for x in (args.input_file, args.url, args.playlist, args.file_path) if x) != 1:
        raise ConfigError("exactly one of --input, --url, --playlist, --file is required")
    notebook = _notebook(cfg, args.notebook)
    before = {s["id"] for s in db.get_notebook_sources(cfg.db_path, notebook["id"])}
    ids = fetch_stage(cfg.db_path, notebook["id"], input_file=args.input_file, url=args.url,
                      playlist_url=args.playlist, file_path=args.file_path, log=safe_log(print))
    added = [i for i in ids if i not in before]
    print(f"NOTEBOOK={notebook['slug']}")
    print(f"ADDED={len(added)}")
    print(f"ALREADY={len(ids) - len(added)}")
    print(f"Next: hmd resume {notebook['slug']}" if added else "Nothing new to process.")
    return EXIT_OK


def cmd_dry_run(cfg: Config, args) -> int:
    from pipeline.estimate import estimate_notebook
    from pipeline.pricing import price_for
    notebook = _notebook(cfg, args.notebook)
    provider = args.provider or cfg.llm.provider
    _check_batch(provider, args.batch)
    batch = args.batch if args.batch is not None else (cfg.llm.batch.enabled and provider == "anthropic_api")
    analyze_model = args.analyze_model or cfg.llm.models["analyze"]
    merge_model = args.merge_model or cfg.llm.models["merge"]
    for model in dict.fromkeys((analyze_model, merge_model)):
        if price_for(model) is None:                 # as a budgeted run refuses it (Runner.__init__)
            raise ConfigError(f"no price for model {model!r} in pipeline/pricing.py; add it to estimate a run")
    e = estimate_notebook(cfg, notebook["id"], analyze_model=analyze_model, merge_model=merge_model,
                          batch=batch, force=args.force)
    print(f"NOTEBOOK={notebook['slug']}")
    print(f"ESTIMATE_USD={e.analyze_cost + e.merge_cost:.2f}")
    print(f"ESTIMATE_RANGE={e.low:.2f}-{e.high:.2f}")
    print(f"ESTIMATE_MINUTES={round(e.minutes)}")
    note = "nominal (claude_cli uses your plan limits, not money)" if provider == "claude_cli" else "USD"
    print(f"  {e.sources_to_analyze} of {e.sources} sources to analyze, merge over all; {note}"
          f"{'; batch prices' if batch else ''}")
    if e.from_duration:
        print("  some sources are not transcribed yet: their size is estimated from the video length")
    if e.unknown_sources:
        n = e.unknown_sources
        print(f"  size unknown for {n} source{'' if n == 1 else 's'}; the estimate leaves {'it' if n == 1 else 'them'} out, "
              "so it understates the merge too")
    n = e.calibration.history_runs
    print(f"  calibrated on {n} earlier run{'' if n == 1 else 's'} on these models" if n
          else "  built-in calibration (no comparable anthropic_api run on these models yet)")
    return EXIT_OK


def cmd_status(cfg: Config, args) -> int:
    if not args.notebook:
        return cmd_notebook_list(cfg, args)
    notebook = _notebook(cfg, args.notebook)
    # The topic is untrusted, user-supplied text: keep it off any line that starts with a
    # KEY=value token a skill might parse (see cmd_run's newline-collapsing comment).
    print(f"Notebook {notebook['id']} '{notebook['slug']}'")
    print(f"  topic: {terminal_safe(notebook['topic'])}")
    sources = db.get_notebook_sources(cfg.db_path, notebook["id"])
    if sources:
        print(f"{'id':<5}{'kind':<8}{'source':<23}{'fetch':<10}{'transcribe':<12}{'analyze':<10}title")
        for s in sources:
            fetch_status = s["fetch_status"] or "?"
            transcribe_status = s["transcribe_status"] or "?"
            analyze_status = s["analyze_status"] or "?"
            # repr(), not the raw string: a video title is untrusted, attacker-influenced text
            # (it comes from YouTube) and this is the last column on the line -- repr() escapes any
            # embedded newline so a forged title can never start a fake KEY=value line below it.
            title = repr(s["title"] or "")
            print(f"{s['id']:<5}{s['kind']:<8}{s['video_id']:<23}{fetch_status:<10}{transcribe_status:<12}"
                 f"{analyze_status:<10}{title}")
            if s["error_message"]:
                # one line: a page's or yt-dlp's message can hold newlines (see the title's repr above)
                print(f"  error: {terminal_safe(' '.join(str(s['error_message']).split()))}")
    else:
        print("No sources yet.")
    run = db.latest_run(cfg.db_path, notebook["id"])
    if run:
        budget = run["budget_usd"] or 0
        print(f"Last run {run['id']}: {run['status']} stages={run['stages']} "
             f"provider={run['provider']} budget=${budget:.2f}")
        if run["error"]:
            print(f"  error: {terminal_safe(' '.join(str(run['error']).split()))}")
        print(usage_report(cfg.db_path, run["id"]))
        timing = timing_report(cfg.db_path, notebook["id"], run["id"])
        if timing:
            print(timing)
    quality = quality_report(cfg.db_path, notebook["id"])
    if quality:
        print(quality)
    merge = db.get_latest_merge(cfg.db_path, notebook["id"])
    if merge and merge.get("report_path"):
        print(f"REPORT={merge['report_path']}")
        twin = html_twin_path(merge["report_path"])
        # a twin older than its report was left by an earlier render (e.g. a later --no-html one)
        if (os.path.isfile(twin) and os.path.isfile(merge["report_path"])
                and os.path.getmtime(twin) >= os.path.getmtime(merge["report_path"])):
            print(f"REPORT_HTML={twin}")
    return EXIT_OK


def configure_console() -> None:
    """Write stdout/stderr as UTF-8 and replace anything unencodable, so printing model or YouTube
    text can never raise. Only streams that support `reconfigure` (io.TextIOWrapper) are touched;
    line buffering and newline handling are left as they are. A stream that refuses (already
    detached, or one that cannot change encoding) is left alone."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError, AttributeError):
            pass


def _run_delegate(cmd: list[str]) -> int:
    """Run another venv's hmd with this process's arguments and environment; its exit code."""
    import subprocess
    return subprocess.call(cmd)


def main(argv: list[str] | None = None) -> int:
    configure_console()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config_path, config_source = resolve_config_path(args.config)
        envfile.set_config_path(config_path, explicit=config_source != "default")
        args.config_path, args.config_source, args.mode = config_path, config_source, run_mode()
        if args.mode == "plugin":
            bootstrap.touch_last_used(sys.prefix)       # keeps this venv from idle pruning

        stale = plugin_stale_reason()
        if stale:
            # The launcher <data>/hmd is shared by every session; one still on an older plugin
            # root lands here after /hmd:init re-pointed it. Run the venv built for that root
            # (kept beside the new one) instead of refusing.
            target = bootstrap.installed_hmd(os.environ.get(bootstrap.PLUGIN_ROOT_ENV, "").strip(),
                                             plugin_data_dir())
            if target:
                return _run_delegate([target] + list(sys.argv[1:] if argv is None else argv))
        if stale and args.command != "doctor":
            raise ConfigError(f"this plugin's Python environment is out of date ({stale}); "
                              f"run /hmd:init to update it")

        if args.command == "init":
            # init creates config.yaml, so it must run before the explicit-config existence
            # check and before load_config -- `hmd init --config custom.yaml` targets a path
            # that legitimately does not exist yet.
            args.config = config_path
            mod = _optional_command("pipeline.init_cmd", "init")
            if mod is None:
                return EXIT_USAGE
            return mod.cmd_init(args)

        args.config_missing = config_source != "default" and not os.path.exists(config_path)
        if args.config_missing and args.command == "doctor":
            cfg = Config()          # doctor reports the missing file as a [FAIL] check
        elif args.config_missing:
            hint = {"--config": "", CONFIG_ENV: f" (from {CONFIG_ENV})",
                    "plugin": "; run /hmd:init (or hmd init) first"}[config_source]
            raise ConfigError(f"config file not found: {config_path}{hint}")
        else:
            cfg = load_config(config_path, explicit=config_source != "default")
            if cfg.transcription.allow_private_hosts or "pipeline.transcription.web_text" in sys.modules:
                from pipeline.transcription import web_text     # metadata fetches run in this process
                web_text.set_allow_private_hosts(cfg.transcription.allow_private_hosts)

        if args.command == "models":
            return cmd_models_use(cfg, config_path, args)

        if args.command == "doctor":
            mod = _optional_command("pipeline.doctor", "doctor")
            if mod is None:
                return EXIT_USAGE
            return mod.cmd_doctor(cfg, args)

        db.init_db(cfg.db_path)

        if args.command == "run":
            return cmd_run(cfg, args)
        if args.command == "resume":
            return cmd_resume(cfg, args)
        if args.command == "analyze":
            return cmd_analyze(cfg, args)
        if args.command == "status":
            return cmd_status(cfg, args)
        if args.command == "dry-run":
            return cmd_dry_run(cfg, args)
        if args.command == "notebook":
            # args.notebook_command is required=True with "list" and "add" as the only choices.
            if args.notebook_command == "add":
                return cmd_notebook_add(cfg, args)
            return cmd_notebook_list(cfg, args)
        if args.command == "eval":
            mod = _optional_command("pipeline.evals", "eval")
            if mod is None:
                return EXIT_USAGE
            return mod.cmd_eval(cfg, args)
        # Every subcommand above returns explicitly. This guards against a new subcommand being
        # added to the parser without a matching dispatch branch here -- without it, main() would
        # fall off the end and return None, which sys.exit() treats as a silent exit code 0.
        raise ConfigError(f"unhandled command {args.command!r}")
    except (ConfigError, db.LegacyDatabaseError, LLMError, OSError, ValueError, sqlite3.Error) as e:
        if os.environ.get("HMD_DEBUG"):
            raise
        print(f"hmd: {terminal_safe(e)}", file=sys.stderr)
        return EXIT_USAGE
    except KeyboardInterrupt:
        print("hmd: interrupted", file=sys.stderr)
        return EXIT_INTERRUPTED


if __name__ == "__main__":
    sys.exit(main())
