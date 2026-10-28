"""Skills must be thin wrappers over the `hmd` CLI.

No inline Python, no MVP-era interpolation pattern, no reference to the deleted MVP surface
(pipeline.subagents, pipeline.stages.*, data/hmd.db, PROJECT_ID/project_id).
"""
import pathlib
import re

from tests.helpers import skill_files


def _skill_paths():
    paths = [str(p) for p in skill_files().values()]
    assert paths, "no skills under .claude/skills/hmd-*/ or skills/*/"
    return paths


# The plugin's own hmd (IDEA-310): the public export rewrites the private skills to this form.
PLUGIN_HMD = 'HMD_PLUGIN_ROOT="${CLAUDE_PLUGIN_ROOT}" "${CLAUDE_PLUGIN_DATA}/hmd"'


def _is_private(path) -> bool:
    return pathlib.Path(path).parent.parent.parent.name == ".claude"


def test_skills_never_inline_python_or_interpolate_into_shell_strings():
    for path in _skill_paths():
        text = open(path, encoding="utf-8").read()
        assert "python -c" not in text, path
        assert not re.search(r'--topic "\{', text), path       # the MVP interpolation pattern


def test_private_skills_run_the_checkout_venv_and_plugin_skills_the_plugin_hmd():
    for path in _skill_paths():
        text = open(path, encoding="utf-8").read()
        if _is_private(path):            # development checkout: .claude/skills/hmd-<x>/
            assert ".venv/Scripts/hmd.exe" in text, path
            assert "${CLAUDE_PLUGIN_DATA}" not in text, path
        else:                            # the public plugin export: skills/<x>/
            assert PLUGIN_HMD in text, path
            assert ".venv" not in text and "hmd.exe" not in text, path


def test_every_skill_documents_exit_codes():
    for path in _skill_paths():
        text = open(path, encoding="utf-8").read()
        if "hmd run" in text or "hmd analyze" in text or "hmd resume" in text:
            assert "exit code 3" in text.lower(), path


def test_no_skill_mentions_deleted_mvp_surface():
    forbidden = ["pipeline.subagents", "pipeline.stages.", "data/hmd.db", "PROJECT_ID", "project_id",
                 "python -"]
    forbidden_ci = ["subagent", "agent tool"]
    for path in _skill_paths():
        text = open(path, encoding="utf-8").read()
        for token in forbidden:
            assert token not in text, (path, token)
        lowered = text.lower()
        for token in forbidden_ci:
            assert token not in lowered, (path, token)
