"""Repo hygiene checks: the CI workflow shape and dependency-file
drift. These are static file checks -- they never import pipeline modules and never touch the
network, so they run the same in CI as locally.
"""
from __future__ import annotations

import pathlib
import re

import pytest
import yaml
from packaging.specifiers import SpecifierSet
from packaging.version import Version

# tomllib is stdlib from Python 3.11; pyproject.toml still declares requires-python >= 3.10, so
# skip (not fail) this module's tomllib-dependent tests on an actual 3.10 interpreter rather than
# breaking collection for the whole suite.
tomllib = pytest.importorskip("tomllib")

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "tests.yml"
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"
REQUIREMENTS_PATH = REPO_ROOT / "requirements.txt"
REQUIREMENTS_WHISPER_PATH = REPO_ROOT / "requirements-whisper.txt"
GITIGNORE_PATH = REPO_ROOT / ".gitignore"


# --- helpers -----------------------------------------------------------------

def _load_workflow() -> dict:
    return yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))


def _pkg_name(spec: str) -> str:
    """The package name portion of a requirement spec (e.g. "anthropic>=0.84.0,<1.0" -> "anthropic"),
    normalized the way pip/PyPI compare distribution names: lowercase, "_" folded to "-"."""
    m = re.match(r"[A-Za-z0-9_.\-]+", spec)
    assert m, f"could not parse a package name out of {spec!r}"
    return m.group(0).lower().replace("_", "-")


def _parse_requirements(path: pathlib.Path) -> list[str]:
    """One requirement spec per non-comment, non-blank, non-"-r ..." line, with any trailing
    "# ..." comment and surrounding whitespace stripped."""
    specs = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-r "):
            continue
        specs.append(line)
    return specs


def _pyproject() -> dict:
    with open(PYPROJECT_PATH, "rb") as f:
        return tomllib.load(f)


# --- Step 1 corrections (item A): the workflow shape -------------------------

def test_workflow_yaml_parses_pins_windows_and_matrix():
    # PyYAML's default (YAML 1.1) loader parses the bare `on:` trigger key as the boolean True,
    # not the string "on" -- a well-known GitHub Actions YAML quirk, not a bug in our workflow
    # file or this test; nothing here reads that key.
    doc = _load_workflow()
    job = doc["jobs"]["pytest"]
    assert job["runs-on"] == "windows-latest"
    assert job["strategy"]["fail-fast"] is False
    assert job["strategy"]["matrix"]["python-version"] == ["3.12", "3.14"]


def test_workflow_installs_requirements_then_editable_package():
    doc = _load_workflow()
    steps = doc["jobs"]["pytest"]["steps"]
    runs = [s.get("run", "") for s in steps]
    req_idx = next(i for i, r in enumerate(runs) if "install -r requirements.txt" in r)
    editable_idx = next(i for i, r in enumerate(runs) if "install -e ." in r)
    assert req_idx < editable_idx


def test_workflow_runs_pytest_tests():
    doc = _load_workflow()
    steps = doc["jobs"]["pytest"]["steps"]
    pytest_steps = [s for s in steps if "pytest tests" in s.get("run", "")]
    assert len(pytest_steps) == 1


def test_workflow_pytest_step_sets_empty_anthropic_api_key():
    doc = _load_workflow()
    steps = doc["jobs"]["pytest"]["steps"]
    pytest_step = next(s for s in steps if "pytest tests" in s.get("run", ""))
    assert pytest_step.get("env", {}).get("ANTHROPIC_API_KEY") == ""


# --- item B: dependency-drift (requirements*.txt hold exact pins, IDEA-284; pyproject.toml keeps
# the compatible ranges; same package names on both sides, every pin inside its range) ---------

def _pins(path: pathlib.Path) -> dict[str, str]:
    """name -> pinned version for a requirements file that holds `name==version` lines only."""
    pins = {}
    for spec in _parse_requirements(path):
        name, sep, version = spec.partition("==")
        assert sep and version and not any(c in version for c in "<>=!~*,;"), (
            f"{path.name}: {spec!r} is not an exact pin (IDEA-284 wants name==version)")
        pins[_pkg_name(name)] = version.strip()
    return pins


def _assert_pins_match(pyproject_specs: list[str], pins: dict[str, str], label: str) -> None:
    """Same package names on both sides, and every pin lies inside its pyproject range."""
    ranges = {_pkg_name(s): SpecifierSet(s[len(re.match(r"[A-Za-z0-9_.\-]+", s).group(0)):].strip())
              for s in pyproject_specs}
    assert set(ranges) == set(pins), (
        f"{label} drift: only in pyproject={set(ranges) - set(pins)} "
        f"only in requirements={set(pins) - set(ranges)}")
    outside = {n: (pins[n], str(ranges[n])) for n in pins if Version(pins[n]) not in ranges[n]}
    assert not outside, f"{label}: pinned versions outside the pyproject range: {outside}"


def test_pyproject_dependencies_match_requirements_txt():
    pyproject = _pyproject()
    dev_names = {_pkg_name(s) for s in pyproject["project"]["optional-dependencies"]["dev"]}
    pins = {n: v for n, v in _pins(REQUIREMENTS_PATH).items() if n not in dev_names}
    _assert_pins_match(pyproject["project"]["dependencies"], pins,
                       "pyproject.toml [project].dependencies vs requirements.txt")


def test_pyproject_dev_extra_matches_requirements_txt_test_deps():
    pyproject = _pyproject()
    dev_specs = pyproject["project"]["optional-dependencies"]["dev"]
    dev_names = {_pkg_name(s) for s in dev_specs}
    pins = {n: v for n, v in _pins(REQUIREMENTS_PATH).items() if n in dev_names}
    _assert_pins_match(dev_specs, pins,
                       "pyproject.toml [project.optional-dependencies].dev vs requirements.txt")


def test_pyproject_whisper_extra_matches_requirements_whisper_txt():
    pyproject = _pyproject()
    lines = [l.strip() for l in REQUIREMENTS_WHISPER_PATH.read_text(encoding="utf-8").splitlines()]
    assert "-r requirements.txt" in lines
    _assert_pins_match(pyproject["project"]["optional-dependencies"]["whisper"],
                       _pins(REQUIREMENTS_WHISPER_PATH),
                       "pyproject.toml [project.optional-dependencies].whisper vs requirements-whisper.txt")


def test_gitignore_has_each_required_entry_exactly_once():
    required = ["data/", "build/", "*.egg-info/", ".env", "credentials/", "videos-*.txt"]
    lines = [l.strip() for l in GITIGNORE_PATH.read_text(encoding="utf-8").splitlines()]
    for entry in required:
        count = lines.count(entry)
        assert count == 1, f"{entry!r} appears {count} times in .gitignore, expected exactly 1"


# --- item C: README.md, CLAUDE.md and the skills stay ASCII (redirected Windows stdout raises on
# a Unicode lookalike it can't encode) ------------------------------------------------------

def test_claude_md_and_skills_are_ascii():
    # A development checkout has .claude/skills/ and a CLAUDE.md; the public plugin export has
    # skills/ and no CLAUDE.md.
    guides = [REPO_ROOT / name for name in ("README.md", "CLAUDE.md") if (REPO_ROOT / name).is_file()]
    files = (guides + sorted((REPO_ROOT / ".claude" / "skills").glob("*/SKILL.md"))
             + sorted((REPO_ROOT / "skills").glob("*/SKILL.md")))
    assert len(files) > 1,"expected README.md plus at least one skills SKILL.md"
    offenses = []
    for f in files:
        text = f.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if any(ord(c) > 127 for c in line):
                offenses.append(f"{f.relative_to(REPO_ROOT)}:{lineno}: {line!r}")
    assert not offenses, "non-ASCII line(s):\n" + "\n".join(offenses)


# --- transitive pins (blocker 3 of Public Preview 0.1): constraints.txt -------------------------

CONSTRAINTS_PATH = REPO_ROOT / "constraints.txt"
_CONSTRAINT_LINE = re.compile(r"^[A-Za-z0-9._-]+==[^\s;]+(\s*;.*)?$")


def _constraint_lines() -> list[str]:
    return [raw.split("#", 1)[0].strip() for raw in CONSTRAINTS_PATH.read_text(encoding="utf-8").splitlines()
            if raw.split("#", 1)[0].strip()]


def test_constraints_pin_every_line():
    lines = _constraint_lines()
    assert lines, "constraints.txt is empty"
    bad = [line for line in lines if not _CONSTRAINT_LINE.match(line)]
    assert not bad, bad


def test_constraints_cover_requirements():
    pins = {}
    for line in _constraint_lines():
        name, rest = line.split("==", 1)
        pins.setdefault(_pkg_name(name), set()).add(rest.split(";", 1)[0].strip())
    for spec in _parse_requirements(REQUIREMENTS_PATH):
        name, version = spec.split("==", 1)
        assert pins.get(_pkg_name(name)) == {version.strip()}, (name, pins.get(_pkg_name(name)))


def test_workflow_and_bootstrap_install_with_the_constraints():
    runs = [s.get("run", "") for s in _load_workflow()["jobs"]["pytest"]["steps"]]
    assert any("-r requirements.txt" in r and "-c constraints.txt" in r for r in runs), runs
    bootstrap_src = (REPO_ROOT / "pipeline" / "bootstrap.py").read_text(encoding="utf-8")
    assert '"-c", os.path.join(root, "constraints.txt")' in bootstrap_src


def test_python_floor_agrees_everywhere():
    # The pins need 3.11+ (constraints.txt is compiled for it); CI and the README say 3.12+.
    # /hmd:init checks requires-python, so a lower floor would let pip fail obscurely later.
    from pipeline import bootstrap
    floor = re.match(r">=\s*(\d+)\.(\d+)", _pyproject()["project"]["requires-python"])
    assert floor and (int(floor.group(1)), int(floor.group(2))) == (3, 12)
    assert bootstrap.DEFAULT_MIN_PYTHON == (3, 12)
    assert "--python-version 3.11" in CONSTRAINTS_PATH.read_text(encoding="utf-8")
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "Python 3.10" not in readme and "Python 3.12 or newer" in readme


def test_readme_privacy_list_names_every_host_hmd_contacts():
    # README "Cost and privacy" claims nothing else is sent anywhere; keep it complete.
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    section = readme.split("## Cost and privacy", 1)[1].split("\n## ", 1)[0]
    for needle in ("Anthropic API", "claude -p", "YouTube", "Supadata", "Hugging Face", "PyPI",
                   "hmd doctor", "platform.claude.com", "Models API"):
        assert needle in section, needle


def test_claude_md_layout_lists_every_stage_module():
    claude_md = REPO_ROOT / "CLAUDE.md"
    if not claude_md.exists():                       # private file, absent in the public export
        pytest.skip("CLAUDE.md not present")
    line = next(l for l in claude_md.read_text(encoding="utf-8").splitlines() if "pipeline/stages/{" in l)
    listed = set(re.search(r"pipeline/stages/\{([^}]*)\}", line).group(1).split(","))
    modules = {p.stem for p in (REPO_ROOT / "pipeline" / "stages").glob("*.py") if p.stem != "__init__"}
    assert modules <= listed, sorted(modules - listed)
