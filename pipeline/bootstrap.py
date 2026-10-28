"""Plugin-mode environment (IDEA-310): `/hmd:init` gives an installed plugin its own venv.

An installed Claude Code plugin has no `.venv`: its files live under `${CLAUDE_PLUGIN_ROOT}`, which
changes on every update, while `${CLAUDE_PLUGIN_DATA}` survives updates (and is deleted on
uninstall). The /hmd:init skill runs this file as a script with any Python 3 on PATH:

    python3 "<plugin root>/pipeline/bootstrap.py" "<plugin data dir>"

and it

1. checks the running Python against `requires-python` in the plugin's pyproject.toml and stops
   with a clear message (exit 4) when it is too old;
2. keys the venv by a digest of what it is built from (`env_digest`: plugin version,
   pyproject.toml, requirements.txt, constraints.txt, requirements.lock): `<data>/venv/<digest>`;
3. returns at once when that venv holds a matching install marker and the launcher exists
   (`BOOTSTRAP=ok`);
4. otherwise takes the lock `<data>/venv/.lock` (a second /hmd:init waits for it; a lock older
   than 30 minutes is stale), builds the new venv BESIDE any older one (a session still running
   hmd from the old venv keeps working), installs the hashed requirements.lock (`--require-hashes`;
   without one, requirements.txt with constraints.txt), then the package itself (non-editable,
   `--no-build-isolation` against the locked setuptools, from a temporary copy so nothing is
   written into the plugin folder), writes the marker `<venv>/hmd-plugin.json`, and only then
   points the launcher `<data>/hmd` (a POSIX sh script that runs the venv's hmd) at the new venv;
   prints `BOOTSTRAP=installed`;
5. removes other digests' venvs idle for over a day (`last_used` in their marker, touched by every
   `hmd` start in plugin mode);
6. prints where the config lives and a suggested home for the reports (`~/HiveMindDigest`,
   absolute): the data home must not be the plugin data dir, which uninstalling deletes.

The marker lives inside the venv, so `hmd` finds it through `sys.prefix`: that is how `hmd` knows
it runs in plugin mode, where its default config is `<data>/config.yaml` (see
pipeline/config.py::resolve_config_path). When the skills pass `HMD_PLUGIN_ROOT`, `hmd` compares
the marker with that root and refuses to run a stale install (pipeline/cli.py).

Run as a script, the plugin's `pipeline/` directory is sys.path[0], so this file imports the
standard library only and no module in `pipeline/` may shadow one it imports
(tests/test_bootstrap.py checks both). It also never imports pydantic or anything else from
requirements.txt: it runs before they exist.
"""
import argparse
import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

MARKER_NAME = "hmd-plugin.json"
VENV_DIRNAME = "venv"
LAUNCHER_NAME = "hmd"
CONFIG_NAME = "config.yaml"
PLUGIN_ROOT_ENV = "HMD_PLUGIN_ROOT"
SUGGESTED_HOME = os.path.join("~", "HiveMindDigest")   # outside the plugin data dir on purpose
DEFAULT_MIN_PYTHON = (3, 12)
LOCK_NAME = ".lock"
LOCK_STALE_SECONDS = 30 * 60        # a lock this old belongs to a crashed /hmd:init
LOCK_WAIT_SECONDS = 10 * 60         # how long a second /hmd:init waits for the first one
LOCK_POLL_SECONDS = 2
IDLE_SECONDS = 24 * 60 * 60         # other digests' venvs unused this long are removed
_DIGEST_RE = re.compile(r"^[0-9a-f]{12}$")
_DIGEST_FILES = ("pyproject.toml", "requirements.txt", "constraints.txt", "requirements.lock")

EXIT_OK = 0
EXIT_ERROR = 2      # venv creation or pip failed
EXIT_USAGE = 4      # Python too old, or the plugin root is not a plugin root

# Files of the plugin root that make up the installable package (pyproject's packages.find
# includes pipeline* only); copied to a temp dir so pip never writes build/ or *.egg-info into
# the plugin folder.
_PACKAGE_FILES = ("pyproject.toml",)
_PACKAGE_DIRS = ("pipeline",)
_SKIP_DIRS = ("__pycache__",)


def default_plugin_root():
    """The plugin (or checkout) root this file belongs to: pipeline/bootstrap.py -> root."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read_text(path):
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except OSError:
        return None


def min_python(root):
    """(major, minor) from `requires-python = ">=X.Y"` in root/pyproject.toml, else the default."""
    text = _read_text(os.path.join(root, "pyproject.toml")) or ""
    m = re.search(r'^\s*requires-python\s*=\s*["\']\s*>=\s*(\d+)\.(\d+)', text, re.M)
    return (int(m.group(1)), int(m.group(2))) if m else DEFAULT_MIN_PYTHON


def plugin_version(root):
    """The plugin's version: `.claude-plugin/plugin.json` when present (an installed plugin),
    else `version` in pyproject.toml (a development checkout), else "unknown"."""
    text = _read_text(os.path.join(root, ".claude-plugin", "plugin.json"))
    if text:
        try:
            version = json.loads(text).get("version")
            if version:
                return str(version)
        except ValueError:
            pass
    text = _read_text(os.path.join(root, "pyproject.toml")) or ""
    m = re.search(r'^\s*version\s*=\s*["\']([^"\']+)["\']', text, re.M)
    return m.group(1) if m else "unknown"


def requirements_sha256(root):
    """sha256 of requirements.txt with line endings normalized (a CRLF checkout of the same file
    must not look changed); "" when the file is missing."""
    try:
        with open(os.path.join(root, "requirements.txt"), "rb") as f:
            data = f.read()
    except OSError:
        return ""
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def desired_state(root):
    """What an up-to-date install of this plugin root records in its marker."""
    return {"plugin_version": plugin_version(root), "requirements_sha256": requirements_sha256(root)}


def env_digest(root):
    """First 12 hex of a sha256 over the plugin version and pyproject.toml, requirements.txt and
    constraints.txt (line endings normalized; a missing file counts as empty)."""
    h = hashlib.sha256(plugin_version(root).encode("utf-8") + b"\0")
    for name in _DIGEST_FILES:
        try:
            with open(os.path.join(root, name), "rb") as f:
                data = f.read()
        except OSError:
            data = b""
        h.update(name.encode("ascii") + b"\0" + data.replace(b"\r\n", b"\n") + b"\0")
    return h.hexdigest()[:12]


def venvs_dir(data_dir):
    return os.path.join(data_dir, VENV_DIRNAME)


def venv_dir(data_dir, digest):
    return os.path.join(venvs_dir(data_dir), digest)


def data_dir_of(prefix):
    """The plugin data dir a venv belongs to: `<data>/venv/<digest>` -> `<data>`; the pre-0.1 flat
    layout `<data>/venv` -> `<data>`."""
    prefix = os.path.abspath(prefix)
    parent = os.path.dirname(prefix)
    if os.path.basename(parent) == VENV_DIRNAME and _DIGEST_RE.match(os.path.basename(prefix)):
        return os.path.dirname(parent)
    return parent


def _bin_dir(venv):
    return os.path.join(venv, "Scripts" if os.name == "nt" else "bin")


def venv_python(venv):
    return os.path.join(_bin_dir(venv), "python.exe" if os.name == "nt" else "python")


def venv_hmd(venv):
    return os.path.join(_bin_dir(venv), "hmd.exe" if os.name == "nt" else "hmd")


def read_marker(venv):
    """The install marker of a venv as a dict, or None (no venv, no marker, unreadable)."""
    text = _read_text(os.path.join(venv, MARKER_NAME))
    if text is None:
        return None
    try:
        data = json.loads(text)
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def stale_reason(marker, desired):
    """None when `marker` matches `desired`; otherwise one short reason for a re-install."""
    if not marker:
        return "not installed"
    if marker.get("plugin_version") != desired["plugin_version"]:
        return "plugin version {} -> {}".format(marker.get("plugin_version"), desired["plugin_version"])
    if marker.get("requirements_sha256") != desired["requirements_sha256"]:
        return "requirements.txt changed"
    return None


def _write_marker(venv, marker):
    # A temp name per process: two hmd starts touching the marker at once must not write into
    # the same temp file (a corrupt marker would silently drop plugin mode).
    tmp = os.path.join(venv, "{}.{}.tmp".format(MARKER_NAME, os.getpid()))
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(marker, f, indent=2)
            f.write("\n")
        os.replace(tmp, os.path.join(venv, MARKER_NAME))
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def touch_last_used(venv, now=None):
    """Record in the venv's marker that hmd ran from it now (plugin mode); never raises."""
    marker = read_marker(venv)
    if marker is None:
        return
    marker["last_used"] = (now or datetime.datetime.now(datetime.timezone.utc)).isoformat()
    try:
        _write_marker(venv, marker)
    except OSError:
        pass


def _last_used(venv):
    marker = read_marker(venv) or {}
    for key in ("last_used", "installed_at"):
        try:
            when = datetime.datetime.fromisoformat(marker[key])
        except (KeyError, TypeError, ValueError):
            continue
        return when if when.tzinfo else when.replace(tzinfo=datetime.timezone.utc)
    try:
        return datetime.datetime.fromtimestamp(os.path.getmtime(venv), datetime.timezone.utc)
    except OSError:
        return None


def prune_old_venvs(data_dir, keep, now=None, idle_seconds=IDLE_SECONDS):
    """Remove the venvs of digests other than `keep` last used over `idle_seconds` ago; returns
    the removed paths. Only `<data>/venv/<12 hex>` directories are considered."""
    now = now or datetime.datetime.now(datetime.timezone.utc)
    base = venvs_dir(data_dir)
    try:
        names = sorted(os.listdir(base))
    except OSError:
        return []
    removed = []
    for name in names:
        path = os.path.join(base, name)
        if name == keep or not _DIGEST_RE.match(name) or not os.path.isdir(path):
            continue
        when = _last_used(path)
        if when is not None and (now - when).total_seconds() < idle_seconds:
            continue
        try:
            _rmtree(path)
            removed.append(path)
        except OSError:
            pass                        # still in use on Windows: the next /hmd:init retries
    return removed


def _acquire_lock(path, sleep, clock):
    """Create-exclusive lock file; one older than LOCK_STALE_SECONDS is taken over. True when
    acquired, False after waiting LOCK_WAIT_SECONDS."""
    deadline = clock() + LOCK_WAIT_SECONDS
    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                age = clock() - os.path.getmtime(path)
            except OSError:
                continue                # released meanwhile
            if age > LOCK_STALE_SECONDS:
                try:
                    os.remove(path)
                except OSError:
                    pass
                continue
            if clock() >= deadline:
                return False
            sleep(LOCK_POLL_SECONDS)
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write("{}\n".format(os.getpid()))
        return True


def installed_hmd(root, data_dir):
    """The hmd of an up-to-date venv for plugin root `root` under `data_dir`, or None (no root,
    no data dir, not built, or stale)."""
    if not root or not data_dir or not os.path.isfile(os.path.join(root, "pyproject.toml")):
        return None
    if install_status(root, data_dir) is not None:
        return None
    return venv_hmd(venv_dir(data_dir, env_digest(root)))


def install_status(root, data_dir):
    """None when `data_dir` holds an up-to-date install for `root`, else the reason it does not."""
    venv = venv_dir(data_dir, env_digest(root))
    if not os.path.isfile(venv_hmd(venv)):
        return "not installed"
    return stale_reason(read_marker(venv), desired_state(root))


def launcher_text(venv):
    """A POSIX sh launcher for the venv's hmd. Absolute paths with forward slashes work in Git Bash
    on Windows and in a native Windows program alike."""
    target = venv_hmd(os.path.abspath(venv)).replace("\\", "/")
    return ("#!/bin/sh\n"
            "# Written by /hmd:init (pipeline/bootstrap.py): runs the plugin's own hmd.\n"
            "exec \"{}\" \"$@\"\n".format(target))


def _copy_package(root, dest):
    for name in _PACKAGE_FILES:
        shutil.copy2(os.path.join(root, name), os.path.join(dest, name))
    for name in _PACKAGE_DIRS:
        shutil.copytree(os.path.join(root, name), os.path.join(dest, name),
                        ignore=shutil.ignore_patterns(*_SKIP_DIRS, "*.pyc"))


def _rmtree(path):
    def retry(func, p, _exc):
        os.chmod(p, 0o700)       # Windows: a read-only file
        func(p)
    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=retry)
    else:
        shutil.rmtree(path, onerror=retry)


def _run(cmd, runner, log):
    proc = runner(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        tail = "\n".join(((proc.stdout or "") + (proc.stderr or "")).strip().splitlines()[-30:])
        log("command failed (exit {}): {}\n{}".format(proc.returncode, " ".join(cmd), tail))
        if "Long Path" in tail:
            log("The plugin data folder's path is too long for the Windows 260-character path limit. "
                "Enable long paths (https://pip.pypa.io/warnings/enable-long-paths) and run "
                "/hmd:init again.")
        return False
    return True


def _write_launcher(launcher, venv):
    with open(launcher, "w", encoding="utf-8", newline="\n") as f:
        f.write(launcher_text(venv))
    os.chmod(launcher, 0o755)


def _remove_flat_venv(base):
    """The pre-0.1 layout had the venv itself at `<data>/venv`; keep only the digest dirs and the
    lock beside it."""
    for name in os.listdir(base):
        path = os.path.join(base, name)
        if name == LOCK_NAME or (_DIGEST_RE.match(name) and os.path.isdir(path)):
            continue
        if os.path.isdir(path):
            _rmtree(path)
        else:
            os.remove(path)


def _install(root, venv, python, runner, log, version_info, now):
    """Build `venv` from scratch; True when the venv, the requirements and the package installed."""
    if os.path.isdir(venv):
        _rmtree(venv)                   # a half-built venv of this digest: nobody runs from it
    if not _run([python, "-m", "venv", venv], runner, log):
        return False
    vpy = venv_python(venv)
    pip = [vpy, "-m", "pip", "install", "--disable-pip-version-check", "--quiet"]
    lock = os.path.join(root, "requirements.lock")
    if os.path.isfile(lock):
        # Every runtime pin and setuptools with its hashes; the package below then builds against
        # that setuptools, so pip downloads nothing unhashed (release security review, 0.1.1).
        reqs = ["--require-hashes", "--no-deps", "-r", lock]
        package = ["--no-deps", "--no-build-isolation"]
    else:                               # an older plugin root without the lock
        reqs = ["-r", os.path.join(root, "requirements.txt")]
        if os.path.isfile(os.path.join(root, "constraints.txt")):
            reqs += ["-c", os.path.join(root, "constraints.txt")]   # transitive pins
        package = []
    if not _run(pip + reqs, runner, log):
        return False
    tmp = tempfile.mkdtemp(prefix="hmd-install-")
    try:
        _copy_package(root, tmp)
        if not _run(pip + package + [tmp], runner, log):
            return False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    stamp = (now or datetime.datetime.now(datetime.timezone.utc)).isoformat()
    marker = dict(desired_state(root))
    marker["python"] = "{}.{}.{}".format(*version_info[:3])
    marker["installed_at"] = stamp
    marker["last_used"] = stamp
    _write_marker(venv, marker)
    return True


def ensure(root, data_dir, runner=subprocess.run, python=None, version_info=None, log=print, now=None,
           sleep=time.sleep, clock=time.time):
    """Bring `data_dir` to an up-to-date install of `root`. Returns an exit code; prints KEY=value
    lines (PYTHON=, BOOTSTRAP=, VENV=, HMD=, CONFIG=, SUGGESTED_HOME=) for the skill."""
    python = python or sys.executable
    version_info = version_info or sys.version_info
    root = os.path.abspath(root)
    data_dir = os.path.abspath(os.path.expanduser(data_dir))
    need = min_python(root)
    have = tuple(version_info[:2])
    log("PYTHON={}.{}.{} ({})".format(version_info[0], version_info[1], version_info[2], python))
    if have < need:
        log("BOOTSTRAP=fail")
        log("hmd needs Python {}.{} or newer; this is {}.{} ({}). Install a newer Python from "
            "python.org (or your package manager) and run /hmd:init again.".format(
                need[0], need[1], have[0], have[1], python))
        return EXIT_USAGE
    if not os.path.isfile(os.path.join(root, "requirements.txt")) or \
            not os.path.isfile(os.path.join(root, "pyproject.toml")):
        log("BOOTSTRAP=fail")
        log("{} is not a Hive Mind Digest plugin root (no requirements.txt / pyproject.toml)".format(root))
        return EXIT_USAGE

    digest = env_digest(root)
    venv = venv_dir(data_dir, digest)
    launcher = os.path.join(data_dir, LAUNCHER_NAME)
    config = os.path.join(data_dir, CONFIG_NAME)
    if install_status(root, data_dir) is None:
        if _read_text(launcher) != launcher_text(venv):
            _write_launcher(launcher, venv)
        log("BOOTSTRAP=ok")
    else:
        os.makedirs(venvs_dir(data_dir), exist_ok=True)
        lock = os.path.join(venvs_dir(data_dir), LOCK_NAME)
        if not _acquire_lock(lock, sleep, clock):
            log("BOOTSTRAP=fail")
            log("another /hmd:init is installing into {} (lock {}); wait for it to finish and run "
                "/hmd:init again".format(venvs_dir(data_dir), lock))
            return EXIT_ERROR
        try:
            reason = install_status(root, data_dir)      # the other /hmd:init may have built it
            if reason is not None:
                log("installing hmd into {} ({}); this takes a minute or two...".format(venv, reason))
                if read_marker(venvs_dir(data_dir)) is not None:
                    _remove_flat_venv(venvs_dir(data_dir))
                if not _install(root, venv, python, runner, log, version_info, now):
                    log("BOOTSTRAP=fail")
                    return EXIT_ERROR
            _write_launcher(launcher, venv)             # only after the install succeeded
        finally:
            try:
                os.remove(lock)
            except OSError:
                pass
        log("BOOTSTRAP=installed" if reason is not None else "BOOTSTRAP=ok")
    prune_old_venvs(data_dir, keep=digest, now=now)
    log("VENV={}".format(venv))
    log("HMD={}".format(launcher))
    log("CONFIG={}{}".format(config, "" if os.path.isfile(config) else " (not written yet)"))
    log("SUGGESTED_HOME={}".format(os.path.abspath(os.path.expanduser(SUGGESTED_HOME))))
    return EXIT_OK


def main(argv=None):
    ap = argparse.ArgumentParser(prog="bootstrap.py",
                                 description="Create or refresh the plugin's own venv (IDEA-310).")
    ap.add_argument("data_dir", help="the plugin data directory (${CLAUDE_PLUGIN_DATA})")
    ap.add_argument("--root", default=None, help="plugin root (default: the folder holding pipeline/)")
    args = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="backslashreplace")
    return ensure(args.root or default_plugin_root(), args.data_dir)


if __name__ == "__main__":
    sys.exit(main())
