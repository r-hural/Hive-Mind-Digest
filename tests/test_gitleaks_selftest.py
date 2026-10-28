"""The secret scanners actually catch an Anthropic key (Evidence-Echo lesson): the key is built at
run time, so no literal key-shaped string is ever committed, and the pre-commit hook scans what is
staged."""
from __future__ import annotations

import pathlib
import secrets
import shutil
import string
import subprocess

import pytest
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_KEY_ALPHABET = string.ascii_letters + string.digits + "_-"


def _runtime_key() -> str:
    body = "".join(secrets.choice(_KEY_ALPHABET) for _ in range(93))
    return "sk-" + "ant-api03-" + body + "AA"


@pytest.mark.skipif(shutil.which("gitleaks") is None, reason="gitleaks not on PATH")
def test_gitleaks_detects_runtime_built_anthropic_key(tmp_path):
    (tmp_path / "leak.txt").write_text(f"ANTHROPIC_API_KEY={_runtime_key()}\n", encoding="utf-8")
    proc = subprocess.run(["gitleaks", "detect", "--no-git", "--source", str(tmp_path), "--redact",
                           "--no-banner", "--exit-code", "1"], capture_output=True, text=True)
    assert proc.returncode == 1, proc.stdout + proc.stderr


def test_precommit_hook_scans_staged():
    config = yaml.safe_load((REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8"))
    hooks = [h for repo in config["repos"] for h in repo["hooks"]]
    gitleaks = [h for h in hooks if h["id"] == "gitleaks"]
    assert len(gitleaks) == 1
    hook = gitleaks[0]
    assert hook["entry"].startswith("gitleaks protect --staged")
    assert hook["language"] == "system"
    assert hook["pass_filenames"] is False and hook["always_run"] is True
