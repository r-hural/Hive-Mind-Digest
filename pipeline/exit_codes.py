"""Process exit codes shared by the `hmd` CLI (pipeline/cli.py) and the evals runner
(pipeline/evals.py), so both agree on one meaning without importing one from the other.

0 done, 2 stage/assertion error, 3 paused on budget, 4 configuration/usage error, 130 interrupted.
"""
from __future__ import annotations

EXIT_OK = 0
EXIT_ERROR = 2
EXIT_PAUSED = 3
EXIT_USAGE = 4
EXIT_INTERRUPTED = 130
