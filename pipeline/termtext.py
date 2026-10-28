"""Text that came from outside (an error body, a page title, a file name) is printed only after
terminal_safe: an ESC or other control character could otherwise clear the screen, retitle the
window, forge a link or hide text (release security review, 2026-10-28). No imports beyond re."""
from __future__ import annotations

import re

# C0 controls except tab and newline, DEL, C1 controls, and the bidi embeddings/overrides/isolates.
_UNSAFE = re.compile("[\x00-\x08\x0b-\x1f\x7f-\x9f‪-‮⁦-⁩]")


def terminal_safe(text: str) -> str:
    """`text` with every control character except newline and tab replaced by "?"."""
    return _UNSAFE.sub("?", str(text))
