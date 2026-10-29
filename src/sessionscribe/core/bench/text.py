"""Text normalisation and error rates for scoring (jiwer is a dev-only dependency)."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

BRACKETED = re.compile(r"\[[^\]]*\]")
KEEP_INSIDE_WORD = {"-", "'", chr(0x2019)}  # hyphen, apostrophe, typographic apostrophe


class BenchDependencyError(RuntimeError):
    pass


def require_jiwer() -> Any:
    try:
        import jiwer
    except ImportError as exc:  # pragma: no cover - dev group installs it
        raise BenchDependencyError(
            "benchmarks need the dev dependencies: run 'uv sync' (jiwer is missing)"
        ) from exc
    return jiwer


def normalize(text: str) -> str:
    """NFC, lower case, ``[tags]`` removed, punctuation -> space, accents kept."""
    text = BRACKETED.sub(" ", unicodedata.normalize("NFC", text)).lower()
    chars = list(text)
    out: list[str] = []
    for i, ch in enumerate(chars):
        cat = unicodedata.category(ch)
        if cat[0] in "PS":
            inside = (
                ch in KEEP_INSIDE_WORD
                and 0 < i < len(chars) - 1
                and chars[i - 1].isalnum()
                and chars[i + 1].isalnum()
            )
            out.append(ch if inside else " ")
        else:
            out.append(ch)
    return " ".join("".join(out).split())


def cer(reference: str, hypothesis: str) -> float:
    """Character error rate on normalised text; empty reference -> 0 or 1."""
    ref, hyp = normalize(reference), normalize(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return float(require_jiwer().cer(ref, hyp))
