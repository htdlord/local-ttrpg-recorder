"""Theme tokens and speaker colours as plain data (no Qt), with a WCAG contrast check."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Tokens:
    background: str
    surface: str
    surface_alt: str
    border: str
    text: str
    text_muted: str
    accent: str
    accent_text: str  # text on the accent colour
    warning: str
    error: str
    success: str
    speakers: tuple[str, ...]


# One accent (deep teal) in both themes; speaker colours from a colour-blind friendly
# set (Okabe-Ito based), adjusted per theme for >= 4.5:1 contrast on the surface.
LIGHT = Tokens(
    background="#f4f6f8",
    surface="#ffffff",
    surface_alt="#eef2f5",
    border="#d5dce3",
    text="#1c2430",
    text_muted="#5c6b7a",
    accent="#0f7c8c",
    accent_text="#ffffff",
    warning="#9a6200",
    error="#b3261e",
    success="#1e7a3c",
    speakers=(
        "#0b6aa8",
        "#b35400",
        "#007a5e",
        "#a8326e",
        "#5b4bb7",
        "#7a6200",
        "#c0392b",
        "#3d5a66",
    ),
)
DARK = Tokens(
    background="#14181d",
    surface="#1c2128",
    surface_alt="#232a33",
    border="#323b46",
    text="#e6ebf0",
    text_muted="#93a1b0",
    accent="#3fb5c4",
    accent_text="#0b1215",
    warning="#e8b04a",
    error="#f2867d",
    success="#6cc58a",
    speakers=(
        "#6cb8f0",
        "#f0a35e",
        "#4fd1ae",
        "#ef8fc0",
        "#a99cf2",
        "#e3c95a",
        "#f28b7f",
        "#9fc1cf",
    ),
)


def _luminance(hex_color: str) -> float:
    h = hex_color.lstrip("#")
    channels = [int(h[i : i + 2], 16) / 255 for i in (0, 2, 4)]
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def contrast(a: str, b: str) -> float:
    la, lb = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def speaker_color(tokens: Tokens, index: int, override: str | None = None) -> str:
    return override or tokens.speakers[index % len(tokens.speakers)]
