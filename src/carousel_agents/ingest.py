from __future__ import annotations

import re
from pathlib import Path


def load_text(path: Path) -> tuple[str, str]:
    """
    Returns (text, detected_type).
    MVP supports: .txt, .md, .srt, .vtt
    """
    suffix = path.suffix.lower().lstrip(".")
    raw = path.read_text(encoding="utf-8", errors="ignore")

    if suffix in {"srt", "vtt"}:
        return _strip_caption_noise(raw), "transcript"
    return raw, suffix or "txt"


def _strip_caption_noise(raw: str) -> str:
    # Remove common caption artifacts: cue numbers, timestamps, WEBVTT headers.
    lines: list[str] = []
    ts = re.compile(r"^\d{2}:\d{2}:\d{2}[,.]\d{3}\s+-->\s+\d{2}:\d{2}:\d{2}[,.]\d{3}")
    for line in raw.splitlines():
        s = line.strip()
        if not s:
            lines.append("")
            continue
        if s.upper() == "WEBVTT":
            continue
        if s.isdigit():
            continue
        if ts.match(s):
            continue
        lines.append(line)
    return "\n".join(lines).strip()

