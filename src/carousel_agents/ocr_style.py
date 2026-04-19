from __future__ import annotations

import json
from pathlib import Path


def sample_ocr_style_snippets(
    *,
    ocr_jsonl: Path = Path("exports/instagram/_dataset/ocr_slides.jsonl"),
    max_snippets: int = 6,
    max_chars_per_snippet: int = 220,
) -> list[str]:
    """
    Returns short OCR excerpts as *style* references (cadence/texture), not facts.
    We prefer slides that look like: short headline + dense subtext.
    """
    if not ocr_jsonl.exists():
        return []

    snippets: list[str] = []
    try:
        lines = ocr_jsonl.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []

    # Simple heuristic: multi-line OCR with a short first line and some dense lines after.
    for raw in lines:
        if len(snippets) >= max_snippets:
            break
        s = raw.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except Exception:
            continue
        text = (obj.get("text") or "").strip()
        if not text:
            continue

        parts = [p.strip() for p in text.splitlines() if p.strip()]
        if len(parts) < 3:
            continue
        headline = parts[0]
        body = " ".join(parts[1:])
        if len(headline) > 28:
            continue
        if len(body) < 60:
            continue

        snippet = f"{headline}\n{body}"
        snippet = snippet[:max_chars_per_snippet].strip()
        snippets.append(snippet)

    return snippets

