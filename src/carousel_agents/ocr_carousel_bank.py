from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class OcrCarouselBankPaths:
    """
    V1 schema (one JSON object per line) for `ocr_carousel_bank.jsonl`:
    {
      "asset_id": "10",
      "shortcode": "DR17bjkjMef",
      "utc_datetime": "2025-12-04T13:07:27Z",
      "slide_count": 6,
      "cover_text": "...",
      "slides": [ {"slide_index": 1, "text": "..."}, ... ],
      "derived": {
        "avg_chars_per_slide": 123.4,
        "has_checklist_bullets": false,
        "has_myth_truth_pattern": true,
        "has_numbered_steps": false,
        "cta_present": true
      }
    }
    """

    carousel_bank_jsonl: Path


def default_bank_paths(*, outdir: Path) -> OcrCarouselBankPaths:
    return OcrCarouselBankPaths(carousel_bank_jsonl=outdir / "ocr_carousel_bank.jsonl")


_BULLET_RE = re.compile(r"(^|\n)\s*[-•]\s+", re.MULTILINE)
_MYTH_RE = re.compile(r"\bmyth\s*:", re.IGNORECASE)
_TRUTH_RE = re.compile(r"\b(truth|reality)\s*:", re.IGNORECASE)
_NUMBERED_RE = re.compile(r"(^|\n)\s*(\d+[\).]|step\s+\d+)\s+", re.IGNORECASE)
_CTA_RE = re.compile(r"\b(save|share|send|dm|comment|ask your gp|book|track your symptoms)\b", re.IGNORECASE)


def _as_bool(m: Any) -> bool:
    return bool(m)


def build_ocr_carousel_bank(
    *,
    ocr_slides_jsonl: Path,
    out_carousel_bank_jsonl: Path,
    overwrite: bool = True,
    max_text_per_slide: int = 4000,
) -> OcrCarouselBankPaths:
    out_carousel_bank_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if not ocr_slides_jsonl.exists():
        raise FileNotFoundError(str(ocr_slides_jsonl))

    by_asset: dict[str, list[dict[str, Any]]] = {}
    for raw in ocr_slides_jsonl.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s:
            continue
        obj = json.loads(s)
        asset_id = str(obj.get("asset_id") or "").strip()
        if not asset_id:
            continue
        by_asset.setdefault(asset_id, []).append(obj)

    out_rows: list[dict[str, Any]] = []
    for asset_id, items in by_asset.items():
        items = [i for i in items if isinstance(i.get("slide_index"), int)]
        items.sort(key=lambda i: i["slide_index"])
        if not items:
            continue

        shortcode = str(items[0].get("shortcode") or "")
        utc_datetime = items[0].get("utc_datetime")

        slides: list[dict[str, Any]] = []
        total_chars = 0
        cover_text = ""
        for i in items:
            t = (i.get("text") or "").strip()
            if not t:
                continue
            t = t[:max_text_per_slide].strip()
            slide_index = int(i["slide_index"])
            if slide_index == 1 and not cover_text:
                cover_text = t
            slides.append({"slide_index": slide_index, "text": t})
            total_chars += len(t)

        if not slides:
            continue

        all_text = "\n\n".join([s["text"] for s in slides]).strip()
        slide_count = len(slides)
        derived = {
            "avg_chars_per_slide": (total_chars / slide_count) if slide_count else 0.0,
            "has_checklist_bullets": _as_bool(_BULLET_RE.search(all_text)),
            "has_myth_truth_pattern": _as_bool(_MYTH_RE.search(all_text) and _TRUTH_RE.search(all_text)),
            "has_numbered_steps": _as_bool(_NUMBERED_RE.search(all_text)),
            "cta_present": _as_bool(_CTA_RE.search(all_text)),
        }

        out_rows.append(
            {
                "asset_id": asset_id,
                "shortcode": shortcode,
                "utc_datetime": utc_datetime,
                "slide_count": slide_count,
                "cover_text": cover_text,
                "slides": slides,
                "derived": derived,
            }
        )

    mode = "w" if overwrite else "a"
    with out_carousel_bank_jsonl.open(mode, encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return OcrCarouselBankPaths(carousel_bank_jsonl=out_carousel_bank_jsonl)


def sample_ocr_carousel_structures(
    *,
    carousel_bank_jsonl: Path = Path("exports/instagram/_dataset/ocr_carousel_bank.jsonl"),
    max_examples: int = 4,
    max_chars_per_example: int = 500,
) -> list[str]:
    """
    Returns compact carousel examples to guide *structure*, not content copying.
    Each example is: cover headline + slide_count + a few slide-role hints.
    """
    if not carousel_bank_jsonl.exists():
        return []
    try:
        lines = carousel_bank_jsonl.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []

    out: list[str] = []
    for raw in lines:
        if len(out) >= max_examples:
            break
        s = raw.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except Exception:
            continue
        cover = (obj.get("cover_text") or "").strip()
        slides = obj.get("slides") or []
        if not cover or not isinstance(slides, list) or len(slides) < 4:
            continue

        slide_count = obj.get("slide_count") or len(slides)
        # Pull first lines from a few slides as a “progression sketch”.
        first_lines: list[str] = []
        for slide in slides[: min(6, len(slides))]:
            t = (slide.get("text") or "").strip()
            if not t:
                continue
            first = t.splitlines()[0].strip()
            if first:
                first_lines.append(first[:60])
        sketch = " | ".join(first_lines[:6])
        example = f"Cover: {cover.splitlines()[0].strip()}\nSlides: {slide_count}\nProgression: {sketch}"
        out.append(example[:max_chars_per_example].strip())
    return out

