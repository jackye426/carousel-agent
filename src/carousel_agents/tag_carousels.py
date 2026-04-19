from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .llm import OpenAICompatibleClient


ALLOWED_PILLARS = ["recognition", "validation", "access_or_decision"]
ALLOWED_FORMATS = ["checklist", "myth_vs_truth", "what_to_do_next", "quote_led", "story", "other"]
ALLOWED_HOOK_STYLES = [
    "contrarian",
    "numbered",
    "myth_bust",
    "how_to",
    "identity",
    "validation",
    "checklist",
    "what_to_do_next",
    "quote_led",
    "statistic",
]


def tag_carousels(
    *,
    tracker_csv: Path,
    ocr_jsonl: Path,
    out_jsonl: Path,
    model: str,
    limit: int | None = None,
    overwrite: bool = False,
) -> Path:
    """
    Create consistent tags (pillar/format/hook_style) per carousel using OCR + caption.
    Writes JSONL rows keyed by asset_id.
    """
    client = OpenAICompatibleClient()
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    # Build OCR index by asset_id
    ocr_by_asset = _load_ocr_by_asset(ocr_jsonl) if ocr_jsonl.exists() else {}

    # Determine already-tagged
    done: set[str] = set()
    if out_jsonl.exists() and not overwrite:
        for line in out_jsonl.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
                if obj.get("asset_id"):
                    done.add(str(obj["asset_id"]))
            except Exception:
                continue

    rows = _load_tracker_carousels(tracker_csv)
    tagged = 0
    out_mode = "w" if overwrite else "a"
    with out_jsonl.open(out_mode, encoding="utf-8") as f_out:
        for r in rows:
            if limit is not None and tagged >= limit:
                break
            asset_id = r["asset_id"]
            if asset_id in done:
                continue

            ocr = ocr_by_asset.get(asset_id, {})
            cover_text = (ocr.get("cover_text") or "").strip()
            slide_text = (ocr.get("all_text") or "").strip()

            resp = client.chat_json(
                model=model,
                temperature=0.0,
                system=_system_tagger(),
                user=_user_tagger(
                    asset_id=asset_id,
                    topic=r.get("topic", ""),
                    content_bucket=r.get("content_bucket", ""),
                    caption=r.get("caption", ""),
                    hook_cover_text=r.get("hook_cover_text", ""),
                    cover_ocr=cover_text,
                    slides_ocr=slide_text,
                ),
            )

            out_row = {
                "asset_id": asset_id,
                "ig_permalink": r.get("ig_permalink", ""),
                "publish_date": r.get("publish_date", ""),
                "topic": r.get("topic", ""),
                "source": {
                    "content_bucket": r.get("content_bucket", ""),
                    "hook_cover_text": r.get("hook_cover_text", ""),
                    "cover_ocr": cover_text,
                },
                "tags": {
                    "pillar": resp.get("pillar"),
                    "format": resp.get("format"),
                    "hook_style": resp.get("hook_style"),
                },
                "notes": resp.get("notes"),
                "confidence": resp.get("confidence", 0.0),
                "model": model,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }

            # Light normalisation
            out_row["tags"]["pillar"] = _pick(resp.get("pillar"), ALLOWED_PILLARS, default="recognition")
            out_row["tags"]["format"] = _pick(resp.get("format"), ALLOWED_FORMATS, default="other")
            out_row["tags"]["hook_style"] = _pick(resp.get("hook_style"), ALLOWED_HOOK_STYLES, default="how_to")

            f_out.write(json.dumps(out_row, ensure_ascii=False))
            f_out.write("\n")
            tagged += 1

    return out_jsonl


def get_tagger_model() -> str:
    return os.environ.get(
        "MODEL_TAGGER",
        os.environ.get("MODEL_IDEATION") or os.environ.get("MODEL_JUDGE", "openai/gpt-4.1-mini"),
    )


def _load_ocr_by_asset(path: Path) -> dict[str, dict[str, str]]:
    by: dict[str, list[dict[str, Any]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        obj = json.loads(s)
        asset_id = str(obj.get("asset_id") or "")
        if not asset_id:
            continue
        by.setdefault(asset_id, []).append(obj)

    out: dict[str, dict[str, str]] = {}
    for asset_id, items in by.items():
        # sort by slide_index
        items = [i for i in items if isinstance(i.get("slide_index"), int)]
        items.sort(key=lambda i: i["slide_index"])
        cover = ""
        texts: list[str] = []
        for i in items:
            t = (i.get("text") or "").strip()
            if not t:
                continue
            if i.get("slide_index") == 1 and not cover:
                cover = t
            texts.append(f"[slide {i.get('slide_index')}] {t}")
        out[asset_id] = {"cover_text": cover, "all_text": "\n\n".join(texts)[:6000]}
    return out


def _load_tracker_carousels(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if (r.get("Asset type") or "").strip().lower() != "carousel":
                continue
            rows.append(
                {
                    "asset_id": (r.get("Asset_ID") or "").strip(),
                    "publish_date": (r.get("Publish_Date") or "").strip(),
                    "topic": (r.get("Topic") or "").strip(),
                    "content_bucket": (r.get("Content_Bucket") or "").strip(),
                    "caption": (r.get("Caption") or "").strip(),
                    "hook_cover_text": (r.get("Hook_Cover_Text") or "").strip(),
                    "ig_permalink": (r.get("IG_Permalink") or "").strip(),
                }
            )
    return rows


def _system_tagger() -> str:
    return (
        "You are a strict classifier for Instagram carousel posts.\n"
        "Return JSON only.\n"
        "Choose tags based on what is actually on the carousel (OCR) and caption.\n"
    )


def _user_tagger(
    *,
    asset_id: str,
    topic: str,
    content_bucket: str,
    caption: str,
    hook_cover_text: str,
    cover_ocr: str,
    slides_ocr: str,
) -> str:
    return (
        f"Asset_ID: {asset_id}\n"
        f"Tracker topic: {topic}\n"
        f"Tracker Content_Bucket: {content_bucket}\n"
        f"Tracker Hook_Cover_Text (may be empty): {hook_cover_text}\n\n"
        f"Cover OCR text (slide 1): {cover_ocr}\n\n"
        f"Slides OCR text (truncated):\n{slides_ocr}\n\n"
        "Tag the post with:\n"
        f"- pillar: one of {ALLOWED_PILLARS}\n"
        f"- format: one of {ALLOWED_FORMATS}\n"
        f"- hook_style: one of {ALLOWED_HOOK_STYLES}\n\n"
        "Guidance:\n"
        "- recognition: symptoms, myths, wrongly normalised experiences\n"
        "- validation: emotional/social reality, dismissal, feeling minimised\n"
        "- access_or_decision: NHS vs private, specialist choice, questions to ask, next steps\n"
        "- format: checklist if it's lists/steps; myth_vs_truth if explicitly myth vs truth; what_to_do_next if step-by-step next actions; quote_led if mostly quotes; story if narrative.\n"
        "- hook_style should be based on the COVER hook.\n\n"
        "Return JSON with EXACT shape:\n"
        '{ "pillar": "recognition", "format": "checklist", "hook_style": "numbered", "notes": "short", "confidence": 0.0 }\n'
    )


def _pick(value: Any, allowed: list[str], *, default: str) -> str:
    v = (str(value or "").strip() or "").lower()
    v = v.replace("-", "_").replace(" ", "_")
    if v in allowed:
        return v
    # some common aliases
    aliases = {
        "myth_bust": "myth_bust",
        "mythbust": "myth_bust",
        "myth": "myth_bust",
        "myth_vs_truth": "myth_vs_truth",
        "what_to_do_next": "what_to_do_next",
        "access": "access_or_decision",
        "access_decision": "access_or_decision",
        "access_or_decision": "access_or_decision",
    }
    if v in aliases and aliases[v] in allowed:
        return aliases[v]
    return default

