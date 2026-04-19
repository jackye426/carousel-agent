from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .llm import OpenAICompatibleClient, image_file_to_data_url


def ocr_slides_dataset(
    *,
    slides_jsonl: Path,
    repo_root: Path,
    out_jsonl: Path,
    model: str,
    limit: int | None = None,
    overwrite: bool = False,
) -> Path:
    """
    OCR each slide image referenced in slides_dataset.jsonl and write results to JSONL.
    """
    client = OpenAICompatibleClient()
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    seen = set()
    if out_jsonl.exists() and not overwrite:
        for line in out_jsonl.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
                key = (obj.get("asset_id"), obj.get("shortcode"), obj.get("slide_index"))
                seen.add(key)
            except Exception:
                continue

    processed = 0
    out_mode = "w" if overwrite else "a"
    with slides_jsonl.open("r", encoding="utf-8") as f_in, out_jsonl.open(out_mode, encoding="utf-8") as f_out:
        for line in f_in:
            if limit is not None and processed >= limit:
                break
            raw = line.strip()
            if not raw:
                continue
            row = json.loads(raw)
            key = (row.get("asset_id"), row.get("shortcode"), row.get("slide_index"))
            if key in seen:
                continue

            rel_path = str(row.get("file_path") or "")
            img_path = (repo_root / rel_path).resolve()
            if not img_path.exists():
                # skip missing files
                continue

            result = _ocr_one_image(client=client, model=model, image_path=str(img_path))
            out_row = {
                "asset_id": row.get("asset_id"),
                "shortcode": row.get("shortcode"),
                "slide_index": row.get("slide_index"),
                "utc_datetime": row.get("utc_datetime"),
                "file_path": rel_path,
                "model": model,
                "text": result.get("text", ""),
                "confidence": result.get("confidence", 0.0),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            f_out.write(json.dumps(out_row, ensure_ascii=False))
            f_out.write("\n")

            processed += 1

    return out_jsonl


def _ocr_one_image(*, client: OpenAICompatibleClient, model: str, image_path: str) -> dict[str, Any]:
    data_url = image_file_to_data_url(image_path)
    messages = [
        {
            "role": "system",
            "content": "You extract text from carousel slide images. Return JSON only.",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "OCR this Instagram carousel slide.\n"
                        "Rules:\n"
                        "- Extract ALL visible text, preserving line breaks where reasonable.\n"
                        "- Do not add commentary.\n"
                        "- If text is unreadable, return an empty string.\n\n"
                        'Return JSON: {"text": "...", "confidence": 0.0}\n'
                        "Confidence is 0.0–1.0."
                    ),
                },
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        },
    ]
    return client.chat_json_messages(model=model, messages=messages, temperature=0.0, timeout_s=180.0)


def get_ocr_model() -> str:
    return os.environ.get("MODEL_OCR", "google/gemini-3-flash-preview")

