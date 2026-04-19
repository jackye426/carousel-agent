from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .performance import compute_derived
from .schemas import PerformanceLog, PerformanceObserved


def _to_int(x: Any) -> int | None:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    try:
        # allow "1,234"
        s = s.replace(",", "")
        return int(float(s))
    except Exception:
        return None


def _parse_date(d: str) -> datetime | None:
    s = (d or "").strip()
    if not s:
        return None
    # tracker uses dd/mm/yyyy
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            continue
    return None


@dataclass(frozen=True)
class TagInfo:
    pillar: str | None
    format: str | None
    hook_style: str | None


def _load_tags(tags_jsonl: Path) -> dict[str, TagInfo]:
    out: dict[str, TagInfo] = {}
    if not tags_jsonl.exists():
        return out
    for line in tags_jsonl.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except Exception:
            continue
        aid = str(obj.get("asset_id") or "").strip()
        if not aid:
            continue
        tags = obj.get("tags") or {}
        out[aid] = TagInfo(
            pillar=(tags.get("pillar") if isinstance(tags, dict) else None),
            format=(tags.get("format") if isinstance(tags, dict) else None),
            hook_style=(tags.get("hook_style") if isinstance(tags, dict) else None),
        )
    return out


def import_performance_from_tracker(
    *,
    tracker_csv: Path,
    tags_jsonl: Path | None,
    out_jsonl: Path,
    overwrite: bool = False,
    carousel_only: bool = True,
    platform: str = "instagram",
) -> tuple[int, int]:
    """
    Build PerformanceLog rows from the marketing tracker CSV (IG metrics).
    Canonical ID: post_id = Asset_ID (string).

    Returns (written, skipped).
    """
    tags = _load_tags(tags_jsonl) if tags_jsonl else {}

    existing_ids: set[str] = set()
    if out_jsonl.exists() and not overwrite:
        for line in out_jsonl.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
                pid = str(obj.get("post_id") or "").strip()
                if pid:
                    existing_ids.add(pid)
            except Exception:
                continue

    rows_out: list[str] = []
    if out_jsonl.exists() and not overwrite:
        rows_out = [ln for ln in out_jsonl.read_text(encoding="utf-8").splitlines() if ln.strip()]

    written = 0
    skipped = 0

    with tracker_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row:
                continue
            asset_id = str(row.get("Asset_ID") or "").strip()
            if not asset_id:
                continue
            if asset_id in existing_ids:
                skipped += 1
                continue

            asset_type = str(row.get("Asset type") or "").strip().lower()
            if carousel_only and asset_type != "carousel":
                continue

            observed = PerformanceObserved(
                impressions=_to_int(row.get("IG_Reach")) or _to_int(row.get("IG_Views")),
                likes=_to_int(row.get("IG_Likes")),
                comments=_to_int(row.get("IG_Comments")),
                shares=_to_int(row.get("IG_Shares")),
                saves=_to_int(row.get("IG_Saves")),
                profile_visits=_to_int(row.get("IG_Profile_Visits")),
                follows=_to_int(row.get("IG_Follows")),
                dms=None,
            )

            tag = tags.get(asset_id)
            pillar = (tag.pillar if tag and tag.pillar else None) or str(row.get("Content_Bucket") or "").strip().lower() or "unknown"
            fmt = (tag.format if tag and tag.format else None) or ("carousel" if asset_type == "carousel" else "other")
            hook_style = (tag.hook_style if tag and tag.hook_style else None)

            published_at = _parse_date(str(row.get("Publish_Date") or ""))
            notes = str(row.get("Topic") or "").strip() or None

            perf = PerformanceLog(
                post_id=asset_id,
                platform=platform,  # type: ignore[arg-type]
                published_at=published_at or datetime.utcnow(),
                run_id=None,
                document_title=None,
                idea_id=f"asset_{asset_id}",
                pillar=pillar,
                format_suggestion=fmt,
                hook_id=None,
                hook_style=hook_style,
                predicted_total=None,
                predicted_scores={},
                observed=observed,
                derived=compute_derived(observed),
                notes=notes,
            )

            rows_out.append(perf.model_dump_json())
            existing_ids.add(asset_id)
            written += 1

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_jsonl.write_text("\n".join(rows_out) + ("\n" if rows_out else ""), encoding="utf-8")
    return written, skipped

