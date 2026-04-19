from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse


def _norm_ig_permalink(url: str) -> str:
    """
    Normalise IG permalink for join keys:
    - drop query params and fragments
    - drop trailing slashes
    - keep scheme+host+path only
    """
    u = (url or "").strip()
    if not u:
        return ""
    try:
        p = urlparse(u)
        scheme = p.scheme or "https"
        netloc = p.netloc
        path = (p.path or "").rstrip("/")
        if not netloc or not path:
            return u.rstrip("/")
        return f"{scheme}://{netloc}{path}"
    except Exception:
        return u.rstrip("/")


@dataclass(frozen=True)
class TrackerIndex:
    permalink_to_asset_id: dict[str, str]


def build_tracker_index(*, tracker_csv: Path) -> TrackerIndex:
    """
    Build a mapping from IG permalink -> Asset_ID using the marketing tracker CSV.
    Expected columns: Asset_ID, IG_Permalink (case-insensitive match).
    """
    permalink_to_asset_id: dict[str, str] = {}
    if not tracker_csv.exists():
        return TrackerIndex(permalink_to_asset_id=permalink_to_asset_id)

    with tracker_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        # tolerate header variants
        for row in reader:
            if not row:
                continue
            aid = (row.get("Asset_ID") or row.get("asset_id") or "").strip()
            pl = (row.get("IG_Permalink") or row.get("ig_permalink") or "").strip()
            if not aid or not pl:
                continue
            key = _norm_ig_permalink(pl)
            if key:
                permalink_to_asset_id[key] = aid
    return TrackerIndex(permalink_to_asset_id=permalink_to_asset_id)


def resolve_to_asset_id(raw: str, tracker: TrackerIndex | None) -> str:
    """
    If `raw` is an IG permalink present in the tracker, return Asset_ID; otherwise return stripped `raw`.
    Use this to join performance rows (which may still store URLs) to tag/OCR datasets keyed by Asset_ID.
    """
    s = (raw or "").strip()
    if not s or not tracker:
        return s
    low = s.lower()
    if low.startswith("http://") or low.startswith("https://"):
        key = _norm_ig_permalink(s)
        if key and key in tracker.permalink_to_asset_id:
            return tracker.permalink_to_asset_id[key]
    return s


def resolve_post_id(
    *,
    post_id: Optional[str],
    ig_permalink: Optional[str],
    tracker: TrackerIndex | None,
) -> str:
    """
    Canonical post_id for performance logging.
    Convention: use tracker Asset_ID (string).
    """
    if post_id and str(post_id).strip():
        return str(post_id).strip()
    if ig_permalink and tracker:
        key = _norm_ig_permalink(ig_permalink)
        if key and key in tracker.permalink_to_asset_id:
            return tracker.permalink_to_asset_id[key]
    return str(post_id or ig_permalink or "").strip()

