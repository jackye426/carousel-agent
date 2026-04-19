from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class SlideRow:
    asset_id: str
    shortcode: str
    slide_index: int
    utc_datetime: Optional[str]
    file_path: str
    file_name: str
    extension: str


_DIR_RE = re.compile(r"^(?P<asset_id>\d+?)_(?P<shortcode>[A-Za-z0-9_-]+)$")

# Instaloader naming with our filename_pattern: {date_utc}_UTC_{shortcode}[_<n>].ext
_FILE_RE = re.compile(
    r"^(?P<dt>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_UTC_(?P<shortcode>[A-Za-z0-9_-]+?)(?:_(?P<slide>\d+))?\.(?P<ext>\w+)$"
)


def _parse_utc(dt_str: str) -> Optional[str]:
    try:
        dt = datetime.strptime(dt_str, "%Y-%m-%d_%H-%M-%S")
        return dt.replace(microsecond=0).isoformat() + "Z"
    except Exception:
        return None


def _load_carousel_asset_ids_from_tracker(tracker_csv: Path) -> set[str]:
    """
    Returns Asset_IDs where 'Asset type' is 'Carousel' (case-insensitive).
    """
    keep: set[str] = set()
    with tracker_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            asset_id = (r.get("Asset_ID") or "").strip()
            asset_type = (r.get("Asset type") or "").strip().lower()
            if asset_id and asset_type == "carousel":
                keep.add(asset_id)
    return keep


def build_slides_dataset(
    *,
    exports_dir: Path = Path("exports"),
    tracker_csv: Optional[Path] = None,
    carousel_only: bool = False,
    out_csv: Optional[Path] = None,
    out_jsonl: Optional[Path] = None,
) -> tuple[Path, Path]:
    """
    Scans {exports_dir}/instagram/* for slide images and writes a dataset.

    The dataset is "slide-centric": one row per image file (typically carousel slides).
    """
    ig_dir = exports_dir / "instagram"
    if not ig_dir.exists():
        raise FileNotFoundError(f"Instagram exports folder not found: {ig_dir}")

    dataset_dir = ig_dir / "_dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_csv or (dataset_dir / "slides_dataset.csv")
    out_jsonl = out_jsonl or (dataset_dir / "slides_dataset.jsonl")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: list[SlideRow] = []
    exts = {".jpg", ".jpeg", ".png", ".webp"}

    carousel_asset_ids: Optional[set[str]] = None
    if tracker_csv and carousel_only:
        carousel_asset_ids = _load_carousel_asset_ids_from_tracker(tracker_csv)

    for post_dir in sorted([p for p in ig_dir.iterdir() if p.is_dir()], key=lambda p: p.name):
        mdir = _DIR_RE.match(post_dir.name)
        if not mdir:
            continue
        asset_id = mdir.group("asset_id")
        shortcode = mdir.group("shortcode")

        if carousel_asset_ids is not None and asset_id not in carousel_asset_ids:
            continue

        for f in sorted(post_dir.iterdir(), key=lambda p: p.name):
            if not f.is_file():
                continue
            if f.suffix.lower() not in exts:
                continue

            mfile = _FILE_RE.match(f.name)
            if mfile:
                # Trust filename shortcode when present, but fall back to dir shortcode
                sc = mfile.group("shortcode") or shortcode
                slide_index = int(mfile.group("slide") or "1")
                utc_dt = _parse_utc(mfile.group("dt"))
                ext = mfile.group("ext") or f.suffix.lstrip(".")
            else:
                sc = shortcode
                slide_index = 1
                utc_dt = None
                ext = f.suffix.lstrip(".")

            rows.append(
                SlideRow(
                    asset_id=asset_id,
                    shortcode=sc,
                    slide_index=slide_index,
                    utc_datetime=utc_dt,
                    file_path=str(f),
                    file_name=f.name,
                    extension=ext.lower(),
                )
            )

    # Write CSV
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "asset_id",
                "shortcode",
                "slide_index",
                "utc_datetime",
                "file_path",
                "file_name",
                "extension",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "asset_id": r.asset_id,
                    "shortcode": r.shortcode,
                    "slide_index": r.slide_index,
                    "utc_datetime": r.utc_datetime or "",
                    "file_path": r.file_path,
                    "file_name": r.file_name,
                    "extension": r.extension,
                }
            )

    # Write JSONL (handy for streaming OCR pipelines)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(
                json.dumps(
                    {
                        "asset_id": r.asset_id,
                        "shortcode": r.shortcode,
                        "slide_index": r.slide_index,
                        "utc_datetime": r.utc_datetime,
                        "file_path": r.file_path,
                        "file_name": r.file_name,
                        "extension": r.extension,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    return out_csv, out_jsonl

