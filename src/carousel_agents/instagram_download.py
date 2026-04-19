from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import instaloader


_IG_SHORTCODE_RE = re.compile(r"/(?:p|reel|tv)/([^/?#]+)/?", re.IGNORECASE)


@dataclass(frozen=True)
class IGDownloadRow:
    asset_id: str
    ig_permalink: str


def extract_shortcode(ig_permalink: str) -> Optional[str]:
    m = _IG_SHORTCODE_RE.search((ig_permalink or "").strip())
    return m.group(1) if m else None


def load_rows_from_tracker_csv(csv_path: Path) -> list[IGDownloadRow]:
    rows: list[IGDownloadRow] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            asset_id = (r.get("Asset_ID") or "").strip()
            ig_permalink = (r.get("IG_Permalink") or "").strip()
            if not asset_id or not ig_permalink:
                continue
            rows.append(IGDownloadRow(asset_id=asset_id, ig_permalink=ig_permalink))
    return rows


def _safe_dir_name(s: str) -> str:
    # Keep Windows-friendly folder names.
    s = re.sub(r"[^\w\-. ]+", "_", s.strip())
    s = re.sub(r"\s+", " ", s).strip()
    return s[:120] if len(s) > 120 else s


def download_from_tracker(
    *,
    csv_path: Path,
    out_dir: Path,
    ig_username: Optional[str] = None,
    ig_password: Optional[str] = None,
    session_dir: Optional[Path] = None,
    overwrite: bool = False,
    limit: Optional[int] = None,
) -> Path:
    """
    Downloads each IG permalink in the tracker CSV using Instaloader.

    Output structure:
      {out_dir}/instagram/{Asset_ID}_{shortcode}/...instaloader files...
      {out_dir}/instagram/_dataset/index.json  (download status + mapping)
    """
    rows = load_rows_from_tracker_csv(csv_path)
    if limit is not None:
        rows = rows[: max(0, int(limit))]

    base = out_dir / "instagram"
    base.mkdir(parents=True, exist_ok=True)

    L = instaloader.Instaloader(
        download_comments=False,
        save_metadata=True,
        compress_json=False,
        dirname_pattern=str(base / "{target}"),
        filename_pattern="{date_utc}_UTC_{shortcode}",
        quiet=False,
    )

    # Optional login for private content / higher reliability.
    if ig_username and ig_password:
        # Use a stable session file so repeated runs don't re-login constantly.
        sess_dir = session_dir or (out_dir / ".instaloader_sessions")
        sess_dir.mkdir(parents=True, exist_ok=True)
        sessionfile = str(sess_dir / f"session-{ig_username}")
        try:
            L.load_session_from_file(ig_username, sessionfile=sessionfile)
        except Exception:
            L.login(ig_username, ig_password)
            L.save_session_to_file(sessionfile=sessionfile)

    index: dict[str, object] = {
        "csv_path": str(csv_path),
        "out_dir": str(out_dir),
        "count_total_rows_with_permalink": len(rows),
        "items": [],
    }

    for row in rows:
        shortcode = extract_shortcode(row.ig_permalink)
        item: dict[str, object] = {
            "asset_id": row.asset_id,
            "ig_permalink": row.ig_permalink,
            "shortcode": shortcode,
            "status": "pending",
        }
        if not shortcode:
            item["status"] = "skipped_bad_permalink"
            index["items"].append(item)
            continue

        target = _safe_dir_name(f"{row.asset_id}_{shortcode}")
        target_dir = base / target

        if target_dir.exists() and any(target_dir.iterdir()) and not overwrite:
            item["status"] = "skipped_exists"
            item["target"] = target
            item["target_dir"] = str(target_dir)
            index["items"].append(item)
            continue

        try:
            post = instaloader.Post.from_shortcode(L.context, shortcode)
            L.download_post(post, target=target)
            item["status"] = "downloaded"
            item["target"] = target
            item["target_dir"] = str(target_dir)
        except Exception as e:
            item["status"] = "error"
            item["error"] = f"{type(e).__name__}: {e}"
            item["target"] = target
            item["target_dir"] = str(target_dir)

        index["items"].append(item)

    dataset_dir = base / "_dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    index_path = dataset_dir / "index.json"
    index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
    return index_path

