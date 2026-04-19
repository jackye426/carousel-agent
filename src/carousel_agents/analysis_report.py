from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TrackerPost:
    asset_id: str
    publish_date: str
    topic: str
    asset_type: str
    content_bucket: str
    hook_cover_text: str
    ig_permalink: str

    ig_views: int | None
    ig_reach: int | None
    ig_likes: int | None
    ig_saves: int | None
    ig_shares: int | None
    ig_profile_visits: int | None

    tt_views: int | None
    tt_likes: int | None
    tt_saves: int | None
    tt_shares: int | None


def generate_report(
    *,
    tracker_csv: Path,
    out_path: Path,
    follower_count_instagram: int = 139,
    tags_jsonl: Path | None = None,
) -> Path:
    posts = _load_tracker(tracker_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tags_by_asset = _load_tags(tags_jsonl) if tags_jsonl and tags_jsonl.exists() else {}

    # Focus on carousels for utility engine
    car = [p for p in posts if p.asset_type.lower() == "carousel"]

    # Derived metrics
    rows = []
    for p in car:
        tags = tags_by_asset.get(p.asset_id, {})
        rows.append(
            {
                "asset_id": p.asset_id,
                "topic": p.topic,
                "bucket": p.content_bucket,
                "pillar": tags.get("pillar"),
                "format": tags.get("format"),
                "hook_style": tags.get("hook_style"),
                "hook_cover_text": p.hook_cover_text,
                "ig_views": p.ig_views,
                "ig_reach": p.ig_reach,
                "ig_saves": p.ig_saves,
                "ig_shares": p.ig_shares,
                "ig_profile_visits": p.ig_profile_visits,
                "ig_save_rate": _rate(p.ig_saves, p.ig_views),
                "ig_share_rate": _rate(p.ig_shares, p.ig_views),
                "ig_profile_visit_rate": _rate(p.ig_profile_visits, p.ig_views),
                "tt_views": p.tt_views,
                "tt_saves": p.tt_saves,
                "tt_shares": p.tt_shares,
                "tt_save_rate": _rate(p.tt_saves, p.tt_views),
                "tt_share_rate": _rate(p.tt_shares, p.tt_views),
            }
        )

    # Group by tracker Content_Bucket (best available label today)
    by_bucket = _group(rows, key="bucket")

    md = []
    md += [
        "## Carousel performance report",
        "",
        f"- **generated_at**: `{datetime.now(timezone.utc).isoformat()}Z`",
        f"- **tracker**: `{tracker_csv.name}`",
        f"- **instagram_followers**: `{follower_count_instagram}`",
        f"- **carousel_posts**: `{len(rows)}`",
        "",
        "### What this report does",
        "",
        "- Computes **rates** (saves/views, shares/views, profile_visits/views) to reduce follower-count noise.",
        "- Breaks down performance by your existing tracker label: `Content_Bucket`.",
        "- Highlights best/worst posts to review for patterns.",
        "",
        "### Leaderboards (by Content_Bucket)",
        "",
    ]

    md += _render_bucket_table(by_bucket, metric="ig_save_rate", title="IG save_rate (saves / views)")
    md += [""] + _render_bucket_table(by_bucket, metric="ig_share_rate", title="IG share_rate (shares / views)")
    md += [""] + _render_bucket_table(by_bucket, metric="ig_profile_visit_rate", title="IG profile_visit_rate (profile_visits / views)")

    # TikTok subsets
    tt_rows = [r for r in rows if r.get("tt_views") is not None]
    if tt_rows:
        by_bucket_tt = _group(tt_rows, key="bucket")
        md += [
            "",
            "### TikTok leaderboards (subset where TT stats exist)",
            "",
        ]
        md += _render_bucket_table(by_bucket_tt, metric="tt_save_rate", title="TT save_rate (saves / views)")
        md += [""] + _render_bucket_table(by_bucket_tt, metric="tt_share_rate", title="TT share_rate (shares / views)")

    # Tag-based leaderboards (preferred)
    tagged_rows = [r for r in rows if r.get("pillar") or r.get("format") or r.get("hook_style")]
    if tagged_rows:
        md += [
            "",
            "### Leaderboards (tagged: pillar / format / hook_style)",
            "",
            "_These use the auto-tagging output (more consistent than `Content_Bucket`)._",
            "",
        ]
        md += _render_group_table(_group(tagged_rows, key="pillar"), metric="ig_save_rate", title="IG save_rate by pillar")
        md += [""] + _render_group_table(_group(tagged_rows, key="format"), metric="ig_save_rate", title="IG save_rate by format")
        md += [""] + _render_group_table(_group(tagged_rows, key="hook_style"), metric="ig_save_rate", title="IG save_rate by hook_style")

        md += [""] + _render_group_table(_group(tagged_rows, key="pillar"), metric="ig_profile_visit_rate", title="IG profile_visit_rate by pillar")
        md += [""] + _render_group_table(_group(tagged_rows, key="format"), metric="ig_profile_visit_rate", title="IG profile_visit_rate by format")
        md += [""] + _render_group_table(_group(tagged_rows, key="hook_style"), metric="ig_profile_visit_rate", title="IG profile_visit_rate by hook_style")

    # Top/bottom examples
    md += [
        "",
        "### Top carousels by IG save_rate",
        "",
    ]
    md += _render_top_posts(rows, metric="ig_save_rate", n=5)

    md += [
        "",
        "### Bottom carousels by IG save_rate",
        "",
    ]
    md += _render_bottom_posts(rows, metric="ig_save_rate", n=5)

    # Recommendations (simple heuristics)
    md += [
        "",
        "### Recommendations (next iteration)",
        "",
        "- **Double down on the buckets with the highest median IG save_rate** (utility engine).",
        "- **Keep one access/decision-style carousel weekly** to drive profile visits and intent (even if saves are lower).",
        "- **Standardise tags next**: add `pillar`, `format`, `hook_style` columns to the tracker (or derive them automatically) so we can report by those dimensions instead of `Content_Bucket`.\n",
    ]

    out_path.write_text("\n".join(md).strip() + "\n", encoding="utf-8")
    return out_path


def _load_tags(path: Path | None) -> dict[str, dict[str, str]]:
    if not path or not path.exists():
        return {}
    out: dict[str, dict[str, str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        obj = json.loads(s)
        asset_id = str(obj.get("asset_id") or "")
        tags = (obj.get("tags") or {}) if isinstance(obj.get("tags"), dict) else {}
        if asset_id:
            out[asset_id] = {
                "pillar": str(tags.get("pillar") or ""),
                "format": str(tags.get("format") or ""),
                "hook_style": str(tags.get("hook_style") or ""),
            }
    return out


def _render_bucket_table(by_bucket: dict[str, list[dict[str, Any]]], *, metric: str, title: str) -> list[str]:
    rows = []
    for bucket, items in by_bucket.items():
        vals = [i.get(metric) for i in items if isinstance(i.get(metric), float)]
        if not vals:
            continue
        rows.append(
            {
                "bucket": bucket or "(blank)",
                "n": len(vals),
                "median": statistics.median(vals),
                "mean": statistics.fmean(vals),
            }
        )
    rows.sort(key=lambda r: r["median"], reverse=True)

    out = [f"#### {title}", "", "| bucket | n | median | mean |", "|---|---:|---:|---:|"]
    for r in rows:
        out.append(f"| {r['bucket']} | {r['n']} | {r['median']:.4f} | {r['mean']:.4f} |")
    return out


def _render_group_table(by_group: dict[str, list[dict[str, Any]]], *, metric: str, title: str) -> list[str]:
    rows = []
    for group, items in by_group.items():
        vals = [i.get(metric) for i in items if isinstance(i.get(metric), float)]
        if not vals:
            continue
        rows.append(
            {
                "group": group or "(blank)",
                "n": len(vals),
                "median": statistics.median(vals),
                "mean": statistics.fmean(vals),
            }
        )
    rows.sort(key=lambda r: r["median"], reverse=True)

    out = [f"#### {title}", "", "| group | n | median | mean |", "|---|---:|---:|---:|"]
    for r in rows:
        out.append(f"| {r['group']} | {r['n']} | {r['median']:.4f} | {r['mean']:.4f} |")
    return out


def _render_top_posts(rows: list[dict[str, Any]], *, metric: str, n: int) -> list[str]:
    vals = [r for r in rows if isinstance(r.get(metric), float)]
    vals.sort(key=lambda r: r[metric], reverse=True)
    out = ["| asset_id | bucket | ig_views | saves | shares | profile_visits | save_rate | topic |", "|---:|---|---:|---:|---:|---:|---:|---|"]
    for r in vals[:n]:
        out.append(
            f"| {r['asset_id']} | {r['bucket']} | {r['ig_views']} | {r['ig_saves']} | {r['ig_shares']} | {r['ig_profile_visits']} | {r[metric]:.4f} | {r['topic']} |"
        )
    return out


def _render_bottom_posts(rows: list[dict[str, Any]], *, metric: str, n: int) -> list[str]:
    vals = [r for r in rows if isinstance(r.get(metric), float)]
    vals.sort(key=lambda r: r[metric])
    out = ["| asset_id | bucket | ig_views | saves | shares | profile_visits | save_rate | topic |", "|---:|---|---:|---:|---:|---:|---:|---|"]
    for r in vals[:n]:
        out.append(
            f"| {r['asset_id']} | {r['bucket']} | {r['ig_views']} | {r['ig_saves']} | {r['ig_shares']} | {r['ig_profile_visits']} | {r[metric]:.4f} | {r['topic']} |"
        )
    return out


def _group(rows: list[dict[str, Any]], *, key: str) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        out[str(r.get(key) or "")].append(r)
    return out


def _rate(num: int | None, den: int | None) -> float | None:
    if num is None or den is None or den <= 0:
        return None
    return float(num) / float(den)


def _load_tracker(path: Path) -> list[TrackerPost]:
    posts: list[TrackerPost] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            posts.append(
                TrackerPost(
                    asset_id=(r.get("Asset_ID") or "").strip(),
                    publish_date=(r.get("Publish_Date") or "").strip(),
                    topic=(r.get("Topic") or "").strip(),
                    asset_type=(r.get("Asset type") or "").strip(),
                    content_bucket=(r.get("Content_Bucket") or "").strip(),
                    hook_cover_text=(r.get("Hook_Cover_Text") or "").strip(),
                    ig_permalink=(r.get("IG_Permalink") or "").strip(),
                    ig_views=_to_int(r.get("IG_Views")),
                    ig_reach=_to_int(r.get("IG_Reach")),
                    ig_likes=_to_int(r.get("IG_Likes")),
                    ig_saves=_to_int(r.get("IG_Saves")),
                    ig_shares=_to_int(r.get("IG_Shares")),
                    ig_profile_visits=_to_int(r.get("IG_Profile_Visits")),
                    tt_views=_to_int(r.get("TT_Views")),
                    tt_likes=_to_int(r.get("TT_Likes")),
                    tt_saves=_to_int(r.get("TT_Saves")),
                    tt_shares=_to_int(r.get("TT_Shares")),
                )
            )
    return posts


def _to_int(s: str | None) -> int | None:
    if s is None:
        return None
    t = str(s).strip()
    if t == "" or t.lower() in {"nan", "none", "null"}:
        return None
    try:
        return int(float(t))
    except Exception:
        return None

