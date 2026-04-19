from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from carousel_agents.llm import OpenAICompatibleClient, image_file_to_data_url


@dataclass(frozen=True)
class TrackerRow:
    asset_id: str
    shortcode: str
    publish_date: str
    topic: str
    asset_type: str
    content_bucket: str
    caption: str
    hook_cover_text: str
    ig_permalink: str
    ig_views: int | None
    ig_likes: int | None
    ig_saves: int | None
    tt_views: int | None
    tt_likes: int | None
    tt_saves: int | None


def load_tracker_rows(path: Path) -> list[TrackerRow]:
    rows: list[TrackerRow] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                TrackerRow(
                    asset_id=(r.get("Asset_ID") or "").strip(),
                    shortcode=_shortcode_from_permalink((r.get("IG_Permalink") or "").strip()),
                    publish_date=(r.get("Publish_Date") or "").strip(),
                    topic=(r.get("Topic") or "").strip(),
                    asset_type=(r.get("Asset type") or "").strip(),
                    content_bucket=(r.get("Content_Bucket") or "").strip(),
                    caption=(r.get("Caption") or "").strip(),
                    hook_cover_text=(r.get("Hook_Cover_Text") or "").strip(),
                    ig_permalink=(r.get("IG_Permalink") or "").strip(),
                    ig_views=_to_int(r.get("IG_Views")),
                    ig_likes=_to_int(r.get("IG_Likes")),
                    ig_saves=_to_int(r.get("IG_Saves")),
                    tt_views=_to_int(r.get("TT_Views")),
                    tt_likes=_to_int(r.get("TT_Likes")),
                    tt_saves=_to_int(r.get("TT_Saves")),
                )
            )
    return rows


def sample_carousels(rows: Iterable[TrackerRow], n: int) -> list[TrackerRow]:
    car = [r for r in rows if r.asset_type.lower() == "carousel" and r.ig_views is not None]
    car.sort(key=lambda r: r.publish_date, reverse=True)
    return car[:n]


def sample_carousels_with_tiktok(rows: Iterable[TrackerRow], n: int) -> list[TrackerRow]:
    car = [
        r
        for r in rows
        if r.asset_type.lower() == "carousel" and r.ig_views is not None and r.tt_views is not None
    ]
    car.sort(key=lambda r: r.publish_date, reverse=True)
    return car[:n]


def run_experiment(
    *,
    tracker_csv: Path,
    n: int,
    out_dir: Path,
    models: list[str],
    audience_hint: str = "UK English (en-GB), UK-specific endometriosis account. Prioritise saves/shares utility. Instagram followers: 139.",
) -> Path:
    client = OpenAICompatibleClient()
    rows = sample_carousels(load_tracker_rows(tracker_csv), n=n)

    run_id = datetime.now(timezone.utc).strftime("exp_%Y%m%d_%H%M%S")
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for row in rows:
        item: dict[str, Any] = {
            "asset_id": row.asset_id,
            "publish_date": row.publish_date,
            "topic": row.topic,
            "content_bucket": row.content_bucket,
            "asset_type": row.asset_type,
            "ig_permalink": row.ig_permalink,
            "inputs": {
                "hook_cover_text": row.hook_cover_text,
                "caption": row.caption,
            },
            "actual": {
                "ig_views": row.ig_views,
                "ig_likes": row.ig_likes,
                "ig_saves": row.ig_saves,
                "tt_views": row.tt_views,
                "tt_likes": row.tt_likes,
                "tt_saves": row.tt_saves,
            },
            "by_model": {},
        }

        for model in models:
            resp = client.chat_json(
                model=model,
                temperature=0.2,
                system=_system_experiment_judge(),
                user=_user_experiment_judge(
                    audience_hint=audience_hint,
                    hook_cover_text=row.hook_cover_text,
                    caption=row.caption,
                    topic=row.topic,
                ),
            )
            item["by_model"][model] = resp

        results.append(item)

    json_path = run_dir / "results.json"
    json_path.write_text(json.dumps({"run_id": run_id, "results": results}, indent=2), encoding="utf-8")

    md_path = run_dir / "report.md"
    md_path.write_text(_render_report_md(run_id=run_id, results=results, models=models), encoding="utf-8")
    return md_path


def _system_experiment_judge() -> str:
    return (
        "You are a strict social media performance judge.\n"
        "Return JSON only.\n"
        "Be honest about uncertainty: you are estimating.\n"
    )


def _user_experiment_judge(*, audience_hint: str, hook_cover_text: str, caption: str, topic: str) -> str:
    return (
        "We are evaluating a past CAROUSEL post.\n"
        "We do NOT have the actual slide text (no OCR). Judge based only on cover hook text + caption.\n\n"
        f"Audience/context: {audience_hint}\n\n"
        f"Topic: {topic}\n"
        f"Cover hook text: {hook_cover_text}\n"
        f"Caption: {caption}\n\n"
        "Task:\n"
        "- Write ONE sentence: what it did well.\n"
        "- Write ONE sentence: what needs improving.\n"
        "- Predict performance metrics for Instagram and TikTok (absolute counts):\n"
        "  - Instagram: views, likes, saves\n"
        "  - TikTok: views, likes, saves\n\n"
        "Return JSON with EXACT shape:\n"
        '{\n'
        '  "did_well": "one sentence",\n'
        '  "improve": "one sentence",\n'
        '  "predicted": {\n'
        '    "instagram": { "views": 0, "likes": 0, "saves": 0 },\n'
        '    "tiktok": { "views": 0, "likes": 0, "saves": 0 }\n'
        "  },\n"
        '  "confidence": 0.0\n'
        "}\n"
        "Confidence is 0.0–1.0.\n"
    )


def _render_report_md(*, run_id: str, results: list[dict[str, Any]], models: list[str]) -> str:
    lines: list[str] = []
    lines += [
        "## Experiment report",
        "",
        f"- **run_id**: `{run_id}`",
        f"- **n**: {len(results)}",
        f"- **models**: {', '.join(f'`{m}`' for m in models)}",
        "",
        "### Summary table (actual vs predicted)",
        "",
    ]

    for item in results:
        lines += [
            f"## Asset {item['asset_id']} — {item.get('topic','').strip()}",
            "",
            f"- **IG permalink**: `{item.get('ig_permalink','')}`",
            f"- **Actual IG**: views={item['actual']['ig_views']} likes={item['actual']['ig_likes']} saves={item['actual']['ig_saves']}",
            f"- **Actual TT**: views={item['actual']['tt_views']} likes={item['actual']['tt_likes']} saves={item['actual']['tt_saves']}",
            "",
        ]
        for m in models:
            r = (item.get("by_model") or {}).get(m) or {}
            pred = r.get("predicted") or {}
            ig = pred.get("instagram") or {}
            tt = pred.get("tiktok") or {}
            lines += [
                f"### {m}",
                "",
                f"- **did_well**: {r.get('did_well','')}",
                f"- **improve**: {r.get('improve','')}",
                f"- **Pred IG**: views={ig.get('views')} likes={ig.get('likes')} saves={ig.get('saves')}",
                f"- **Pred TT**: views={tt.get('views')} likes={tt.get('likes')} saves={tt.get('saves')}",
                f"- **confidence**: {r.get('confidence')}",
                "",
            ]
        lines.append("---\n")

    return "\n".join(lines).strip() + "\n"


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


def _shortcode_from_permalink(url: str) -> str:
    if not url:
        return ""
    parts = url.split("/")
    for i, p in enumerate(parts):
        if p in {"p", "reel"} and i + 1 < len(parts):
            return parts[i + 1]
    return ""


def run_vision_experiment(
    *,
    tracker_csv: Path,
    exports_dir: Path,
    n: int,
    out_dir: Path,
    models: list[str],
    max_slides: int = 10,
    audience_hint: str = "UK English (en-GB), UK endometriosis account. Prioritise saves/shares utility. Instagram followers: 139.",
) -> Path:
    client = OpenAICompatibleClient()
    rows = sample_carousels(load_tracker_rows(tracker_csv), n=n)

    run_id = datetime.now(timezone.utc).strftime("vision_%Y%m%d_%H%M%S")
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for row in rows:
        slide_paths = _find_slide_images(exports_dir=exports_dir, asset_id=row.asset_id, shortcode=row.shortcode, max_slides=max_slides)
        item: dict[str, Any] = {
            "asset_id": row.asset_id,
            "shortcode": row.shortcode,
            "publish_date": row.publish_date,
            "topic": row.topic,
            "content_bucket": row.content_bucket,
            "asset_type": row.asset_type,
            "ig_permalink": row.ig_permalink,
            "slide_paths": [str(p) for p in slide_paths],
            "actual": {
                "ig_views": row.ig_views,
                "ig_likes": row.ig_likes,
                "ig_saves": row.ig_saves,
                "tt_views": row.tt_views,
                "tt_likes": row.tt_likes,
                "tt_saves": row.tt_saves,
            },
            "by_model": {},
        }

        for model in models:
            resp = _vision_judge_one(
                client=client,
                model=model,
                audience_hint=audience_hint,
                topic=row.topic,
                caption=row.caption,
                hook_cover_text=row.hook_cover_text,
                slide_paths=slide_paths,
            )
            item["by_model"][model] = resp

        results.append(item)

    json_path = run_dir / "results.json"
    json_path.write_text(json.dumps({"run_id": run_id, "results": results}, indent=2), encoding="utf-8")

    md_path = run_dir / "report.md"
    md_path.write_text(_render_vision_report_md(run_id=run_id, results=results, models=models), encoding="utf-8")
    return md_path


def run_vision_experiment_tt_only(
    *,
    tracker_csv: Path,
    exports_dir: Path,
    n: int,
    out_dir: Path,
    models: list[str],
    max_slides: int = 10,
    audience_hint: str = "UK English (en-GB), UK endometriosis account. Prioritise saves/shares utility. Instagram followers: 139.",
) -> Path:
    client = OpenAICompatibleClient()
    rows = sample_carousels_with_tiktok(load_tracker_rows(tracker_csv), n=n)

    run_id = datetime.now(timezone.utc).strftime("vision_tt_%Y%m%d_%H%M%S")
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for row in rows:
        slide_paths = _find_slide_images(exports_dir=exports_dir, asset_id=row.asset_id, shortcode=row.shortcode, max_slides=max_slides)
        item: dict[str, Any] = {
            "asset_id": row.asset_id,
            "shortcode": row.shortcode,
            "publish_date": row.publish_date,
            "topic": row.topic,
            "content_bucket": row.content_bucket,
            "asset_type": row.asset_type,
            "ig_permalink": row.ig_permalink,
            "slide_paths": [str(p) for p in slide_paths],
            "actual": {
                "ig_views": row.ig_views,
                "ig_likes": row.ig_likes,
                "ig_saves": row.ig_saves,
                "tt_views": row.tt_views,
                "tt_likes": row.tt_likes,
                "tt_saves": row.tt_saves,
            },
            "by_model": {},
        }

        for model in models:
            resp = _vision_judge_one(
                client=client,
                model=model,
                audience_hint=audience_hint,
                topic=row.topic,
                caption=row.caption,
                hook_cover_text=row.hook_cover_text,
                slide_paths=slide_paths,
            )
            item["by_model"][model] = resp

        results.append(item)

    json_path = run_dir / "results.json"
    json_path.write_text(json.dumps({"run_id": run_id, "results": results}, indent=2), encoding="utf-8")

    md_path = run_dir / "report.md"
    md_path.write_text(_render_vision_report_md(run_id=run_id, results=results, models=models), encoding="utf-8")
    return md_path


def _find_slide_images(*, exports_dir: Path, asset_id: str, shortcode: str, max_slides: int) -> list[Path]:
    folder = exports_dir / "instagram" / f"{asset_id}_{shortcode}"
    if not folder.exists():
        return []
    imgs = sorted(folder.glob("*.jpg"))

    def slide_num(p: Path) -> int:
        name = p.stem
        try:
            return int(name.split("_")[-1])
        except Exception:
            return 999

    imgs.sort(key=slide_num)
    return imgs[:max_slides]


def _vision_judge_one(
    *,
    client: OpenAICompatibleClient,
    model: str,
    audience_hint: str,
    topic: str,
    caption: str,
    hook_cover_text: str,
    slide_paths: list[Path],
) -> dict[str, Any]:
    parts: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "Evaluate this Instagram carousel as a social media judge.\n"
                "You can see the slide images.\n\n"
                f"Audience/context: {audience_hint}\n"
                f"Topic: {topic}\n"
                f"Cover hook text (from tracker, may differ): {hook_cover_text}\n"
                f"Caption: {caption}\n\n"
                "Give:\n"
                "- ONE sentence: what it did well (include composition/readability/copy).\n"
                "- ONE sentence: what to improve (include composition/readability/copy).\n"
                "- Predict performance metrics (absolute counts):\n"
                "  - Instagram: views, likes, saves\n"
                "  - TikTok: views, likes, saves\n\n"
                "Return JSON with EXACT shape:\n"
                '{\n'
                '  "did_well": "one sentence",\n'
                '  "improve": "one sentence",\n'
                '  "predicted": {\n'
                '    "instagram": { "views": 0, "likes": 0, "saves": 0 },\n'
                '    "tiktok": { "views": 0, "likes": 0, "saves": 0 }\n'
                "  },\n"
                '  "confidence": 0.0\n'
                "}\n"
            ),
        }
    ]
    for p in slide_paths:
        parts.append({"type": "image_url", "image_url": {"url": image_file_to_data_url(str(p))}})

    messages = [
        {"role": "system", "content": "You are a strict social media performance judge. Return JSON only."},
        {"role": "user", "content": parts},
    ]
    return client.chat_json_messages(model=model, messages=messages, temperature=0.2, timeout_s=240.0)


def _render_vision_report_md(*, run_id: str, results: list[dict[str, Any]], models: list[str]) -> str:
    lines: list[str] = []
    lines += [
        "## Full vision judging report",
        "",
        f"- **run_id**: `{run_id}`",
        f"- **n**: {len(results)}",
        f"- **models**: {', '.join(f'`{m}`' for m in models)}",
        "",
    ]
    for item in results:
        lines += [
            f"## Asset {item['asset_id']} — {item.get('topic','').strip()}",
            "",
            f"- **IG permalink**: `{item.get('ig_permalink','')}`",
            f"- **Slides attached**: {len(item.get('slide_paths') or [])}",
            f"- **Actual IG**: views={item['actual']['ig_views']} likes={item['actual']['ig_likes']} saves={item['actual']['ig_saves']}",
            f"- **Actual TT**: views={item['actual']['tt_views']} likes={item['actual']['tt_likes']} saves={item['actual']['tt_saves']}",
            "",
        ]
        for m in models:
            r = (item.get("by_model") or {}).get(m) or {}
            pred = r.get("predicted") or {}
            ig = pred.get("instagram") or {}
            tt = pred.get("tiktok") or {}
            lines += [
                f"### {m}",
                "",
                f"- **did_well**: {r.get('did_well','')}",
                f"- **improve**: {r.get('improve','')}",
                f"- **Pred IG**: views={ig.get('views')} likes={ig.get('likes')} saves={ig.get('saves')}",
                f"- **Pred TT**: views={tt.get('views')} likes={tt.get('likes')} saves={tt.get('saves')}",
                f"- **confidence**: {r.get('confidence')}",
                "",
            ]
        lines.append("---\n")
    return "\n".join(lines).strip() + "\n"
