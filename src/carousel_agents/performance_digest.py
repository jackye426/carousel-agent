from __future__ import annotations

import json
import math
import os
import csv
from pathlib import Path
from statistics import median
from typing import Any

from .defaults import (
    experiments_dir_default,
    marketing_tracker_csv_path,
    ocr_slides_jsonl_path,
    performance_digest_json_path,
    performance_jsonl_path,
    tags_jsonl_path,
)
from .id_resolution import TrackerIndex, build_tracker_index, resolve_to_asset_id
from .performance import compute_derived, load_performance_jsonl


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            out.append(json.loads(s))
        except Exception:
            continue
    return out


def _safe_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        f = float(x)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except Exception:
        return None


def _summarise_rates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    save_rates: list[float] = []
    share_rates: list[float] = []
    pvr_rates: list[float] = []
    for r in rows:
        d = r.get("derived") or {}
        sr = _safe_float(d.get("save_rate"))
        shr = _safe_float(d.get("share_rate"))
        pvr = _safe_float(d.get("profile_visit_rate"))
        if sr is not None:
            save_rates.append(sr)
        if shr is not None:
            share_rates.append(shr)
        if pvr is not None:
            pvr_rates.append(pvr)

    def _stats(xs: list[float]) -> dict[str, Any]:
        if not xs:
            return {"n": 0, "median": None}
        return {"n": len(xs), "median": float(median(xs))}

    return {
        "save_rate": _stats(save_rates),
        "share_rate": _stats(share_rates),
        "profile_visit_rate": _stats(pvr_rates),
    }


def _index_tags(tags_jsonl: Path) -> dict[str, dict[str, Any]]:
    """
    Index by asset_id (string) to tag payload.
    """
    items = _read_jsonl(tags_jsonl)
    out: dict[str, dict[str, Any]] = {}
    for it in items:
        aid = str(it.get("asset_id") or "").strip()
        if not aid:
            continue
        out[aid] = it
    return out


def _index_ocr_first_slide(ocr_jsonl: Path) -> dict[str, str]:
    """
    Return asset_id -> slide 1 OCR text (best-effort).
    """
    items = _read_jsonl(ocr_jsonl)
    out: dict[str, str] = {}
    for it in items:
        aid = str(it.get("asset_id") or "").strip()
        if not aid:
            continue
        try:
            slide_idx = int(it.get("slide_index") or 0)
        except Exception:
            slide_idx = 0
        if slide_idx != 1:
            continue
        text = str(it.get("text") or "").strip()
        if text:
            out[aid] = text
    return out


def _tracker_index(tracker_csv: Path | None) -> TrackerIndex | None:
    if tracker_csv is None or not tracker_csv.exists():
        return None
    return build_tracker_index(tracker_csv=tracker_csv)


def _index_tracker_captions(tracker_csv: Path | None) -> dict[str, str]:
    """
    Best-effort Asset_ID -> Caption text from the marketing tracker CSV.
    Used only for writer formatting hints (not factual grounding).
    """
    if tracker_csv is None or not tracker_csv.exists():
        return {}
    out: dict[str, str] = {}
    try:
        with tracker_csv.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row:
                    continue
                aid = str(row.get("Asset_ID") or row.get("asset_id") or "").strip()
                cap = str(row.get("Caption") or row.get("caption") or "").strip()
                if aid and cap:
                    out[aid] = cap
    except Exception:
        return {}
    return out


def _tag_for_post(
    pid_raw: str,
    *,
    tags_by_asset: dict[str, dict[str, Any]],
    tracker: TrackerIndex | None,
) -> dict[str, Any]:
    pid_asset = resolve_to_asset_id(pid_raw, tracker) if tracker else pid_raw
    return tags_by_asset.get(pid_asset) or tags_by_asset.get((pid_raw or "").strip()) or {}


def build_performance_digest(
    *,
    performance_jsonl: Path,
    tags_jsonl: Path | None = None,
    ocr_jsonl: Path | None = None,
    tracker_csv: Path | None = None,
    top_n: int = 3,
    followers_instagram: int | None = 139,
) -> dict[str, Any]:
    """
    Build a small, prompt-friendly digest of what has performed for this account.
    Deterministic: no LLM calls.

    Join strategy (best-effort):
    - performance.post_id MAY match carousel_tags.asset_id (string) in your dataset.
    - If it doesn't, the digest still contains performance rows without OCR/tags.
    """
    logs = load_performance_jsonl(performance_jsonl)
    perf_rows: list[dict[str, Any]] = []
    for r in logs:
        d = r.derived.model_dump() if r.derived else compute_derived(r.observed).model_dump()
        perf_rows.append(
            {
                "post_id": r.post_id,
                "platform": r.platform,
                "published_at": (r.published_at.isoformat() if r.published_at else None),
                "pillar": r.pillar,
                "format_suggestion": r.format_suggestion,
                "hook_style": r.hook_style,
                "observed": (r.observed.model_dump() if r.observed else {}),
                "derived": d,
                "notes": r.notes,
            }
        )

    tags_by_asset: dict[str, dict[str, Any]] = _index_tags(tags_jsonl) if tags_jsonl else {}
    ocr_first: dict[str, str] = _index_ocr_first_slide(ocr_jsonl) if ocr_jsonl else {}
    tracker = _tracker_index(tracker_csv)
    captions_by_asset = _index_tracker_captions(tracker_csv)

    def _score_key(row: dict[str, Any]) -> float:
        d = row.get("derived") or {}
        sr = _safe_float(d.get("save_rate")) or 0.0
        shr = _safe_float(d.get("share_rate")) or 0.0
        # weighted to your stated north stars (saves > shares)
        return (sr * 1.0) + (shr * 0.6)

    top = sorted(perf_rows, key=_score_key, reverse=True)[: max(0, int(top_n))]
    top_examples: list[dict[str, Any]] = []
    for r in top:
        pid = str(r.get("post_id") or "")
        tag = _tag_for_post(pid, tags_by_asset=tags_by_asset, tracker=tracker)
        pid_asset = resolve_to_asset_id(pid, tracker) if tracker else pid
        # prefer tag cover_ocr if available, else OCR slide 1 if post_id matches asset_id
        cover = ""
        try:
            cover = str(((tag.get("source") or {}).get("cover_ocr") or "")).strip()
        except Exception:
            cover = ""
        if not cover:
            cover = ocr_first.get(pid_asset, "") or ocr_first.get(pid, "")
        cap = captions_by_asset.get(pid_asset) or captions_by_asset.get(pid) or ""
        top_examples.append(
            {
                "post_id": pid,
                "asset_id_resolved": pid_asset if pid_asset != pid else None,
                "platform": r.get("platform"),
                "published_at": r.get("published_at"),
                "pillar": r.get("pillar"),
                "format_suggestion": r.get("format_suggestion"),
                "hook_style": r.get("hook_style"),
                "tags": (tag.get("tags") or {}),
                "topic": tag.get("topic"),
                "cover_text": cover[:800] if cover else None,
                "caption_text": cap[:2200] if cap else None,
                "observed": r.get("observed"),
                "derived": r.get("derived"),
            }
        )

    digest: dict[str, Any] = {
        "version": "v1",
        "account_context": {
            "followers_instagram": followers_instagram,
            "north_star_metrics": ["save_rate", "share_rate", "profile_visit_rate"],
            "note": "Use as weak priors only; do not overfit. Novel ideas can still win if strongly supported by the source document.",
        },
        "baseline_rates": _summarise_rates(perf_rows),
        "top_examples": top_examples,
    }
    return digest


def save_digest(digest: dict[str, Any], *, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(digest, ensure_ascii=False, indent=2), encoding="utf-8")


def digest_prompt_body(digest: dict[str, Any]) -> str:
    """Stable JSON text for ideation prompts (extract/score/review)."""
    minimal = (os.environ.get("CONTEXT_MINIMAL") or "").strip().lower() in ("1", "true", "yes", "on")
    top = digest.get("top_examples") or []
    if minimal and isinstance(top, list):
        top = top[:2]
    return json.dumps(
        {
            "account_context": digest.get("account_context"),
            "baseline_rates": digest.get("baseline_rates"),
            "top_examples": top,
        },
        ensure_ascii=False,
        indent=2 if not minimal else None,
    )


def digest_writer_hints(digest: dict[str, Any]) -> str:
    """
    Compact lines for hooks/CTA/slides: north stars + a few top-post snapshots.
    """
    minimal = (os.environ.get("CONTEXT_MINIMAL") or "").strip().lower() in ("1", "true", "yes", "on")
    max_ex = 2 if minimal else 5
    lines: list[str] = []
    ac = digest.get("account_context") or {}
    metrics = ac.get("north_star_metrics") or ["save_rate", "share_rate", "profile_visit_rate"]
    lines.append(f"North-star metrics (optimise for these in tone/structure, not vanity reach): {metrics}.")
    br = digest.get("baseline_rates") or {}
    for name in ("save_rate", "share_rate", "profile_visit_rate"):
        st = br.get(name) or {}
        med = st.get("median")
        n = st.get("n", 0)
        if med is not None and n:
            lines.append(f"- Baseline median {name}: {med} (n={n})")
    lines.append("Top posts (structure/voice hints only — do NOT copy cover lines):")
    for ex in (digest.get("top_examples") or [])[:max_ex]:
        tags = ex.get("tags") if isinstance(ex.get("tags"), dict) else {}
        hook = tags.get("hook_style") or ex.get("hook_style")
        pil = tags.get("pillar") or ex.get("pillar")
        fmt = tags.get("format") or ex.get("format_suggestion")
        cov = (ex.get("cover_text") or "").strip().replace("\n", " ")
        cap = (ex.get("caption_text") or "").strip()
        if len(cov) > 140:
            cov = cov[:137] + "..."
        cap_line = ""
        if cap:
            one = cap.replace("\r", "").strip()
            one = " ".join(one.split())
            if len(one) > 180:
                one = one[:177] + "..."
            cap_line = f" | caption_formatting: {one}"
        lines.append(
            f"  • pillar={pil!s} format={fmt!s} hook_style={hook!s} | cover_texture: {cov!s}{cap_line}"
        )
    lines.extend(_experiment_finding_lines(experiments_dir_default()))
    return "\n".join(lines)


def _experiment_finding_lines(experiments_dir: Path, *, max_items: int = 4) -> list[str]:
    try:
        from .experiment_analysis import load_experiment_results_from_dir
    except Exception:
        return []
    rows = load_experiment_results_from_dir(experiments_dir)
    if not rows:
        return []
    tail = rows[-max_items:]
    out: list[str] = [
        "Recent concluded A/B experiments (weak priors only — do not overfit to a single test):",
    ]
    for r in tail:
        eid = r.get("experiment_id", "")
        w = r.get("winner", "")
        m = r.get("primary_metric", "")
        meets = r.get("meets_min_impressions", False)
        out.append(f"  • {eid}: {m} winner={w!s} (min_impressions_met={meets})")
    return out


def load_or_build_performance_digest(
    *,
    explicit_path: Path | None = None,
    default_json: Path | None = None,
    performance_jsonl: Path | None = None,
    tags_jsonl: Path | None = None,
    ocr_jsonl: Path | None = None,
    tracker_csv: Path | None = None,
    top_n: int = 3,
    followers_instagram: int | None = 139,
    disabled: bool = False,
) -> dict[str, Any] | None:
    """
    Resolve digest for a pipeline run:
    1) explicit_path if provided (must exist)
    2) else default_json if it exists (typically outputs/performance_digest.json)
    3) else build from performance_jsonl when non-empty

    Optional paths default to repo-conventional locations; missing files are skipped.
    """
    if disabled:
        return None
    perf_path = performance_jsonl or performance_jsonl_path()
    out_default = default_json or performance_digest_json_path()
    tags_p = tags_jsonl or tags_jsonl_path()
    ocr_p = ocr_jsonl or ocr_slides_jsonl_path()
    track_p = tracker_csv or marketing_tracker_csv_path()

    if explicit_path is not None:
        if not explicit_path.exists():
            raise FileNotFoundError(str(explicit_path))
        return json.loads(explicit_path.read_text(encoding="utf-8"))
    if out_default.exists():
        return json.loads(out_default.read_text(encoding="utf-8"))
    if not perf_path.exists():
        return None
    logs = load_performance_jsonl(perf_path)
    if not logs:
        return None
    return build_performance_digest(
        performance_jsonl=perf_path,
        tags_jsonl=tags_p if tags_p.exists() else None,
        ocr_jsonl=ocr_p if ocr_p.exists() else None,
        tracker_csv=track_p if track_p.exists() else None,
        top_n=top_n,
        followers_instagram=followers_instagram,
    )

