"""
Match a published IG post (caption ± OCR slide text) to RunState candidates when platform ids differ.

Uses token Jaccard + character bigram overlap (no extra ML dependencies).
"""

from __future__ import annotations

import glob as std_glob
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .schemas import CandidateIdea, RunState


_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_MENTION_RE = re.compile(r"@[\w.]+")
_MULTI_SPACE = re.compile(r"\s+")

# Minimal stopwords so short captions still compare meaningfully
_STOP = frozenset(
    "a an the and or but in on at to for of as is was are were be been being it its this that these those with from by not no yes so if we you your my our they their".split()
)


def normalize_for_match(text: str) -> str:
    t = (text or "").strip().lower()
    t = _URL_RE.sub(" ", t)
    t = _MENTION_RE.sub(" ", t)
    t = t.replace("#", "")
    t = re.sub(r"[^\w\s]+", " ", t, flags=re.UNICODE)
    t = _MULTI_SPACE.sub(" ", t).strip()
    return t


def tokenize(normalized: str) -> set[str]:
    parts = [p for p in normalized.split() if len(p) >= 2 and p not in _STOP]
    return set(parts)


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return float(inter) / float(union) if union else 0.0


def dice_bigrams(a: str, b: str) -> float:
    """Sørensen–Dice on character bigrams; robust for short / noisy strings."""
    if not a or not b:
        return 0.0
    if len(a) < 2 or len(b) < 2:
        return 1.0 if a == b else 0.0

    def bigrams(s: str) -> list[str]:
        return [s[i : i + 2] for i in range(len(s) - 1)]

    ca, cb = Counter(bigrams(a)), Counter(bigrams(b))
    inter = sum((ca & cb).values())
    denom = sum(ca.values()) + sum(cb.values())
    return (2.0 * inter / denom) if denom else 0.0


def text_similarity(query: str, target: str) -> float:
    qn, tn = normalize_for_match(query), normalize_for_match(target)
    if not qn or not tn:
        return 0.0
    jac = jaccard(tokenize(qn), tokenize(tn))
    dice = dice_bigrams(qn, tn)
    return max(jac, dice)


def caption_affinity(caption: str, c: CandidateIdea) -> float:
    parts: list[str] = [
        f"{c.topic} {c.angle} {c.core_claim} {c.audience_pain} {c.promise}",
    ]
    rb = getattr(c, "reader_benefit", None)
    if rb and str(rb).strip():
        parts.append(str(rb).strip())
    cap_d = getattr(c, "caption", None)
    if cap_d and str(cap_d).strip():
        parts.append(str(cap_d).strip())
    for h in c.hooks:
        parts.append(h.text)
    if c.best_cta:
        parts.append(c.best_cta)
    if c.carousel_draft and c.carousel_draft.slides:
        for s in c.carousel_draft.slides:
            parts.append(f"{s.main_text} {s.subtext}")
    scores = [text_similarity(caption, p) for p in parts if p.strip()]
    return max(scores) if scores else 0.0


def ocr_affinity(ocr: str, c: CandidateIdea) -> float:
    if not ocr.strip():
        return 0.0
    if c.carousel_draft and c.carousel_draft.slides:
        draft = "\n".join((f"{s.main_text} {s.subtext}").strip() for s in c.carousel_draft.slides)
        return text_similarity(ocr, draft)
    # No draft yet: compare to hooks + topic (weak)
    fallback = "\n".join([f"{c.topic} {c.angle}"] + [h.text for h in c.hooks])
    return text_similarity(ocr, fallback) if fallback.strip() else 0.0


def _first_slide_ocr_chunk(ocr: str) -> str:
    t = (ocr or "").strip()
    if not t:
        return ""
    parts = [p.strip() for p in t.split("\n\n") if p.strip()]
    return parts[0] if parts else t[:800]


def cover_affinity(ocr: str, c: CandidateIdea) -> float:
    """Match first-slide OCR to slide 1 draft (packaging / cover framing)."""
    if not ocr.strip():
        return 0.0
    if not (c.carousel_draft and c.carousel_draft.slides):
        return 0.0
    cov = _first_slide_ocr_chunk(ocr)
    s0 = c.carousel_draft.slides[0]
    draft = f"{s0.main_text} {s0.subtext}"
    return text_similarity(cov, draft)


def hook_line_affinity(hook_line: str, c: CandidateIdea) -> float:
    """Optional first-line / on-image hook text from the published post."""
    if not (hook_line or "").strip():
        return 0.0
    bh = ""
    if c.best_hook_id:
        for h in c.hooks:
            if h.hook_id == c.best_hook_id:
                bh = (h.text or "").strip()
                break
    if not bh and c.hooks:
        bh = (c.hooks[0].text or "").strip()
    if not bh:
        return 0.0
    return text_similarity(hook_line, bh)


def combined_score(
    *,
    caption: str | None,
    ocr: str | None,
    c: CandidateIdea,
    caption_weight: float = 0.4,
    ocr_weight: float = 0.6,
    hook_line: str | None = None,
    hook_weight: float = 0.0,
) -> tuple[float, float, float, float, float]:
    """
    Returns (total, caption_score, ocr_score, hook_score, cover_score).
    OCR score uses max(full-carousel OCR vs draft, first-slide vs slide 1) when a draft exists.
    """
    cap_s = caption_affinity(caption or "", c) if (caption or "").strip() else 0.0
    ocr_blob = (ocr or "").strip()
    ocr_full_s = ocr_affinity(ocr_blob, c) if ocr_blob else 0.0
    cover_s = cover_affinity(ocr_blob, c) if ocr_blob else 0.0
    if c.carousel_draft and c.carousel_draft.slides and ocr_blob:
        ocr_s = max(ocr_full_s, cover_s)
    else:
        ocr_s = ocr_full_s
    hook_s = hook_line_affinity(hook_line or "", c) if (hook_line or "").strip() else 0.0

    has_c = bool((caption or "").strip())
    has_o = bool(ocr_blob)
    has_h = bool((hook_line or "").strip()) and hook_weight > 0.0

    if has_c and has_o and has_h:
        wsum = caption_weight + ocr_weight + hook_weight
        total = (caption_weight * cap_s + ocr_weight * ocr_s + hook_weight * hook_s) / wsum
    elif has_c and has_o:
        wsum = caption_weight + ocr_weight
        total = (caption_weight * cap_s + ocr_weight * ocr_s) / wsum
    elif has_c and has_h:
        wsum = caption_weight + hook_weight
        total = (caption_weight * cap_s + hook_weight * hook_s) / wsum
    elif has_o and has_h:
        wsum = ocr_weight + hook_weight
        total = (ocr_weight * ocr_s + hook_weight * hook_s) / wsum
    elif has_c:
        total = cap_s
    elif has_o:
        total = ocr_s
    elif has_h:
        total = hook_s
    else:
        total = 0.0
    return total, cap_s, ocr_s, hook_s, cover_s


@dataclass(frozen=True)
class PostMatch:
    run_path: Path | None
    document_id: str
    document_title: str | None
    generation_run_id: str | None
    idea_id: str
    score: float
    caption_score: float
    ocr_score: float
    hook_score: float
    cover_score: float
    pillar: str
    format_suggestion: str
    hook_id: str | None
    hook_style: str | None
    ab_variant: str | None = None
    experiment_id: str | None = None
    treatment_key: str | None = None
    base_idea_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_path": str(self.run_path) if self.run_path else None,
            "document_id": self.document_id,
            "document_title": self.document_title,
            "generation_run_id": self.generation_run_id,
            "idea_id": self.idea_id,
            "score": self.score,
            "caption_score": self.caption_score,
            "ocr_score": self.ocr_score,
            "hook_score": self.hook_score,
            "cover_score": self.cover_score,
            "pillar": self.pillar,
            "format_suggestion": self.format_suggestion,
            "hook_id": self.hook_id,
            "hook_style": self.hook_style,
            "ab_variant": self.ab_variant,
            "experiment_id": self.experiment_id,
            "treatment_key": self.treatment_key,
            "base_idea_id": self.base_idea_id,
        }


def _hook_style_for_candidate(c: CandidateIdea) -> str | None:
    if c.best_hook_id:
        for h in c.hooks:
            if h.hook_id == c.best_hook_id:
                return (h.style or "").strip() or None
    if c.hooks:
        return (c.hooks[0].style or "").strip() or None
    return None


def iter_matches_for_state(
    rs: RunState,
    *,
    run_path: Path | None,
    caption: str | None,
    ocr: str | None,
    hook_line: str | None,
    caption_weight: float,
    ocr_weight: float,
    hook_weight: float,
) -> Iterator[PostMatch]:
    doc_id = rs.document.document_id
    title = rs.document.title
    gen_id = rs.generation_run_id
    for c in rs.candidates:
        total, cs, os_, hs, covs = combined_score(
            caption=caption,
            ocr=ocr,
            c=c,
            caption_weight=caption_weight,
            ocr_weight=ocr_weight,
            hook_line=hook_line,
            hook_weight=hook_weight,
        )
        yield PostMatch(
            run_path=run_path,
            document_id=doc_id,
            document_title=title,
            generation_run_id=gen_id,
            idea_id=c.idea_id,
            score=total,
            caption_score=cs,
            ocr_score=os_,
            hook_score=hs,
            cover_score=covs,
            pillar=c.content_pillar,
            format_suggestion=c.format_suggestion,
            hook_id=c.best_hook_id,
            hook_style=_hook_style_for_candidate(c),
            ab_variant=c.ab_variant,
            experiment_id=c.experiment_id,
            treatment_key=c.treatment_key,
            base_idea_id=c.base_idea_id,
        )


def match_post_to_runs(
    *,
    states: list[tuple[Path | None, RunState]],
    caption: str | None = None,
    ocr: str | None = None,
    hook_line: str | None = None,
    caption_weight: float = 0.4,
    ocr_weight: float = 0.6,
    hook_weight: float = 0.0,
    generation_run_id: str | None = None,
    top_n: int = 15,
    min_score: float = 0.0,
) -> list[PostMatch]:
    want = (generation_run_id or "").strip()
    filtered: list[tuple[Path | None, RunState]] = states
    if want:
        filtered = [(p, rs) for p, rs in states if (rs.generation_run_id or "") == want]
    out: list[PostMatch] = []
    for path, rs in filtered:
        out.extend(
            iter_matches_for_state(
                rs,
                run_path=path,
                caption=caption,
                ocr=ocr,
                hook_line=hook_line,
                caption_weight=caption_weight,
                ocr_weight=ocr_weight,
                hook_weight=hook_weight,
            )
        )
    out.sort(key=lambda m: m.score, reverse=True)
    out = [m for m in out if m.score >= min_score]
    return out[:top_n]


def load_run_state_path(path: Path) -> RunState:
    return RunState.model_validate_json(path.read_text(encoding="utf-8"))


def discover_run_json_files(paths: list[Path], globs: list[str]) -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []
    for p in paths:
        rp = p.expanduser().resolve()
        if rp.is_file() and rp.suffix.lower() == ".json":
            if rp not in seen:
                seen.add(rp)
                out.append(rp)
        elif rp.is_dir():
            for f in sorted(rp.rglob("*.json")):
                if f not in seen:
                    seen.add(f)
                    out.append(f)
    for pattern in globs:
        pat = str(Path(pattern).expanduser())
        for raw in std_glob.glob(pat, recursive=True):
            f = Path(raw).resolve()
            if f.is_file() and f.suffix.lower() == ".json" and f not in seen:
                seen.add(f)
                out.append(f)
    return sorted(out, key=lambda x: str(x).lower())


def read_ocr_slides_text_for_asset(ocr_jsonl: Path, asset_id: str) -> str:
    """Concatenate slide OCR lines for one asset_id (slide_index order)."""
    rows: list[tuple[int, str]] = []
    aid = str(asset_id).strip()
    if not ocr_jsonl.is_file():
        return ""
    for line in ocr_jsonl.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except json.JSONDecodeError:
            continue
        if str(obj.get("asset_id") or "").strip() != aid:
            continue
        try:
            idx = int(obj.get("slide_index") or 0)
        except (TypeError, ValueError):
            idx = 0
        t = (obj.get("text") or "").strip()
        rows.append((idx, t))
    rows.sort(key=lambda x: x[0])
    return "\n\n".join(t for _, t in rows if t)


def read_ocr_carousel_bank_row(carousel_bank_jsonl: Path, asset_id: str) -> str | None:
    """Return concatenated slide text from ocr_carousel_bank.jsonl for one asset_id."""
    aid = str(asset_id).strip()
    if not carousel_bank_jsonl.is_file():
        return None
    for line in carousel_bank_jsonl.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except json.JSONDecodeError:
            continue
        if str(obj.get("asset_id") or "").strip() != aid:
            continue
        slides = obj.get("slides") or []
        if isinstance(slides, list):
            parts = []
            for sl in sorted(
                (x for x in slides if isinstance(x, dict)),
                key=lambda d: int(d.get("slide_index") or 0),
            ):
                parts.append(str(sl.get("text") or "").strip())
            return "\n\n".join(p for p in parts if p)
        return (obj.get("cover_text") or "").strip() or None
    return None
