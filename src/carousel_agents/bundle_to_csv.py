from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .schemas import CandidateIdea, RunState


@dataclass(frozen=True)
class BundleMeta:
    run_id: str | None
    document: str | None
    generated_at: str | None


@dataclass(frozen=True)
class CarouselRow:
    run_id: str | None
    generation_run_id: str | None
    document: str | None
    generated_at: str | None
    bundle_path: str
    carousel_number: int
    title: str
    pillar: str | None
    format: str | None
    hook_best: str | None
    cta: str | None
    disclaimer: str | None
    slide_count: int
    idea_id: str | None = None


@dataclass(frozen=True)
class SlideRow:
    run_id: str | None
    generation_run_id: str | None
    document: str | None
    generated_at: str | None
    bundle_path: str
    carousel_number: int
    carousel_title: str
    slide_number: int
    slide_title: str
    slide_body: str
    cta: str | None = None
    idea_id: str | None = None


CSV_CAROUSEL_FIELDNAMES = [
    "run_id",
    "generation_run_id",
    "document",
    "generated_at",
    "bundle_path",
    "carousel_number",
    "idea_id",
    "title",
    "pillar",
    "format",
    "hook_best",
    "cta",
    "disclaimer",
    "slide_count",
]

CSV_SLIDE_FIELDNAMES = [
    "run_id",
    "generation_run_id",
    "document",
    "generated_at",
    "bundle_path",
    "carousel_number",
    "idea_id",
    "carousel_title",
    "slide_number",
    "slide_title",
    "slide_body",
    "cta",
]


_RE_RUN_ID = re.compile(r"^- \*\*run_id\*\*: `([^`]+)`\s*$")
_RE_DOC = re.compile(r"^- \*\*document\*\*: `([^`]+)`\s*$")
_RE_GEN = re.compile(r"^- \*\*generated_at\*\*: `([^`]+)`\s*$")
_RE_CAROUSEL = re.compile(r"^### #(\d+)\s+—\s+(.*)\s*$")
_RE_PILLAR = re.compile(r"^- \*\*pillar\*\*: `([^`]+)`\s*$")
_RE_FORMAT = re.compile(r"^- \*\*format\*\*: `([^`]+)`\s*$")
_RE_HOOK = re.compile(r"^#### Hook \(best\)\s*$")
_RE_SLIDES = re.compile(r"^#### Slides\s*$")
_RE_SLIDE = re.compile(r"^\*\*Slide (\d+)\*\*\s*$")
_RE_CTA = re.compile(r"^\*\*CTA\*\*\s*$")
_RE_DISCLAIMER = re.compile(r"^\*\*Disclaimer\*\*\s*$")
_RE_RULE = re.compile(r"^---\s*$")


def _read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8", errors="ignore").splitlines()


def _collect_paragraph(lines: list[str], i: int) -> tuple[str, int]:
    """
    Collect text until blank line or rule line.
    Returns (paragraph, new_index).
    """
    parts: list[str] = []
    while i < len(lines):
        s = lines[i].rstrip()
        if not s.strip():
            break
        if _RE_RULE.match(s.strip()):
            break
        parts.append(s)
        i += 1
    text = "\n".join(parts).strip()
    # advance past blank lines
    while i < len(lines) and not lines[i].strip():
        i += 1
    return text, i


def parse_bundle_markdown(path: Path) -> tuple[BundleMeta, list[CarouselRow], list[SlideRow]]:
    lines = _read_lines(path)
    meta = BundleMeta(run_id=None, document=None, generated_at=None)

    carousels: list[CarouselRow] = []
    slides: list[SlideRow] = []

    i = 0
    current_num: int | None = None
    current_title: str | None = None
    pillar: str | None = None
    fmt: str | None = None
    hook: str | None = None
    cta: str | None = None
    disclaimer: str | None = None
    slide_count = 0

    def flush_carousel() -> None:
        nonlocal current_num, current_title, pillar, fmt, hook, cta, disclaimer, slide_count
        if current_num is None or current_title is None:
            return
        carousels.append(
            CarouselRow(
                run_id=meta.run_id,
                generation_run_id=None,
                document=meta.document,
                generated_at=meta.generated_at,
                bundle_path=str(path),
                carousel_number=current_num,
                title=current_title,
                pillar=pillar,
                format=fmt,
                hook_best=hook,
                cta=cta,
                disclaimer=disclaimer,
                slide_count=slide_count,
                idea_id=None,
            )
        )
        current_num = None
        current_title = None
        pillar = None
        fmt = None
        hook = None
        cta = None
        disclaimer = None
        slide_count = 0

    while i < len(lines):
        s = lines[i].rstrip()

        if (m := _RE_RUN_ID.match(s)):
            meta = BundleMeta(run_id=m.group(1), document=meta.document, generated_at=meta.generated_at)
            i += 1
            continue
        if (m := _RE_DOC.match(s)):
            meta = BundleMeta(run_id=meta.run_id, document=m.group(1), generated_at=meta.generated_at)
            i += 1
            continue
        if (m := _RE_GEN.match(s)):
            meta = BundleMeta(run_id=meta.run_id, document=meta.document, generated_at=m.group(1))
            i += 1
            continue

        if (m := _RE_CAROUSEL.match(s)):
            # starting a new carousel
            flush_carousel()
            current_num = int(m.group(1))
            current_title = m.group(2).strip()
            i += 1
            continue

        if (m := _RE_PILLAR.match(s)):
            pillar = m.group(1)
            i += 1
            continue
        if (m := _RE_FORMAT.match(s)):
            fmt = m.group(1)
            i += 1
            continue

        if _RE_HOOK.match(s.strip()):
            i += 1
            # skip blanks
            while i < len(lines) and not lines[i].strip():
                i += 1
            hook_text, i = _collect_paragraph(lines, i)
            hook = hook_text or hook
            continue

        if _RE_SLIDES.match(s.strip()):
            i += 1
            continue

        if (m := _RE_SLIDE.match(s.strip())) and current_num is not None and current_title is not None:
            slide_num = int(m.group(1))
            i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            slide_title, i = _collect_paragraph(lines, i)
            slide_body, i = _collect_paragraph(lines, i)
            slides.append(
                SlideRow(
                    run_id=meta.run_id,
                    generation_run_id=None,
                    document=meta.document,
                    generated_at=meta.generated_at,
                    bundle_path=str(path),
                    carousel_number=current_num,
                    carousel_title=current_title,
                    slide_number=slide_num,
                    slide_title=slide_title,
                    slide_body=slide_body,
                    cta=None,
                    idea_id=None,
                )
            )
            slide_count = max(slide_count, slide_num)
            continue

        if _RE_CTA.match(s.strip()):
            i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            cta_text, i = _collect_paragraph(lines, i)
            cta = cta_text or cta
            continue

        if _RE_DISCLAIMER.match(s.strip()):
            i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            d_text, i = _collect_paragraph(lines, i)
            # keep the last disclaimer (bundles sometimes repeat)
            disclaimer = d_text or disclaimer
            continue

        i += 1

    flush_carousel()
    return meta, carousels, slides


def _best_hook_text(c: CandidateIdea) -> str:
    if not c.hooks:
        return ""
    if c.best_hook_id:
        for h in c.hooks:
            if h.hook_id == c.best_hook_id:
                return (h.text or "").strip()
    return (c.hooks[0].text or "").strip()


def _effective_cta(c: CandidateIdea) -> str:
    if c.carousel_draft and (c.carousel_draft.cta or "").strip():
        return (c.carousel_draft.cta or "").strip()
    return (c.best_cta or "").strip()


def write_export_csvs_from_state(
    *, state: RunState, run_dir: Path, generated_at: str | None = None
) -> tuple[Path, Path]:
    """
    Write bundle__carousels.csv and bundle__slides.csv from RunState.
    CTA is taken from carousel_draft.cta when present, otherwise best_cta (hooks-only flow).
    Every slide row repeats the same cta for that idea for easy filtering in spreadsheets.
    """
    selected = [c for c in state.candidates if c.selected]
    selected.sort(key=lambda c: c.rank or 10_000)

    run_id = state.document.document_id
    gen_id = state.generation_run_id
    document = state.document.title or ""
    ts = generated_at if generated_at is not None else datetime.utcnow().isoformat() + "Z"
    bundle_path = str((run_dir / "bundle.md").resolve())

    carousels_csv = run_dir / "bundle__carousels.csv"
    slides_csv = run_dir / "bundle__slides.csv"

    carousel_rows: list[CarouselRow] = []
    slide_rows: list[SlideRow] = []

    for c in selected:
        rank = c.rank or 0
        hook = _best_hook_text(c) or None
        cd = c.carousel_draft
        slides_list = list(cd.slides) if cd and cd.slides else []
        slide_count = len(slides_list)
        cta_str = _effective_cta(c) or None
        disclaimer: str | None = None
        if cd:
            disclaimer = (cd.disclaimer or "").strip() or None

        carousel_rows.append(
            CarouselRow(
                run_id=run_id,
                generation_run_id=gen_id,
                document=document,
                generated_at=ts,
                bundle_path=bundle_path,
                carousel_number=rank,
                title=c.topic,
                pillar=c.content_pillar,
                format=c.format_suggestion,
                hook_best=hook,
                cta=cta_str,
                disclaimer=disclaimer,
                slide_count=slide_count,
                idea_id=c.idea_id,
            )
        )

        for s in slides_list:
            slide_rows.append(
                SlideRow(
                    run_id=run_id,
                    generation_run_id=gen_id,
                    document=document,
                    generated_at=ts,
                    bundle_path=bundle_path,
                    carousel_number=rank,
                    carousel_title=c.topic,
                    slide_number=s.slide_number,
                    slide_title=(s.main_text or "").strip(),
                    slide_body=(s.subtext or "").strip(),
                    cta=cta_str,
                    idea_id=c.idea_id,
                )
            )

    run_dir.mkdir(parents=True, exist_ok=True)

    with carousels_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_CAROUSEL_FIELDNAMES)
        w.writeheader()
        for r in carousel_rows:
            w.writerow({k: getattr(r, k) for k in CSV_CAROUSEL_FIELDNAMES})

    with slides_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_SLIDE_FIELDNAMES)
        w.writeheader()
        for r in slide_rows:
            w.writerow({k: getattr(r, k) for k in CSV_SLIDE_FIELDNAMES})

    return carousels_csv, slides_csv


def write_bundle_csvs(*, bundle_path: Path, out_dir: Path | None = None) -> tuple[Path, Path]:
    meta, carousels, slides = parse_bundle_markdown(bundle_path)
    _ = meta

    out_dir = out_dir or bundle_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    carousels_csv = out_dir / f"{bundle_path.stem}__carousels.csv"
    slides_csv = out_dir / f"{bundle_path.stem}__slides.csv"

    cta_by_carousel = {c.carousel_number: c.cta for c in carousels}

    with carousels_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_CAROUSEL_FIELDNAMES)
        w.writeheader()
        for r in carousels:
            w.writerow({k: getattr(r, k) for k in CSV_CAROUSEL_FIELDNAMES})

    with slides_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_SLIDE_FIELDNAMES)
        w.writeheader()
        for r in slides:
            row = {k: getattr(r, k) for k in CSV_SLIDE_FIELDNAMES}
            row["cta"] = cta_by_carousel.get(r.carousel_number, row.get("cta"))
            w.writerow(row)

    return carousels_csv, slides_csv

