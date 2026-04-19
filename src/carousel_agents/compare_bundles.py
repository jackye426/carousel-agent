from __future__ import annotations

import csv
from pathlib import Path

from .bundle_to_csv import CarouselRow, SlideRow, parse_bundle_markdown


def write_comparison_csvs(
    *,
    bundles: list[Path],
    out_dir: Path,
    overview_name: str = "comparison_overview.csv",
    detail_name: str = "comparison_detail_slides.csv",
) -> tuple[Path, Path]:
    """
    Create two comparison CSVs across multiple bundle markdown files:
    - overview: one row per carousel per run
    - detail: one row per slide per carousel per run
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    overview_rows: list[dict] = []
    detail_rows: list[dict] = []

    for b in bundles:
        meta, carousels, slides = parse_bundle_markdown(b)

        def _run_label() -> str:
            # Human-friendly label, stable across runs
            stem = b.stem
            if stem.startswith("bundle__"):
                return stem[len("bundle__") :]
            return stem

        run_label = _run_label()
        cta_by_carousel = {c.carousel_number: c.cta for c in carousels}

        for c in carousels:
            overview_rows.append(
                {
                    "run_label": run_label,
                    "run_id": meta.run_id,
                    "document": meta.document,
                    "generated_at": meta.generated_at,
                    "bundle_path": str(b),
                    "carousel_number": c.carousel_number,
                    "topic": c.title,
                    "pillar": c.pillar,
                    "format": c.format,
                    "hook_best": c.hook_best,
                    "cta": c.cta,
                    "disclaimer": c.disclaimer,
                    "slide_count": c.slide_count,
                }
            )

        for s in slides:
            detail_rows.append(
                {
                    "run_label": run_label,
                    "run_id": meta.run_id,
                    "document": meta.document,
                    "generated_at": meta.generated_at,
                    "bundle_path": str(b),
                    "carousel_number": s.carousel_number,
                    "topic": s.carousel_title,
                    "slide_number": s.slide_number,
                    "slide_title": s.slide_title,
                    "slide_body": s.slide_body,
                    "cta": cta_by_carousel.get(s.carousel_number),
                }
            )

    overview_path = out_dir / overview_name
    detail_path = out_dir / detail_name

    # Deterministic ordering for easier comparison
    overview_rows.sort(key=lambda r: (r["run_label"], r["carousel_number"]))
    detail_rows.sort(key=lambda r: (r["run_label"], r["carousel_number"], r["slide_number"]))

    with overview_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "run_label",
                "run_id",
                "document",
                "generated_at",
                "bundle_path",
                "carousel_number",
                "topic",
                "pillar",
                "format",
                "hook_best",
                "cta",
                "disclaimer",
                "slide_count",
            ],
        )
        w.writeheader()
        for r in overview_rows:
            w.writerow(r)

    with detail_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "run_label",
                "run_id",
                "document",
                "generated_at",
                "bundle_path",
                "carousel_number",
                "topic",
                "slide_number",
                "slide_title",
                "slide_body",
                "cta",
            ],
        )
        w.writeheader()
        for r in detail_rows:
            w.writerow(r)

    return overview_path, detail_path

