"""
Canonical reporting helpers for performance JSONL and tracker CSV.

Used by `python -m carousel_agents metrics ...` and by legacy CLI aliases.
"""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.table import Table

from .analysis_report import generate_report
from .performance import load_performance_jsonl
from .schemas import PerformanceLog


def _avg(xs: list[float | None]) -> float:
    vals = [x for x in xs if x is not None]
    return sum(vals) / max(1, len(vals))


def print_performance_summary_table(
    *,
    path: Path,
    group_by: str,
    min_n: int,
    console: Console | None = None,
) -> None:
    """Print Rich table: aggregate PerformanceLog rows by hook_style, pillar, format, or persona."""
    c = console or Console()
    rows = load_performance_jsonl(path)
    if not rows:
        c.print(f"[yellow]No rows found at[/yellow] {path}")
        return

    def key(r: PerformanceLog) -> str:
        if group_by == "pillar":
            return r.pillar or "(none)"
        if group_by == "format":
            return r.format_suggestion or "(none)"
        if group_by == "persona":
            return r.persona_id or "(none)"
        if group_by == "experiment_id":
            return r.experiment_id or "(none)"
        return r.hook_style or "(none)"

    groups: dict[str, list[PerformanceLog]] = {}
    for r in rows:
        groups.setdefault(key(r), []).append(r)

    table = Table(title=f"Performance by {group_by}")
    table.add_column(group_by)
    table.add_column("n", justify="right")
    table.add_column("avg_save_rate", justify="right")
    table.add_column("avg_share_rate", justify="right")
    table.add_column("avg_profile_visit_rate", justify="right")

    for k, g in sorted(groups.items(), key=lambda kv: len(kv[1]), reverse=True):
        if len(g) < min_n:
            continue
        sr = _avg([r.derived.save_rate for r in g])
        shr = _avg([r.derived.share_rate for r in g])
        pvr = _avg([r.derived.profile_visit_rate for r in g])
        table.add_row(k, str(len(g)), f"{sr:.4f}", f"{shr:.4f}", f"{pvr:.4f}")

    c.print(table)


def write_carousel_tracker_report(
    *,
    tracker_csv: Path,
    out: Path,
    ig_followers: int,
    tags_jsonl: Path | None,
) -> Path:
    """Markdown report from tracker (+ optional tags JSONL)."""
    return generate_report(
        tracker_csv=tracker_csv,
        out_path=out,
        follower_count_instagram=ig_followers,
        tags_jsonl=tags_jsonl,
    )
