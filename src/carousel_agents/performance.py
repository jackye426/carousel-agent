from __future__ import annotations

import json
from pathlib import Path

from .schemas import PerformanceDerived, PerformanceLog, PerformanceObserved


def append_performance_log(*, path: Path, row: PerformanceLog) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(row.model_dump_json())
        f.write("\n")


def compute_derived(observed: PerformanceObserved) -> PerformanceDerived:
    imp = observed.impressions or 0
    if imp <= 0:
        return PerformanceDerived()
    return PerformanceDerived(
        save_rate=_safe_div(observed.saves, imp),
        share_rate=_safe_div(observed.shares, imp),
        profile_visit_rate=_safe_div(observed.profile_visits, imp),
    )


def load_performance_jsonl(path: Path) -> list[PerformanceLog]:
    if not path.exists():
        return []
    rows: list[PerformanceLog] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        rows.append(PerformanceLog.model_validate(json.loads(s)))
    return rows


def _safe_div(num: int | None, den: int | None) -> float | None:
    if not num or not den:
        return 0.0
    return round(float(num) / float(den), 6)

