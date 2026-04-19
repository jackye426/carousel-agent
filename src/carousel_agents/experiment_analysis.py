"""
Analyse A/B experiment outcomes from performance JSONL (no third-party stats deps).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .performance import load_performance_jsonl
from .schemas import PerformanceLog


def _metric_value(r: PerformanceLog, metric: str) -> float | None:
    d = r.derived
    if metric == "save_rate":
        return d.save_rate
    if metric == "share_rate":
        return d.share_rate
    if metric == "profile_visit_rate":
        return d.profile_visit_rate
    return None


def _counts_for_metric(r: PerformanceLog, metric: str) -> tuple[int | None, int | None]:
    """
    Return (numerator, denominator) for rate-based metrics when derivable from observed counts.
    For save_rate: saves, impressions
    """
    obs = r.observed
    im = obs.impressions
    if metric == "save_rate":
        return obs.saves, im
    if metric == "share_rate":
        return obs.shares, im
    if metric == "profile_visit_rate":
        return obs.profile_visits, im
    return None, None


def two_proportion_z_test(
    *,
    x1: int,
    n1: int,
    x2: int,
    n2: int,
) -> tuple[float | None, float | None]:
    """
    Pooled two-proportion z-test (two-sided p-value using normal approx).
    Returns (z, p_two_sided) or (None, None) if undefined.
    """
    if n1 <= 0 or n2 <= 0:
        return None, None
    p1 = x1 / n1
    p2 = x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    denom = p_pool * (1.0 - p_pool) * (1.0 / n1 + 1.0 / n2)
    if denom <= 0:
        return None, None
    se = math.sqrt(denom)
    z = (p1 - p2) / se
    abs_z = abs(z)
    p_two = 2.0 * (1.0 - _std_normal_cdf(abs_z))
    return z, max(0.0, min(1.0, p_two))


def _std_normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def analyze_experiment(
    *,
    experiment_id: str,
    performance_jsonl: Path,
    primary_metric: str = "save_rate",
    min_impressions_per_arm: int = 2000,
) -> dict[str, Any]:
    allowed_m = {"save_rate", "share_rate", "profile_visit_rate"}
    if primary_metric not in allowed_m:
        raise ValueError(f"primary_metric must be one of {sorted(allowed_m)}")
    rows = [r for r in load_performance_jsonl(performance_jsonl) if (r.experiment_id or "") == experiment_id]
    by_v: dict[str, list[PerformanceLog]] = {"A": [], "B": []}
    for r in rows:
        v = r.variant
        if v in by_v:
            by_v[v].append(r)

    arms: dict[str, Any] = {}
    for label in ("A", "B"):
        g = by_v[label]
        imps = [r.observed.impressions for r in g if r.observed.impressions is not None]
        sum_imp = sum(imps) if imps else None
        tot_num = 0
        for r in g:
            num, den = _counts_for_metric(r, primary_metric)
            if num is not None:
                tot_num += int(num)
        rates = [_metric_value(r, primary_metric) for r in g if _metric_value(r, primary_metric) is not None]
        avg_rate = sum(rates) / len(rates) if rates else None
        arms[label] = {
            "n_posts": len(g),
            "sum_impressions": sum_imp,
            "sum_metric_numerator": tot_num,
            "avg_metric_rate": avg_rate,
            "post_ids": [r.post_id for r in g],
        }

    a_rows, b_rows = by_v["A"], by_v["B"]
    x1, n1 = _pool_counts(a_rows, primary_metric)
    x2, n2 = _pool_counts(b_rows, primary_metric)
    z, p_val = (
        two_proportion_z_test(x1=x1, n1=n1, x2=x2, n2=n2)
        if n1 and n2
        else (None, None)
    )

    meets = (
        n1 >= min_impressions_per_arm and n2 >= min_impressions_per_arm if n1 and n2 else False
    )

    winner: str = "inconclusive"
    if p_val is not None and meets and z is not None:
        if p_val < 0.05:
            p1 = x1 / n1
            p2 = x2 / n2
            if p1 > p2:
                winner = "A"
            elif p2 > p1:
                winner = "B"
            else:
                winner = "tie"

    return {
        "experiment_id": experiment_id,
        "primary_metric": primary_metric,
        "min_impressions_per_arm": min_impressions_per_arm,
        "meets_min_impressions": meets,
        "arms_detail": arms,
        "test": {
            "pooled_A": {"x": x1, "n": n1, "rate": (x1 / n1) if n1 else None},
            "pooled_B": {"x": x2, "n": n2, "rate": (x2 / n2) if n2 else None},
            "difference_B_minus_A": ((x2 / n2) - (x1 / n1)) if n1 and n2 else None,
            "z": z,
            "p_value_two_sided": p_val,
        },
        "winner": winner,
        "notes": (
            "Need at least one logged row per arm with experiment_id and variant."
            if len(a_rows) == 0 or len(b_rows) == 0
            else ""
        ),
    }


def _pool_counts(rows: list[PerformanceLog], metric: str) -> tuple[int, int]:
    """Sum numerator and impressions across rows."""
    tot_x = 0
    tot_n = 0
    for r in rows:
        num, den = _counts_for_metric(r, metric)
        if den is None or den <= 0:
            continue
        if num is None:
            continue
        tot_x += int(num)
        tot_n += int(den)
    return tot_x, tot_n


def save_experiment_result(result: dict[str, Any], *, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out_path


def load_experiment_results_from_dir(experiments_dir: Path) -> list[dict[str, Any]]:
    if not experiments_dir.exists():
        return []
    out: list[dict[str, Any]] = []
    for p in sorted(experiments_dir.glob("*_result.json")):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            continue
    return out
