"""A/B experiment analysis and pipeline split (no API)."""

from __future__ import annotations

import json
from pathlib import Path

from carousel_agents.experiment_analysis import analyze_experiment, two_proportion_z_test
from carousel_agents.performance import append_performance_log
from carousel_agents.pipeline import _apply_experiment_ab_split, _pick_two_distinct_hooks
from carousel_agents.schemas import (
    CandidateIdea,
    Citation,
    DocumentMeta,
    ExperimentSpec,
    HookOption,
    PerformanceLog,
    PerformanceObserved,
    RunState,
    ScoreFields,
    Shortlist,
)
from carousel_agents.validation import enforce_selection_gate


def test_two_proportion_z_symmetric() -> None:
    z, p = two_proportion_z_test(x1=50, n1=5000, x2=50, n2=5000)
    assert z is not None and p is not None
    assert abs(z) < 1e-6
    assert p > 0.99


def test_analyze_experiment_from_jsonl(tmp_path: Path) -> None:
    p = tmp_path / "perf.jsonl"
    append_performance_log(
        path=p,
        row=PerformanceLog(
            post_id="p1",
            idea_id="i__expA",
            pillar="recognition",
            format_suggestion="steps",
            experiment_id="exp_test",
            variant="A",
            hook_style="how_to",
            observed=PerformanceObserved(impressions=5000, saves=200),
        ),
    )
    append_performance_log(
        path=p,
        row=PerformanceLog(
            post_id="p2",
            idea_id="i__expB",
            pillar="recognition",
            format_suggestion="steps",
            experiment_id="exp_test",
            variant="B",
            hook_style="myth_bust",
            observed=PerformanceObserved(impressions=5000, saves=250),
        ),
    )
    out = analyze_experiment(
        experiment_id="exp_test",
        performance_jsonl=p,
        primary_metric="save_rate",
        min_impressions_per_arm=2000,
    )
    assert out["meets_min_impressions"] is True
    assert out["winner"] == "B"


def test_pick_two_distinct_hooks() -> None:
    a, b = _pick_two_distinct_hooks(
        [
            HookOption(hook_id="h1", text="a", style="how_to"),
            HookOption(hook_id="h2", text="b", style="myth_bust"),
        ]
    )
    assert a.style != b.style


def test_apply_experiment_ab_split() -> None:
    base = CandidateIdea(
        idea_id="i001",
        content_pillar="habits",
        topic="t",
        angle="a",
        core_claim="c",
        audience_pain="p",
        promise="pr",
        format_suggestion="steps",
        source_citations=[Citation(chunk_id="c1", excerpt="x", note="n")],
        safety_flags=[],
        scores=ScoreFields(),
        rank=1,
        selected=True,
        hooks=[
            HookOption(hook_id="h1", text="one", style="how_to"),
            HookOption(hook_id="h2", text="two", style="myth_bust"),
        ],
        best_hook_id="h1",
    )
    state = RunState(
        document=DocumentMeta(document_id="doc_x"),
        candidates=[base],
        shortlist=Shortlist(selected_idea_ids=["i001"], selection_frozen_at=__import__("datetime").datetime.utcnow()),
        experiment=ExperimentSpec(
            experiment_id="exp_unit",
            hypothesis="h",
            idea_id="i001",
            treatment_key="t",
        ),
    )
    enforce_selection_gate(state)
    _apply_experiment_ab_split(state)
    ids = [c.idea_id for c in state.candidates if c.selected]
    assert "i001__expA" in ids and "i001__expB" in ids
    assert state.shortlist.selected_idea_ids == ["i001__expA", "i001__expB"]

