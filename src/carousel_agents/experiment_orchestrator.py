"""
Experiment program orchestrator: performance signals, queue matching, LLM suggestions,
and wiring into the pipeline after the shortlist is frozen.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .defaults import experiments_dir_default
from .experiment_analysis import load_experiment_results_from_dir
from .experiment_program_store import load_program, save_program
from .llm import OpenAICompatibleClient, get_orchestrator_config
from .llm_response_models import OrchestratorSuggestResponse
from .performance import load_performance_jsonl
from .performance_digest import load_or_build_performance_digest
from .prompts import system_orchestrator, user_orchestrator_suggest
from .schemas import CandidateIdea, ExperimentProgramState, ExperimentSpec, QueuedHypothesis, RunState


def build_orchestrator_signals(
    *,
    performance_jsonl: Path | None = None,
    performance_digest_path: Path | None = None,
    experiments_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Compact bundle for rule + LLM planners: aggregates + digest slice + recent experiment results.
    """
    from .defaults import performance_digest_json_path, performance_jsonl_path

    pj = performance_jsonl or performance_jsonl_path()
    digest = None
    dp = performance_digest_path
    if dp is None:
        digest = load_or_build_performance_digest(explicit_path=None, disabled=False)
    elif dp.exists():
        digest = json.loads(dp.read_text(encoding="utf-8"))

    rows = load_performance_jsonl(pj) if pj.exists() else []
    by_hook: dict[str, list[float]] = defaultdict(list)
    by_pillar: dict[str, int] = defaultdict(int)
    exp_rows = [r for r in rows if r.experiment_id]
    for r in rows:
        hs = (r.hook_style or "(none)").strip()
        if r.derived and r.derived.save_rate is not None:
            by_hook[hs].append(float(r.derived.save_rate))
        by_pillar[(r.pillar or "(none)").strip()] += 1

    hook_summary = {
        k: {"n": len(vs), "median_save_rate": sorted(vs)[len(vs) // 2] if vs else None}
        for k, vs in by_hook.items()
    }

    ex_dir = experiments_dir or experiments_dir_default()
    recent_results = load_experiment_results_from_dir(ex_dir)[-6:]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "performance_row_counts": {"total": len(rows), "with_experiment_id": len(exp_rows)},
        "hook_style_aggregate": hook_summary,
        "pillar_post_counts": dict(by_pillar),
        "performance_digest": digest,
        "recent_experiment_results": recent_results,
    }


def _selected_candidates(state: RunState) -> list[CandidateIdea]:
    return [c for c in state.candidates if c.selected]


def match_hypothesis_to_candidate(candidates: list[CandidateIdea], qh: QueuedHypothesis) -> CandidateIdea | None:
    if not candidates:
        return None
    pool = list(candidates)
    pillar = (qh.target_pillar or "").strip().lower()
    if pillar:
        filtered = [c for c in pool if (c.content_pillar or "").strip().lower() == pillar]
        if filtered:
            pool = filtered
    sub = (qh.topic_substring or "").strip().lower()
    if sub:
        filtered = [c for c in pool if sub in (c.topic or "").lower()]
        if filtered:
            pool = filtered
    pool.sort(key=lambda c: (c.rank or 10_000, c.idea_id))
    return pool[0]


def try_apply_orchestrator(state: RunState, program_path: Path) -> bool:
    """
    If the queue has a `queued` item, match it to the current shortlist and set `state.experiment`.
    Persists program updates when an experiment is attached. Returns True if state.experiment was set.
    """
    prog = load_program(program_path)
    head = next((q for q in prog.queue if q.status == "queued"), None)
    if head is None:
        return False
    cand = match_hypothesis_to_candidate(_selected_candidates(state), head)
    if cand is None:
        return False

    exp_id = f"exp_{head.id}"
    state.experiment = ExperimentSpec(
        experiment_id=exp_id,
        hypothesis=head.hypothesis,
        idea_id=cand.idea_id,
        treatment_key=head.treatment_key or "hook_style_contrast",
        primary_metric=head.primary_metric,
        min_impressions_per_arm=head.min_impressions_per_arm,
        status="planned",
    )
    head.status = "attached"
    head.attached_experiment_id = exp_id
    head.attached_idea_id = cand.idea_id
    head.attached_run_id = state.document.document_id
    note = (
        f" Orchestrator attached queue item {head.id!r} → experiment {exp_id!r} on idea {cand.idea_id!r}."
    )
    state.shortlist.notes = (state.shortlist.notes or "").strip() + note
    save_program(prog, path=program_path)
    return True


def suggest_rules_based(*, signals: dict[str, Any]) -> QueuedHypothesis:
    """
    Deterministic fallback: prioritise a hook-style contrast test on under-sampled pillars
    (heuristic, not optimal — LLM path is richer).
    """
    pillar_counts: dict[str, int] = signals.get("pillar_post_counts") or {}
    # Prefer access_or_decision if we have few posts there vs recognition (example heuristic)
    access_n = pillar_counts.get("access_or_decision", 0)
    rec_n = pillar_counts.get("recognition", 0)
    pillar = "access_or_decision" if access_n <= rec_n else "recognition"
    return QueuedHypothesis(
        hypothesis=(
            f"Test two contrasting hook styles on {pillar} content to improve save_rate "
            "(e.g. direct how-to vs myth-bust framing)."
        ),
        treatment_key="hook_style_contrast",
        target_pillar=pillar,
        topic_substring=None,
        primary_metric="save_rate",
        source="rule",
        rationale="Heuristic: balance learning across pillars using logged post counts.",
    )


def suggest_with_llm(
    *,
    signals: dict[str, Any],
    program: ExperimentProgramState,
    client: Any | None = None,
) -> tuple[list[QueuedHypothesis], str | None]:
    cl = client or OpenAICompatibleClient()
    cfg = get_orchestrator_config()
    queue_summary = json.dumps([q.model_dump(mode="json") for q in program.queue], ensure_ascii=False, indent=2)
    signals_json = json.dumps(signals, ensure_ascii=False, default=str, indent=2)
    out = cl.chat_structured(
        response_model=OrchestratorSuggestResponse,
        model=cfg.model,
        temperature=cfg.temperature,
        system=system_orchestrator(),
        user=user_orchestrator_suggest(signals_json=signals_json, queue_summary=queue_summary),
    )
    queued: list[QueuedHypothesis] = []
    for h in out.hypotheses:
        tp = h.target_pillar.strip().lower() if h.target_pillar and h.target_pillar.strip() else None
        th = h.topic_hint.strip() if h.topic_hint and h.topic_hint.strip() else None
        queued.append(
            QueuedHypothesis(
                hypothesis=h.hypothesis.strip(),
                treatment_key=(h.treatment_key or "hook_style_contrast").strip(),
                target_pillar=tp,
                topic_substring=th,
                primary_metric="save_rate",
                source="llm",
                rationale=(h.rationale or "").strip(),
            )
        )
    notes = (out.program_notes or "").strip() or None
    return queued, notes


def append_suggestions_to_program(
    program: ExperimentProgramState,
    items: list[QueuedHypothesis],
    *,
    path: Path,
) -> None:
    program.queue.extend(items)
    save_program(program, path=path)
