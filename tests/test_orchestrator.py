"""Experiment orchestrator matching (no API)."""

from __future__ import annotations

from carousel_agents.experiment_orchestrator import match_hypothesis_to_candidate, try_apply_orchestrator
from carousel_agents.experiment_program_store import load_program, save_program
from carousel_agents.schemas import (
    CandidateIdea,
    Citation,
    DocumentMeta,
    ExperimentProgramState,
    QueuedHypothesis,
    RunState,
    ScoreFields,
    Shortlist,
)


def _cand(idea_id: str, pillar: str, topic: str, rank: int) -> CandidateIdea:
    return CandidateIdea(
        idea_id=idea_id,
        content_pillar=pillar,
        topic=topic,
        angle="a",
        core_claim="c",
        audience_pain="p",
        promise="pr",
        format_suggestion="steps",
        source_citations=[Citation(chunk_id="c1", excerpt="x", note="n")],
        safety_flags=[],
        scores=ScoreFields(),
        rank=rank,
        selected=True,
    )


def test_match_hypothesis_pillar_and_topic() -> None:
    c1 = _cand("i1", "recognition", "Symptom tracking basics", 1)
    c2 = _cand("i2", "access_or_decision", "How to prepare for laparoscopy", 2)
    qh = QueuedHypothesis(
        hypothesis="h",
        target_pillar="access_or_decision",
        topic_substring="laparoscopy",
    )
    assert match_hypothesis_to_candidate([c1, c2], qh) == c2


def test_try_apply_orchestrator_writes_spec(tmp_path) -> None:
    qh = QueuedHypothesis(
        hypothesis="test",
        treatment_key="t",
        target_pillar="access_or_decision",
    )
    prog = ExperimentProgramState(queue=[qh])
    path = tmp_path / "program.json"
    save_program(prog, path=path)

    c = _cand("i9", "access_or_decision", "NHS vs private", 1)
    state = RunState(
        document=DocumentMeta(document_id="doc_test"),
        candidates=[c],
        shortlist=Shortlist(selected_idea_ids=["i9"], selection_frozen_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc)),
    )
    assert try_apply_orchestrator(state, path) is True
    assert state.experiment is not None
    assert state.experiment.idea_id == "i9"
    assert state.experiment.experiment_id == f"exp_{qh.id}"

    reloaded = load_program(path)
    assert reloaded.queue[0].status == "attached"
