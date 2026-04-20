"""Tests for Streamlit RunState load / phase inference helpers."""

from __future__ import annotations

from carousel_agents.schemas import (
    CandidateIdea,
    CarouselDraft,
    Citation,
    DocumentMeta,
    DraftSlide,
    HookOption,
    RunState,
    Shortlist,
)
from carousel_agents.ui.run_state_session import infer_ui_phase


def _minimal_candidate(idea_id: str = "i1", **kwargs: object) -> CandidateIdea:
    base = dict(
        idea_id=idea_id,
        content_pillar="p",
        topic="t",
        angle="a",
        core_claim="c",
        audience_pain="ap",
        promise="pr",
        format_suggestion="carousel",
        source_citations=[Citation(chunk_id="ch1", excerpt="e", note="")],
    )
    base.update(kwargs)
    return CandidateIdea.model_validate(base)


def _minimal_state(**kwargs: object) -> RunState:
    doc = DocumentMeta(document_id="doc1", title="T")
    base = dict(
        document=doc,
        candidates=[_minimal_candidate()],
        shortlist=Shortlist(),
        awaiting_human_review=True,
        human_shortlist_curated=False,
    )
    base.update(kwargs)
    return RunState.model_validate(base)


def test_infer_select_when_awaiting_review() -> None:
    s = _minimal_state(awaiting_human_review=True)
    assert infer_ui_phase(s) == "select"


def test_infer_select_when_not_curated() -> None:
    s = _minimal_state(
        awaiting_human_review=False,
        human_shortlist_curated=False,
        shortlist=Shortlist(selected_idea_ids=["i1"]),
    )
    assert infer_ui_phase(s) == "select"


def test_infer_clarify_curated_no_hooks_no_transcript() -> None:
    s = _minimal_state(
        awaiting_human_review=False,
        human_shortlist_curated=True,
        shortlist=Shortlist(selected_idea_ids=["i1"]),
        candidates=[_minimal_candidate(idea_id="i1", hooks=[])],
        writer_clarification_transcript=None,
    )
    assert infer_ui_phase(s) == "clarify"


def test_infer_produce_when_transcript_but_no_hooks() -> None:
    s = _minimal_state(
        awaiting_human_review=False,
        human_shortlist_curated=True,
        shortlist=Shortlist(selected_idea_ids=["i1"]),
        candidates=[_minimal_candidate(idea_id="i1", hooks=[])],
        writer_clarification_transcript="Q1: x\nReviewer answers:\ny",
    )
    assert infer_ui_phase(s) == "produce"


def test_infer_produce_when_hooks_incomplete_slides() -> None:
    s = _minimal_state(
        awaiting_human_review=False,
        human_shortlist_curated=True,
        shortlist=Shortlist(selected_idea_ids=["i1"]),
        candidates=[
            _minimal_candidate(
                idea_id="i1",
                hooks=[HookOption(hook_id="h1", text="t", style="s")],
                carousel_draft=None,
            )
        ],
    )
    assert infer_ui_phase(s) == "produce"


def test_infer_done_when_all_slides() -> None:
    draft = CarouselDraft(
        slides=[DraftSlide(slide_number=1, main_text="m", subtext="s", citations=[])],
        cta="cta",
    )
    s = _minimal_state(
        awaiting_human_review=False,
        human_shortlist_curated=True,
        shortlist=Shortlist(selected_idea_ids=["i1"]),
        candidates=[_minimal_candidate(idea_id="i1", carousel_draft=draft)],
    )
    assert infer_ui_phase(s) == "done"
