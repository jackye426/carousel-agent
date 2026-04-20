"""Human-in-the-loop selection (no API)."""

from __future__ import annotations

from carousel_agents.human_review import HumanSelectionFile, apply_human_selection_to_state
from carousel_agents.schemas import (
    RUN_STATE_SCHEMA_VERSION,
    CandidateIdea,
    Citation,
    DocumentMeta,
    RunState,
    Shortlist,
)


def _idea(iid: str) -> CandidateIdea:
    return CandidateIdea(
        idea_id=iid,
        content_pillar="recognition",
        topic="t",
        angle="a",
        core_claim="c",
        audience_pain="p",
        promise="pr",
        format_suggestion="steps",
        source_citations=[
            Citation(chunk_id="c1", excerpt="x", note="n"),
            Citation(chunk_id="c2", excerpt="y", note="n"),
        ],
        safety_flags=[],
    )


def test_apply_human_selection_sets_shortlist_and_overrides() -> None:
    state = RunState(
        schema_version=RUN_STATE_SCHEMA_VERSION,
        document=DocumentMeta(document_id="d", title="t", type="md", language="en"),
        candidates=[_idea("i001"), _idea("i002")],
        shortlist=Shortlist(),
        awaiting_human_review=True,
        proposed_editorial_direction="Model proposal.",
    )
    sel = HumanSelectionFile(
        selected_idea_ids=["i002"],
        editorial_direction="Human override direction.",
        idea_overrides={"i002": {"angle": "New angle"}},
    )
    apply_human_selection_to_state(state, sel)
    assert state.shortlist.selected_idea_ids == ["i002"]
    assert state.shortlist.selection_frozen_at is not None
    assert state.human_editorial_direction == "Human override direction."
    assert state.human_shortlist_curated is True
    assert state.awaiting_human_review is False
    i2 = next(c for c in state.candidates if c.idea_id == "i002")
    assert i2.selected is True
    assert i2.angle == "New angle"
    i1 = next(c for c in state.candidates if c.idea_id == "i001")
    assert i1.selected is False
