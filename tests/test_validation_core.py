"""Unit tests for deterministic validation and ranking (no API calls)."""

from __future__ import annotations

from carousel_agents.pipeline import _coerce_candidate
from carousel_agents.schemas import (
    RUN_STATE_SCHEMA_VERSION,
    CandidateIdea,
    Chunk,
    Citation,
    DocumentMeta,
    RunState,
    ScoreFields,
    Shortlist,
    SourceLocation,
    Weights,
)
from carousel_agents.validation import (
    compute_weighted_totals_and_rank,
    finalize_shortlist_ids,
    pick_stratified_shortlist,
    repair_citation_chunk_ids,
)


def _minimal_idea(*, idea_id: str, pillar: str, rank: int) -> CandidateIdea:
    return CandidateIdea(
        idea_id=idea_id,
        content_pillar=pillar,
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
        rank=rank,
    )


def test_pick_stratified_shortlist_one_per_pillar_then_rank() -> None:
    """Top of pool is all one pillar; stratified should still pull other pillars when available."""
    pool = [
        _minimal_idea(idea_id="a1", pillar="access_or_decision", rank=1),
        _minimal_idea(idea_id="a2", pillar="access_or_decision", rank=2),
        _minimal_idea(idea_id="r1", pillar="recognition", rank=3),
        _minimal_idea(idea_id="v1", pillar="validation", rank=4),
    ]
    out = pick_stratified_shortlist(pool, 3, pillar_order=["recognition", "validation", "access_or_decision"])
    ids = [c.idea_id for c in out]
    assert ids == ["r1", "v1", "a1"]


def _chunk(cid: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=cid,
        text=text,
        source_location=SourceLocation(),
        section_path=[],
        char_start=0,
        char_end=len(text),
    )


def test_compute_weighted_totals_tiebreak_order() -> None:
    w = Weights()
    a = CandidateIdea(
        idea_id="i_b",
        content_pillar="validation",
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
        scores=ScoreFields(
            hook_strength=4,
            clarity=3,
            save_share=3,
            strategic_fit=3,
            source_support=5,
            total_weighted=None,
        ),
    )
    b = CandidateIdea(
        idea_id="i_a",
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
        scores=ScoreFields(
            hook_strength=4,
            clarity=3,
            save_share=3,
            strategic_fit=3,
            source_support=5,
            total_weighted=None,
        ),
    )
    state = RunState(
        schema_version=RUN_STATE_SCHEMA_VERSION,
        document=DocumentMeta(document_id="d", title="t", type="md", language="en"),
        candidates=[a, b],
        shortlist=Shortlist(),
    )
    compute_weighted_totals_and_rank(state, w)
    assert state.candidates[0].idea_id == "i_a"
    assert state.candidates[0].rank == 1
    assert state.candidates[1].idea_id == "i_b"


def test_finalize_shortlist_ids_respects_pool_order() -> None:
    c1 = CandidateIdea(
        idea_id="i1",
        content_pillar="validation",
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
        rank=1,
    )
    c2 = CandidateIdea(
        idea_id="i2",
        content_pillar="validation",
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
        rank=2,
    )
    pool = [c1, c2]
    out = finalize_shortlist_ids(["bad", "i2", "i1"], pool=pool, select_n=2)
    assert out == ["i2", "i1"]


def test_repair_citation_chunk_ids() -> None:
    idea = CandidateIdea(
        idea_id="i1",
        content_pillar="validation",
        topic="t",
        angle="a",
        core_claim="c",
        audience_pain="p",
        promise="pr",
        format_suggestion="steps",
        source_citations=[
            Citation(chunk_id="c_wrong", excerpt="only in b", note="n"),
            Citation(chunk_id="c_wrong", excerpt="stays", note="n"),
        ],
        safety_flags=[],
    )
    state = RunState(
        schema_version=RUN_STATE_SCHEMA_VERSION,
        document=DocumentMeta(document_id="d", title="t", type="md", language="en"),
        chunks=[
            _chunk("c_wrong", "stays here"),
            _chunk("c_right", "prefix only in b middle"),
        ],
        candidates=[idea],
        shortlist=Shortlist(),
    )
    n = repair_citation_chunk_ids(state)
    assert n == 1
    assert idea.source_citations[0].chunk_id == "c_right"


def test_coerce_candidate_verbatim_to_excerpt() -> None:
    raw = {
        "idea_id": "i1",
        "content_pillar": "validation",
        "topic": "t",
        "angle": "a",
        "core_claim": "c",
        "audience_pain": "p",
        "promise": "pr",
        "format_suggestion": "steps",
        "source_citations": [{"chunk_id": "c1", "verbatim": "hello", "note": "n"}],
        "safety_flags": [],
    }
    fixed = _coerce_candidate(raw)
    assert fixed["source_citations"][0]["excerpt"] == "hello"


def test_run_state_default_schema_version() -> None:
    st = RunState.model_validate_json(
        '{"document": {"document_id": "x", "title": "t", "type": "md", "language": "en"}, '
        '"chunks": [], "candidates": [], "shortlist": {}}'
    )
    assert st.schema_version == RUN_STATE_SCHEMA_VERSION
