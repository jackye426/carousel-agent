"""Tests for caption/OCR → RunState linkage."""

from __future__ import annotations

from pathlib import Path

from carousel_agents.post_linkage import (
    caption_affinity,
    match_post_to_runs,
    normalize_for_match,
    ocr_affinity,
    read_ocr_slides_text_for_asset,
    text_similarity,
    tokenize,
)
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


def _cand(**kwargs: object) -> CandidateIdea:
    base = dict(
        idea_id="i1",
        content_pillar="recognition",
        topic="Vitamin D myths",
        angle="What research actually says",
        core_claim="Supplements help some, not all",
        audience_pain="Confusion",
        promise="Clarity",
        format_suggestion="carousel",
        source_citations=[Citation(chunk_id="c1", excerpt="e", note="")],
        hooks=[
            HookOption(hook_id="h1", text="Stop blaming vitamin D for every symptom", style="contrarian"),
        ],
        carousel_draft=CarouselDraft(
            slides=[
                DraftSlide(slide_number=1, main_text="The myth you keep hearing", subtext="It's not that simple", citations=[]),
                DraftSlide(slide_number=2, main_text="What studies measure", subtext="Dosage matters", citations=[]),
            ],
            cta="Save for your GP visit",
        ),
    )
    base.update(kwargs)
    return CandidateIdea.model_validate(base)


def test_normalize_strips_url_and_hashtags() -> None:
    s = normalize_for_match("Check this out https://x.com/y #endo #health")
    assert "http" not in s
    assert "#" not in s


def test_text_similarity_identical() -> None:
    assert text_similarity("vitamin D myths carousel", "vitamin D myths carousel") > 0.9


def test_caption_affinity_hooks() -> None:
    c = _cand()
    cap = "Stop blaming vitamin D for every symptom"
    assert caption_affinity(cap, c) > 0.5


def test_ocr_affinity_draft() -> None:
    c = _cand()
    ocr = "The myth you keep hearing\n\nWhat studies measure"
    assert ocr_affinity(ocr, c) > 0.3


def test_match_post_picks_correct_idea(tmp_path: Path) -> None:
    weak = _cand(idea_id="weak", topic="Unrelated topic about cats", hooks=[], carousel_draft=None)
    strong = _cand(idea_id="strong")
    rs = RunState.model_validate(
        {
            "document": DocumentMeta(document_id="doc_x", title="T"),
            "candidates": [weak, strong],
            "shortlist": Shortlist(),
            "awaiting_human_review": False,
            "human_shortlist_curated": True,
        }
    )
    p = tmp_path / "run.json"
    p.write_text(rs.model_dump_json(), encoding="utf-8")
    matches = match_post_to_runs(
        states=[(p, rs)],
        caption="Stop blaming vitamin D for every symptom",
        ocr="The myth you keep hearing",
        top_n=5,
    )
    assert matches[0].idea_id == "strong"


def test_read_ocr_slides_jsonl(tmp_path: Path) -> None:
    j = tmp_path / "ocr.jsonl"
    j.write_text(
        '{"asset_id":"99","slide_index":2,"text":"Second slide"}\n'
        '{"asset_id":"99","slide_index":1,"text":"First slide"}\n',
        encoding="utf-8",
    )
    t = read_ocr_slides_text_for_asset(j, "99")
    assert "First slide" in t
    assert t.index("First slide") < t.index("Second slide")


def test_jaccard_tokenize() -> None:
    a = tokenize(normalize_for_match("hello world test"))
    b = tokenize(normalize_for_match("hello world other"))
    assert len(a & b) >= 2
