"""
Pydantic response shapes for Instructor-backed `chat_structured` calls.
These mirror LLM JSON outputs before domain coercion (`_coerce_candidate`, etc.).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .schemas import (
    CarouselDraft,
    CitationIn,
    HookOption,
    JudgeDecision,
    OutlineSlide,
    QAResult,
)


class ExtractCandidateItem(BaseModel):
    """One extracted idea from the ideation extract pass."""

    model_config = ConfigDict(extra="ignore")

    idea_id: str = ""
    content_pillar: str = ""
    topic: str = ""
    angle: str = ""
    core_claim: str = ""
    audience_pain: str = ""
    promise: str = ""
    format_suggestion: str = ""
    reader_benefit: str = ""
    source_citations: list[CitationIn] = Field(default_factory=list)
    safety_flags: list[str] = Field(default_factory=list)


class ExtractCandidatesResponse(BaseModel):
    candidates: list[ExtractCandidateItem] = Field(default_factory=list)
    editorial_direction: str = ""


class ScoreRow(BaseModel):
    """Per-idea score row from the scoring pass."""

    model_config = ConfigDict(extra="ignore")

    idea_id: str
    hook_strength: int | None = None
    recognition: int | None = None
    clarity: int | None = None
    slideability: int | None = None
    save_share: int | None = None
    emotional_pull: int | None = None
    novelty: int | None = None
    strategic_fit: int | None = None
    source_support: int | None = None
    total_weighted: float | None = None


class ScoreCandidatesResponse(BaseModel):
    scores: list[ScoreRow] = Field(default_factory=list)


class HooksWriterResponse(BaseModel):
    hooks: list[HookOption] = Field(default_factory=list)
    best_hook_id: str | None = None
    outline: list[OutlineSlide] = Field(default_factory=list)


class CTAResponse(BaseModel):
    cta_options: list[str] = Field(default_factory=list)
    best_cta: str | None = None


class CaptionResponse(BaseModel):
    """IG caption draft for a single selected idea."""

    caption: str = ""


class SlidesWriterResponse(BaseModel):
    carousel_draft: CarouselDraft


class QAEditResponse(BaseModel):
    carousel_draft: CarouselDraft | None = None
    qa: QAResult | None = None


class WriterPreflightResponse(BaseModel):
    """Writer asks for clarification before hooks/slides (UI or automated)."""

    need_clarification: bool = False
    questions: list[str] = Field(default_factory=list)
    assistant_message: str = ""
    """Short message to show the human (summary + tone)."""


HookPolicyJudgeResponse = JudgeDecision


class OrchestratorHypothesisItem(BaseModel):
    """One proposed hypothesis from the orchestrator LLM."""

    hypothesis: str = ""
    treatment_key: str = "hook_style_contrast"
    target_pillar: str | None = None
    topic_hint: str | None = None
    rationale: str = ""


class OrchestratorSuggestResponse(BaseModel):
    hypotheses: list[OrchestratorHypothesisItem] = Field(default_factory=list)
    program_notes: str = ""
