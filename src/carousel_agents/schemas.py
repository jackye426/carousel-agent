from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


def _new_queued_hypothesis_id() -> str:
    return f"qh_{uuid.uuid4().hex[:10]}"


class SourceLocation(BaseModel):
    page: Optional[int] = None
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    timestamp_start: Optional[float] = None
    timestamp_end: Optional[float] = None


class Chunk(BaseModel):
    chunk_id: str
    text: str
    source_location: SourceLocation
    section_path: list[str] = Field(default_factory=list)
    char_start: int
    char_end: int


class Citation(BaseModel):
    chunk_id: str
    excerpt: str
    note: str


class CitationIn(BaseModel):
    """
    Input-friendly citation shape. Some models will use `verbatim` instead of `excerpt`.
    We normalise it during parsing so the rest of the pipeline can rely on `excerpt`.
    """

    chunk_id: str
    excerpt: str | None = None
    verbatim: str | None = None
    note: str = ""

    def to_citation(self) -> Citation:
        ex = (self.excerpt or self.verbatim or "").strip()
        return Citation(chunk_id=self.chunk_id, excerpt=ex, note=self.note)


class ScoreFields(BaseModel):
    """Five scored dimensions (0–5). Extra keys from older runs are ignored."""

    model_config = ConfigDict(extra="ignore")

    hook_strength: int | None = None
    clarity: int | None = None
    save_share: int | None = None
    strategic_fit: int | None = None
    source_support: int | None = None
    total_weighted: float | None = None


class HookOption(BaseModel):
    hook_id: str
    text: str
    style: str
    score: float | None = None
    rationale: str | None = None


class OutlineSlide(BaseModel):
    slide: int
    purpose: str
    key_point: str
    citations: list[str] = Field(default_factory=list)


class DraftSlide(BaseModel):
    slide_number: int
    main_text: str
    subtext: str
    citations: list[str] = Field(default_factory=list)
    design_note: str | None = None


class CarouselDraft(BaseModel):
    slides: list[DraftSlide]
    cta: str
    disclaimer: str | None = None


class QAItem(BaseModel):
    type: str
    detail: str
    slide: int | None = None


class QAResult(BaseModel):
    issues: list[QAItem] = Field(default_factory=list)
    passed: bool | None = None


class JudgeDecision(BaseModel):
    """
    Governance decision (ideation shortlist / carousel review; name kept for JSON compatibility).
    - approve: good to proceed/publish
    - revise: must be rewritten (with provided directives)
    - veto: do not proceed/publish
    """

    status: Literal["approve", "revise", "veto"]
    reason: str = ""
    directives: list[str] = Field(default_factory=list)


class JudgeShortlistReviewItem(BaseModel):
    idea_id: str
    decision: JudgeDecision


class JudgeShortlistReview(BaseModel):
    reviewed: list[JudgeShortlistReviewItem] = Field(default_factory=list)
    notes: str | None = None


class JudgeSlideIssue(BaseModel):
    slide_number: int | None = None
    type: str
    detail: str


class JudgeCarouselReview(BaseModel):
    """
    Review of a drafted carousel.
    """

    idea_id: str
    decision: JudgeDecision
    issues: list[JudgeSlideIssue] = Field(default_factory=list)
    one_sentence_good: str | None = None
    one_sentence_fix: str | None = None


class CandidateIdea(BaseModel):
    idea_id: str
    content_pillar: str
    topic: str
    angle: str
    core_claim: str
    audience_pain: str
    promise: str
    format_suggestion: str
    # Patient-facing "why save/share" line from ideation extract (not raw service description).
    reader_benefit: str | None = None
    source_citations: list[Citation]
    safety_flags: list[str] = Field(default_factory=list)
    scores: ScoreFields = Field(default_factory=ScoreFields)
    rank: int | None = None
    selected: bool = False
    selection_reason: str | None = None
    hooks: list[HookOption] = Field(default_factory=list)
    best_hook_id: str | None = None
    cta_options: list[str] = Field(default_factory=list)
    best_cta: str | None = None
    outline: list[OutlineSlide] = Field(default_factory=list)
    carousel_draft: CarouselDraft | None = None
    qa: QAResult | None = None

    # governance (ideation / hook policy; keys unchanged for saved run JSON)
    judge_vetoed: bool = False
    judge_veto_reason: str | None = None
    judge_carousel_review: JudgeCarouselReview | None = None

    # hook language governance
    hook_policy_flags: list[str] = Field(default_factory=list)
    hook_rewrite_rounds_used: int = 0

    # A/B experiment arms (set when one shortlisted idea is split into two publishable variants)
    experiment_id: str | None = None
    ab_variant: Literal["A", "B"] | None = None
    base_idea_id: str | None = None


class ExperimentSpec(BaseModel):
    """
    When attached to RunState, the pipeline splits `idea_id` into two candidates after hooks+CTA
    (different hook styles when possible). Both arms share one export folder / run_id.
    """

    experiment_id: str
    hypothesis: str = ""
    idea_id: str
    treatment_key: str = ""
    primary_metric: Literal["save_rate", "share_rate", "profile_visit_rate"] = "save_rate"
    min_impressions_per_arm: int = 2000
    status: Literal["planned", "live", "concluded"] = "planned"
    schedule_arm_order: Literal["A_first", "B_first"] | None = None


class QueuedHypothesis(BaseModel):
    """
    One row in the experiment program queue. Matched to a shortlisted idea after ideation.
    """

    id: str = Field(default_factory=_new_queued_hypothesis_id)
    hypothesis: str = ""
    treatment_key: str = "hook_style_contrast"
    target_pillar: str | None = None
    topic_substring: str | None = None
    primary_metric: Literal["save_rate", "share_rate", "profile_visit_rate"] = "save_rate"
    min_impressions_per_arm: int = 2000
    status: Literal["queued", "attached", "cancelled", "concluded"] = "queued"
    attached_experiment_id: str | None = None
    attached_idea_id: str | None = None
    attached_run_id: str | None = None
    source: Literal["llm", "rule", "manual"] = "manual"
    rationale: str = ""


class ExperimentProgramState(BaseModel):
    """Persisted experiment queue + metadata (data/experiment_program.json)."""

    version: str = "v1"
    queue: list[QueuedHypothesis] = Field(default_factory=list)
    notes: str | None = None


class AudienceConfig(BaseModel):
    platform: Literal["instagram", "linkedin", "tiktok"] = "instagram"
    locale: str = "en-GB"
    region: str = "UK"
    english_variant: str = "UK"
    tone: str = "direct"
    reading_level: str = "grade_6_to_8"
    target_persona: str = "UK women with suspected/diagnosed endometriosis"
    pillars_allowed: list[str] = Field(default_factory=lambda: ["recognition", "validation", "access_or_decision"])
    pillars_priority: dict[str, float] = Field(
        default_factory=lambda: {"recognition": 1.0, "validation": 0.9, "access_or_decision": 1.0}
    )
    constraints: dict[str, Any] = Field(default_factory=dict)


class DocumentMeta(BaseModel):
    document_id: str
    title: str | None = None
    type: str = "txt"
    language: str = "en"
    ingested_at: datetime = Field(default_factory=datetime.utcnow)
    source_metadata: dict[str, Any] = Field(default_factory=dict)


class Shortlist(BaseModel):
    selected_idea_ids: list[str] = Field(default_factory=list)
    selection_frozen_at: datetime | None = None
    rubric_version: str = "v1"
    notes: str | None = None


class PipelineOutputs(BaseModel):
    final_selected_idea_id: str | None = None
    final_carousel: CarouselDraft | None = None


# Bump when RunState JSON shape changes incompatibly; loaders may migrate old files.
RUN_STATE_SCHEMA_VERSION = 3


class RunState(BaseModel):
    schema_version: int = RUN_STATE_SCHEMA_VERSION
    document: DocumentMeta
    audience: AudienceConfig = Field(default_factory=AudienceConfig)
    chunks: list[Chunk] = Field(default_factory=list)
    candidates: list[CandidateIdea] = Field(default_factory=list)
    shortlist: Shortlist = Field(default_factory=Shortlist)
    outputs: PipelineOutputs = Field(default_factory=PipelineOutputs)
    # Ideation: unified = same model for extract+score+governance; split = extract vs score (+ governance uses score model)
    ideation_mode: Literal["unified", "split"] = "unified"
    ideation_extract_model: str | None = None
    ideation_score_model: str | None = None
    # Optional persona card id (filename stem under docs/personas/*.json); Writer injection only.
    audience_preset: str | None = None
    experiment: ExperimentSpec | None = None
    # Human-in-the-loop (stop after scoring → review → continue with human_selection.json)
    awaiting_human_review: bool = False
    proposed_editorial_direction: str | None = None
    human_editorial_direction: str | None = None
    human_shortlist_curated: bool = False
    # UI / human reviewer freeform (hooks + slides)
    reviewer_brief_global: str | None = None
    reviewer_brief_by_idea: dict[str, str] = Field(default_factory=dict)
    writer_clarification_transcript: str | None = None


class PerformanceObserved(BaseModel):
    impressions: int | None = None
    likes: int | None = None
    comments: int | None = None
    shares: int | None = None
    saves: int | None = None
    profile_visits: int | None = None
    follows: int | None = None
    dms: int | None = None


class PerformanceDerived(BaseModel):
    save_rate: float | None = None
    share_rate: float | None = None
    profile_visit_rate: float | None = None


class PerformanceLog(BaseModel):
    """
    One row per published post, linked back to idea + hook tags.
    Store as JSONL for append-only simplicity.
    """

    model_config = ConfigDict(extra="ignore")

    post_id: str
    platform: Literal["instagram", "linkedin", "tiktok"] = "instagram"
    published_at: datetime = Field(default_factory=datetime.utcnow)

    # lineage
    run_id: str | None = None
    document_title: str | None = None
    idea_id: str
    pillar: str
    format_suggestion: str
    hook_id: str | None = None
    hook_style: str | None = None

    # A/B experiment (optional; pair rows share experiment_id + distinct variant)
    experiment_id: str | None = None
    variant: Literal["A", "B"] | None = None
    treatment_key: str | None = None
    base_idea_id: str | None = None

    # audience segment (optional; see docs/audience-and-personas-plan.md)
    persona_id: str | None = None
    persona_source: Literal["explicit", "rules", "unclassified"] | None = None

    # prediction snapshot (optional)
    predicted_total: float | None = None
    predicted_scores: dict[str, int] = Field(default_factory=dict)

    observed: PerformanceObserved = Field(default_factory=PerformanceObserved)
    derived: PerformanceDerived = Field(default_factory=PerformanceDerived)
    notes: str | None = None


@dataclass(frozen=True)
class Weights:
    """Weights for the five-dimension rubric (normalised in compute_weighted_totals)."""

    hook_strength: float = 1.35
    source_support: float = 1.35
    strategic_fit: float = 1.1
    save_share: float = 1.1
    clarity: float = 1.1

