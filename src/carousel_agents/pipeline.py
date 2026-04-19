from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .chunking import ChunkingConfig, ChunkStrategyName, chunk_document
from .ingest import load_text
from .llm import (
    MockClient,
    OpenAICompatibleClient,
    get_hook_judge_config,
    get_ideation_extract_config,
    get_ideation_score_config,
    get_model_config,
)
from .llm_response_models import (
    CTAResponse,
    ExtractCandidatesResponse,
    HookPolicyJudgeResponse,
    HooksWriterResponse,
    QAEditResponse,
    ScoreCandidatesResponse,
    SlidesWriterResponse,
)
from .prompts import (
    system_editor,
    system_hook_judge,
    system_ideation,
    system_ideation_scoring,
    system_writer,
    user_draft_slides,
    user_extract_candidates,
    user_generate_hooks,
    user_generate_ctas,
    user_rewrite_hooks_from_policy,
    user_review_shortlist,
    user_ideation_carousel_review,
    user_rewrite_slides_from_ideation,
    user_qa_edit,
    user_score_candidates,
)
from .schemas import (
    CandidateIdea,
    CarouselDraft,
    CitationIn,
    DocumentMeta,
    ExperimentSpec,
    HookOption,
    JudgeCarouselReview,
    JudgeShortlistReview,
    QAResult,
    RUN_STATE_SCHEMA_VERSION,
    RunState,
    Weights,
)
from .hook_policy import find_policy_violations
from .context_pack import (
    WriterContextPack,
    build_writer_context_pack_for_cta_only,
    build_writer_context_pack_for_hooks_and_slides,
    build_writer_source_evidence,
    sample_ocr_style_for_hook_policy,
)
from .defaults import max_parallel_writer_workers
from .observability import configure_logging, log_stage, set_run_context
from .persona_card import apply_persona_preset_to_writer_idea
from .export_clean import export_selected_markdown
from .validation import (
    ValidationError,
    compute_weighted_totals_and_rank,
    enforce_selection_gate,
    repair_citation_chunk_ids,
    repair_citation_excerpts,
    validate_citations_verbatim,
)

def _normalise_hook_style(style: str) -> str:
    s = (style or "").strip().lower()
    s = s.replace("-", "_").replace(" ", "_")
    # pick the strongest signal if the model returns compound tags like "numbered/how_to"
    if "/" in s:
        s = s.split("/")[0].strip()

    mapping = {
        "myth_bust": "myth_bust",
        "mythbust": "myth_bust",
        "myth": "myth_bust",
        "myth_vs_truth": "myth_bust",
        "myth_vs_fact": "myth_bust",
        "mythbusting": "myth_bust",
        "numbered": "numbered",
        "list": "numbered",
        "steps": "how_to",
        "how_to": "how_to",
        "howto": "how_to",
        "identity": "identity",
        "validation": "validation",
        "contrarian": "contrarian",
        "checklist": "checklist",
        "what_to_do_next": "what_to_do_next",
        "quote_led": "quote_led",
        "quote": "quote_led",
        "statistic": "statistic",
        "stats": "statistic",
        "awareness": "statistic",
    }
    return mapping.get(s, s or "how_to")


def _reapply_frozen_shortlist(state: RunState) -> None:
    """Reset `selected` from `shortlist.selected_idea_ids` and clear veto flags for a clean hooks/CTA re-run."""
    want = set(state.shortlist.selected_idea_ids or [])
    for c in state.candidates:
        c.selected = c.idea_id in want
        if c.selected:
            c.judge_vetoed = False
            c.judge_veto_reason = None


def _clear_hooks_cta_for_selected(state: RunState) -> None:
    for c in state.candidates:
        if not c.selected:
            continue
        c.hooks = []
        c.best_hook_id = None
        c.outline = []
        c.cta_options = []
        c.best_cta = None
        c.hook_policy_flags = []
        c.hook_rewrite_rounds_used = 0


def _write_hooks_for_candidate(
    state: RunState,
    c: CandidateIdea,
    client: Any,
    *,
    writer: Any,
    hook_judge: Any,
    ctx: WriterContextPack,
    hook_count: int,
    judge_govern: bool,
    hook_policy: bool,
    hook_rewrite_rounds: int,
    performance_digest: dict[str, Any] | None,
) -> None:
    writer_src = build_writer_source_evidence(state.chunks, c)
    idea_for_prompt = apply_persona_preset_to_writer_idea(
        ctx.apply_to_idea_dict(c.model_dump()),
        state.audience_preset,
    )
    hook_out = client.chat_structured(
        response_model=HooksWriterResponse,
        model=writer.model,
        temperature=writer.temperature,
        system=system_writer(),
        user=user_generate_hooks(
            idea=idea_for_prompt,
            hook_count=hook_count,
            audience=state.audience.model_dump(),
            performance_digest=performance_digest,
            source_evidence=writer_src,
        ),
    )
    c.hooks = [
        HookOption(
            hook_id=h.hook_id or f"h{idx+1:03d}",
            text=h.text,
            style=_normalise_hook_style(h.style),
            rationale=h.rationale,
            score=h.score,
        )
        for idx, h in enumerate(hook_out.hooks)
    ]
    c.best_hook_id = hook_out.best_hook_id
    c.outline = list(hook_out.outline)

    if hook_policy and c.hooks:
        rounds = max(0, int(hook_rewrite_rounds))
        ocr_snips = sample_ocr_style_for_hook_policy()

        def _violations_map() -> dict[str, list[str]]:
            v: dict[str, list[str]] = {}
            for h in c.hooks:
                matches = find_policy_violations(h.text)
                if matches:
                    v[h.hook_id] = [m.matched for m in matches]
            return v

        for r in range(rounds + 1):
            viol = _violations_map()
            if not viol:
                break

            c.hook_policy_flags = sorted({m for ms in viol.values() for m in ms})
            c.hook_rewrite_rounds_used = r

            if not judge_govern:
                break

            judge_review = client.chat_structured(
                response_model=HookPolicyJudgeResponse,
                model=hook_judge.model,
                temperature=hook_judge.temperature,
                system=system_hook_judge(),
                user=(
                    "You are enforcing a hard hook language policy.\n"
                    "If hooks contain banned phrases, you must return status revise (with directives) or veto.\n\n"
                    "Return JSON shape:\n"
                    '{ "status": "revise|veto", "reason": "...", "directives": ["..."] }\n\n'
                    f"Idea:\n{apply_persona_preset_to_writer_idea(c.model_dump(), state.audience_preset)}\n\n"
                    f"Violations by hook_id:\n{viol}\n"
                ),
            )
            status = str(judge_review.status or "revise").strip().lower()
            directives = [str(d).strip() for d in (judge_review.directives or []) if str(d).strip()]

            if status == "veto":
                c.selected = False
                c.judge_vetoed = True
                c.judge_veto_reason = str(judge_review.reason or "vetoed_by_hook_policy")
                c.hooks = []
                c.best_hook_id = None
                c.outline = []
                break

            rewrite_out = client.chat_structured(
                response_model=HooksWriterResponse,
                model=writer.model,
                temperature=writer.temperature,
                system=system_writer(),
                user=user_rewrite_hooks_from_policy(
                    idea=apply_persona_preset_to_writer_idea(
                        ctx.apply_to_idea_dict(c.model_dump()),
                        state.audience_preset,
                    ),
                    hook_count=hook_count,
                    previous_hooks=[h.model_dump() for h in c.hooks],
                    violations=viol,
                    judge_directives=directives,
                    ocr_style_snippets=ocr_snips,
                    audience=state.audience.model_dump(),
                    performance_digest=performance_digest,
                    source_evidence=writer_src,
                ),
            )
            if not rewrite_out.hooks:
                break
            c.hooks = [
                HookOption(
                    hook_id=h.hook_id or f"h{idx+1:03d}",
                    text=h.text,
                    style=_normalise_hook_style(h.style),
                    rationale=h.rationale,
                    score=h.score,
                )
                for idx, h in enumerate(rewrite_out.hooks)
            ]
            c.best_hook_id = rewrite_out.best_hook_id or c.best_hook_id


def _write_cta_for_candidate(
    state: RunState,
    c: CandidateIdea,
    client: Any,
    *,
    writer: Any,
    ctx_cta: WriterContextPack,
    performance_digest: dict[str, Any] | None,
) -> None:
    writer_src = build_writer_source_evidence(state.chunks, c)
    idea_for_prompt = apply_persona_preset_to_writer_idea(
        ctx_cta.apply_to_idea_dict(c.model_dump()),
        state.audience_preset,
    )
    cta_out = client.chat_structured(
        response_model=CTAResponse,
        model=writer.model,
        temperature=writer.temperature,
        system=system_writer(),
        user=user_generate_ctas(
            idea=idea_for_prompt,
            cta_count=5,
            audience=state.audience.model_dump(),
            performance_digest=performance_digest,
            source_evidence=writer_src,
        ),
    )
    c.cta_options = [str(x).strip() for x in (cta_out.cta_options or []) if str(x).strip()][:8]
    if cta_out.best_cta and str(cta_out.best_cta).strip():
        c.best_cta = str(cta_out.best_cta).strip()


def _stage_hooks_and_cta(
    state: RunState,
    client: Any,
    *,
    do_hooks: bool,
    do_cta: bool,
    hook_count: int,
    judge_govern: bool,
    hook_policy: bool,
    hook_rewrite_rounds: int,
    performance_digest: dict[str, Any] | None = None,
    mock: bool = False,
) -> None:
    selected = [c for c in state.candidates if c.selected]
    if not selected:
        return

    max_w = max_parallel_writer_workers()
    parallel = max_w > 1 and len(selected) > 1 and not mock

    def process_candidate(cand: CandidateIdea, cl: Any) -> None:
        writer = get_model_config("writer")
        if do_hooks:
            _write_hooks_for_candidate(
                state,
                cand,
                cl,
                writer=writer,
                hook_judge=get_hook_judge_config(),
                ctx=build_writer_context_pack_for_hooks_and_slides(),
                hook_count=hook_count,
                judge_govern=judge_govern,
                hook_policy=hook_policy,
                hook_rewrite_rounds=hook_rewrite_rounds,
                performance_digest=performance_digest,
            )
        if do_cta and cand.selected:
            _write_cta_for_candidate(
                state,
                cand,
                cl,
                writer=writer,
                ctx_cta=build_writer_context_pack_for_cta_only(),
                performance_digest=performance_digest,
            )

    if parallel:
        log_stage("hooks_cta_parallel", max_workers=min(max_w, len(selected)), n=len(selected))

        def worker(cand: CandidateIdea) -> None:
            cl = MockClient() if mock else OpenAICompatibleClient()
            process_candidate(cand, cl)

        with ThreadPoolExecutor(max_workers=min(max_w, len(selected))) as ex:
            list(ex.map(worker, selected))
        return

    writer = get_model_config("writer")
    hook_judge = get_hook_judge_config()
    ctx = build_writer_context_pack_for_hooks_and_slides()
    ctx_cta = build_writer_context_pack_for_cta_only()
    for c in selected:
        if do_hooks:
            _write_hooks_for_candidate(
                state,
                c,
                client,
                writer=writer,
                hook_judge=hook_judge,
                ctx=ctx,
                hook_count=hook_count,
                judge_govern=judge_govern,
                hook_policy=hook_policy,
                hook_rewrite_rounds=hook_rewrite_rounds,
                performance_digest=performance_digest,
            )
        if do_cta and c.selected:
            _write_cta_for_candidate(
                state,
                c,
                client,
                writer=writer,
                ctx_cta=ctx_cta,
                performance_digest=performance_digest,
            )


def run_hooks_cta_from_state(
    *,
    state_path: Path,
    out_path: Path,
    export_dir: Path | None,
    export_document_id_suffix: str,
    hook_count: int = 12,
    judge_govern: bool = True,
    hook_policy: bool = False,
    hook_rewrite_rounds: int = 2,
    mock: bool = False,
    performance_digest: dict[str, Any] | None = None,
) -> RunState:
    """
    Re-run hooks + CTA only from a saved RunState, keeping the frozen shortlist.
    Uses MODEL_WRITER for copy; hook policy uses MODEL_HOOK_JUDGE (default mistral-large if unset).
    """
    client = MockClient() if mock else OpenAICompatibleClient()
    state = RunState.model_validate_json(state_path.read_text(encoding="utf-8"))
    if state.experiment and any(c.ab_variant for c in state.candidates):
        raise ValidationError(
            "This RunState is already A/B split. run-hooks-cta would regenerate hooks independently per arm "
            "and break the experiment pairing. Use a RunState saved before the split, or omit experiment from JSON."
        )
    _reapply_frozen_shortlist(state)
    _clear_hooks_cta_for_selected(state)
    base_id = state.document.document_id
    state.document.document_id = f"{base_id}__{export_document_id_suffix}"
    configure_logging()
    set_run_context(run_id=state.document.document_id, document_id=state.document.document_id)
    _stage_hooks_and_cta(
        state,
        client,
        do_hooks=True,
        do_cta=True,
        hook_count=hook_count,
        judge_govern=judge_govern,
        hook_policy=hook_policy,
        hook_rewrite_rounds=hook_rewrite_rounds,
        performance_digest=performance_digest,
        mock=mock,
    )
    _apply_experiment_ab_split(state)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(state.model_dump_json(indent=2), encoding="utf-8")
    if export_dir is not None:
        export_selected_markdown(state=state, export_root=export_dir)
    return state


def run_pipeline(
    *,
    input_path: Path,
    out_path: Path,
    candidate_count: int = 30,
    select_n: int = 5,
    select_pool_k: int = 10,
    performance_digest: dict[str, Any] | None = None,
    hook_count: int = 12,
    max_slides: int = 10,
    do_hooks: bool = True,
    do_cta: bool = True,
    do_slides: bool = True,
    do_qa: bool = False,
    judge_govern: bool = True,
    judge_rewrite_rounds: int = 1,
    hook_policy: bool = False,
    hook_rewrite_rounds: int = 2,
    mock: bool = False,
    export_dir: Path | None = None,
    split_ideation: bool = False,
    document_id: str | None = None,
    chunk_strategy: ChunkStrategyName = "auto",
    chunk_max_chars: int | None = None,
    chunk_overlap: int | None = None,
    audience_preset: str | None = None,
    require_audience_preset: bool = False,
    experiment: ExperimentSpec | None = None,
    orchestrate_program_path: Path | None = None,
) -> RunState:
    configure_logging()
    client = MockClient() if mock else OpenAICompatibleClient()

    text, detected_type = load_text(input_path)
    chunks = chunk_document(
        text,
        config=ChunkingConfig(
            strategy=chunk_strategy,
            max_chars=chunk_max_chars,
            overlap_chars=chunk_overlap,
        ),
        ingest_type=detected_type,
        source_path=input_path,
    )

    ideation_extract = get_ideation_extract_config(split_ideation=split_ideation)
    ideation_score = get_ideation_score_config(split_ideation=split_ideation)
    ideation = ideation_score

    doc_id = document_id or f"doc_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    preset_s = str(audience_preset).strip() if audience_preset else ""
    if require_audience_preset and not preset_s:
        raise ValidationError("require_audience_preset is set but --audience-preset was not provided.")

    set_run_context(run_id=doc_id, document_id=doc_id)
    log_stage("pipeline_start", input=str(input_path), mock=mock, split_ideation=split_ideation)

    state = RunState(
        schema_version=RUN_STATE_SCHEMA_VERSION,
        document=DocumentMeta(
            document_id=doc_id,
            title=input_path.stem,
            type=detected_type,
            language="en",
        ),
        chunks=chunks,
        ideation_mode="split" if split_ideation else "unified",
        ideation_extract_model=ideation_extract.model,
        ideation_score_model=ideation_score.model,
        audience_preset=preset_s if preset_s else None,
        experiment=experiment,
    )

    # ---- Stage: Extract candidates (Ideation extract pass)
    log_stage("extract_candidates", model=ideation_extract.model)
    extract_out = client.chat_structured(
        response_model=ExtractCandidatesResponse,
        model=ideation_extract.model,
        temperature=ideation_extract.temperature,
        system=system_ideation(),
        user=user_extract_candidates(
            chunks=[c.model_dump() for c in state.chunks],
            candidate_count=candidate_count,
            audience=state.audience.model_dump(),
            performance_digest=performance_digest,
        ),
    )
    candidates_raw = [item.model_dump() for item in extract_out.candidates]
    state.candidates = [CandidateIdea(**_coerce_candidate(_ensure_idea_id(i, idx))) for idx, i in enumerate(candidates_raw)]

    # Some models occasionally attach a correct excerpt to the wrong chunk_id (often due to chunk boundaries).
    # Repair by searching the document and re-pointing the citation to the chunk that contains the excerpt.
    repair_citation_chunk_ids(state)
    # If excerpts were paraphrased, replace with best-match source substring (fallback).
    repair_citation_excerpts(state, min_ratio=0.75)
    validate_citations_verbatim(state, strict=False)

    # ---- Stage: Score candidates (Ideation score pass; governance below uses same score model)
    log_stage("score_candidates", model=ideation_score.model)
    weights = Weights()
    score_system = system_ideation_scoring() if split_ideation else system_ideation()
    score_out = client.chat_structured(
        response_model=ScoreCandidatesResponse,
        model=ideation_score.model,
        temperature=ideation_score.temperature,
        system=score_system,
        user=user_score_candidates(
            candidates=[c.model_dump() for c in state.candidates],
            weights=weights.__dict__,
            audience=state.audience.model_dump(),
            performance_digest=performance_digest,
        ),
    )
    by_id: dict[str, dict[str, Any]] = {row.idea_id: row.model_dump() for row in score_out.scores if row.idea_id}
    for c in state.candidates:
        s = by_id.get(c.idea_id)
        if not s:
            continue
        for k, v in s.items():
            if k == "idea_id":
                continue
            if v is None:
                continue
            if hasattr(c.scores, k):
                setattr(c.scores, k, int(v))

    compute_weighted_totals_and_rank(state, weights)

    # ---- Stage: Select shortlist (deterministic — no LLM)
    # Take the first `select_n` ideas among the top `select_pool_k` by weighted rank (candidates are already sorted).
    pool_k = max(1, min(int(select_pool_k), len(state.candidates)))
    selection_pool = state.candidates[:pool_k]
    n_pick = min(select_n, len(selection_pool))
    picked = selection_pool[:n_pick]
    selected_ids = [c.idea_id for c in picked]
    state.shortlist.selected_idea_ids = selected_ids
    split_note = (
        f" Ideation: split — extract={ideation_extract.model}, score={ideation_score.model}."
        if split_ideation
        else ""
    )
    state.shortlist.notes = (
        f"Deterministic shortlist: top {n_pick} by rank among positions 1–{pool_k} "
        f"(rank = weighted total, then source_support, hook_strength, content_pillar, idea_id) "
        f"(no selection LLM).{split_note}"
    )
    state.shortlist.selection_frozen_at = datetime.now(timezone.utc)

    for c in state.candidates:
        c.selected = False
    for c in picked:
        c.selected = True
        c.selection_reason = (c.selection_reason or "").strip() or "Selected by weighted rank (deterministic)."

    # ---- Stage: Ideation governance over shortlist (optional)
    if judge_govern:
        selected = [c for c in state.candidates if c.selected]
        review = client.chat_structured(
            response_model=JudgeShortlistReview,
            model=ideation.model,
            temperature=0.0,
            system=system_ideation(),
            user=user_review_shortlist(
                selected=[c.model_dump() for c in selected],
                audience=state.audience.model_dump(),
                performance_digest=performance_digest,
            ),
        )
        decisions_by_id = {r.idea_id: r.decision for r in review.reviewed}

        # Apply vetoes to current selection
        for c in state.candidates:
            if not c.selected:
                continue
            d = decisions_by_id.get(c.idea_id)
            if d and d.status == "veto":
                c.selected = False
                c.judge_vetoed = True
                c.judge_veto_reason = d.reason or "vetoed_by_judge"

        # Backfill from next-ranked ideas to maintain select_n if possible
        selected_now = [c for c in state.candidates if c.selected]
        if len(selected_now) < select_n:
            need = select_n - len(selected_now)
            ranked_remaining = sorted([c for c in state.candidates if not c.selected and not c.judge_vetoed], key=lambda x: x.rank or 10_000)
            for c in ranked_remaining[:need]:
                c.selected = True
            selected_ids = [c.idea_id for c in state.candidates if c.selected]
            state.shortlist.selected_idea_ids = selected_ids

    if orchestrate_program_path is not None and state.experiment is None:
        from .experiment_orchestrator import try_apply_orchestrator

        try_apply_orchestrator(state, orchestrate_program_path)

    if state.experiment:
        _validate_experiment_idea_in_shortlist(state)

    enforce_selection_gate(state)

    if state.experiment and not do_hooks:
        raise ValidationError("RunState has experiment set but do_hooks is false; A/B split requires hooks.")

    _stage_hooks_and_cta(
        state,
        client,
        do_hooks=do_hooks,
        do_cta=do_cta,
        hook_count=hook_count,
        judge_govern=judge_govern,
        hook_policy=hook_policy,
        hook_rewrite_rounds=hook_rewrite_rounds,
        performance_digest=performance_digest,
        mock=mock,
    )

    _apply_experiment_ab_split(state)

    # ---- Stage: Slides (Writer) - selected only
    if do_slides:
        log_stage("draft_slides")
        writer = get_model_config("writer")
        ctx_slides = build_writer_context_pack_for_hooks_and_slides()
        for c in state.candidates:
            if not c.selected:
                continue
            writer_src = build_writer_source_evidence(state.chunks, c)
            idea_for_prompt = apply_persona_preset_to_writer_idea(
                ctx_slides.apply_to_idea_dict(c.model_dump()),
                state.audience_preset,
            )
            slides_out = client.chat_structured(
                response_model=SlidesWriterResponse,
                model=writer.model,
                temperature=writer.temperature,
                system=system_writer(),
                user=user_draft_slides(
                    idea=idea_for_prompt,
                    max_slides=max_slides,
                    audience=state.audience.model_dump(),
                    performance_digest=performance_digest,
                    source_evidence=writer_src,
                ),
            )
            c.carousel_draft = slides_out.carousel_draft

            # Ideation critique -> rewrite loop (optional governance)
            if judge_govern and c.carousel_draft:
                rounds = max(0, int(judge_rewrite_rounds))
                for _ in range(rounds + 1):
                    c.judge_carousel_review = client.chat_structured(
                        response_model=JudgeCarouselReview,
                        model=ideation.model,
                        temperature=0.0,
                        system=system_ideation(),
                        user=user_ideation_carousel_review(
                            idea=apply_persona_preset_to_writer_idea(dict(c.model_dump()), state.audience_preset),
                            carousel_draft=c.carousel_draft.model_dump(),
                            audience=state.audience.model_dump(),
                            source_evidence=writer_src,
                        ),
                    )

                    if not c.judge_carousel_review:
                        break

                    status = c.judge_carousel_review.decision.status
                    if status == "approve":
                        break
                    if status == "veto":
                        c.selected = False
                        c.judge_vetoed = True
                        c.judge_veto_reason = c.judge_carousel_review.decision.reason or "vetoed_by_judge_carousel"
                        c.carousel_draft = None
                        break
                    # revise: rewrite once, then re-review (loop continues)
                    rewrite_slides = client.chat_structured(
                        response_model=SlidesWriterResponse,
                        model=writer.model,
                        temperature=writer.temperature,
                        system=system_writer(),
                        user=user_rewrite_slides_from_ideation(
                            idea=apply_persona_preset_to_writer_idea(dict(c.model_dump()), state.audience_preset),
                            carousel_draft=c.carousel_draft.model_dump(),
                            ideation_review=c.judge_carousel_review.model_dump(),
                            max_slides=max_slides,
                            audience=state.audience.model_dump(),
                            performance_digest=performance_digest,
                            source_evidence=writer_src,
                        ),
                    )
                    c.carousel_draft = rewrite_slides.carousel_draft

    # ---- Stage: QA (Editor) - selected only
    if do_qa:
        log_stage("qa_edit")
        editor = get_model_config("editor")
        for c in state.candidates:
            if not c.selected or not c.carousel_draft:
                continue
            qa_src = build_writer_source_evidence(state.chunks, c)
            qa_out = client.chat_structured(
                response_model=QAEditResponse,
                model=editor.model,
                temperature=editor.temperature,
                system=system_editor(),
                user=user_qa_edit(
                    draft={"idea_id": c.idea_id, "carousel_draft": c.carousel_draft},
                    source_evidence=qa_src,
                ),
            )
            if qa_out.carousel_draft is not None:
                c.carousel_draft = qa_out.carousel_draft
            if qa_out.qa is not None:
                c.qa = qa_out.qa

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(state.model_dump_json(indent=2), encoding="utf-8")

    if export_dir is not None:
        export_selected_markdown(state=state, export_root=export_dir)
    return state


def _validate_experiment_idea_in_shortlist(state: RunState) -> None:
    exp = state.experiment
    assert exp is not None
    if exp.idea_id not in (state.shortlist.selected_idea_ids or []):
        raise ValidationError(
            f"experiment.idea_id {exp.idea_id!r} is not in the frozen shortlist "
            f"(selected: {state.shortlist.selected_idea_ids}). "
            "Use an idea_id that is actually selected (check rank / pool size)."
        )


def _pick_two_distinct_hooks(hooks: list[HookOption]) -> tuple[HookOption, HookOption]:
    if len(hooks) < 2:
        raise ValidationError("A/B experiment requires at least two hook options on the target idea.")
    by_style: dict[str, HookOption] = {}
    for h in hooks:
        st = (h.style or "").strip().lower()
        if st not in by_style:
            by_style[st] = h
        if len(by_style) >= 2:
            a, b = list(by_style.values())[:2]
            return a, b
    return hooks[0], hooks[1]


def _apply_experiment_ab_split(state: RunState) -> None:
    """
    After hooks+CTA, replace the single experiment target candidate with two selected arms
    (different hooks when possible). Idempotent if split already applied.
    """
    exp = state.experiment
    if not exp:
        return
    base = exp.idea_id
    if any(c.base_idea_id == base and c.ab_variant is not None for c in state.candidates):
        return
    idx = next((i for i, c in enumerate(state.candidates) if c.idea_id == base), None)
    if idx is None:
        raise ValidationError(
            f"experiment.idea_id {base!r} not found among candidates after hooks "
            "(split may already be inconsistent, or idea_id does not match)."
        )
    base_cand = state.candidates[idx]
    if not base_cand.hooks:
        raise ValidationError(f"Cannot split experiment for {base}: no hooks generated.")
    ha, hb = _pick_two_distinct_hooks(base_cand.hooks)
    ca = base_cand.model_copy(deep=True)
    cb = base_cand.model_copy(deep=True)
    ca.idea_id = f"{base}__expA"
    cb.idea_id = f"{base}__expB"
    ca.hooks = [ha]
    cb.hooks = [hb]
    ca.best_hook_id = ha.hook_id
    cb.best_hook_id = hb.hook_id
    ca.experiment_id = exp.experiment_id
    cb.experiment_id = exp.experiment_id
    ca.ab_variant = "A"
    cb.ab_variant = "B"
    ca.base_idea_id = base
    cb.base_idea_id = base
    ca.outline = []
    cb.outline = []
    ca.carousel_draft = None
    cb.carousel_draft = None
    ca.qa = None
    cb.qa = None
    ca.judge_carousel_review = None
    cb.judge_carousel_review = None
    ca.judge_vetoed = False
    cb.judge_vetoed = False
    ca.judge_veto_reason = None
    cb.judge_veto_reason = None
    state.candidates.pop(idx)
    state.candidates.insert(idx, ca)
    state.candidates.insert(idx + 1, cb)
    new_ids: list[str] = []
    for iid in state.shortlist.selected_idea_ids or []:
        if iid == base:
            new_ids.extend([ca.idea_id, cb.idea_id])
        else:
            new_ids.append(iid)
    state.shortlist.selected_idea_ids = new_ids
    note = (
        f" A/B split: experiment {exp.experiment_id!r} — arms {ca.idea_id!r} vs {cb.idea_id!r} "
        f"(hooks {ha.style!r} vs {hb.style!r})."
    )
    state.shortlist.notes = (state.shortlist.notes or "").strip() + note
    enforce_selection_gate(state)


def _ensure_idea_id(d: dict[str, Any], idx: int) -> dict[str, Any]:
    if "idea_id" not in d or not d["idea_id"]:
        d = dict(d)
        d["idea_id"] = f"i{idx+1:03d}"
    return d


def _coerce_candidate(d: dict[str, Any]) -> dict[str, Any]:
    """
    Normalise common model output quirks to our strict schema.
    - Map source_citations[*].verbatim -> excerpt
    - Ensure required keys exist
    """
    out = dict(d)
    citations = out.get("source_citations") or []
    fixed: list[dict[str, Any]] = []
    for c in citations:
        if not isinstance(c, dict):
            continue
        ci = CitationIn(**c).to_citation()
        fixed.append(ci.model_dump())
    out["source_citations"] = fixed
    if out.get("safety_flags") is None:
        out["safety_flags"] = []
    elif not isinstance(out["safety_flags"], list):
        out["safety_flags"] = [str(out["safety_flags"])]
    return out

