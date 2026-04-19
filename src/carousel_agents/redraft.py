from __future__ import annotations

from pathlib import Path
from typing import Any

from .llm import OpenAICompatibleClient, get_hook_judge_config, get_model_config
from .llm_response_models import HookPolicyJudgeResponse, HooksWriterResponse, QAEditResponse, SlidesWriterResponse
from .prompts import (
    system_editor,
    system_hook_judge,
    system_ideation,
    system_writer,
    user_draft_slides,
    user_ideation_carousel_review,
    user_qa_edit,
    user_rewrite_slides_from_ideation,
)
from .pipeline import _normalise_hook_style
from .schemas import JudgeCarouselReview, RunState, HookOption
from .hook_policy import find_policy_violations
from .context_pack import (
    build_writer_context_pack_for_hooks_and_slides,
    build_writer_source_evidence,
    sample_ocr_style_for_hook_policy,
)
from .persona_card import apply_persona_preset_to_writer_idea
from .prompts import user_generate_hooks, user_rewrite_hooks_from_policy


def redraft_selected_slides(
    *,
    state: RunState,
    max_slides: int = 10,
    do_qa: bool = True,
    judge_govern: bool = True,
    judge_rewrite_rounds: int = 1,
    hook_policy: bool = False,
    hook_rewrite_rounds: int = 2,
    performance_digest: dict[str, Any] | None = None,
) -> RunState:
    """
    Re-run only the Writer/Editor stages for already-selected ideas.
    This is used for iterative improvement without re-extracting/re-scoring/re-selecting.
    """
    client = OpenAICompatibleClient()
    writer = get_model_config("writer")
    editor = get_model_config("editor")
    ideation = get_model_config("ideation")
    hook_judge = get_hook_judge_config()
    ctx = build_writer_context_pack_for_hooks_and_slides()

    for c in state.candidates:
        if not c.selected:
            continue

        writer_src = build_writer_source_evidence(state.chunks, c)

        # Re-generate hooks with policy enforcement (optional) so slide drafting uses the improved best hook.
        hook_out = client.chat_structured(
            response_model=HooksWriterResponse,
            model=writer.model,
            temperature=writer.temperature,
            system=system_writer(),
            user=user_generate_hooks(
                idea=apply_persona_preset_to_writer_idea(
                    ctx.apply_to_idea_dict(c.model_dump()),
                    state.audience_preset,
                ),
                hook_count=12,
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
        raw_hooks: list[dict[str, Any]] = [h.model_dump() for h in hook_out.hooks]
        rewrite_out: HooksWriterResponse | None = None

        if hook_policy:
            rounds = max(0, int(hook_rewrite_rounds))
            ocr_snips = sample_ocr_style_for_hook_policy()

            def _violations_map(hooks_in: list[dict[str, Any]]) -> dict[str, list[str]]:
                v: dict[str, list[str]] = {}
                for idx, h in enumerate(hooks_in):
                    text = str((h or {}).get("text", ""))
                    matches = find_policy_violations(text)
                    if matches:
                        v[f"h{idx+1:03d}"] = [m.matched for m in matches]
                return v

            for _r in range(rounds + 1):
                viol = _violations_map(raw_hooks)
                if not viol:
                    break
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
                    raw_hooks = []
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
                        hook_count=12,
                        previous_hooks=raw_hooks,
                        violations=viol,
                        judge_directives=directives,
                        ocr_style_snippets=ocr_snips,
                        audience=state.audience.model_dump(),
                        performance_digest=performance_digest,
                        source_evidence=writer_src,
                    ),
                )
                raw_hooks = [h.model_dump() for h in rewrite_out.hooks]
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

        slides_out = client.chat_structured(
            response_model=SlidesWriterResponse,
            model=writer.model,
            temperature=writer.temperature,
            system=system_writer(),
            user=user_draft_slides(
                idea=apply_persona_preset_to_writer_idea(
                    ctx.apply_to_idea_dict(c.model_dump()),
                    state.audience_preset,
                ),
                max_slides=max_slides,
                audience=state.audience.model_dump(),
                performance_digest=performance_digest,
                source_evidence=writer_src,
            ),
        )
        c.carousel_draft = slides_out.carousel_draft

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

        if do_qa and c.carousel_draft:
            qa_src = build_writer_source_evidence(state.chunks, c)
            qa_out = client.chat_structured(
                response_model=QAEditResponse,
                model=editor.model,
                temperature=editor.temperature,
                system=system_editor(),
                user=user_qa_edit(
                    draft={"idea_id": c.idea_id, "carousel_draft": c.carousel_draft.model_dump()},
                    source_evidence=qa_src,
                ),
            )
            if qa_out.carousel_draft is not None:
                c.carousel_draft = qa_out.carousel_draft
            if qa_out.qa is not None:
                c.qa = qa_out.qa

    return state


def load_run_state(path: Path) -> RunState:
    return RunState.model_validate_json(path.read_text(encoding="utf-8"))


def save_run_state(*, state: RunState, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(state.model_dump_json(indent=2), encoding="utf-8")

