"""
Human-in-the-loop: artifacts after ideation scoring, and applying `human_selection.json` to continue the pipeline.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from .schemas import RunState
from .validation import ValidationError

ALLOWED_IDEA_OVERRIDE_KEYS = frozenset(
    {
        "content_pillar",
        "topic",
        "angle",
        "core_claim",
        "audience_pain",
        "promise",
        "format_suggestion",
    }
)


class HumanSelectionFile(BaseModel):
    """Edit a copy of the template next to your RunState JSON, then pass it to `continue`."""

    selected_idea_ids: list[str] = Field(default_factory=list)
    editorial_direction: str | None = Field(default=None, description="Campaign direction for Writer (optional).")
    global_notes: str | None = Field(default=None, description="Extra reviewer notes for Writer context.")
    idea_notes: dict[str, str] = Field(default_factory=dict, description="Per idea_id: freeform notes for the Writer.")
    idea_overrides: dict[str, dict[str, str]] = Field(default_factory=dict)


def apply_human_selection_to_state(state: RunState, sel: HumanSelectionFile) -> None:
    allowed = {c.idea_id for c in state.candidates}
    picked = [str(iid).strip() for iid in sel.selected_idea_ids if str(iid).strip()]
    if not picked:
        raise ValidationError("human_selection.selected_idea_ids must include at least one idea_id.")
    for iid in picked:
        if iid not in allowed:
            raise ValidationError(
                f"Unknown idea_id in human selection: {iid!r}. Valid ids: {sorted(allowed)}"
            )

    by_id = {c.idea_id: c for c in state.candidates}
    for iid, patch in (sel.idea_overrides or {}).items():
        if iid not in by_id:
            raise ValidationError(f"idea_overrides references unknown idea_id: {iid!r}")
        if not isinstance(patch, dict):
            raise ValidationError(f"idea_overrides[{iid!r}] must be an object.")
        for k, v in patch.items():
            if k not in ALLOWED_IDEA_OVERRIDE_KEYS:
                raise ValidationError(
                    f"idea_overrides[{iid!r}] unsupported key {k!r}. "
                    f"Allowed: {sorted(ALLOWED_IDEA_OVERRIDE_KEYS)}"
                )
            setattr(by_id[iid], k, str(v).strip())

    picked_set = set(picked)
    for c in state.candidates:
        c.selected = c.idea_id in picked_set
        if c.selected:
            c.selection_reason = (c.selection_reason or "").strip() or "Selected via human review (human_selection.json)."

    state.shortlist.selected_idea_ids = list(picked)
    state.shortlist.selection_frozen_at = datetime.now(timezone.utc)
    state.shortlist.notes = "Human-curated shortlist from human_selection.json."

    if sel.editorial_direction is not None and str(sel.editorial_direction).strip():
        state.human_editorial_direction = str(sel.editorial_direction).strip()
    else:
        state.human_editorial_direction = None

    if sel.global_notes is not None and str(sel.global_notes).strip():
        state.reviewer_brief_global = str(sel.global_notes).strip()
    else:
        state.reviewer_brief_global = None

    state.reviewer_brief_by_idea = {
        k: str(v).strip() for k, v in (sel.idea_notes or {}).items() if str(v).strip()
    }

    state.human_shortlist_curated = True
    state.awaiting_human_review = False


def _trunc(s: str | None, n: int) -> str:
    t = (s or "").replace("\n", " ").strip()
    return t if len(t) <= n else t[: n - 1] + "…"


def build_human_review_markdown(state: RunState, *, review_top_n: int = 20) -> str:
    lines: list[str] = [
        "## Ideation review (human checkpoint)",
        "",
        f"- **run / document_id**: `{state.document.document_id}`",
        f"- **source title**: {state.document.title or '_(untitled)_'}",
        "",
    ]
    prop = (state.proposed_editorial_direction or "").strip()
    if prop:
        lines += [
            "### Proposed editorial direction (from model)",
            "",
            prop,
            "",
        ]
    else:
        lines += [
            "### Proposed editorial direction (from model)",
            "",
            "_(none returned — you can still set `editorial_direction` in the JSON file when continuing.)_",
            "",
        ]

    ranked = sorted(state.candidates, key=lambda c: (c.rank or 10_000))
    lines += [
        f"### Ranked ideas (showing top **{min(review_top_n, len(ranked))}** of {len(ranked)})",
        "",
        "| rank | idea_id | pillar | total | topic |",
        "|---:|---|---|---:|---|",
    ]
    for c in ranked[:review_top_n]:
        tw = c.scores.total_weighted if c.scores and c.scores.total_weighted is not None else ""
        lines.append(
            f"| {c.rank or ''} | `{c.idea_id}` | `{c.content_pillar}` | {tw} | {_trunc(c.topic, 70)} |"
        )
    lines.append("")
    lines += [
        "### Reader benefit (ideation)",
        "",
        "One line per idea — patient-facing payoff the model attached at extract (see RunState for full text).",
        "",
    ]
    for c in ranked[:review_top_n]:
        rb = getattr(c, "reader_benefit", None)
        if rb and str(rb).strip():
            lines.append(f"- `{c.idea_id}`: {_trunc(str(rb), 160)}")
    if all(not (str(getattr(c, "reader_benefit", None) or "").strip()) for c in ranked[:review_top_n]):
        lines.append("_(none in this run — older extracts or model omitted `reader_benefit`.)_")
    lines.append("")
    lines += [
        "### Next step",
        "",
        "1. Copy `*_human_selection.template.json` → e.g. `my_selection.json`.",
        "2. Set `selected_idea_ids` to the ideas you want hooks/slides for.",
        "3. Optionally set `editorial_direction` (overrides the model proposal for Writer hooks/slides).",
        "4. Optionally edit `idea_overrides` to tweak angle/topic/etc. per idea_id.",
        "5. Run: `python -m carousel_agents continue --from-state <this-run.json> --human-selection my_selection.json --out ...`",
        "",
    ]
    return "\n".join(lines).strip() + "\n"


def write_human_review_artifacts(
    state: RunState,
    out_path: Path,
    *,
    review_top_n: int = 20,
) -> tuple[Path, Path]:
    md_path = out_path.with_name(f"{out_path.stem}_human_review.md")
    json_path = out_path.with_name(f"{out_path.stem}_human_selection.template.json")
    md_path.write_text(build_human_review_markdown(state, review_top_n=review_top_n), encoding="utf-8")
    example = HumanSelectionFile(
        selected_idea_ids=["i001", "i002"],
        editorial_direction="Optional: your campaign direction for Writer (omit to keep the model's proposal).",
        global_notes="Optional: extra constraints (tone, taboos, format).",
        idea_notes={"i001": "Optional freeform note for this idea."},
        idea_overrides={"i001": {"angle": "Optional: tighten the angle for this idea only."}},
    )
    json_path.write_text(
        json.dumps(example.model_dump(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return md_path, json_path
