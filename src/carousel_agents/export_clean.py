from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Literal

from .bundle_to_csv import write_export_csvs_from_state
from .schemas import CandidateIdea, DraftSlide, RunState


def export_selected_markdown(*, state: RunState, export_root: Path) -> Path:
    """
    Writes a clean, readable Markdown export of selected ideas.
    Returns the run export directory path (outputs/<run_id>/).
    """
    run_id = state.document.document_id
    run_dir = export_root / run_id
    selected_dir = run_dir / "selected"
    selected_dir.mkdir(parents=True, exist_ok=True)

    generated_at = datetime.utcnow().isoformat() + "Z"

    selected = [c for c in state.candidates if c.selected]
    selected.sort(key=_selected_export_sort_key)

    # Index
    idx_lines = [
        "## Selected ideas (clean export)",
        "",
        f"- **run_id**: `{run_id}`",
        f"- **document**: `{state.document.title or ''}`",
        f"- **generated_at**: `{generated_at}`",
        "",
    ]
    if state.experiment:
        exp = state.experiment
        idx_lines += [
            "### A/B experiment",
            "",
            f"- **experiment_id**: `{exp.experiment_id}`",
            f"- **treatment**: `{exp.treatment_key or '(unset)'}`",
            f"- **hypothesis**: {exp.hypothesis or '_(none)_'}",
            f"- **primary_metric**: `{exp.primary_metric}`",
            "",
            "See `schedule.md` for suggested weekly order and spacing. Machine-readable: `experiment.json`.",
            "",
        ]
    idx_lines += [
        "### Shortlist",
        "",
    ]
    for c in selected:
        fname = _selected_file_basename(c) + ".md"
        arm = ""
        if c.ab_variant:
            arm = f" **arm {c.ab_variant}**"
        idx_lines.append(
            f"- **#{c.rank}** `{c.idea_id}` ({c.content_pillar}){arm}: [{c.topic}](selected/{fname})"
        )
    (run_dir / "index.md").write_text("\n".join(idx_lines) + "\n", encoding="utf-8")

    # Per-idea files
    for c in selected:
        fname = _selected_file_basename(c) + ".md"
        (selected_dir / fname).write_text(_idea_markdown(c), encoding="utf-8")

    # Bundle file (single readable file with topic/hook/slides only)
    (run_dir / "bundle.md").write_text(_bundle_markdown(state, generated_at=generated_at), encoding="utf-8")

    if state.experiment:
        _write_experiment_json(run_dir=run_dir, state=state, selected=selected)
        _write_schedule_md(run_dir=run_dir, state=state, selected=selected)

    write_export_csvs_from_state(state=state, run_dir=run_dir, generated_at=generated_at)

    return run_dir


def _norm_ws_export(s: str) -> str:
    return " ".join(s.split()).strip()


def _skip_redundant_disclaimer_slide(slide: DraftSlide, disclaimer: str) -> bool:
    """
    Avoid exporting a disclaimer-only slide when the same text is already in carousel_draft.disclaimer.
    """
    disc = disclaimer.strip()
    if not disc:
        return False
    main = (slide.main_text or "").strip().lower()
    if main in ("disclaimer", "disclaimer:", "legal", "legal notice"):
        return True
    body = _norm_ws_export((slide.main_text or "") + " " + (slide.subtext or "")).lower()
    nd = _norm_ws_export(disc).lower()
    if len(nd) >= 32 and nd in body:
        return True
    if len(body) >= 32 and body in nd:
        return True
    return False


def _selected_export_sort_key(c: CandidateIdea) -> tuple:
    r = c.rank or 10_000
    arm = 2
    if c.ab_variant == "A":
        arm = 0
    elif c.ab_variant == "B":
        arm = 1
    return (r, arm, c.idea_id)


def _selected_file_basename(c: CandidateIdea) -> str:
    return _safe_filename(f"{c.rank:02d}_{c.idea_id}_{c.content_pillar}_{c.topic}")


def _schedule_arm_order(state: RunState) -> Literal["A_first", "B_first"] | None:
    exp = state.experiment
    if not exp:
        return None
    if exp.schedule_arm_order:
        return exp.schedule_arm_order
    rng = random.Random(state.document.document_id)
    return "A_first" if rng.random() < 0.5 else "B_first"


def _write_experiment_json(*, run_dir: Path, state: RunState, selected: list[CandidateIdea]) -> None:
    exp = state.experiment
    assert exp is not None
    arms: dict[str, str] = {}
    for c in selected:
        if c.ab_variant and c.idea_id:
            arms[c.ab_variant] = c.idea_id
    tk = exp.treatment_key or ""
    log_hint = (
        "python -m carousel_agents log-performance --post-id <Asset_ID> "
        f"--experiment-id {exp.experiment_id!r} --variant A|B --idea-id <arm idea_id>"
    )
    if tk.strip():
        log_hint += f" --treatment-key {tk!r}"
    log_hint += " ..."
    payload = {
        "spec": exp.model_dump(mode="json"),
        "run_id": state.document.document_id,
        "resolved_arms": arms,
        "schedule_arm_order": _schedule_arm_order(state),
        "log_performance_hint": log_hint,
    }
    (run_dir / "experiment.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_schedule_md(*, run_dir: Path, state: RunState, selected: list[CandidateIdea]) -> None:
    exp = state.experiment
    assert exp is not None
    order = _schedule_arm_order(state)
    arms = [c for c in selected if c.ab_variant]
    buffers = [c for c in selected if not c.ab_variant]
    lines: list[str] = [
        "## Suggested weekly schedule",
        "",
        f"- **experiment_id**: `{exp.experiment_id}`",
        f"- **treatment_key**: `{exp.treatment_key}`",
        f"- **randomised publish order** (arms only): **{order}** — use this so the test is not always A-then-B.",
        "",
        "### Spacing rules (operational)",
        "",
        "- Leave **at least 2–3 days** between arm A and arm B, or split **early week vs late week**.",
        "- Schedule **at least one non-experiment post** between the two arms when possible (see buffer ideas below).",
        "- Covers for A and B should **not** read as duplicates in the feed.",
        "",
        "### Experiment arms (publish these two; order as above)",
        "",
    ]
    if order == "A_first":
        lines.append("Sequence: **arm A → (buffers) → arm B** within the week when possible.")
    else:
        lines.append("Sequence: **arm B → (buffers) → arm A** within the week when possible.")
    lines.append("")
    by_v = {c.ab_variant: c for c in arms if c.ab_variant}
    for label in ("A", "B"):
        c = by_v.get(label)  # type: ignore[arg-type]
        if not c:
            continue
        st = c.hooks[0].style if c.hooks else "?"
        lines.append(f"- **Arm {label}** — `{c.idea_id}` — {c.topic} — hook style: `{st}`")
    lines += [
        "",
        "### Buffer ideas (same export — use between arms)",
        "",
    ]
    if not buffers:
        lines.append("_No other selected ideas in this run; pull a buffer from backlog or another export._")
    else:
        for c in sorted(buffers, key=_selected_export_sort_key):
            lines.append(f"- `{c.idea_id}` — {c.topic} ({c.content_pillar})")
    lines.append("")
    (run_dir / "schedule.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _idea_markdown(c: CandidateIdea) -> str:
    lines: list[str] = []
    lines += [
        f"## #{c.rank} — {c.topic}",
        "",
        f"- **idea_id**: `{c.idea_id}`",
    ]
    if c.experiment_id and c.ab_variant:
        lines += [
            f"- **experiment_id**: `{c.experiment_id}`",
            f"- **A/B arm**: `{c.ab_variant}`",
            f"- **base_idea_id** (pairing): `{c.base_idea_id}`",
        ]
    lines += [
        f"- **pillar**: `{c.content_pillar}`",
        f"- **format**: `{c.format_suggestion}`",
        "",
        "### Angle",
        c.angle.strip(),
        "",
        "### Core claim",
        c.core_claim.strip(),
        "",
        "### Audience pain",
        c.audience_pain.strip(),
        "",
        "### Promise",
        c.promise.strip(),
        "",
        "### Selection reason",
        (c.selection_reason or "").strip() or "_(none)_",
        "",
        "### Scores",
        "",
        "| dimension | score |",
        "|---|---:|",
        f"| hook_strength | {c.scores.hook_strength} |",
        f"| clarity | {c.scores.clarity} |",
        f"| save_share | {c.scores.save_share} |",
        f"| strategic_fit | {c.scores.strategic_fit} |",
        f"| source_support | {c.scores.source_support} |",
        f"| total_weighted | {c.scores.total_weighted} |",
        "",
    ]

    if c.safety_flags:
        lines += ["### Flags", ""] + [f"- `{f}`" for f in c.safety_flags] + [""]

    lines += ["### Citations (verbatim excerpts)", ""]
    for cit in c.source_citations:
        lines += [
            f"- **{cit.chunk_id}**: {cit.note}",
            "",
            "> " + cit.excerpt.strip().replace("\n", "\n> "),
            "",
        ]

    if c.hooks:
        lines += ["### Hooks", ""]
        for h in c.hooks:
            best = " **(best)**" if c.best_hook_id and h.hook_id == c.best_hook_id else ""
            lines += [f"- `{h.hook_id}` ({h.style}){best}: {h.text}", f"  - rationale: {h.rationale or ''}".rstrip()]
        lines += [""]

    if c.carousel_draft and c.carousel_draft.slides:
        lines += ["### Carousel draft", ""]
        disc_d = (c.carousel_draft.disclaimer or "").strip()
        slides_md = [s for s in c.carousel_draft.slides if not _skip_redundant_disclaimer_slide(s, disc_d)]
        for s in slides_md:
            cit = f" (cites {', '.join(s.citations)})" if s.citations else ""
            lines += [
                f"#### Slide {s.slide_number}{cit}",
                "",
                s.main_text.strip(),
                "",
                s.subtext.strip(),
                "",
            ]
        lines += [f"**CTA**: {c.carousel_draft.cta}", ""]
        cap = (getattr(c, "caption", None) or "").strip()
        if cap:
            lines += ["**Caption**", "", cap, ""]
        if c.carousel_draft.disclaimer:
            lines += [f"**Disclaimer**: {c.carousel_draft.disclaimer}", ""]

    if c.qa:
        lines += ["### QA", ""]
        lines += [f"- **passed**: `{c.qa.passed}`"]
        for issue in c.qa.issues:
            where = f" (slide {issue.slide})" if issue.slide is not None else ""
            lines += [f"- `{issue.type}`{where}: {issue.detail}"]
        lines += [""]

    return "\n".join(lines).strip() + "\n"


def _bundle_markdown(state: RunState, *, generated_at: str) -> str:
    run_id = state.document.document_id
    selected = [c for c in state.candidates if c.selected]
    selected.sort(key=_selected_export_sort_key)

    lines: list[str] = [
        "## Carousel bundle (clean)",
        "",
        f"- **run_id**: `{run_id}`",
        f"- **document**: `{state.document.title or ''}`",
        f"- **generated_at**: `{generated_at}`",
        "",
    ]
    if state.experiment:
        lines += [
            f"- **experiment_id**: `{state.experiment.experiment_id}`",
            f"- **schedule**: see `schedule.md`",
            "",
        ]

    for c in selected:
        lines += _bundle_one(c)

    return "\n".join(lines).strip() + "\n"


def _bundle_one(c: CandidateIdea) -> list[str]:
    best_hook_text = _pick_best_hook_text(c)
    lines: list[str] = [
        f"### #{c.rank} — {c.topic}",
        "",
        f"- **pillar**: `{c.content_pillar}`",
        f"- **format**: `{c.format_suggestion}`",
        "",
    ]
    if c.ab_variant:
        lines += [
            f"- **A/B arm**: `{c.ab_variant}` — `experiment_id`: `{c.experiment_id}` — `base_idea_id`: `{c.base_idea_id}`",
            "",
        ]
    lines += [
        "#### Hook (best)",
        "",
        best_hook_text or "_(no hook generated)_",
        "",
    ]

    if not c.carousel_draft or not c.carousel_draft.slides:
        # Hooks + CTA-only mode
        best_cta = (c.best_cta or "").strip()
        if best_cta:
            lines += ["**CTA**", "", best_cta, "", "---", ""]
        else:
            lines += ["**CTA**", "", "_(no CTA generated)_", "", "---", ""]
        cap = (getattr(c, "caption", None) or "").strip()
        if cap:
            lines += ["**Caption**", "", cap, "", "---", ""]
        return lines

    lines += ["#### Slides", ""]

    disclaimer = (c.carousel_draft.disclaimer or "").strip()
    slides_out = [s for s in c.carousel_draft.slides if not _skip_redundant_disclaimer_slide(s, disclaimer)]
    for i, s in enumerate(slides_out, start=1):
        main = (s.main_text or "").strip()
        sub = (s.subtext or "").strip()
        lines += [
            f"**Slide {i}**",
            "",
            main or "_(empty)_",
            "",
        ]
        if sub:
            lines += [sub, ""]
        lines += ["---", ""]

    cta = (c.carousel_draft.cta or "").strip()
    if cta:
        lines += ["**CTA**", "", cta, "", "---", ""]

    cap = (getattr(c, "caption", None) or "").strip()
    if cap:
        lines += ["**Caption**", "", cap, "", "---", ""]

    if disclaimer:
        lines += ["**Disclaimer**", "", disclaimer, "", "---", ""]

    return lines


def _pick_best_hook_text(c: CandidateIdea) -> str:
    if not c.hooks:
        return ""
    if c.best_hook_id:
        for h in c.hooks:
            if h.hook_id == c.best_hook_id:
                return (h.text or "").strip()
    # fallback
    return (c.hooks[0].text or "").strip()


def _safe_filename(s: str) -> str:
    keep = []
    for ch in s:
        if ch.isalnum() or ch in {"-", "_"}:
            keep.append(ch)
        elif ch in {" ", ".", ","}:
            keep.append("_")
    out = "".join(keep).strip("_")
    return out[:140] if len(out) > 140 else out

