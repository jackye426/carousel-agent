"""
Helpers for loading a persisted RunState into the Streamlit UI and inferring which step to show.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from carousel_agents.schemas import CandidateIdea, RunState

UiPhase = Literal["upload", "select", "clarify", "produce", "done"]


def _selected_ids(rs: RunState) -> list[str]:
    ids = [str(i).strip() for i in (rs.shortlist.selected_idea_ids or []) if str(i).strip()]
    if ids:
        return ids
    return [c.idea_id for c in rs.candidates if c.selected]


def _selected_candidates(rs: RunState) -> list[CandidateIdea]:
    ids = set(_selected_ids(rs))
    if not ids:
        return []
    return [c for c in rs.candidates if c.idea_id in ids]


def _has_slides(c: CandidateIdea) -> bool:
    cd = c.carousel_draft
    return bool(cd and cd.slides)


def infer_ui_phase(rs: RunState) -> UiPhase:
    """
    Map persisted pipeline/UI fields to the Streamlit phase so users can reopen a JSON run
    without rerunning ideation.
    """
    if rs.awaiting_human_review:
        return "select"

    ids = _selected_ids(rs)
    if not rs.human_shortlist_curated or not ids:
        return "select"

    sel = _selected_candidates(rs)
    if not sel:
        return "select"

    if all(_has_slides(c) for c in sel):
        return "done"

    has_hooks = any(c.hooks for c in sel)
    if not has_hooks:
        tr = (rs.writer_clarification_transcript or "").strip()
        if not tr:
            return "clarify"
        return "produce"

    return "produce"


def widget_defaults_from_run_state(rs: RunState) -> dict[str, Any]:
    """
    Key/value pairs to merge into st.session_state so pick_* / note_* match the loaded RunState.
    """
    ids = set(_selected_ids(rs))
    out: dict[str, Any] = {}
    for c in rs.candidates:
        out[f"pick_{c.idea_id}"] = c.idea_id in ids
        out[f"note_{c.idea_id}"] = (rs.reviewer_brief_by_idea or {}).get(c.idea_id, "") or ""
    return out


def chat_log_seed_for_loaded(rs: RunState, phase: UiPhase) -> list[tuple[str, str]]:
    """Minimal chat history so the clarify step is not empty after a cold load."""
    if phase == "select":
        return [
            (
                "assistant",
                "Loaded a saved run. Review ideas below, adjust your shortlist if needed, then continue.",
            )
        ]
    if phase == "clarify":
        return [
            (
                "assistant",
                "Loaded a saved run after shortlist selection. Writer clarification runs next when you continue.",
            )
        ]
    if phase == "produce":
        return [
            (
                "assistant",
                "Loaded a saved run ready for production (hooks, slides). Run production when you are ready.",
            )
        ]
    if phase == "done":
        title = (rs.document.title or rs.document.document_id or "document").strip()
        return [("assistant", f"Loaded completed run for “{title}”. Export paths are shown below.")]
    return []


def append_recent_run_index(
    *,
    index_path: Path,
    run_path: Path,
    rs: RunState,
    phase: UiPhase,
    max_entries: int = 40,
) -> None:
    """Append one entry to a small JSON list for quick reopen (best-effort)."""
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "path": str(run_path.resolve()),
        "document_id": rs.document.document_id,
        "title": (rs.document.title or "")[:200],
        "phase": phase,
    }
    try:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        data: list[dict[str, Any]] = []
        if index_path.exists():
            try:
                data = json.loads(index_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                data = []
        if not isinstance(data, list):
            data = []
        data.append(entry)
        data = data[-max_entries:]
        index_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError:
        pass
