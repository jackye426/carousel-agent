"""
Auto-save Carousel Studio (Streamlit) draft: writer notes + editorial fields into RunState JSON.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from carousel_agents.schemas import RunState


def resolve_runstate_path(raw: str, *, cwd: Path | None = None) -> Path:
    """
    Resolve a user-typed path to a RunState JSON file.
    Handles quotes, expands ~, tries cwd-relative paths on Windows/Linux.
    """
    s = (raw or "").strip().strip('\'"')
    if not s:
        return Path()
    base = cwd or Path.cwd()
    p = Path(s).expanduser()
    if p.is_file():
        return p.resolve()
    # Absolute path missing — e.g. typo
    if p.is_absolute() and not p.is_file():
        return p
    # Relative: try as-is from cwd
    p2 = (base / s).resolve()
    if p2.is_file():
        return p2
    # Normalise slashes for pasted paths
    p3 = (base / s.replace("\\", "/")).resolve()
    if p3.is_file():
        return p3
    return p.resolve() if p.is_absolute() else p2


def merge_select_ui_into_run_state(
    rs: RunState,
    *,
    ranked: list[Any],
    session_state: Any,
    editorial_key: str = "ui_editorial",
    global_notes_key: str = "ui_global_notes",
) -> None:
    """
    Copy Streamlit widget values into RunState without applying human shortlist (no gate).
    """
    ed = str(session_state.get(editorial_key, "")).strip()
    if ed:
        rs.human_editorial_direction = ed
    else:
        rs.human_editorial_direction = None

    gn = str(session_state.get(global_notes_key, "")).strip()
    rs.reviewer_brief_global = gn if gn else None

    by_idea = dict(rs.reviewer_brief_by_idea or {})
    for c in ranked:
        nk = f"note_{c.idea_id}"
        if nk in session_state:
            t = str(session_state[nk]).strip()
            if t:
                by_idea[c.idea_id] = t
            else:
                by_idea.pop(c.idea_id, None)
    rs.reviewer_brief_by_idea = by_idea


def default_autosave_path(rs: RunState, out_parent: Path) -> Path:
    out_parent = out_parent.expanduser()
    out_parent.mkdir(parents=True, exist_ok=True)
    return out_parent / f"{rs.document.document_id}_studio.json"


def save_run_state_json(rs: RunState, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rs.model_dump_json(indent=2), encoding="utf-8")
