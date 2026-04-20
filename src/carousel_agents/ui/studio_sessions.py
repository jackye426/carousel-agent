"""
ChatGPT-style session storage: one bundle per conversation (RunState + phase + chat_log + paths).

Stored under <output_folder>/.carousel_studio/sessions/
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from carousel_agents.schemas import RunState

BUNDLE_VERSION = 1
MAX_INDEX_ENTRIES = 60


def studio_root(out_parent: Path) -> Path:
    root = out_parent.expanduser() / ".carousel_studio"
    root.mkdir(parents=True, exist_ok=True)
    (root / "sessions").mkdir(parents=True, exist_ok=True)
    return root


def new_session_id() -> str:
    return f"ses_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


def _index_path(out_parent: Path) -> Path:
    return studio_root(out_parent) / "index.json"


def _bundle_path(out_parent: Path, session_id: str) -> Path:
    return studio_root(out_parent) / "sessions" / f"{session_id}.json"


@dataclass(frozen=True)
class SessionListItem:
    session_id: str
    title: str
    updated_at: str
    bundle_path: Path


def list_sessions(out_parent: Path, *, limit: int = 40) -> list[SessionListItem]:
    idx = _index_path(out_parent)
    if not idx.is_file():
        return []
    try:
        data = json.loads(idx.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    rows = data.get("sessions") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        return []
    out: list[SessionListItem] = []
    for row in rows[:limit]:
        if not isinstance(row, dict):
            continue
        sid = str(row.get("session_id") or "").strip()
        bp = str(row.get("bundle_path") or "").strip()
        if not sid or not bp:
            continue
        p = Path(bp)
        if not p.is_file():
            continue
        out.append(
            SessionListItem(
                session_id=sid,
                title=str(row.get("title") or sid)[:80],
                updated_at=str(row.get("updated_at") or "")[:25],
                bundle_path=p.resolve(),
            )
        )
    return out


def _write_index(out_parent: Path, sessions: list[dict[str, Any]]) -> None:
    ip = _index_path(out_parent)
    ip.parent.mkdir(parents=True, exist_ok=True)
    ip.write_text(json.dumps({"version": 1, "sessions": sessions[:MAX_INDEX_ENTRIES]}, indent=2), encoding="utf-8")


def save_studio_session(
    *,
    out_parent: Path,
    session_id: str,
    rs: RunState,
    phase: str,
    chat_log: list[tuple[str, str]],
    out_path: Path | None,
    final_out: Path | None = None,
    title: str | None = None,
) -> Path:
    """Write bundle JSON and update index. Returns path to bundle file."""
    out_parent = out_parent.expanduser()
    bp = _bundle_path(out_parent, session_id)
    now = datetime.now(timezone.utc).isoformat()
    doc_title = title or (rs.document.title or rs.document.document_id or "Untitled")
    payload = {
        "version": BUNDLE_VERSION,
        "session_id": session_id,
        "title": str(doc_title)[:200],
        "updated_at": now,
        "phase": phase,
        "chat_log": [[a, b] for a, b in chat_log],
        "run_state": json.loads(rs.model_dump_json()),
        "out_path": str(out_path.resolve()) if out_path else None,
        "final_out": str(final_out.resolve()) if final_out else None,
    }
    bp.parent.mkdir(parents=True, exist_ok=True)
    bp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # Index: merge by session_id, newest first
    idx_path = _index_path(out_parent)
    existing: list[dict[str, Any]] = []
    if idx_path.is_file():
        try:
            raw = json.loads(idx_path.read_text(encoding="utf-8"))
            existing = raw.get("sessions") if isinstance(raw, dict) else []
        except (json.JSONDecodeError, OSError):
            existing = []
    if not isinstance(existing, list):
        existing = []

    row = {
        "session_id": session_id,
        "title": str(doc_title)[:120],
        "updated_at": now,
        "bundle_path": str(bp.resolve()),
    }
    merged = [r for r in existing if isinstance(r, dict) and str(r.get("session_id")) != session_id]
    merged.insert(0, row)
    merged.sort(key=lambda r: str(r.get("updated_at") or ""), reverse=True)
    _write_index(out_parent, merged[:MAX_INDEX_ENTRIES])
    return bp


def load_studio_bundle(bundle_path: Path) -> dict[str, Any]:
    return json.loads(Path(bundle_path).read_text(encoding="utf-8"))


def bundle_to_run_state(bundle: dict[str, Any]) -> RunState:
    rs = bundle.get("run_state")
    if not isinstance(rs, dict):
        raise ValueError("bundle missing run_state")
    return RunState.model_validate(rs)


def format_session_label(item: SessionListItem) -> str:
    try:
        dt = item.updated_at[:10] if item.updated_at else ""
    except Exception:
        dt = ""
    t = item.title[:48] + ("…" if len(item.title) > 48 else "")
    tail = item.session_id[-8:] if len(item.session_id) > 8 else item.session_id
    if dt:
        return f"{t} · {dt} · {tail}"
    return f"{t} · {tail}"
