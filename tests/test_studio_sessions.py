"""Studio session bundle save/load."""

from __future__ import annotations

from pathlib import Path

from carousel_agents.schemas import DocumentMeta, RunState, Shortlist
from carousel_agents.ui.studio_sessions import (
    bundle_to_run_state,
    load_studio_bundle,
    list_sessions,
    new_session_id,
    save_studio_session,
)


def test_save_list_load_roundtrip(tmp_path: Path) -> None:
    rs = RunState(
        document=DocumentMeta(document_id="d1", title="Hello"),
        candidates=[],
        shortlist=Shortlist(),
    )
    sid = new_session_id()
    bp = save_studio_session(
        out_parent=tmp_path,
        session_id=sid,
        rs=rs,
        phase="select",
        chat_log=[("assistant", "hi")],
        out_path=tmp_path / "x.json",
        final_out=None,
    )
    assert bp.is_file()
    items = list_sessions(tmp_path)
    assert len(items) >= 1
    raw = load_studio_bundle(items[0].bundle_path)
    rs2 = bundle_to_run_state(raw)
    assert rs2.document.document_id == "d1"
    assert raw.get("phase") == "select"
    assert raw.get("chat_log") == [["assistant", "hi"]]
