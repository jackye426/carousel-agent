"""
Carousel Studio — chat-style flow: ideation → select ideas + notes → writer clarification → hooks/slides + optional A/B.

Run (from repo root, with optional venv):
  pip install -e ".[ui]"
  streamlit run src/carousel_agents/ui/streamlit_app.py
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from carousel_agents.defaults import export_dir_default
from carousel_agents.human_review import HumanSelectionFile, apply_human_selection_to_state
from carousel_agents.llm import MockClient, OpenAICompatibleClient
from carousel_agents.pipeline import run_pipeline, run_pipeline_production_tail
from carousel_agents.schemas import ExperimentSpec, RunState
from carousel_agents.ui.run_state_session import (
    append_recent_run_index,
    chat_log_seed_for_loaded,
    infer_ui_phase,
    widget_defaults_from_run_state,
)
from carousel_agents.ui.studio_persist import (
    default_autosave_path,
    merge_select_ui_into_run_state,
    resolve_runstate_path,
    save_run_state_json,
)
from carousel_agents.ui.studio_sessions import (
    bundle_to_run_state,
    format_session_label,
    load_studio_bundle,
    list_sessions,
    new_session_id,
    save_studio_session,
)
from carousel_agents.writer_preflight import run_writer_preflight


def _client(mock: bool) -> Any:
    return MockClient() if mock else OpenAICompatibleClient()


def _short(text: str | None, n: int = 220) -> str:
    t = (text or "").strip()
    if not t:
        return "—"
    return t if len(t) <= n else t[: n - 1] + "…"


def _fmt_qa_block(questions: list[str], answer_text: str) -> str:
    lines = []
    for i, q in enumerate(questions):
        lines.append(f"Q{i + 1}: {q}")
    lines.append("Reviewer answers:")
    lines.append(answer_text.strip())
    return "\n".join(lines)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _api_key_configured() -> bool:
    return bool(
        (os.getenv("OPENROUTER_API_KEY") or "").strip()
        or (os.getenv("OPENAI_API_KEY") or "").strip()
    )


def _apply_loaded_bundle_to_session(bundle: dict[str, Any]) -> None:
    """Restore RunState, phase, chat, paths from a studio session bundle."""
    rs = bundle_to_run_state(bundle)
    st.session_state.run_state = rs
    st.session_state.phase = str(bundle.get("phase") or "upload")
    raw_log = bundle.get("chat_log") or []
    st.session_state.chat_log = [(str(a), str(b)) for a, b in raw_log] if raw_log else []
    op = bundle.get("out_path") or ""
    st.session_state.out_path = Path(op) if op else None
    fo = bundle.get("final_out") or ""
    st.session_state.final_out = Path(fo) if fo else None
    st.session_state.studio_session_id = str(bundle.get("session_id") or "")
    for k in ("last_preflight", "preflight_ready"):
        st.session_state.pop(k, None)
    st.session_state._sync_widgets_from_run = True


def _persist_conversation_snapshot(out_parent_str: str | None) -> None:
    """Save current UI conversation to .carousel_studio (ChatGPT-style history)."""
    rs = st.session_state.get("run_state")
    if rs is None:
        return
    raw = (out_parent_str or st.session_state.get("out_parent_sidebar") or str(Path.cwd() / "outputs")).strip()
    out_parent = Path(raw).expanduser()
    sid = st.session_state.get("studio_session_id")
    if not sid:
        sid = new_session_id()
        st.session_state.studio_session_id = sid
    phase = str(st.session_state.get("phase") or "upload")
    chat_log: list[tuple[str, str]] = list(st.session_state.get("chat_log") or [])
    op = st.session_state.get("out_path")
    fo = getattr(st.session_state, "final_out", None)
    try:
        save_studio_session(
            out_parent=out_parent,
            session_id=sid,
            rs=rs,
            phase=phase,
            chat_log=chat_log,
            out_path=Path(op) if op else None,
            final_out=Path(fo) if fo else None,
        )
    except OSError:
        pass


def _new_conversation() -> None:
    for k in (
        "run_state",
        "phase",
        "chat_log",
        "out_path",
        "final_out",
        "studio_session_id",
        "last_preflight",
        "preflight_ready",
        "_active_studio_doc_id",
        "_sync_widgets_from_run",
        "_last_run_upload_marker",
        "ui_editorial",
        "ui_global_notes",
    ):
        st.session_state.pop(k, None)
    st.session_state.run_state = None
    st.session_state.phase = "upload"
    st.session_state.chat_log = []


def main() -> None:
    load_dotenv(_project_root() / ".env")
    st.set_page_config(page_title="Carousel Studio", layout="wide")
    st.title("Carousel Studio")
    st.caption("Ideation → your selection & notes → writer clarification → hooks, slides, optional A/B.")

    if "run_state" not in st.session_state:
        st.session_state.run_state = None
    if "phase" not in st.session_state:
        st.session_state.phase = "upload"
    if "chat_log" not in st.session_state:
        st.session_state.chat_log: list[tuple[str, str]] = []

    if st.session_state.pop("_sync_widgets_from_run", False):
        rs0 = st.session_state.run_state
        if rs0 is not None:
            for k, v in widget_defaults_from_run_state(rs0).items():
                st.session_state[k] = v
            # Reset campaign/global text areas so they pick up loaded RunState
            st.session_state.pop("ui_editorial", None)
            st.session_state.pop("ui_global_notes", None)

    rs = st.session_state.run_state

    with st.sidebar:
        out_parent = st.text_input(
            "Output folder",
            value=str(Path.cwd() / "outputs"),
            help="Ideation JSON, autosaves, and your conversation history are stored here (under .carousel_studio/).",
        )
        out_base = Path(out_parent).expanduser()

        st.subheader("Conversations")
        st.caption("Like ChatGPT: previous runs are saved under **Output folder** → `.carousel_studio/sessions/`.")
        if st.button("New conversation", use_container_width=True, help="Start fresh (upload a new document)."):
            _new_conversation()
            st.rerun()

        sess_items = list_sessions(out_base)
        sess_labels = ["—"] + [format_session_label(s) for s in sess_items]
        pick_i = st.selectbox(
            "Open a previous conversation",
            options=list(range(len(sess_labels))),
            format_func=lambda i: sess_labels[i],
            index=0,
            key="studio_pick_conversation",
        )
        if st.button("Open selected", use_container_width=True, disabled=(pick_i == 0)):
            item = sess_items[pick_i - 1] if pick_i > 0 and pick_i <= len(sess_items) else None
            if item is not None and item.bundle_path.is_file():
                try:
                    bundle = load_studio_bundle(item.bundle_path)
                    _apply_loaded_bundle_to_session(bundle)
                    st.success("Loaded conversation.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not load session: {e}")

        st.divider()
        st.subheader("Settings")
        mock = st.checkbox("Mock mode (no API keys)", value=False)
        if not mock and not _api_key_configured():
            st.error(
                "No API key found. Set **OPENROUTER_API_KEY** or **OPENAI_API_KEY** in `.env` "
                "(see `.env.example`) or export it in your environment, then refresh."
            )
        candidate_count = st.number_input("Ideas to extract", min_value=5, max_value=60, value=20, step=1)
        export_root = st.text_input("Export root", value=str(export_dir_default()))

        st.divider()
        st.subheader("Import RunState file")
        st.caption(
            "Open a RunState `.json` from a past ideation or production run. "
            "Paste the **full path** (from File Explorer: copy as path). "
            f"Your **Output folder** is: `{Path(out_parent).expanduser()}` — runs are usually `ui_<filename>.json` there."
        )
        load_path = st.text_input("Path to RunState JSON", value="", placeholder=r"C:\Users\…\outputs\ui_doc.json")
        if st.button("Load from path", use_container_width=True):
            raw_in = (load_path or "").strip()
            if not raw_in:
                st.warning(
                    "Paste the **full path** to a `.json` RunState file first (the box above was empty). "
                    "Or use **Or upload RunState JSON** below — no typing needed."
                )
            else:
                p = resolve_runstate_path(load_path, cwd=Path.cwd())
                if not p.is_file():
                    tried = p.resolve()
                    st.error(
                        "File not found.\n\n"
                        f"- Looked here: `{tried}`\n"
                        "- Check spelling; paste the **full** path from Explorer (**Copy as path**), "
                        "or use **upload** below."
                    )
                else:
                    try:
                        raw = p.read_text(encoding="utf-8")
                        loaded = RunState.model_validate_json(raw)
                        phase = infer_ui_phase(loaded)
                        st.session_state.run_state = loaded
                        st.session_state.phase = phase
                        st.session_state.chat_log = chat_log_seed_for_loaded(loaded, phase)
                        st.session_state.out_path = p
                        for k in ("last_preflight", "preflight_ready"):
                            st.session_state.pop(k, None)
                        st.session_state._sync_widgets_from_run = True
                        if phase == "done":
                            st.session_state.final_out = p
                        else:
                            st.session_state.pop("final_out", None)
                        idx = Path(out_parent).expanduser() / ".carousel_studio_index.json"
                        append_recent_run_index(index_path=idx, run_path=p, rs=loaded, phase=phase)
                        st.session_state.studio_session_id = new_session_id()
                        _persist_conversation_snapshot(out_parent)
                        st.success(f"Loaded — step: **{phase}**")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Could not load RunState: {e}")

        up = st.file_uploader("Or upload RunState JSON", type=["json"], key="runstate_upload")
        if up is not None:
            marker = f"{up.name}:{up.size}"
            if st.session_state.get("_last_run_upload_marker") != marker:
                try:
                    loaded = RunState.model_validate_json(up.getvalue().decode("utf-8"))
                    phase = infer_ui_phase(loaded)
                    st.session_state.run_state = loaded
                    st.session_state.phase = phase
                    st.session_state.chat_log = chat_log_seed_for_loaded(loaded, phase)
                    out_dir = Path(out_parent).expanduser()
                    out_dir.mkdir(parents=True, exist_ok=True)
                    snapshot = out_dir / f"{loaded.document.document_id}_loaded.json"
                    snapshot.write_text(loaded.model_dump_json(indent=2), encoding="utf-8")
                    st.session_state.out_path = snapshot
                    for k in ("last_preflight", "preflight_ready"):
                        st.session_state.pop(k, None)
                    st.session_state._sync_widgets_from_run = True
                    st.session_state._last_run_upload_marker = marker
                    if phase == "done":
                        st.session_state.final_out = snapshot
                    else:
                        st.session_state.pop("final_out", None)
                    idx = out_dir / ".carousel_studio_index.json"
                    append_recent_run_index(index_path=idx, run_path=snapshot, rs=loaded, phase=phase)
                    st.session_state.studio_session_id = new_session_id()
                    _persist_conversation_snapshot(str(out_dir))
                    st.success(f"Loaded **{up.name}** — step: **{phase}**")
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not load RunState: {e}")

        idx_path = Path(out_parent).expanduser() / ".carousel_studio_index.json"
        recent_rows: list[dict[str, Any]] = []
        if idx_path.is_file():
            try:
                raw_list = json.loads(idx_path.read_text(encoding="utf-8"))
                if isinstance(raw_list, list):
                    recent_rows = [r for r in raw_list if isinstance(r, dict)]
            except (json.JSONDecodeError, OSError):
                recent_rows = []
        recent_options: list[tuple[str, Path]] = []
        for row in reversed(recent_rows[-15:]):
            p_str = str(row.get("path") or "").strip()
            if not p_str:
                continue
            pth = Path(p_str)
            if not pth.is_file():
                continue
            title = str(row.get("title") or row.get("document_id") or pth.name)[:48]
            ph = str(row.get("phase") or "?")
            recent_options.append((f"{title} · {ph} · {pth.name}", pth))
        if recent_options:
            label_list = ["—"] + [x[0] for x in recent_options]
            pick_recent = st.selectbox("Recent (this output folder)", options=label_list, index=0)
            path_by_label = dict(recent_options)
            if st.button(
                "Open selected recent",
                use_container_width=True,
                disabled=(pick_recent == "—"),
            ):
                p2 = path_by_label.get(pick_recent)
                if p2 is not None and p2.is_file():
                    try:
                        loaded = RunState.model_validate_json(p2.read_text(encoding="utf-8"))
                        phase = infer_ui_phase(loaded)
                        st.session_state.run_state = loaded
                        st.session_state.phase = phase
                        st.session_state.chat_log = chat_log_seed_for_loaded(loaded, phase)
                        st.session_state.out_path = p2
                        for k in ("last_preflight", "preflight_ready"):
                            st.session_state.pop(k, None)
                        st.session_state._sync_widgets_from_run = True
                        if phase == "done":
                            st.session_state.final_out = p2
                        else:
                            st.session_state.pop("final_out", None)
                        st.session_state.studio_session_id = new_session_id()
                        _persist_conversation_snapshot(str(Path(out_parent).expanduser()))
                        st.success(f"Loaded — step: **{phase}**")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Could not load: {e}")

        rs_sidebar = st.session_state.run_state
        if rs_sidebar is not None:
            st.divider()
            st.download_button(
                "Download current RunState (JSON)",
                data=rs_sidebar.model_dump_json(indent=2).encode("utf-8"),
                file_name=f"{rs_sidebar.document.document_id}_snapshot.json",
                key="download_runstate_snapshot",
            )

        st.session_state.out_parent_sidebar = out_parent

    st.session_state.mock = mock
    api_ok = mock or _api_key_configured()

    # ---- Phase: upload / ideate
    if st.session_state.phase == "upload":
        f = st.file_uploader("Upload source (.md, .txt)", type=["md", "txt", "markdown"])
        if st.button("Run ideation", type="primary", disabled=f is None or not api_ok):
            if f is None:
                st.warning("Upload a document first.")
            else:
                suffix = Path(f.name).suffix or ".txt"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(f.getvalue())
                    tmp_path = Path(tmp.name)
                try:
                    out_path = Path(out_parent) / f"ui_{Path(f.name).stem}.json"
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    with st.spinner("Extracting & scoring candidates…"):
                        state = run_pipeline(
                            input_path=tmp_path,
                            out_path=out_path,
                            candidate_count=int(candidate_count),
                            select_n=5,
                            mock=mock,
                            export_dir=None,
                            stop_after_scoring=True,
                        )
                    st.session_state.run_state = state
                    st.session_state.out_path = out_path
                    st.session_state.phase = "select"
                    st.session_state.chat_log = [
                        (
                            "assistant",
                            "Here are ranked ideas from your document. Select the ones you want below, "
                            "add direction and notes, then continue.",
                        )
                    ]
                    idx = Path(out_parent).expanduser() / ".carousel_studio_index.json"
                    append_recent_run_index(index_path=idx, run_path=out_path, rs=state, phase="select")
                    st.session_state.studio_session_id = new_session_id()
                    st.session_state.out_parent_sidebar = out_parent
                    _persist_conversation_snapshot(out_parent)
                    st.rerun()
                finally:
                    try:
                        tmp_path.unlink(missing_ok=True)  # type: ignore[arg-type]
                    except OSError:
                        pass

    # ---- Phase: select
    if st.session_state.phase == "select" and rs is not None:
        did = rs.document.document_id
        if st.session_state.get("_active_studio_doc_id") != did:
            st.session_state.ui_editorial = rs.human_editorial_direction or rs.proposed_editorial_direction or ""
            st.session_state.ui_global_notes = rs.reviewer_brief_global or ""
            st.session_state._active_studio_doc_id = did

        st.subheader("1 — Review ideas")
        if rs.proposed_editorial_direction:
            with st.expander("Model’s proposed editorial direction", expanded=True):
                st.write(rs.proposed_editorial_direction)

        ranked = sorted(rs.candidates, key=lambda c: (c.rank or 10_000))
        st.markdown(
            "Expand each row for **direction** (angle), **claims**, and **how the carousel will expand**. "
            "Tick **Include in production** at the bottom of an idea to shortlist it — a **Writer notes** field appears there."
        )
        for c in ranked:
            tw = ""
            if c.scores is not None and c.scores.total_weighted is not None:
                tw = f" · weighted **{c.scores.total_weighted}**"
            exp_title = f"#{c.rank} `{c.idea_id}` — {_short(c.topic, 90)}"
            picked = bool(st.session_state.get(f"pick_{c.idea_id}", False))
            top3 = c.rank is not None and c.rank <= 3
            with st.expander(exp_title, expanded=top3 or picked):
                st.markdown(
                    f"**Pillar** `{c.content_pillar}` · **Suggested format** `{c.format_suggestion}`{tw}"
                )
                st.markdown("**Angle / direction**")
                st.write(_short(c.angle, 600))
                st.markdown("**Core claim**")
                st.write(_short(c.core_claim, 600))
                st.markdown("**Audience pain**")
                st.write(_short(c.audience_pain, 500))
                st.markdown("**Promise** (what the reader gets)")
                st.write(_short(c.promise, 500))
                rb = getattr(c, "reader_benefit", None)
                if rb and str(rb).strip():
                    st.markdown("**Reader benefit** (ideation — patient-facing payoff)")
                    st.success(_short(str(rb), 600))
                st.markdown("**How this expands into a carousel**")
                st.info(
                    f"The Writer will shape this as a **{c.format_suggestion or 'carousel'}** using the angle above, "
                    "ground claims in **verbatim citations** from your source document, and keep one clear pillar focus. "
                    "Slide count adapts to the idea (not a fixed slide quota)."
                )
                if c.source_citations:
                    ex0 = c.source_citations[0]
                    st.caption(
                        f"Example citation ({ex0.chunk_id}): “{_short(ex0.excerpt, 160)}”"
                    )
                if c.safety_flags:
                    st.warning("Flags: " + ", ".join(c.safety_flags))

                st.divider()
                st.checkbox(
                    "Include in production",
                    key=f"pick_{c.idea_id}",
                    help="Shortlist this idea for the Writer (hooks, CTA, slides).",
                )
                if st.session_state.get(f"pick_{c.idea_id}", False):
                    st.text_area(
                        "Writer notes (optional)",
                        key=f"note_{c.idea_id}",
                        height=72,
                        placeholder="Tone, taboos, emphasis, or tweaks for this idea only…",
                    )

        st.text_area(
            "Campaign / editorial direction",
            height=100,
            help="Overrides or refines the model proposal for Writer. Auto-saved with your notes.",
            key="ui_editorial",
        )
        st.text_area(
            "Extra reviewer notes (constraints, tone, taboos)",
            height=80,
            key="ui_global_notes",
        )

        out_dir = Path(out_parent).expanduser()
        save_path = st.session_state.get("out_path")
        if save_path is None:
            save_path = default_autosave_path(rs, out_dir)
        else:
            save_path = Path(save_path)
            if not save_path.parent.exists():
                save_path = out_dir / save_path.name
        try:
            merge_select_ui_into_run_state(rs, ranked=ranked, session_state=st.session_state)
            save_run_state_json(rs, save_path)
            st.session_state.out_path = save_path
            st.caption(f"Draft auto-saved (writer notes + direction): `{save_path}`")
            _persist_conversation_snapshot(st.session_state.get("out_parent_sidebar"))
        except OSError as e:
            st.warning(f"Could not auto-save draft: {e}")

        if st.button("Continue to writer clarification", type="primary"):
            picked_ids = [c.idea_id for c in ranked if st.session_state.get(f"pick_{c.idea_id}", False)]
            if not picked_ids:
                st.error("Select at least one idea (use **Include in production** inside an idea).")
            else:
                idea_notes_out = {
                    iid: str(st.session_state.get(f"note_{iid}", "")).strip() for iid in picked_ids
                }
                idea_notes_out = {k: v for k, v in idea_notes_out.items() if v}
                editorial = str(st.session_state.get("ui_editorial", "")).strip()
                global_notes = str(st.session_state.get("ui_global_notes", "")).strip()
                sel = HumanSelectionFile(
                    selected_idea_ids=picked_ids,
                    editorial_direction=editorial or None,
                    global_notes=global_notes or None,
                    idea_notes=idea_notes_out,
                )
                apply_human_selection_to_state(rs, sel)
                if "last_preflight" in st.session_state:
                    del st.session_state.last_preflight
                if "preflight_ready" in st.session_state:
                    del st.session_state.preflight_ready
                st.session_state.phase = "clarify"
                st.session_state.chat_log.append(("user", f"Selected {len(sel.selected_idea_ids)} idea(s) with notes."))
                st.rerun()

    # ---- Phase: writer clarification
    if st.session_state.phase == "clarify" and rs is not None:
        st.subheader("2 — Writer clarification")
        for role, text in st.session_state.chat_log[-12:]:
            with st.chat_message(role):
                st.markdown(text)

        client = _client(st.session_state.mock)
        if "last_preflight" not in st.session_state:
            with st.spinner("Writer is checking whether anything needs clarification…"):
                pf = run_writer_preflight(rs, client, prior_qa=None)
            st.session_state.last_preflight = pf
            msg = pf.assistant_message or ""
            if pf.questions:
                msg += "\n\n**Questions:**\n" + "\n".join(f"- {q}" for q in pf.questions)
            elif not msg:
                msg = "No follow-up questions — you can generate content whenever you're ready."
            st.session_state.chat_log.append(("assistant", msg))
            st.rerun()

        pf = st.session_state.last_preflight
        if pf.need_clarification and pf.questions:
            ans = st.text_area("Your answers (address each question)", height=150, key="clar_ans")
            if st.button("Save answers & continue", type="primary"):
                block = _fmt_qa_block(pf.questions, ans)
                prev = (rs.writer_clarification_transcript or "").strip()
                rs.writer_clarification_transcript = (prev + "\n\n" + block).strip() if prev else block
                st.session_state.chat_log.append(("user", ans))
                st.session_state.phase = "produce"
                st.rerun()
        else:
            if st.button("Continue to production", type="primary"):
                st.session_state.phase = "produce"
                st.rerun()

        op = st.session_state.get("out_path")
        if op:
            try:
                save_run_state_json(rs, Path(op))
            except OSError:
                pass
        _persist_conversation_snapshot(st.session_state.get("out_parent_sidebar"))

    # ---- Phase: produce (A/B + tail)
    if st.session_state.phase == "produce" and rs is not None:
        st.subheader("3 — Produce content + optional A/B")
        st.info("Hooks, CTAs, slides, and optional experiment split run next.")

        do_ab = st.checkbox("Attach A/B experiment (split one idea into two arms after hooks)", value=False)
        first_id = (rs.shortlist.selected_idea_ids or ["idea_id"])[0]
        exp_json = st.text_area(
            "ExperimentSpec JSON (idea_id must be one of your selected ids)",
            value=json.dumps(
                {
                    "experiment_id": "exp_ui_001",
                    "hypothesis": "Myth-bust vs how-to hook",
                    "idea_id": first_id,
                    "treatment_key": "hook_style_contrast",
                    "primary_metric": "save_rate",
                },
                indent=2,
            ),
            height=180,
        )

        if st.button("Run production", type="primary"):
            if do_ab:
                try:
                    exp = ExperimentSpec.model_validate_json(exp_json)
                    if exp.idea_id not in (rs.shortlist.selected_idea_ids or []):
                        st.error(f"experiment.idea_id must be one of: {rs.shortlist.selected_idea_ids}")
                        return
                    rs.experiment = exp
                except Exception as e:
                    st.error(f"Invalid experiment JSON: {e}")
                    return
            else:
                rs.experiment = None

            exp_path = Path(export_root)
            out_final = Path(out_parent) / f"{rs.document.document_id}_produced.json"
            with st.spinner("Generating hooks, CTAs, slides…"):
                run_pipeline_production_tail(
                    rs,
                    client=_client(st.session_state.mock),
                    mock=st.session_state.mock,
                    performance_digest=None,
                    hook_count=12,
                    max_slides=10,
                    do_hooks=True,
                    do_cta=True,
                    do_slides=True,
                    do_qa=False,
                    judge_govern=True,
                    export_dir=exp_path,
                    out_path=out_final,
                )
            st.session_state.phase = "done"
            st.session_state.final_out = out_final
            st.session_state.out_path = out_final
            idx = Path(out_parent).expanduser() / ".carousel_studio_index.json"
            append_recent_run_index(index_path=idx, run_path=out_final, rs=rs, phase="done")
            _persist_conversation_snapshot(st.session_state.get("out_parent_sidebar"))
            st.success("Done.")
            st.rerun()

        op = st.session_state.get("out_path")
        if op and rs is not None:
            try:
                save_run_state_json(rs, Path(op))
            except OSError:
                pass
        _persist_conversation_snapshot(st.session_state.get("out_parent_sidebar"))

    if st.session_state.phase == "done" and getattr(st.session_state, "final_out", None):
        st.subheader("Output")
        p = st.session_state.final_out
        rs_done: RunState | None = st.session_state.run_state
        doc_id = rs_done.document.document_id if rs_done else ""
        st.write(f"RunState: `{p}`")
        st.write(f"Exports under: `{export_root}` / `{doc_id}`")
        if p and Path(p).exists():
            st.download_button("Download RunState JSON", data=Path(p).read_bytes(), file_name=Path(p).name)

    if st.sidebar.button("Reset session"):
        _new_conversation()
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()


if __name__ == "__main__":
    main()
