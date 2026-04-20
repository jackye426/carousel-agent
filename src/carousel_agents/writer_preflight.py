"""
Writer preflight: optional clarification questions before hooks/slides (UI or API).
"""

from __future__ import annotations

from typing import Any

from .llm import get_model_config
from .llm_response_models import WriterPreflightResponse
from .prompts import system_writer, user_writer_preflight_clarification
from .schemas import RunState


def run_writer_preflight(
    state: RunState,
    client: Any,
    *,
    prior_qa: str | None = None,
) -> WriterPreflightResponse:
    selected = [c for c in state.candidates if c.selected]
    if not selected:
        return WriterPreflightResponse(need_clarification=False, questions=[], assistant_message="No ideas selected.")
    writer = get_model_config("writer")
    return client.chat_structured(
        response_model=WriterPreflightResponse,
        model=writer.model,
        temperature=writer.temperature,
        system=system_writer(),
        user=user_writer_preflight_clarification(
            selected_ideas=[c.model_dump() for c in selected],
            audience=state.audience.model_dump(),
            editorial_direction=state.human_editorial_direction or state.proposed_editorial_direction,
            reviewer_notes_global=state.reviewer_brief_global,
            prior_qa=prior_qa,
        ),
    )
