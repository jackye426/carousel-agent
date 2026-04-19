"""Load/save experiment program queue JSON."""

from __future__ import annotations

import json
from pathlib import Path

from .schemas import ExperimentProgramState


def load_program(path: Path) -> ExperimentProgramState:
    if not path.exists():
        return ExperimentProgramState()
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return ExperimentProgramState()
    return ExperimentProgramState.model_validate_json(raw)


def save_program(state: ExperimentProgramState, *, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(state.model_dump_json(indent=2) + "\n", encoding="utf-8")
