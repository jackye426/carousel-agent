"""Load persona cards from JSON files (simple, direct targeting)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


def personas_dir() -> Path:
    return Path(os.environ.get("PERSONAS_DIR", "docs/personas")).resolve()


class PersonaCard(BaseModel):
    """One segment; keep fields short — prompts merge a capped `to_prompt_dict()`."""

    id: str = Field(..., description="Stable slug; matches filename without .json")
    name: str = ""
    summary: str = ""
    primary_fear: str = ""
    primary_hope: str = ""
    lexicon: list[str] = Field(default_factory=list)
    dos: list[str] = Field(default_factory=list)
    donts: list[str] = Field(default_factory=list)
    content_job: str = ""  # e.g. validate | educate | next_step

    def to_prompt_dict(self) -> dict[str, Any]:
        """Capped payload injected under `persona_preset` on writer idea dicts."""
        return {
            "id": self.id,
            "name": self.name.strip(),
            "summary": _cap(self.summary, 400),
            "primary_fear": _cap(self.primary_fear, 240),
            "primary_hope": _cap(self.primary_hope, 240),
            "lexicon": [_cap(x, 220) for x in self.lexicon[:12] if str(x).strip()],
            "dos": [_cap(x, 160) for x in self.dos[:5] if str(x).strip()],
            "donts": [_cap(x, 160) for x in self.donts[:5] if str(x).strip()],
            "content_job": (self.content_job or "").strip(),
        }


def _cap(s: str, n: int) -> str:
    t = (s or "").strip()
    return t if len(t) <= n else t[: n - 1] + "…"


def load_persona_card(preset_id: str) -> PersonaCard | None:
    """Load `{personas_dir()}/{preset_id}.json`. Returns None if missing or invalid."""
    pid = (preset_id or "").strip()
    if not pid or pid.lower() in ("none", "null"):
        return None
    path = personas_dir() / f"{pid}.json"
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(raw, dict):
        return None
    if raw.get("id") != pid:
        raw = {**raw, "id": pid}
    try:
        return PersonaCard.model_validate(raw)
    except Exception:
        return None


def apply_persona_preset_to_writer_idea(idea: dict[str, Any], preset_id: str | None) -> dict[str, Any]:
    """Merge `persona_preset` into a writer-bound idea dict (copy)."""
    out = dict(idea)
    if not preset_id or not str(preset_id).strip():
        return out
    card = load_persona_card(str(preset_id).strip())
    if card is None:
        out["persona_preset_error"] = f"persona_card_not_found:{preset_id}"
        return out
    out["persona_preset"] = card.to_prompt_dict()
    return out
