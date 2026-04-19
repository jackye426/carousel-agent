"""Rule-based persona proxy when explicit persona is not logged."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

PersonaSource = str  # "explicit" | "rules" | "unclassified"


def persona_rules_path() -> Path:
    return Path(os.environ.get("PERSONA_RULES_PATH", "data/persona_rules.json")).resolve()


def _norm(s: str | None) -> str:
    return (s or "").strip().lower().replace("-", "_").replace(" ", "_")


def load_persona_rules() -> dict[str, Any]:
    p = persona_rules_path()
    if not p.exists():
        return {"version": 1, "default_persona_id": "unclassified", "rules": []}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"version": 1, "default_persona_id": "unclassified", "rules": []}


def _rule_matches(
    rule: dict[str, Any],
    *,
    pillar_n: str,
    hook_n: str,
    format_n: str,
) -> bool:
    """All fields present on the rule must match; omitted fields are wildcards."""
    if rule.get("pillar") is not None and str(rule["pillar"]).strip():
        if _norm(str(rule["pillar"])) != pillar_n:
            return False
    if rule.get("hook_style") is not None and str(rule["hook_style"]).strip():
        if not hook_n or _norm(str(rule["hook_style"])) != hook_n:
            return False
    if rule.get("format_suggestion") is not None and str(rule["format_suggestion"]).strip():
        if not format_n or _norm(str(rule["format_suggestion"])) != format_n:
            return False
    return True


def resolve_persona_proxy(
    *,
    pillar: str | None,
    hook_style: str | None,
    format_suggestion: str | None,
) -> tuple[str, PersonaSource]:
    """
    Return (persona_id, source). First matching rule wins; else default_persona_id;
    if still unclassified, return ("unclassified", "unclassified").
    """
    data = load_persona_rules()
    rules = data.get("rules") or []
    default_id = str(data.get("default_persona_id") or "unclassified").strip() or "unclassified"
    pillar_n = _norm(pillar)
    hook_n = _norm(hook_style)
    format_n = _norm(format_suggestion)

    for rule in rules:
        if not isinstance(rule, dict):
            continue
        if not _rule_matches(rule, pillar_n=pillar_n, hook_n=hook_n, format_n=format_n):
            continue
        pid = str(rule.get("persona_id") or "").strip()
        if pid:
            return pid, "rules"

    if default_id and default_id != "unclassified":
        return default_id, "rules"
    return "unclassified", "unclassified"
