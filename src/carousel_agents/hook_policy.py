from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


def _norm(s: str) -> str:
    # Normalise to make detection robust across punctuation/quotes/case.
    s = (s or "").lower()
    s = s.replace("’", "'")
    s = re.sub(r"\s+", " ", s).strip()
    return s


# Canonical banned phrases (normalised, lowercased)
BANNED_PHRASES: list[str] = [
    "here's what no one tells you",
    "changes everything",
    "you're not alone",
    "every second count",
    "no more endless gp ping-pong",
    "here's why",
]


_RE_CAPS_NEED = re.compile(r"\bYOU\s+NEED\s+TO\b")


@dataclass(frozen=True)
class HookPolicyMatch:
    kind: str  # phrase|caps_emphasis
    matched: str


def find_policy_violations(text: str) -> list[HookPolicyMatch]:
    t_raw = text or ""
    t = _norm(t_raw)
    out: list[HookPolicyMatch] = []

    for p in BANNED_PHRASES:
        if p in t:
            out.append(HookPolicyMatch(kind="phrase", matched=p))

    # caps emphasis pattern (keep it simple and explicit)
    if _RE_CAPS_NEED.search(t_raw):
        out.append(HookPolicyMatch(kind="caps_emphasis", matched="YOU NEED TO"))

    return out


def hooks_have_violations(hooks: Iterable[dict]) -> bool:
    for h in hooks:
        if not isinstance(h, dict):
            continue
        if find_policy_violations(str(h.get("text", ""))):
            return True
    return False

