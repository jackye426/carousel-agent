from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class PatientVoiceBankPaths:
    """
    Bank files are JSONL for append-friendly, dataset-style workflows.

    V1 schemas (one JSON object per line):

    - patient_voice_messages.jsonl
      {
        "thread_id": "...",
        "source": "...",
        "source_path": "...",
        "timestamp": "...",
        "text": "...",
        "char_len": 123,
        "has_question": true,
        "contains_numbers": false
      }

    - patient_voice_snippets.jsonl
      {
        "snippet_id": "pv_000001",
        "text": "...",
        "source_thread_id": "...",
        "source_message_timestamp": "...",
        "tags": { "topic": ["..."], "intent": ["..."] }
      }
    """

    messages_jsonl: Path
    snippets_jsonl: Path


_WHATSAPP_SYSTEM_EXACT = {
    "‎Messages and calls are end-to-end encrypted. Only people in this chat can read, listen to, or share them.",
    "You deleted this message.",
    "‎You deleted this message.",
}

_EDITED_SUFFIX_RE = re.compile(r"\s*‎?<This message was edited>\s*$", re.IGNORECASE)

# Very lightweight keyword tags (v1). Keep small + robust.
_TOPIC_KEYWORDS: dict[str, tuple[str, ...]] = {
    "bowel": ("bowel", "toilet", "constipation", "rectal", "poo", "bleed rect", "opening my bowels"),
    "bladder": ("bladder", "urine", "pee", "wee", "urgency", "empty bladder"),
    "fatigue": ("fatigue", "tired", "exhaust", "brain fog"),
    "fertility": ("fertility", "pregnan", "freeze my eggs", "egg freezing", "ivf", "children"),
    "nhs_waits": ("nhs", "waiting list", "waitlist", "gp", "referral"),
    "insurance": ("insurance", "axa", "bupa", "aviva", "self funding", "self-funding"),
    "pain": ("pain", "cramps", "aching", "stabbing", "pressure", "can’t walk", "cant walk"),
    "surgery": ("surgery", "laparoscopy", "excision", "ablation", "hysterectomy", "robotic"),
}

_INTENT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "find_specialist": ("find", "specialist", "recommend", "who do you", "where do you"),
    "second_opinion": ("second opinion", "another opinion", "sense-check", "compare"),
    "symptom_management": ("manage", "management", "pain relief", "support", "physio", "dietitian"),
    "surgery": ("excision", "surgery", "hysterectomy", "laparoscopy"),
}


def default_bank_paths(*, outdir: Path) -> PatientVoiceBankPaths:
    return PatientVoiceBankPaths(
        messages_jsonl=outdir / "patient_voice_messages.jsonl",
        snippets_jsonl=outdir / "patient_voice_snippets.jsonl",
    )


def _clean_text(s: str) -> str:
    t = (s or "").strip()
    if not t:
        return ""
    if t in _WHATSAPP_SYSTEM_EXACT:
        return ""
    t = _EDITED_SUFFIX_RE.sub("", t).strip()
    # normalise whitespace but keep intentional newlines
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    return t


def _iter_patient_messages(threads_jsonl: Path) -> Iterable[dict[str, Any]]:
    for raw in threads_jsonl.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s:
            continue
        obj = json.loads(s)
        for m in obj.get("messages", []) or []:
            if (m.get("role") or "").strip().lower() != "patient":
                continue
            txt = m.get("text")
            if not isinstance(txt, str):
                continue
            cleaned = _clean_text(txt)
            if not cleaned:
                continue
            yield {
                "thread_id": obj.get("thread_id"),
                "source": obj.get("source"),
                "source_path": obj.get("source_path"),
                "timestamp": m.get("timestamp"),
                "text": cleaned,
            }


def _topic_tags(text: str) -> list[str]:
    t = text.lower()
    tags: list[str] = []
    for topic, kws in _TOPIC_KEYWORDS.items():
        if any(kw in t for kw in kws):
            tags.append(topic)
    return tags


def _intent_tags(text: str) -> list[str]:
    t = text.lower()
    tags: list[str] = []
    for intent, kws in _INTENT_KEYWORDS.items():
        if any(kw in t for kw in kws):
            tags.append(intent)
    return tags


def _sentenceish_split(text: str) -> list[str]:
    """
    Rule-based splitter; we prefer not to add heavy NLP deps.
    Keeps 1–2 sentence snippets; merges tiny fragments.
    """
    t = re.sub(r"\s+", " ", text).strip()
    if not t:
        return []
    parts = re.split(r"(?<=[.!?])\s+", t)
    parts = [p.strip() for p in parts if p and p.strip()]
    # Merge very short fragments into previous
    merged: list[str] = []
    for p in parts:
        if merged and len(p) < 24:
            merged[-1] = f"{merged[-1]} {p}".strip()
        else:
            merged.append(p)
    return merged


def build_patient_voice_bank(
    *,
    in_threads_jsonl: Path,
    out_messages_jsonl: Path,
    out_snippets_jsonl: Path,
    overwrite: bool = True,
    min_message_chars: int = 12,
    min_snippet_chars: int = 28,
    max_snippet_chars: int = 260,
) -> PatientVoiceBankPaths:
    """
    Build patient voice banks from deduped, patient-only thread JSONL.
    Intended input: `threads.patient_only.dedup_clean.jsonl`.
    """
    out_messages_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_snippets_jsonl.parent.mkdir(parents=True, exist_ok=True)

    messages: list[dict[str, Any]] = []
    snippets: list[dict[str, Any]] = []

    snippet_i = 0
    for row in _iter_patient_messages(in_threads_jsonl):
        txt = row["text"]
        if len(txt) < min_message_chars:
            continue

        has_question = "?" in txt
        contains_numbers = bool(re.search(r"\d", txt))
        msg = dict(row)
        msg["char_len"] = len(txt)
        msg["has_question"] = has_question
        msg["contains_numbers"] = contains_numbers
        messages.append(msg)

        # Create snippets from sentence-like chunks
        sent = _sentenceish_split(txt)
        # Group as 1–2 sentence snippets
        buf: list[str] = []
        for s in sent:
            buf.append(s)
            if len(buf) >= 2 or sum(len(x) for x in buf) >= 170:
                snippet = " ".join(buf).strip()
                buf = []
                if min_snippet_chars <= len(snippet) <= max_snippet_chars:
                    snippet_i += 1
                    snippets.append(
                        {
                            "snippet_id": f"pv_{snippet_i:06d}",
                            "text": snippet,
                            "source_thread_id": row.get("thread_id"),
                            "source_message_timestamp": row.get("timestamp"),
                            "tags": {"topic": _topic_tags(snippet), "intent": _intent_tags(snippet)},
                        }
                    )
        if buf:
            snippet = " ".join(buf).strip()
            if min_snippet_chars <= len(snippet) <= max_snippet_chars:
                snippet_i += 1
                snippets.append(
                    {
                        "snippet_id": f"pv_{snippet_i:06d}",
                        "text": snippet,
                        "source_thread_id": row.get("thread_id"),
                        "source_message_timestamp": row.get("timestamp"),
                        "tags": {"topic": _topic_tags(snippet), "intent": _intent_tags(snippet)},
                    }
                )

    mode = "w" if overwrite else "a"
    with out_messages_jsonl.open(mode, encoding="utf-8") as f:
        for m in messages:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    with out_snippets_jsonl.open(mode, encoding="utf-8") as f:
        for s in snippets:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    return PatientVoiceBankPaths(messages_jsonl=out_messages_jsonl, snippets_jsonl=out_snippets_jsonl)


def sample_patient_voice_snippets(
    *,
    snippets_jsonl: Path = Path("exports/instagram/_dataset/patient_voice_snippets.jsonl"),
    max_snippets: int = 6,
    max_chars_per_snippet: int = 220,
) -> list[str]:
    """
    Returns short patient excerpts as *voice* references (cadence/word choice), not source claims.
    """
    if not snippets_jsonl.exists():
        return []
    try:
        lines = snippets_jsonl.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []

    out: list[str] = []
    for raw in lines:
        if len(out) >= max_snippets:
            break
        s = raw.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except Exception:
            continue
        txt = (obj.get("text") or "").strip()
        if not txt:
            continue
        # Avoid ultra-low-signal one-liners as prompt references.
        if len(txt) < 45:
            continue
        txt = txt.replace("\n", " ").strip()
        out.append(txt[:max_chars_per_snippet].strip())
    return out

