"""
Single injection surface for Writer-stage context (patient voice, OCR structure, style).

Caps are enforced here so prompts do not grow without bound. Set CONTEXT_MINIMAL=1 to omit
non-essential examples (A/B or cost saving).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Sequence

from .ocr_carousel_bank import sample_ocr_carousel_structures
from .ocr_style import sample_ocr_style_snippets
from .patient_voice import sample_patient_voice_snippets
from .schemas import CandidateIdea, Chunk


def _writer_evidence_max_chunk_chars() -> int | None:
    raw = (os.environ.get("WRITER_EVIDENCE_MAX_CHUNK_CHARS") or "").strip()
    if not raw:
        return 16_000
    if raw.lower() in ("0", "none", "off"):
        return None
    try:
        return max(1_000, int(raw))
    except ValueError:
        return 16_000


def collect_writer_chunk_ids(idea: CandidateIdea) -> list[str]:
    """
    Chunk IDs referenced by this idea, in stable order: citations first, then outline slide citations.
    Deduplicated.
    """
    ordered: list[str] = []
    seen: set[str] = set()
    for cit in idea.source_citations:
        cid = (cit.chunk_id or "").strip()
        if cid and cid not in seen:
            seen.add(cid)
            ordered.append(cid)
    for slide in idea.outline:
        for raw in slide.citations:
            cid = str(raw).strip()
            if cid and cid not in seen:
                seen.add(cid)
                ordered.append(cid)
    return ordered


def build_writer_source_evidence(chunks: Sequence[Chunk], idea: CandidateIdea) -> str:
    """
    Full chunk text for every chunk cited by the idea (deduped). Used to ground Writer / QA prompts.
    """
    by_id = {c.chunk_id: c for c in chunks}
    max_chars = _writer_evidence_max_chunk_chars()
    parts: list[str] = []
    for cid in collect_writer_chunk_ids(idea):
        ch = by_id.get(cid)
        if ch is None:
            parts.append(f"--- chunk_id: {cid} ---\n(Missing: chunk not found in run state.)\n")
            continue
        text = ch.text
        if max_chars is not None and len(text) > max_chars:
            text = text[:max_chars] + "\n[... truncated by WRITER_EVIDENCE_MAX_CHUNK_CHARS ...]"
        parts.append(f"--- chunk_id: {cid} ---\n{text}\n")
    return "\n".join(parts).strip()


def _minimal() -> bool:
    v = (os.environ.get("CONTEXT_MINIMAL") or "").strip().lower()
    return v in ("1", "true", "yes", "on")


@dataclass
class WriterContextPack:
    """Structured context merged into idea dicts for hook/slide/CTA prompts."""

    patient_voice_snippets: list[str] = field(default_factory=list)
    ocr_structure_examples: list[str] = field(default_factory=list)

    def apply_to_idea_dict(self, idea: dict[str, Any]) -> dict[str, Any]:
        out = dict(idea)
        if self.patient_voice_snippets:
            out["patient_voice_snippets"] = self.patient_voice_snippets
        if self.ocr_structure_examples:
            out["ocr_structure_examples"] = self.ocr_structure_examples
        return out


def build_writer_context_pack_for_hooks_and_slides() -> WriterContextPack:
    """
    Voice + structure references for hook generation and slide drafting.

    Caps (defaults): align with sample_* module defaults; minimal mode clears both lists.
    """
    if _minimal():
        return WriterContextPack()
    return WriterContextPack(
        patient_voice_snippets=sample_patient_voice_snippets(
            max_snippets=6,
            max_chars_per_snippet=220,
        ),
        ocr_structure_examples=sample_ocr_carousel_structures(
            max_examples=4,
            max_chars_per_example=500,
        ),
    )


def build_writer_context_pack_for_cta_only() -> WriterContextPack:
    """CTA stage uses patient voice only (matches prior pipeline behaviour)."""
    if _minimal():
        return WriterContextPack()
    return WriterContextPack(
        patient_voice_snippets=sample_patient_voice_snippets(
            max_snippets=6,
            max_chars_per_snippet=220,
        ),
    )


def sample_ocr_style_for_hook_policy() -> list[str]:
    """Hook policy rewrite loop; keep separate from structure pack."""
    if _minimal():
        return sample_ocr_style_snippets(max_snippets=3, max_chars_per_snippet=220)
    return sample_ocr_style_snippets(max_snippets=6, max_chars_per_snippet=220)
