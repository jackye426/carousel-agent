from __future__ import annotations

from dataclasses import asdict
from difflib import SequenceMatcher
from typing import Any

from .schemas import CandidateIdea, RunState, Weights


class ValidationError(RuntimeError):
    pass


def _norm_ws(s: str) -> str:
    # Collapse all whitespace for robust "verbatim enough" matching across line wraps.
    return " ".join(s.split())


def repair_citation_chunk_ids(state: RunState) -> int:
    """
    If an excerpt is present in the document but not in the cited chunk, re-point the citation to the
    first chunk that contains the excerpt (whitespace-normalised match).

    Returns the number of repaired citations.
    """
    chunks = state.chunks
    norm_chunk_text = {c.chunk_id: _norm_ws(c.text) for c in chunks}

    repaired = 0
    for idea in state.candidates:
        for cit in idea.source_citations:
            ex = cit.excerpt.strip()
            if not ex or ex.startswith("TODO:"):
                continue
            cited_ok = ex in (state_chunk := (next((c.text for c in chunks if c.chunk_id == cit.chunk_id), ""))) or (
                _norm_ws(ex) in norm_chunk_text.get(cit.chunk_id, "")
            )
            if cited_ok:
                continue

            nex = _norm_ws(ex)
            for chunk_id, ntext in norm_chunk_text.items():
                if nex and nex in ntext:
                    cit.chunk_id = chunk_id
                    repaired += 1
                    break
    return repaired


def repair_citation_excerpts(state: RunState, *, min_ratio: float = 0.82) -> int:
    """
    If a citation excerpt does not appear in the cited chunk, attempt to replace it with a best-match
    substring from that chunk (whitespace-normalised), so provenance remains anchored to source text.

    This is a pragmatic fallback for models that paraphrase inside `excerpt`.
    Returns number of repaired excerpts.
    """
    repaired = 0
    chunks_by_id = {c.chunk_id: c for c in state.chunks}

    for idea in state.candidates:
        for cit in idea.source_citations:
            ex = cit.excerpt.strip()
            if not ex or ex.startswith("TODO:"):
                continue
            chunk = chunks_by_id.get(cit.chunk_id)
            if not chunk:
                continue
            if _excerpt_in_chunk(ex, chunk.text):
                continue

            best = _best_match_substring(ex, chunk.text)
            if best and best[1] >= min_ratio:
                cit.excerpt = best[0]
                # keep original intent but note repair
                cit.note = (cit.note or "").strip()
                cit.note = (cit.note + " " if cit.note else "") + "[excerpt_auto_repaired]"
                repaired += 1

    return repaired


def _best_match_substring(excerpt: str, chunk_text: str) -> tuple[str, float] | None:
    needle = _norm_ws(excerpt)
    hay = _norm_ws(chunk_text)
    if not needle or not hay:
        return None

    # Search windows near the needle length (cap for performance).
    n = len(needle)
    n = min(max(n, 40), 260)  # avoid very short/very long windows
    step = max(10, n // 6)

    best_sub = ""
    best_ratio = 0.0
    for start in range(0, max(1, len(hay) - n), step):
        sub = hay[start : start + n]
        ratio = SequenceMatcher(None, needle, sub).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_sub = sub
    if best_sub:
        return best_sub, best_ratio
    return None


def validate_citations_verbatim(state: RunState, *, strict: bool = True) -> list[str]:
    chunk_by_id = {c.chunk_id: c for c in state.chunks}
    issues: list[str] = []
    for idea in state.candidates:
        if len(idea.source_citations) < 2:
            msg = f"{idea.idea_id} has <2 citations."
            if strict:
                raise ValidationError(msg)
            issues.append(msg)
        for cit in idea.source_citations:
            chunk = chunk_by_id.get(cit.chunk_id)
            if not chunk:
                msg = f"{idea.idea_id} cites missing chunk_id {cit.chunk_id}."
                if strict:
                    raise ValidationError(msg)
                issues.append(msg)
                continue
            ex = cit.excerpt.strip()
            if not ex or ex.startswith("TODO:"):
                continue
            if _excerpt_in_chunk(ex, chunk.text) is False:
                msg = f"{idea.idea_id} excerpt not found in {cit.chunk_id}. Excerpt starts: {cit.excerpt[:80]!r}"
                if strict:
                    raise ValidationError(msg)
                issues.append(msg)
                # mark for downstream scoring/QA
                if "citation_not_verbatim" not in idea.safety_flags:
                    idea.safety_flags.append("citation_not_verbatim")
    return issues


def _excerpt_in_chunk(excerpt: str, chunk_text: str) -> bool:
    """
    True if excerpt appears in chunk text. Uses whitespace-normalised match.
    Also supports legacy excerpts containing "..." by matching all parts in order.
    """
    if excerpt in chunk_text:
        return True
    if _norm_ws(excerpt) in _norm_ws(chunk_text):
        return True

    if "..." not in excerpt:
        return False

    # Handle non-verbatim ellipsis by verifying ordered parts exist.
    parts = [p.strip() for p in excerpt.split("...") if p.strip()]
    if not parts:
        return False
    hay = _norm_ws(chunk_text)
    idx = 0
    for part in parts:
        needle = _norm_ws(part)
        pos = hay.find(needle, idx)
        if pos < 0:
            return False
        idx = pos + len(needle)
    return True


def compute_weighted_totals_and_rank(state: RunState, weights: Weights) -> None:
    w = asdict(weights)
    denom = sum(w.values())

    def total(idea) -> float:
        s = idea.scores
        num = 0.0
        for k, weight in w.items():
            v = getattr(s, k, None)
            if v is None:
                continue
            num += float(v) * float(weight)
        return (num / denom) * 100.0

    for idea in state.candidates:
        idea.scores.total_weighted = round(total(idea), 1)

    # Rank: total desc, then source_support, hook_strength, then stable tie-breakers
    # (content_pillar + idea_id) so ordering is deterministic when scores match.
    def key(idea):
        s = idea.scores
        return (
            -(s.total_weighted or 0.0),
            -(s.source_support or 0),
            -(s.hook_strength or 0),
            str(idea.content_pillar or ""),
            str(idea.idea_id or ""),
        )

    state.candidates.sort(key=key)
    for idx, idea in enumerate(state.candidates):
        idea.rank = idx + 1


def finalize_shortlist_ids(
    selected_ids_raw: list[Any],
    *,
    pool: list[CandidateIdea],
    select_n: int,
) -> list[str]:
    """
    Keep only IDs in `pool` (order = best rank first). Prefer the LLM's order for valid IDs,
    then fill deterministically by pool order until `select_n` (capped by pool size).
    """
    allowed = {c.idea_id for c in pool}
    pool_order = [c.idea_id for c in pool]
    n = max(0, min(int(select_n), len(pool_order)))

    out: list[str] = []
    seen: set[str] = set()
    for raw in selected_ids_raw:
        iid = str(raw).strip()
        if iid in allowed and iid not in seen:
            out.append(iid)
            seen.add(iid)
        if len(out) >= n:
            return out[:n]
    for iid in pool_order:
        if len(out) >= n:
            break
        if iid not in seen:
            out.append(iid)
            seen.add(iid)
    return out[:n]


def enforce_selection_gate(state: RunState) -> None:
    if not state.shortlist.selection_frozen_at or not state.shortlist.selected_idea_ids:
        raise ValidationError("Selection gate not frozen: shortlist.selection_frozen_at or selected_idea_ids missing.")

    selected_set = set(state.shortlist.selected_idea_ids)
    for idea in state.candidates:
        if idea.selected and idea.idea_id not in selected_set:
            raise ValidationError(f"{idea.idea_id} marked selected=true but not present in shortlist.")
        if (idea.idea_id in selected_set) and not idea.selected:
            # normalize to true
            idea.selected = True

