from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .schemas import Chunk, SourceLocation

ChunkStrategyName = Literal["auto", "paragraph", "markdown_headings", "transcript"]


@dataclass(frozen=True)
class ChunkingConfig:
    """
    - paragraph: split on blank lines, pack up to max_chars (legacy behavior).
    - markdown_headings: split on # headings; carry section_path on each chunk.
    - transcript: larger default windows; prefers not to split speaker-labelled turns.
    - auto: choose from file path + ingest type (see resolve_chunk_strategy).
    """

    max_chars: int | None = None
    overlap_chars: int | None = None
    strategy: ChunkStrategyName = "auto"

    # When strategy is transcript and max_chars is None, use this cap.
    transcript_default_max_chars: int = 2800
    transcript_default_overlap: int = 220

    # paragraph / markdown defaults when max_chars is None
    paragraph_default_max_chars: int = 1200
    paragraph_default_overlap: int = 150


def resolve_chunk_strategy(
    strategy: ChunkStrategyName,
    *,
    ingest_type: str,
    source_path: Path | None,
) -> Literal["paragraph", "markdown_headings", "transcript"]:
    if strategy != "auto":
        return strategy  # type: ignore[return-value]
    suf = (source_path.suffix.lower() if source_path else "") or ""
    if suf == ".md" or ingest_type == "md":
        return "markdown_headings"
    if ingest_type == "transcript":
        return "transcript"
    # Plain .txt etc.: treat obvious podcast/transcript filenames as transcript-style windows
    if source_path:
        stem = source_path.stem.lower()
        for hint in ("transcript", "podcast", "episode", "show-notes"):
            if hint in stem:
                return "transcript"
    return "paragraph"


def _effective_limits(
    config: ChunkingConfig,
    resolved: Literal["paragraph", "markdown_headings", "transcript"],
) -> tuple[int, int]:
    if resolved == "transcript":
        mc = config.max_chars if config.max_chars is not None else config.transcript_default_max_chars
        oc = config.overlap_chars if config.overlap_chars is not None else config.transcript_default_overlap
        return mc, oc
    mc = config.max_chars if config.max_chars is not None else config.paragraph_default_max_chars
    oc = config.overlap_chars if config.overlap_chars is not None else config.paragraph_default_overlap
    return mc, oc


def chunk_document(
    text: str,
    *,
    config: ChunkingConfig,
    ingest_type: str = "txt",
    source_path: Path | None = None,
) -> list[Chunk]:
    resolved = resolve_chunk_strategy(config.strategy, ingest_type=ingest_type, source_path=source_path)
    max_chars, overlap_chars = _effective_limits(config, resolved)

    inner = ChunkingConfig(
        max_chars=max_chars,
        overlap_chars=overlap_chars,
        strategy=resolved,
        transcript_default_max_chars=config.transcript_default_max_chars,
        transcript_default_overlap=config.transcript_default_overlap,
        paragraph_default_max_chars=config.paragraph_default_max_chars,
        paragraph_default_overlap=config.paragraph_default_overlap,
    )

    if resolved == "markdown_headings":
        return _chunk_markdown_headings(text, config=inner)
    if resolved == "transcript":
        return _chunk_transcript(text, config=inner)
    return _chunk_paragraph(text, config=inner)


def chunk_text(text: str, *, config: ChunkingConfig) -> list[Chunk]:
    """Backward-compatible alias: fixed paragraph strategy, explicit limits in config."""
    mc = config.max_chars if config.max_chars is not None else config.paragraph_default_max_chars
    oc = config.overlap_chars if config.overlap_chars is not None else config.paragraph_default_overlap
    inner = ChunkingConfig(
        max_chars=mc,
        overlap_chars=oc,
        strategy="paragraph",
        transcript_default_max_chars=config.transcript_default_max_chars,
        transcript_default_overlap=config.transcript_default_overlap,
        paragraph_default_max_chars=config.paragraph_default_max_chars,
        paragraph_default_overlap=config.paragraph_default_overlap,
    )
    return _chunk_paragraph(text, config=inner)


# --- Paragraph (legacy) ---

# Line-start speaker labels: "Sarah: ...", "HOST: ...", "Dr Lee: ..."
_SPEAKER_LINE = re.compile(
    r"^\s*(?P<label>[A-Za-z][A-Za-z0-9 ,.'-]{0,48}):\s*(?P<body>.*)$"
)


def _chunk_paragraph(text: str, *, config: ChunkingConfig) -> list[Chunk]:
    max_chars = config.max_chars or config.paragraph_default_max_chars
    overlap_chars = config.overlap_chars or config.paragraph_default_overlap
    normalized = text.replace("\r\n", "\n")
    paras = _split_paragraphs_with_offsets(normalized)

    chunks: list[Chunk] = []
    cur_parts: list[str] = []
    cur_start: int | None = None
    cur_end: int | None = None

    def flush() -> None:
        nonlocal cur_parts, cur_start, cur_end
        if not cur_parts or cur_start is None or cur_end is None:
            cur_parts = []
            cur_start = None
            cur_end = None
            return
        joined = "\n\n".join(cur_parts).strip()
        if not joined:
            cur_parts = []
            cur_start = None
            cur_end = None
            return
        chunk_id = f"c{len(chunks)+1:03d}"
        line_start, line_end = _char_range_to_line_range(normalized, cur_start, cur_end)
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                text=joined,
                source_location=SourceLocation(line_start=line_start, line_end=line_end),
                section_path=[],
                char_start=cur_start,
                char_end=cur_end,
            )
        )
        if overlap_chars > 0:
            overlap_start = max(0, cur_end - overlap_chars)
            overlap_text = normalized[overlap_start:cur_end].strip()
            cur_parts = [overlap_text] if overlap_text else []
            cur_start = overlap_start if overlap_text else None
            cur_end = cur_end if overlap_text else None
        else:
            cur_parts = []
            cur_start = None
            cur_end = None

    for para_text, para_start, para_end in paras:
        if cur_start is None:
            cur_start = para_start
            cur_end = para_end
            cur_parts = [para_text]
            continue

        tentative = ("\n\n".join(cur_parts) + "\n\n" + para_text).strip()
        if len(tentative) <= max_chars:
            cur_parts.append(para_text)
            cur_end = para_end
        else:
            flush()
            if cur_start is None:
                cur_start = para_start
                cur_parts = [para_text]
            else:
                cur_parts.append(para_text)
            cur_end = para_end

    flush()
    return chunks


def _split_paragraphs_with_offsets(text: str) -> list[tuple[str, int, int]]:
    paras: list[tuple[str, int, int]] = []
    n = len(text)
    i = 0
    while i < n:
        while i < n and text[i] == "\n":
            i += 1
        if i >= n:
            break
        start = i
        while i < n:
            if text[i] == "\n" and i + 1 < n and text[i + 1] == "\n":
                break
            i += 1
        end = i
        para = text[start:end].strip()
        if para:
            paras.append((para, start, end))
        while i < n and text[i] == "\n":
            i += 1
    return paras


def _char_range_to_line_range(text: str, start: int, end: int) -> tuple[int, int]:
    pre = text[:start]
    mid = text[start:end]
    line_start = pre.count("\n") + 1
    line_end = line_start + mid.count("\n")
    return line_start, line_end


# --- Markdown: section boundaries at headings ---

def _markdown_heading_spans(text: str) -> list[tuple[int, int, int, str]]:
    """Returns list of (char_start, char_end, level, title) for each heading line."""
    out: list[tuple[int, int, int, str]] = []
    for m in re.finditer(r"(?m)^(#{1,6})\s+(.+?)\s*$", text):
        level = len(m.group(1))
        title = m.group(2).strip()
        out.append((m.start(), m.end(), level, title))
    return out


def _chunk_markdown_headings(text: str, *, config: ChunkingConfig) -> list[Chunk]:
    normalized = text.replace("\r\n", "\n")

    spans = _markdown_heading_spans(normalized)
    if not spans:
        return _chunk_paragraph(normalized, config=config)

    sections: list[tuple[list[str], str, int, int]] = []
    stack: list[tuple[int, str]] = []

    # Preamble before first heading (keep raw slice so char offsets stay aligned)
    first = spans[0][0]
    if first > 0 and normalized[:first].strip():
        pre = normalized[:first]
        sections.append(([], pre, 0, first))

    for idx, (h_start, h_end, level, title) in enumerate(spans):
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        path = [t for _, t in stack]

        end_pos = spans[idx + 1][0] if idx + 1 < len(spans) else len(normalized)
        block = normalized[h_start:end_pos]
        if not block.strip():
            continue
        sections.append((path, block, h_start, end_pos))

    chunks: list[Chunk] = []
    for path, block, abs_start, _abs_end in sections:
        sub = _chunk_paragraph(block, config=config)
        for c in sub:
            chunks.append(
                Chunk(
                    chunk_id=f"c{len(chunks)+1:03d}",
                    text=c.text,
                    source_location=SourceLocation(
                        line_start=_line_at(normalized, abs_start + c.char_start),
                        line_end=_line_at(normalized, abs_start + c.char_end),
                    ),
                    section_path=path,
                    char_start=abs_start + c.char_start,
                    char_end=abs_start + c.char_end,
                )
            )

    return _reoverlap_chunk_sequence(chunks, normalized)


def _line_at(text: str, char_idx: int) -> int:
    return text[:char_idx].count("\n") + 1


def _reoverlap_chunk_sequence(chunks: list[Chunk], _full_text: str) -> list[Chunk]:
    """Assign sequential chunk ids (overlap is already applied inside per-section paragraph chunking)."""
    return [
        Chunk(
            chunk_id=f"c{i+1:03d}",
            text=c.text,
            source_location=c.source_location,
            section_path=c.section_path,
            char_start=c.char_start,
            char_end=c.char_end,
        )
        for i, c in enumerate(chunks)
    ]


# --- Transcript: optional speaker turns + larger windows ---

def _split_transcript_turns(text: str) -> list[tuple[str | None, str, int, int]]:
    """
    If lines look like 'Name: dialogue', split into turns. Otherwise single stream.
    Returns (speaker_label_or_none, turn_text, start, end) in original string coordinates.
    """
    normalized = text.replace("\r\n", "\n")
    lines = normalized.split("\n")
    turns: list[tuple[str | None, str, int, int]] = []
    # Build char offset for each line start
    offsets: list[int] = []
    pos = 0
    for ln in lines:
        offsets.append(pos)
        pos += len(ln) + 1  # + newline

    speaker_hits = sum(1 for ln in lines if _SPEAKER_LINE.match(ln))
    # Avoid false positives on isolated "Note:" lines — need a dialogue-like pattern
    if speaker_hits < 3:
        return [(None, normalized, 0, len(normalized))]

    i = 0
    while i < len(lines):
        line = lines[i]
        m = _SPEAKER_LINE.match(line)
        start_char = offsets[i]
        if m:
            label = m.group("label")
            body_parts = [m.group("body")]
            j = i + 1
            while j < len(lines) and not _SPEAKER_LINE.match(lines[j]):
                body_parts.append(lines[j])
                j += 1
            block = "\n".join(body_parts).strip()
            end_char = offsets[j - 1] + len(lines[j - 1]) if j > i else start_char + len(line)
            turns.append((label, f"{label}: {block}" if block else f"{label}:", start_char, end_char))
            i = j
        else:
            # Preamble or non-speaker line: merge until next speaker
            j = i
            body_lines = []
            while j < len(lines) and not _SPEAKER_LINE.match(lines[j]):
                body_lines.append(lines[j])
                j += 1
            block = "\n".join(body_lines).strip()
            if block:
                start_c = offsets[i]
                end_c = offsets[j - 1] + len(lines[j - 1])
                turns.append((None, block, start_c, end_c))
            i = j

    return turns if turns else [(None, normalized, 0, len(normalized))]


def _chunk_transcript(text: str, *, config: ChunkingConfig) -> list[Chunk]:
    doc = text.replace("\r\n", "\n")
    max_chars = config.max_chars or config.transcript_default_max_chars
    overlap_chars = config.overlap_chars or config.transcript_default_overlap
    inner = ChunkingConfig(
        max_chars=max_chars,
        overlap_chars=overlap_chars,
        strategy="paragraph",
        transcript_default_max_chars=config.transcript_default_max_chars,
        transcript_default_overlap=config.transcript_default_overlap,
        paragraph_default_max_chars=config.paragraph_default_max_chars,
        paragraph_default_overlap=config.paragraph_default_overlap,
    )

    turns = _split_transcript_turns(doc)
    if len(turns) == 1 and turns[0][0] is None:
        return _chunk_paragraph(turns[0][1], config=inner)

    chunks: list[Chunk] = []
    cur_parts: list[str] = []
    cur_start: int | None = None
    cur_end: int | None = None

    def flush() -> None:
        nonlocal cur_parts, cur_start, cur_end
        if not cur_parts or cur_start is None or cur_end is None:
            cur_parts = []
            cur_start = None
            cur_end = None
            return
        joined = "\n\n".join(cur_parts).strip()
        if not joined:
            cur_parts = []
            cur_start = None
            cur_end = None
            return
        chunk_id = f"c{len(chunks)+1:03d}"
        line_start, line_end = _char_range_to_line_range(doc, cur_start, cur_end)
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                text=joined,
                source_location=SourceLocation(line_start=line_start, line_end=line_end),
                section_path=[],
                char_start=cur_start,
                char_end=cur_end,
            )
        )
        if overlap_chars > 0:
            overlap_start = max(0, cur_end - overlap_chars)
            overlap_text = doc[overlap_start:cur_end].strip()
            cur_parts = [overlap_text] if overlap_text else []
            cur_start = overlap_start if overlap_text else None
            cur_end = cur_end if overlap_text else None
        else:
            cur_parts = []
            cur_start = None
            cur_end = None

    for _label, turn_text, t_start, t_end in turns:
        if cur_start is None:
            cur_start = t_start
            cur_end = t_end
            cur_parts = [turn_text]
            continue
        tentative = ("\n\n".join(cur_parts) + "\n\n" + turn_text).strip()
        if len(tentative) <= max_chars:
            cur_parts.append(turn_text)
            cur_end = t_end
        else:
            flush()
            if cur_start is None:
                cur_start = t_start
                cur_parts = [turn_text]
            else:
                cur_parts.append(turn_text)
            cur_end = t_end

    flush()

    # If speaker packing produced nothing (edge), fall back
    if not chunks:
        return _chunk_paragraph(doc, config=inner)

    # Re-number ids
    return [
        Chunk(
            chunk_id=f"c{i+1:03d}",
            text=c.text,
            source_location=c.source_location,
            section_path=c.section_path,
            char_start=c.char_start,
            char_end=c.char_end,
        )
        for i, c in enumerate(chunks)
    ]
