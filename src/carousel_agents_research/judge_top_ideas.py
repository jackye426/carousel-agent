from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from carousel_agents.chunking import ChunkingConfig, chunk_text
from carousel_agents.ingest import load_text
from carousel_agents.llm import OpenAICompatibleClient
from carousel_agents.llm_response_models import ExtractCandidatesResponse, ScoreCandidatesResponse
from carousel_agents.prompts import system_ideation, user_extract_candidates, user_score_candidates
from carousel_agents.schemas import CandidateIdea, CitationIn, DocumentMeta, RunState, Weights
from carousel_agents.validation import (
    compute_weighted_totals_and_rank,
    repair_citation_chunk_ids,
    repair_citation_excerpts,
    validate_citations_verbatim,
)
from rich import print as rprint


def _coerce_candidate(d: dict[str, Any]) -> dict[str, Any]:
    out = dict(d)
    citations = out.get("source_citations") or []
    fixed: list[dict[str, Any]] = []
    for c in citations:
        if not isinstance(c, dict):
            continue
        ci = CitationIn(**c).to_citation()
        fixed.append(ci.model_dump())
    out["source_citations"] = fixed
    if out.get("safety_flags") is None:
        out["safety_flags"] = []
    elif not isinstance(out["safety_flags"], list):
        out["safety_flags"] = [str(out["safety_flags"])]
    return out


def _idea_summary(c: CandidateIdea) -> dict[str, Any]:
    return {
        "idea_id": c.idea_id,
        "rank": c.rank,
        "content_pillar": c.content_pillar,
        "topic": c.topic,
        "angle": c.angle,
        "core_claim": c.core_claim,
        "audience_pain": c.audience_pain,
        "promise": c.promise,
        "format_suggestion": c.format_suggestion,
        "total_weighted": c.scores.total_weighted,
        "scores": {
            "hook_strength": c.scores.hook_strength,
            "clarity": c.scores.clarity,
            "save_share": c.scores.save_share,
            "strategic_fit": c.scores.strategic_fit,
            "source_support": c.scores.source_support,
        },
    }


def export_top_ideas_per_model(
    *,
    input_path: Path,
    models: list[str],
    top_n: int = 3,
    candidate_count: int = 30,
    temperature: float = 0.1,
    timeout_s: float = 360.0,
    max_chunks: int | None = None,
    out_dir: Path = Path("outputs/experiments"),
) -> Path:
    text, detected_type = load_text(input_path)
    chunks = chunk_text(text, config=ChunkingConfig())
    if max_chunks is not None and max_chunks > 0:
        chunks = chunks[: max_chunks]

    client = OpenAICompatibleClient()
    weights = Weights()
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = out_dir / f"judge_top_ideas_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    by_model: dict[str, Any] = {}

    for model in models:
        rprint(f"[bold]judge-top-ideas[/bold] {model}: extract + score…")
        state = RunState(
            document=DocumentMeta(
                document_id=f"top_{ts}",
                title=input_path.stem,
                type=detected_type,
                language="en",
            ),
            chunks=chunks,
        )

        extract_out = client.chat_structured(
            response_model=ExtractCandidatesResponse,
            model=model,
            temperature=temperature,
            system=system_ideation(),
            user=user_extract_candidates(
                chunks=[c.model_dump() for c in state.chunks],
                candidate_count=candidate_count,
                audience=state.audience.model_dump(),
            ),
            timeout_s=timeout_s,
        )
        candidates_raw = [item.model_dump() for item in extract_out.candidates]
        state.candidates = []
        for idx, i in enumerate(candidates_raw):
            if not isinstance(i, dict):
                continue
            d = dict(i)
            if not d.get("idea_id"):
                d["idea_id"] = f"i{idx+1:03d}"
            state.candidates.append(CandidateIdea(**_coerce_candidate(d)))

        repair_citation_chunk_ids(state)
        repair_citation_excerpts(state, min_ratio=0.75)
        validate_citations_verbatim(state, strict=False)

        score_out = client.chat_structured(
            response_model=ScoreCandidatesResponse,
            model=model,
            temperature=temperature,
            system=system_ideation(),
            user=user_score_candidates(
                candidates=[c.model_dump() for c in state.candidates],
                weights=weights.__dict__,
                audience=state.audience.model_dump(),
            ),
            timeout_s=timeout_s,
        )
        by_id: dict[str, dict[str, Any]] = {row.idea_id: row.model_dump() for row in score_out.scores if row.idea_id}
        for c in state.candidates:
            s = by_id.get(c.idea_id)
            if not s:
                continue
            for k, v in s.items():
                if k == "idea_id":
                    continue
                if hasattr(c.scores, k):
                    try:
                        setattr(c.scores, k, int(v))
                    except Exception:
                        continue

        compute_weighted_totals_and_rank(state, weights)
        top = state.candidates[:top_n]
        by_model[model] = {"top_ideas": [_idea_summary(x) for x in top]}

    payload = {
        "input_path": str(input_path),
        "chunk_count": len(chunks),
        "candidate_count": candidate_count,
        "top_n": top_n,
        "temperature": temperature,
        "models": models,
        "by_model": by_model,
    }
    json_path = run_dir / "top_ideas.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    md_lines: list[str] = [
        "## Top ideas by ideation model",
        "",
        f"- **Source**: `{input_path}`",
        f"- **Chunks used**: {len(chunks)}",
        f"- **Candidate target**: {candidate_count} · **Top N**: {top_n}",
        "",
        "Rank uses the pipeline weighted total after scoring (same rubric as production).",
        "",
    ]
    for model, block in by_model.items():
        md_lines.append(f"### {model}")
        md_lines.append("")
        for idea in block["top_ideas"]:
            md_lines.append(f"**{idea['rank']}. {idea['idea_id']}** — {idea['topic']}")
            md_lines.append(f"- Pillar: `{idea['content_pillar']}` · Format: `{idea['format_suggestion']}`")
            md_lines.append(f"- **Claim**: {idea['core_claim']}")
            md_lines.append(f"- **Weighted total**: {idea['total_weighted']}")
            md_lines.append("")
        md_lines.append("")

    md_path = run_dir / "top_ideas.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    return run_dir
