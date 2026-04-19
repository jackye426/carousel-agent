from __future__ import annotations

import json
import time
from dataclasses import dataclass
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


@dataclass(frozen=True)
class RunResult:
    model: str
    run_idx: int
    elapsed_s: float
    ok: bool
    error: str | None
    selected_idea_ids: list[str]
    top5_idea_ids: list[str]


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def _format_md_table(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    header = rows[0]
    body = rows[1:]
    out = []
    out.append("| " + " | ".join(header) + " |")
    out.append("| " + " | ".join(["---"] * len(header)) + " |")
    for r in body:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def benchmark_judges(
    *,
    input_path: Path,
    models: list[str],
    runs_per_model: int = 2,
    candidate_count: int = 30,
    select_n: int = 5,
    select_pool_k: int = 10,
    temperature: float = 0.1,
    timeout_s: float = 300.0,
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
    run_dir = out_dir / f"judge_benchmark_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    results: list[RunResult] = []

    def coerce_candidate(d: dict[str, Any]) -> dict[str, Any]:
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

    for model in models:
        for run_idx in range(1, runs_per_model + 1):
            t0 = time.time()
            try:
                state = RunState(
                    document=DocumentMeta(
                        document_id=f"bench_{ts}",
                        title=input_path.stem,
                        type=detected_type,
                        language="en",
                    ),
                    chunks=chunks,
                )

                rprint(f"[dim]judge-benchmark[/dim] {model} run {run_idx}: extract candidates…")
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
                    state.candidates.append(CandidateIdea(**coerce_candidate(d)))

                repair_citation_chunk_ids(state)
                repair_citation_excerpts(state, min_ratio=0.75)
                validate_citations_verbatim(state, strict=False)

                rprint(f"[dim]judge-benchmark[/dim] {model} run {run_idx}: score candidates…")
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
                pool_k = max(1, min(int(select_pool_k), len(state.candidates)))
                selection_pool = state.candidates[:pool_k]
                n_pick = min(select_n, len(selection_pool))
                selected_ids = [c.idea_id for c in selection_pool[:n_pick]]
                top5 = [c.idea_id for c in sorted(state.candidates, key=lambda x: (x.rank or 10**9))[:5]]

                elapsed_s = time.time() - t0
                results.append(
                    RunResult(
                        model=model,
                        run_idx=run_idx,
                        elapsed_s=elapsed_s,
                        ok=True,
                        error=None,
                        selected_idea_ids=selected_ids,
                        top5_idea_ids=top5,
                    )
                )
            except Exception as e:
                elapsed_s = time.time() - t0
                results.append(
                    RunResult(
                        model=model,
                        run_idx=run_idx,
                        elapsed_s=elapsed_s,
                        ok=False,
                        error=f"{type(e).__name__}: {e}",
                        selected_idea_ids=[],
                        top5_idea_ids=[],
                    )
                )

    by_model: dict[str, list[RunResult]] = {}
    for r in results:
        by_model.setdefault(r.model, []).append(r)

    model_summaries: dict[str, Any] = {}
    for model, runs in by_model.items():
        oks = [r for r in runs if r.ok]
        errs = [r for r in runs if not r.ok]
        avg_s = sum(r.elapsed_s for r in runs) / max(1, len(runs))
        stab = None
        if len(oks) >= 2:
            js: list[float] = []
            for i in range(len(oks)):
                for j in range(i + 1, len(oks)):
                    js.append(_jaccard(set(oks[i].selected_idea_ids), set(oks[j].selected_idea_ids)))
            stab = sum(js) / max(1, len(js))

        model_summaries[model] = {
            "runs": len(runs),
            "ok": len(oks),
            "errors": len(errs),
            "avg_elapsed_s": round(avg_s, 2),
            "stability_selected_jaccard": None if stab is None else round(stab, 3),
        }

    first_ok: dict[str, RunResult] = {}
    for model, runs in by_model.items():
        for r in runs:
            if r.ok:
                first_ok[model] = r
                break

    pairwise: list[dict[str, Any]] = []
    ms = list(first_ok.keys())
    for i in range(len(ms)):
        for j in range(i + 1, len(ms)):
            a = first_ok[ms[i]]
            b = first_ok[ms[j]]
            pairwise.append(
                {
                    "model_a": a.model,
                    "model_b": b.model,
                    "selected_jaccard": round(_jaccard(set(a.selected_idea_ids), set(b.selected_idea_ids)), 3),
                    "top5_jaccard": round(_jaccard(set(a.top5_idea_ids), set(b.top5_idea_ids)), 3),
                }
            )

    out_json = run_dir / "results.json"
    out_json.write_text(
        json.dumps(
            {
                "input_path": str(input_path),
                "candidate_count": candidate_count,
                "select_n": select_n,
                "temperature": temperature,
                "timeout_s": timeout_s,
                "max_chunks": max_chunks,
                "chunk_count": len(chunks),
                "models": models,
                "runs_per_model": runs_per_model,
                "model_summaries": model_summaries,
                "pairwise_agreement_first_ok": pairwise,
                "runs": [r.__dict__ for r in results],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    md_lines: list[str] = []
    md_lines.append("## Ideation benchmark\n")
    md_lines.append(f"- **Input**: `{input_path}`")
    md_lines.append(f"- **Candidate count**: {candidate_count}")
    md_lines.append(f"- **Select n**: {select_n}")
    md_lines.append(f"- **Runs/model**: {runs_per_model}")
    md_lines.append(f"- **Temp**: {temperature}")
    md_lines.append("")

    md_rows = [["Model", "OK/Total", "Avg time (s)", "Stability (selected Jaccard)"]]
    for model, s in model_summaries.items():
        md_rows.append(
            [
                model,
                f"{s['ok']}/{s['runs']}",
                str(s["avg_elapsed_s"]),
                "" if s["stability_selected_jaccard"] is None else str(s["stability_selected_jaccard"]),
            ]
        )
    md_lines.append("### Summary")
    md_lines.append(_format_md_table(md_rows))
    md_lines.append("")

    if pairwise:
        md_lines.append("### Pairwise agreement (first successful run per model)")
        pr = [["Model A", "Model B", "Selected Jaccard", "Top-5 Jaccard"]]
        for p in pairwise:
            pr.append([p["model_a"], p["model_b"], str(p["selected_jaccard"]), str(p["top5_jaccard"])])
        md_lines.append(_format_md_table(pr))
        md_lines.append("")

    md_lines.append("### Notes")
    md_lines.append("- Higher **stability** usually means the ideation model is more consistent at scoring/selection.")
    md_lines.append(
        "- If you care about ‘better at the job’ beyond consistency, we should add a small **human rating** step "
        "(e.g., label 10 shortlisted ideas as pass/fail) and compute precision-by-model."
    )
    md_lines.append("")

    out_md = run_dir / "report.md"
    out_md.write_text("\n".join(md_lines), encoding="utf-8")

    return run_dir
