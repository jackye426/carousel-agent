# Simplification: full execution plan

This plan implements the agreed directions: **clearer governance defaults**, **no parallel-ideation CLI**, **research code separated from core**, **one metrics story**, **single context-pack for style snippets**, **leaner prompts**, and **lighter documentation sprawl**.  
**Split ideation** (`--split-ideation`) stays as a supported option.

---

## Goals

| Goal | Outcome |
|------|---------|
| **Governance** | One documented “happy path”; optional hooks only when needed. |
| **CLI** | Fewer first-class commands; research and metrics paths are obvious. |
| **Code layout** | Core package is the default mental model; research is importable but not mixed into daily flow. |
| **Prompts** | Shorter, structured inputs; measurable quality check after trim. |
| **Docs** | Single backlog entry point; strategy docs linked, not duplicated. |

## Non-goals (this pass)

- Removing split ideation or `RunState` / re-stage flows.  
- Rewriting the entire pipeline or switching providers.  
- Deleting citation repair/validation (keep as core).

---

## Phase 0 — Baseline and guardrails (0.5–1 day)

**Purpose:** Know what you are protecting before refactors.

1. **Freeze a golden path**  
   Document one command sequence you use today: e.g. `run` → inspect `run.json` → `export-bundle` (or your actual path). Save a **small fixture doc** + **expected bundle shape** (manual checklist is enough for v1).

2. **Snapshot prompt sizes**  
   In `prompts.py`, note approximate **token/char length** of main system + user builders (`user_generate_hooks`, `user_draft_slides`, ideation users). You will compare after Phase 6.

3. **Optional:** run **`--mock`** on the fixture once so you have deterministic JSON for regression after CLI moves.

**Exit:** Written baseline (even a short `docs/golden-path.md` stub) + prompt length notes.

---

## Phase 1 — Governance: defaults and documentation (0.5–1 day)

**Purpose:** Reduce decision fatigue; same code paths, clearer contract.

### 1.1 Default story (product)

| Layer | Default | When to change |
|-------|---------|----------------|
| **Ideation governance** (`judge_govern`) | **On** | Off only for quick drafts / debugging. |
| **Hook language policy** (`hook_policy`) | **Off** | On when compliance/brand bans phrases; document ticket or policy ref. |
| **Editor QA** (`do_qa`) | **Off** for speed; **On** for publish-ready | Document in team norm. |

### 1.2 Code/docs

1. Set **CLI defaults** in `run` / `redraft-slides` / `run-hooks-cta` to match the table if anything diverges today (verify `judge_govern` default is already `True`, `hook_policy` `False`, `do_qa` `False` on `run`).  
2. Add **`docs/governance.md`** (short): one paragraph per flag, when to enable hook policy + QA, link to `hook_policy` module if useful.  
3. Update **Typer help strings** on `run` so the docstring states the default story in 2–3 lines.

**Exit:** Defaults match the table; `docs/governance.md` exists; help text matches.

---

## Phase 2 — Remove `parallel-ideation` (0.25–0.5 day)

**Purpose:** One less duplicate full pipeline run; split ideation remains via `--split-ideation`.

1. **Delete** the `parallel_ideation` command and its block in `__main__.py` (lines ~133–218 area).  
2. **Remove** any references in README / internal docs (search repo for `parallel-ideation`, `parallel_ideation`).  
3. **Document** the replacement workflow: two separate `run` invocations with different `--out` and env if you ever need A/B:  
   - Run A: `MODEL_IDEATION` unified, `--out run_unified.json`  
   - Run B: `--split-ideation`, `--out run_split.json`, set `MODEL_IDEATION_EXTRACT` / `MODEL_IDEATION_SCORE` as needed  

**Exit:** No `parallel-ideation` in CLI; docs explain manual A/B if needed.

---

## Phase 3 — Research cluster extraction (1–2 days)

**Purpose:** Core package = production pipeline + shared libs; research = optional.

### 3.1 Layout (suggested)

```
src/
  carousel_agents/          # core: pipeline, llm, prompts, export, validation, …
  carousel_agents_research/ # or carousel_research/
    __init__.py
    experiment.py           # moved from carousel_agents
    judge_benchmark.py
    judge_top_ideas.py
    __main__.py             # Typer: experiment-predict, experiment-vision, judge-benchmark, judge-top-ideas
```

### 3.2 Steps

1. Move **`experiment.py`**, **`judge_benchmark.py`**, **`judge_top_ideas.py`** into the new package; fix imports to use `from carousel_agents...` for shared code.  
2. Remove their commands from **`carousel_agents/__main__.py`**; register them only on **`carousel_agents_research/__main__.py`** (or single `research` CLI).  
3. **`pyproject.toml`**: add optional extra `[project.optional-dependencies] research = []` if you want `pip install carousel-agents[research]`, or keep one package with two entry points:  
   - `carousel-agents` → `python -m carousel_agents`  
   - document `python -m carousel_agents_research` for experiments  
4. **Root README**: “Core workflow” section lists only core commands; “Research & benchmarks” points to the research module.

**Exit:** Core `__main__.py` has no experiment/judge commands; research runs via separate module; imports resolve; smoke-run one research command.

---

## Phase 4 — Metrics: one canonical path (1–2 days)

**Purpose:** One mental model: **data in → digest/report out**.

### 4.1 Inventory (do first)

List current flows:

- `log-performance`, `import-performance-from-tracker`, `normalize-performance-ids`  
- `build-performance-digest`  
- `report-performance`, `report-carousels`  
- `analysis_report.generate_report` (if distinct)

### 4.2 Target design

| Step | Canonical command / role |
|------|----------------------------|
| **Authoritative table → JSONL** | Prefer **`import-performance-from-tracker`** for bulk; **`log-performance`** for one-off rows. |
| **IDs** | **`normalize-performance-ids`** as maintenance utility (document quarterly or on import issues). |
| **Digest for ideation** | **`build-performance-digest`** (unchanged purpose). |
| **Human-readable insight** | **Single** `metrics report` (new Typer group or one command with subcommands) that can emit: `--kind summary|by-tag|carousels` **or** keep two thin wrappers that call one internal `build_report(kind=...)`. |

### 4.3 Implementation sketch

1. Extract shared logic from `report-performance`, `report-carousels`, and `analysis_report` into **`carousel_agents/metrics_reporting.py`** (or `metrics/__init__.py`) with functions `report_summary(...)`, `report_carousels(...)`, etc.  
2. Replace duplicate CLI with **`metrics` group**:  
   - `metrics summary` → old report-performance style  
   - `metrics carousels` → old report-carousels  
   - Or one command `metrics report --mode summary|carousels`  
3. Deprecate old command names with **`typer` hidden=True** + print deprecation warning pointing to new names for one release, then remove.

**Exit:** README “Metrics” section describes one flow; duplicate markdown reports (if any) generated from one codepath.

---

## Phase 5 — Context pack: one injection surface (1 day)

**Purpose:** `pipeline.py` (and redraft) call **one helper** instead of three ad-hoc samplers.

### 5.1 API (suggested)

**New module:** `carousel_agents/context_pack.py`

```python
@dataclass
class WriterContextPack:
    patient_voice_snippets: list[str]
    ocr_structure_examples: list[dict]  # or typed model
    ocr_style_snippets: list[str]

def build_writer_context_pack() -> WriterContextPack:
    """Single place: caps, ordering, empty defaults."""
```

1. Implement **`build_writer_context_pack()`** calling existing `sample_patient_voice_snippets`, `sample_ocr_carousel_structures`, `sample_ocr_style_snippets` with **documented caps** (e.g. max N snippets per category) so prompts cannot balloon silently.  
2. Replace scattered calls in **`pipeline.py`** / **`redraft.py`** with one pack; pass `pack` into prompt builders or `model_dump`-style dict for JSON prompts.  
3. **Log** (optional, Phase 6 tie-in): debug log total character count of injected context once per stage.

**Exit:** Single import path for writer-side context; hard caps documented in `context_pack.py`.

---

## Phase 6 — Prompt diet (1–3 days)

**Purpose:** Reduce adverse effects from overstuffed prompts; keep grounding.

### 6.1 Audit

1. List **every** block injected into Writer and Ideation: `performance_digest`, audience, chunks, patient voice, OCR examples, hook policy text, etc.  
2. Mark each: **required** | **nice-to-have** | **duplicate of another block**.

### 6.2 Rules

- **Deduplicate:** if `audience` repeats what `persona` card will say later, merge.  
- **Truncate:** performance digest = **top-K bullets**, fixed max chars (already partially true—enforce globally).  
- **Order:** put **task + constraints first**; examples second; metadata last (models often weight early tokens).  
- **Optional toggles:** `CONTEXT_MINIMAL=1` env to disable non-essential examples for A/B comparison.

### 6.3 Verify

1. Re-run **golden path** (Phase 0) on 2–3 real docs; compare bundles for **clarity and citation behavior**, not just length.  
2. If quality regresses, restore **one** section at a time until stable.

**Exit:** Prompt char counts down vs Phase 0 snapshot; short note in `docs/prompts.md` describing sections and caps.

---

## Phase 7 — Documentation consolidation (0.5 day)

**Purpose:** Single entry point; avoid duplicate “open decisions.”

1. **`README.md`** (or `docs/README.md`):  
   - Link **Golden path** → `docs/governance.md` → `docs/audience-and-personas-plan.md` (strategy).  
2. **`TODO.md`**: prepend line: “Strategy and personas: see `docs/audience-and-personas-plan.md`.” Remove overlapping bullets if any duplicate the plan (or merge into one list with sections).  
3. **`docs/simplification-execution-plan.md`** (this file): move to **Completed** checklist at bottom when phases done.

**Exit:** New contributor finds workflow in README in under 5 minutes.

---

## Execution order (recommended)

| Order | Phase | Rationale |
|-------|-------|-----------|
| 1 | Phase 0 | Baseline |
| 2 | Phase 1 | Low risk, clarifies product |
| 3 | Phase 5 | Shrinks prompt *inputs* before rewriting prompt *text* (Phase 6) |
| 4 | Phase 6 | Prompt diet with caps enforced |
| 5 | Phase 2 | Small CLI delete |
| 6 | Phase 4 | Metrics refactor touches CLI; do after core stable |
| 7 | Phase 3 | Research move is broader; optional to swap with Phase 4 if you prefer code layout first |
| 8 | Phase 7 | Last: point docs at finished structure |

**Note:** Phases **3** and **4** can be **swapped** if you want package layout before metrics unification. Phase **5 → 6** should stay adjacent.

---

## Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Research import breaks | Move files incrementally; run `python -m compileall` and one smoke command per step. |
| Metrics refactor drops a metric | Diff CSV/markdown outputs pre/post on a fixed `performance.jsonl`. |
| Prompt trim hurts quality | Golden-path docs + restore one section at a time; optional `CONTEXT_MINIMAL` A/B. |
| Scope creep | Finish Phases 0–1–5–6 before large Phase 3–4 if timeboxed. |

---

## Success criteria (definition of done)

- [x] Governance defaults and **`docs/governance.md`** match team practice.  
- [x] **`parallel-ideation`** removed; replacement documented.  
- [x] **`WriterContextPack`** (`context_pack.py`) used in pipeline/redraft; caps + `CONTEXT_MINIMAL` documented in `docs/prompts.md`.  
- [x] Prompt total size reduced vs baseline **or** explicit decision to keep a section with justification (`HOOK_BRIEF` shortened; digest trims when `CONTEXT_MINIMAL`).  
- [x] Metrics: canonical **`metrics`** subcommand + legacy aliases.  
- [x] Research commands live in **`carousel_agents_research`**.  
- [x] README reflects simplified story; links to golden path and governance.

---

## Completed checklist

_Executed in this repo._

- [x] Phase 0 — Baseline  
- [x] Phase 1 — Governance  
- [x] Phase 2 — Remove parallel-ideation  
- [x] Phase 3 — Research package  
- [x] Phase 4 — Metrics unification  
- [x] Phase 5 — Context pack  
- [x] Phase 6 — Prompt diet  
- [x] Phase 7 — Docs consolidation  
