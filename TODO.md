# Engineering backlog

**Status:** Items below are implemented in code; keep this file as a living checklist for future work.

**Simplification:** completed — see [`docs/simplification-execution-plan.md`](docs/simplification-execution-plan.md).

## Schema & compatibility

- [x] **2 — Version `RunState` (schema version)**  
  `RunState.schema_version` (see `RUN_STATE_SCHEMA_VERSION` in `schemas.py`). New saves include it; older JSON without the field loads with default `1`.

## Tests

- [x] **3 — Focused tests for non-LLM logic**  
  `tests/test_validation_core.py`: ranking tie-breaks, shortlist ID finalization, citation chunk repair, candidate coercion, schema default.

## Selection behaviour

- [x] **4 — Document selection semantics; optional tie-breakers**  
  Ranking tie-breakers: `total_weighted` → `source_support` → `hook_strength` → `content_pillar` → `idea_id`. CLI `--select-pool-k` help documents the shortlist slice. `SCORING_AND_SELECTION.md` remains the long-form reference.

## Configuration & reuse

- [x] **7 — Audience / persona configuration**  
  `--audience-preset` + personas + metrics grouping exist. **`--require-audience-preset`** fails fast when a campaign must load a persona card.

- [x] **5 — Centralize default paths**  
  `src/carousel_agents/defaults.py` + `CAROUSEL_*` env vars (see `.env.example`). `load_or_build_performance_digest` uses the same defaults.

## Operations

- [x] **6 — Observability for production runs**  
  `observability.py`: `CAROUSEL_LOG_LEVEL` (e.g. `INFO`) logs JSON lines for pipeline stages and LLM calls (latency; token `usage` when the chat-completions JSON path returns it).

## Performance

- [x] **8 — Concurrency where safe**  
  `CAROUSEL_MAX_PARALLEL` (default `1`): when `>1`, hooks+CTA for selected ideas run in a thread pool (one `OpenAICompatibleClient` per worker). Ignored for `--mock`.
