# Audience and personas (simple plan)

**Goal:** Content that clearly fits the people we serve, grounded in real language and real outcomes—without bloating prompts or blurring medical accuracy.

**Simple** here means: **few moving parts, each one doing one job**, in the straightest line from data → decision → copy.

---

## The three inputs (do not merge into one blob)

| Input | Job |
|-------|-----|
| **Persona** | Who we are speaking to: tensions, language, what “good” sounds like. |
| **Performance** | What has actually worked (rates by tag; later by segment). |
| **Source document** | What we are allowed to claim (citations, governance). |

Persona and performance **steer** tone and emphasis; the **document** still wins on facts.

---

## What we will actually maintain

1. **Persona cards** — One file per segment (`docs/personas/` or `data/personas/`, team picks one folder and stays there). Each card is short: **id**, **who**, **fear / hope**, **5–10 lexicon lines** (aggregated, anonymized), **do / don’t**, **content job** (validate | educate | next step). No essays.

2. **One optional preset per run** — e.g. environment or a single CLI flag when we add it: load **one** card and pass its fields into the Writer in a **fixed, capped shape** (same idea as `context_pack`: nothing unbounded).

3. **Performance as today** — `data/performance.jsonl`, tags, `metrics summary` / `metrics carousels`, optional digest for ideation. We do **not** duplicate reporting; we extend only when persona labels exist.

4. **Persona proxy on posts (when we measure “for whom”)** — Three cases only: **explicit** at log time if we know the audience; else **rules file** (e.g. pillar + hook_style → persona_id); else **`unclassified`**. No guessing beyond that.

---

## Order of work (direct path)

| Step | Action | Stop when |
|------|--------|-----------|
| A | Write **2–4** persona cards from chat + existing patient voice snippets. | Cards are usable in a meeting and fit on one screen each. |
| B | Use cards **manually** in briefs (“this run is for `persona_x`”) until it hurts. | Team agrees preset names and fields. |
| C | Add **`persona_id` + `persona_source`** to performance logging and optional rules YAML. | `metrics` can group by persona (or export for a sheet). |
| D | Wire **one** preset into the pipeline (load card → structured Writer context). | Voice fit improves without citation regressions. |
| E | Only then tighten **ideation** (extract hints from active preset) or enrich **digest** by segment—if data supports it. | — |

Skipping steps or doing D before B usually wastes code.

---

## What we are not doing in v1

- ML clustering of chats as a gate.  
- Many CLI commands or persona-specific pipelines.  
- Persona text copy-pasted into every system prompt.  
- Treating **pillar** alone as a full persona.

---

## Risks (one line each)

- **Small n** → use performance as tie-breaker, not law.  
- **Stereotypes** → multiple cards, aggregated quotes, human review.  
- **Persona vs truth** → persona never overrides citations or governance.

---

## Open items

- [x] Persona cards folder: **`docs/personas/`** (override with `PERSONAS_DIR`).  
- [ ] Owner for card updates.  
- [x] Rules proxy: **`data/persona_rules.json`** (override with `PERSONA_RULES_PATH`).  
- [ ] Minimum **n** before trusting a “persona × format” claim.

## Implementation (repo)

- **`--audience-preset <id>`** on `run` → loads `docs/personas/<id>.json`, injects capped **`persona_preset`** block into Writer prompts; stored on **`RunState.audience_preset`**.
- **`log-performance`**: optional **`--persona-id`** / **`--persona-source`**; if omitted, **`resolve_persona_proxy`** uses `data/persona_rules.json` unless **`--no-infer-persona`**.
- **`metrics summary --group-by persona`**: aggregates by **`persona_id`**.

---

## Engineering dependencies

Implementation (schema version, paths, tests, observability) stays in **`TODO.md`**. This plan stays **product-first**; code follows the steps above.
