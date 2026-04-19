# Prompt construction notes

## Layers

- **System prompts:** `system_ideation`, `system_writer`, etc. in `carousel_agents/prompts.py`.
- **User prompts:** chunk lists, audience block (`_audience_block`), optional **performance digest** (ideation + writer hints).
- **Writer context pack:** `carousel_agents/context_pack.py` merges patient voice snippets and OCR structure examples with fixed caps.

## Trimming context

- Set **`CONTEXT_MINIMAL=1`** (or `true` / `yes` / `on`) in the environment to:
  - Omit patient voice + OCR structure examples for hooks/slides/CTA (via context pack).
  - Shorten performance digest in `digest_prompt_body` / `digest_writer_hints` (fewer `top_examples`, compact JSON).

Use this for A/B comparisons or when prompts feel overloaded.

## Baseline sizes

Approximate character counts for main static blocks (re-run after large edits):

| Block | ~chars | Location |
|-------|--------|----------|
| `DOCMAP_CONTEXT_SUMMARY` | ~400 | `prompts.py` |
| `HOOK_BRIEF_SUMMARY` | ~280 | `prompts.py` |
| `HOOK_LANGUAGE_POLICY` | ~420 | `prompts.py` |

Ideation/scoring user prompts add variable chunk text and optional digest JSON.

See also `docs/prompt_baseline.md` for a snapshot taken during the simplification pass.
