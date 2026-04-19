## Carousel Agents (Selection-First)

This repo contains a practical pipeline for turning an uploaded document/transcript into high-quality social media carousels:

**document → candidate ideas → scored shortlist → hooks → slides → (optional) QA**

The design is **selection-first** and **citation-based** to preserve source fidelity and reduce hallucinations.

### What’s here

- **`docs/golden-path.md`**: recommended command flow (run → export → metrics)
- **`docs/governance.md`**: defaults for `--judge-govern`, `--hook-policy`, `--qa`
- `SCORING_AND_SELECTION.md`: the rubric, selection gate, and provenance rules
- `src/carousel_agents/`: pipeline code (ingest, chunking, extraction, scoring, hooks, slides, QA)
- `src/carousel_agents_research/`: optional benchmarks and metric-prediction experiments (separate CLI)
- `data/` (optional): run outputs and evaluation logs
- `documents/`: source documents/transcripts to process

### Quick start

1. Create a virtual environment and install deps:

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

### Download your past Instagram content (for analysis)

This uses `instaloader` to download each `IG_Permalink` listed in your tracker CSV.

1. (Recommended) Set IG credentials (for private posts + fewer failures):

```powershell
$env:IG_USERNAME="your_ig_username"
$env:IG_PASSWORD="your_ig_password"
```

2. Run the downloader:

```powershell
python -m carousel_agents ig-download `
  --csv "Marketing - Content - Tracker - Content Tracker.csv" `
  --outdir "exports"
```

Outputs are saved under `exports/instagram/{Asset_ID}_{shortcode}/...` and an index file at `exports/instagram/_dataset/index.json`.

### Build a slide-level dataset (for OCR / analysis)

After downloading, generate a dataset with one row per slide image:

```powershell
python -m carousel_agents ig-slides-dataset `
  --exports-dir "exports" `
  --tracker-csv "Marketing - Content - Tracker - Content Tracker (1).csv" `
  --carousel-only
```

This writes:
- `exports/instagram/_dataset/slides_dataset.csv`
- `exports/instagram/_dataset/slides_dataset.jsonl`

### OCR your carousel slides (turn images into text)

This extracts on-slide text into a reusable dataset for analysis and model judging.

```powershell
python -m carousel_agents ig-ocr --limit 10
```

Output:
- `exports/instagram/_dataset/ocr_slides.jsonl`

2. Set environment variables (OpenAI-compatible API):

```powershell
$env:OPENROUTER_API_KEY="..."
# Optional (recommended by OpenRouter):
$env:OPENROUTER_SITE_URL="https://yourdomain.com"
$env:OPENROUTER_APP_NAME="carousel-agents"

# Models (OpenRouter model ids)
$env:MODEL_IDEATION="mistralai/mistral-large"
$env:MODEL_HOOK_JUDGE="mistralai/mistral-large"
$env:MODEL_WRITER="google/gemini-2.5-pro-preview"
# Optional: legacy alias — if you still use MODEL_JUDGE, it applies when MODEL_IDEATION is unset
$env:MODEL_EDITOR="openai/gpt-4.1-mini"
```

The **ideation** pass (idea extraction + scoring + shortlist) defaults to **`mistralai/mistral-large`**. The **hook judge** defaults to **`mistralai/mistral-large`**, and the **writer** defaults to **`google/gemini-2.5-pro-preview`**. Override any role via `MODEL_*` env vars (e.g. `openai/gpt-4.1-mini` for ideation if you prefer).

**Split ideation** (extract with one model, score/govern with another): `python -m carousel_agents run ... --split-ideation` with `MODEL_IDEATION_EXTRACT` and `MODEL_IDEATION_SCORE` (see `.env.example`). To compare unified vs split, run two separate `run` invocations (see `docs/golden-path.md`).

3. Run the pipeline on a text file:

```bash
python -m carousel_agents run --input "path\\to\\doc.txt" --out "data\\run.json" --export-dir outputs
```

Optional **persona preset** (loads `docs/personas/<id>.json` for Writer tone; see `docs/personas/`):

```bash
python -m carousel_agents run --input "path\\to\\doc.txt" --out "data\\run.json" --export-dir outputs --audience-preset long_road_diagnosis
```

### Track performance (so tags become a flywheel)

Log each published post (especially `hook_style`, `pillar`, `format`) so you can see what works.

Append a row:

```powershell
python -m carousel_agents log-performance `
  --post-id "IG_2026-04-15_endo_001" `
  --run-id "doc_20260415_181236" `
  --document-title "anets-studio-n4lie_liz-aneta-apr-15-2026-003" `
  --idea-id "i014" `
  --pillar "validation" `
  --format "myth_vs_truth" `
  --hook-id "h004" `
  --hook-style "myth_bust" `
  --impressions 12000 `
  --saves 540 `
  --shares 210 `
  --profile-visits 95 `
  --notes "Strong save/share; myth format worked"
```

Quick report (aggregates `PerformanceLog` JSONL by tag):

```powershell
python -m carousel_agents metrics summary --group-by hook_style --min-n 2
```

(`report-performance` still works and forwards to the same implementation.)

### Research and benchmarks (optional)

```powershell
python -m carousel_agents_research judge-benchmark --input documents\\example.txt --models "openai/gpt-4.1-mini"
python -m carousel_agents_research experiment-predict --csv "path\\to\\tracker.csv" --n 6
```

### Notes

- The pipeline enforces a **selection freeze gate**: hooks/slides only run after shortlist selection is frozen.
- Early iteration assumes **you are the evaluator**; you can log your ratings and (later) real-world post performance for a flywheel.

