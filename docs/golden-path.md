# Golden path (production workflow)

Typical sequence:

1. **Run the pipeline** on a source document (`.txt` / supported types):

   ```bash
   python -m carousel_agents run --input path/to/doc.txt --out data/run.json --export-dir outputs
   ```

   Optional persona (see `docs/personas/*.json`): add `--audience-preset long_road_diagnosis` (or your card id).

2. **Inspect** `data/run.json` (or your `--out`) for candidates, shortlist, and any governance flags.

3. **Export** clean markdown (if you did not pass `--export-dir` on `run`, or need to regenerate):

   ```bash
   python -m carousel_agents export-bundle --run-json data/run.json --export-dir outputs
   ```

4. **Metrics** (canonical commands):

   ```bash
   python -m carousel_agents metrics summary --path data/performance.jsonl
   python -m carousel_agents metrics carousels --csv "path/to/tracker.csv" --out outputs/reports/carousel_performance.md
   ```

   Legacy aliases `report-performance` and `report-carousels` still work; they print a tip to prefer `metrics`.

5. **Optional:** `--mock` on `run` for deterministic JSON without API calls (smoke tests).

See `docs/governance.md` for defaults (`--judge-govern`, `--hook-policy`, `--qa`).

## Manual A/B: unified vs split ideation

There is no `parallel-ideation` command. Run two separate invocations:

- **Unified:** `python -m carousel_agents run --input doc.txt --out run_u.json` (default ideation).
- **Split:** set `MODEL_IDEATION_EXTRACT` / `MODEL_IDEATION_SCORE` (and optional `TEMP_*`), then  
  `python -m carousel_agents run --input doc.txt --out run_s.json --split-ideation`.
