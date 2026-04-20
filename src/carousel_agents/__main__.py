from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, cast

import typer
from dotenv import load_dotenv
from rich import print

load_dotenv()

from .defaults import (
    experiment_program_json_path,
    experiments_dir_default,
    export_dir_default,
    marketing_tracker_csv_path,
    ocr_slides_jsonl_path,
    performance_digest_json_path,
    performance_jsonl_path,
    performance_normalized_jsonl_path,
    run_json_default,
    tags_jsonl_path,
)

from .chunking import ChunkStrategyName
from .pipeline import (
    run_hooks_cta_from_state,
    run_pipeline,
    run_pipeline_continue_from_human_selection,
)
from .performance_digest import build_performance_digest, load_or_build_performance_digest, save_digest
from .experiment_analysis import analyze_experiment, save_experiment_result
from .experiment_orchestrator import (
    append_suggestions_to_program,
    build_orchestrator_signals,
    suggest_rules_based,
    suggest_with_llm,
)
from .experiment_program_store import load_program, save_program
from .schemas import ExperimentSpec, RunState
from .instagram_download import download_from_tracker
from .instagram_slides_dataset import build_slides_dataset
from .performance import append_performance_log, compute_derived
from .schemas import PerformanceLog, PerformanceObserved
from .ig_ocr import get_ocr_model, ocr_slides_dataset
from .metrics_reporting import print_performance_summary_table, write_carousel_tracker_report
from .tag_carousels import get_tagger_model, tag_carousels
from .bundle_to_csv import write_bundle_csvs
from .compare_bundles import write_comparison_csvs
from .export_clean import export_selected_markdown
from .redraft import load_run_state, redraft_selected_slides, save_run_state
from .patient_voice import build_patient_voice_bank, default_bank_paths as default_patient_voice_paths
from .ocr_carousel_bank import build_ocr_carousel_bank, default_bank_paths as default_ocr_carousel_paths
from .id_resolution import build_tracker_index, resolve_post_id
from .import_tracker_performance import import_performance_from_tracker
from .persona_rules import resolve_persona_proxy
from .post_linkage import (
    discover_run_json_files,
    load_run_state_path,
    match_post_to_runs,
    read_ocr_carousel_bank_row,
    read_ocr_slides_text_for_asset,
)


app = typer.Typer(add_completion=False)
experiment_app = typer.Typer(add_completion=False, help="A/B experiment templates and analysis.")
orchestrator_app = typer.Typer(add_completion=False, help="Experiment program orchestrator (queue + performance signals + LLM).")
metrics_app = typer.Typer(add_completion=False, help="Reports from performance JSONL and the marketing tracker.")


@app.command()
def run(
    input: Path = typer.Option(..., "--input", exists=True, file_okay=True, dir_okay=False),
    out: Path = typer.Option(default_factory=run_json_default, help="Written RunState JSON."),
    export_dir: Path = typer.Option(
        default_factory=export_dir_default,
        help="Write clean markdown exports here.",
    ),
    candidate_count: int = typer.Option(30, "--candidate-count"),
    select_n: int = typer.Option(5, "--select-n"),
    select_pool_k: int = typer.Option(
        10,
        "--select-pool-k",
        help=(
            "Take --select-n candidates from the top K by rank after scoring "
            "(candidates are sorted by weighted total, then source_support, hook_strength, "
            "content_pillar, idea_id). By default picks are stratified by content pillar; "
            "use --no-stratified-shortlist for strict top-N order. Deterministic; no selection LLM."
        ),
    ),
    stratified_shortlist: bool = typer.Option(
        True,
        "--stratified-shortlist/--no-stratified-shortlist",
        help="Prefer pillar diversity in the shortlist (default: on).",
    ),
    performance_digest: Path | None = typer.Option(
        None,
        "--performance-digest",
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="Explicit JSON digest file. If omitted, uses outputs/performance_digest.json or builds from data/performance.jsonl.",
    ),
    no_performance_digest: bool = typer.Option(
        False,
        "--no-performance-digest",
        help="Disable performance context for this run (overrides auto-load).",
    ),
    hook_count: int = typer.Option(12, "--hook-count"),
    max_slides: int = typer.Option(10, "--max-slides"),
    do_hooks: bool = typer.Option(True, "--hooks/--no-hooks"),
    do_cta: bool = typer.Option(True, "--cta/--no-cta"),
    do_slides: bool = typer.Option(True, "--slides/--no-slides"),
    do_qa: bool = typer.Option(False, "--qa/--no-qa"),
    judge_govern: bool = typer.Option(
        True,
        "--judge-govern/--no-judge-govern",
        help="Allow ideation + hook policy to veto and force rewrites (CLI name unchanged).",
    ),
    judge_rewrite_rounds: int = typer.Option(
        1,
        "--judge-rewrite-rounds",
        help="Max carousel rewrite rounds from the ideation review per draft.",
    ),
    hook_policy: bool = typer.Option(
        False,
        "--hook-policy/--no-hook-policy",
        help="Enable banned-phrase detection + Hook Judge rewrites/vetoes (default: off — Writer hooks only).",
    ),
    hook_rewrite_rounds: int = typer.Option(2, "--hook-rewrite-rounds", help="Max rewrite rounds for hook language policy."),
    mock: bool = typer.Option(False, "--mock", help="Run without API calls using deterministic mock outputs."),
    split_ideation: bool = typer.Option(
        False,
        "--split-ideation/--no-split-ideation",
        help="Separate models for extract (MODEL_IDEATION_EXTRACT) vs score/governance (MODEL_IDEATION_SCORE).",
    ),
    chunk_strategy: str = typer.Option(
        "auto",
        "--chunk-strategy",
        help="Chunking: auto (from path/type), paragraph, markdown_headings (# sections), transcript (larger windows, speaker turns if detected).",
    ),
    chunk_max_chars: int | None = typer.Option(
        None,
        "--chunk-max-chars",
        help="Override max chunk size (defaults: ~1200 paragraph, ~2800 transcript).",
    ),
    chunk_overlap: int | None = typer.Option(
        None,
        "--chunk-overlap",
        help="Override overlap between chunks (defaults: ~150 paragraph, ~220 transcript).",
    ),
    audience_preset: str | None = typer.Option(
        None,
        "--audience-preset",
        help="Persona card id (stem of docs/personas/<id>.json). Sets Writer tone/emphasis; optional.",
    ),
    require_audience_preset: bool = typer.Option(
        False,
        "--require-audience-preset",
        help="Fail the run if --audience-preset is not set (campaigns that must target a persona).",
    ),
    experiment_json: Path | None = typer.Option(
        None,
        "--experiment-json",
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="ExperimentSpec JSON: splits one shortlisted idea_id into two arms after hooks (see: experiment template).",
    ),
    orchestrate: bool = typer.Option(
        False,
        "--orchestrate",
        help="After the shortlist is frozen, attach the next queued hypothesis from the experiment program (see: orchestrate suggest).",
    ),
    orchestrate_program: Path | None = typer.Option(
        None,
        "--orchestrate-program",
        exists=False,
        file_okay=True,
        dir_okay=False,
        help="Path to experiment program JSON (default: data/experiment_program.json).",
    ),
    stop_after_scoring: bool = typer.Option(
        False,
        "--stop-after-scoring",
        help=(
            "Stop after extract+score: save RunState, write *_human_review.md + *_human_selection.template.json, "
            "then exit. Edit/copy the template to your selection JSON and run `continue-from-human`."
        ),
    ),
):
    """
    Run the selection-first carousel pipeline on an input document.

    Default story: ideation governance ON (--judge-govern), hook language policy OFF (--no-hook-policy),
    Editor QA OFF unless you pass --qa for publish-ready passes.

    Models: MODEL_IDEATION (unified extract+score+governance; legacy MODEL_JUDGE), or split via --split-ideation;
    MODEL_WRITER (hooks, slides, CTAs); use --hook-policy for MODEL_HOOK_JUDGE policy loop;
    MODEL_EDITOR (QA when --qa).
    """
    allowed_strategies = {"auto", "paragraph", "markdown_headings", "transcript"}
    if chunk_strategy not in allowed_strategies:
        raise typer.BadParameter(f"--chunk-strategy must be one of: {sorted(allowed_strategies)}")
    strategy_typed = cast(ChunkStrategyName, chunk_strategy)

    load_dotenv()
    if no_performance_digest:
        perf_digest_obj = None
    elif performance_digest is not None:
        perf_digest_obj = json.loads(performance_digest.read_text(encoding="utf-8"))
    else:
        perf_digest_obj = load_or_build_performance_digest(explicit_path=None, disabled=False)
    if experiment_json is not None and orchestrate:
        raise typer.BadParameter("Use either --experiment-json or --orchestrate, not both.")
    exp_obj: ExperimentSpec | None = None
    if experiment_json is not None:
        exp_obj = ExperimentSpec.model_validate_json(experiment_json.read_text(encoding="utf-8"))
    orch_path: Path | None = None
    if orchestrate:
        orch_path = orchestrate_program if orchestrate_program is not None else experiment_program_json_path()
    state = run_pipeline(
        input_path=input,
        out_path=out,
        candidate_count=candidate_count,
        select_n=select_n,
        select_pool_k=select_pool_k,
        stratified_shortlist=stratified_shortlist,
        performance_digest=perf_digest_obj,
        hook_count=hook_count,
        max_slides=max_slides,
        do_hooks=do_hooks,
        do_cta=do_cta,
        do_slides=do_slides,
        do_qa=do_qa,
        judge_govern=judge_govern,
        judge_rewrite_rounds=judge_rewrite_rounds,
        hook_policy=hook_policy,
        hook_rewrite_rounds=hook_rewrite_rounds,
        mock=mock,
        export_dir=export_dir,
        split_ideation=split_ideation,
        chunk_strategy=strategy_typed,
        chunk_max_chars=chunk_max_chars,
        chunk_overlap=chunk_overlap,
        audience_preset=audience_preset,
        require_audience_preset=require_audience_preset,
        experiment=exp_obj,
        orchestrate_program_path=orch_path,
        stop_after_scoring=stop_after_scoring,
    )
    print(f"[bold green]Saved:[/bold green] {out}")
    if state.awaiting_human_review:
        print(
            "[bold yellow]Stopped for human review.[/bold yellow] Open the *_human_review.md file, "
            "fill a human_selection JSON, then: [bold]python -m carousel_agents continue-from-human[/bold] ..."
        )
    print(f"[bold]Candidates:[/bold] {len(state.candidates)}  [bold]Selected:[/bold] {len(state.shortlist.selected_idea_ids)}")
    print(f"[bold]Ideation:[/bold] {state.ideation_mode}  extract={state.ideation_extract_model}  score={state.ideation_score_model}")
    if state.experiment:
        print(
            f"[bold]Experiment:[/bold] {state.experiment.experiment_id!r} "
            f"on idea_id={state.experiment.idea_id!r} — see export schedule.md / experiment.json"
        )


@app.command("continue-from-human")
def continue_from_human(
    from_state: Path = typer.Option(
        ...,
        "--from-state",
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="RunState JSON from a run that used --stop-after-scoring.",
    ),
    human_selection: Path = typer.Option(
        ...,
        "--human-selection",
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="Your edited human_selection JSON (see template next to the stopped run).",
    ),
    out: Path = typer.Option(..., "--out", help="Destination RunState JSON after hooks/slides."),
    export_dir: Path = typer.Option(
        default_factory=export_dir_default,
        help="Write clean markdown exports here.",
    ),
    performance_digest: Path | None = typer.Option(
        None,
        "--performance-digest",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    no_performance_digest: bool = typer.Option(
        False,
        "--no-performance-digest",
        help="Disable performance context for this continue run.",
    ),
    hook_count: int = typer.Option(12, "--hook-count"),
    max_slides: int = typer.Option(10, "--max-slides"),
    do_hooks: bool = typer.Option(True, "--hooks/--no-hooks"),
    do_cta: bool = typer.Option(True, "--cta/--no-cta"),
    do_slides: bool = typer.Option(True, "--slides/--no-slides"),
    do_qa: bool = typer.Option(False, "--qa/--no-qa"),
    judge_govern: bool = typer.Option(True, "--judge-govern/--no-judge-govern"),
    judge_rewrite_rounds: int = typer.Option(1, "--judge-rewrite-rounds"),
    hook_policy: bool = typer.Option(False, "--hook-policy/--no-hook-policy"),
    hook_rewrite_rounds: int = typer.Option(2, "--hook-rewrite-rounds"),
    mock: bool = typer.Option(False, "--mock"),
    orchestrate_program: Path | None = typer.Option(
        None,
        "--orchestrate-program",
        exists=False,
        file_okay=True,
        dir_okay=False,
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Continue even if awaiting_human_review is false (use with care).",
    ),
):
    """Resume the pipeline after human review: apply selected idea_ids + optional direction, then hooks → slides → QA."""
    load_dotenv()
    if no_performance_digest:
        perf_digest_obj = None
    elif performance_digest is not None:
        perf_digest_obj = json.loads(performance_digest.read_text(encoding="utf-8"))
    else:
        perf_digest_obj = load_or_build_performance_digest(explicit_path=None, disabled=False)
    orch_path: Path | None = None
    if orchestrate_program is not None:
        orch_path = orchestrate_program
    state = run_pipeline_continue_from_human_selection(
        state_path=from_state,
        human_selection_path=human_selection,
        out_path=out,
        performance_digest=perf_digest_obj,
        hook_count=hook_count,
        max_slides=max_slides,
        do_hooks=do_hooks,
        do_cta=do_cta,
        do_slides=do_slides,
        do_qa=do_qa,
        judge_govern=judge_govern,
        judge_rewrite_rounds=judge_rewrite_rounds,
        hook_policy=hook_policy,
        hook_rewrite_rounds=hook_rewrite_rounds,
        mock=mock,
        export_dir=export_dir,
        orchestrate_program_path=orch_path,
        force=force,
    )
    print(f"[bold green]Saved:[/bold green] {out}")
    print(f"[bold]Selected:[/bold] {len(state.shortlist.selected_idea_ids)}")


@orchestrator_app.command("status")
def orchestrate_status(
    program: Path = typer.Option(default_factory=experiment_program_json_path),
):
    """Show experiment program queue and paths."""
    p = load_program(program)
    print(f"[bold]Program file:[/bold] {program.resolve()}")
    print(f"[bold]Queue ({len(p.queue)}):[/bold]")
    for q in p.queue:
        print(
            f"  - {q.id} status={q.status} source={q.source} pillar={q.target_pillar!r} "
            f"treatment={q.treatment_key!r} attached={q.attached_experiment_id!r}"
        )
    if p.notes:
        print(f"[bold]Notes:[/bold] {p.notes}")


@orchestrator_app.command("signals")
def orchestrate_signals(
    performance_jsonl: Path = typer.Option(default_factory=performance_jsonl_path),
    performance_digest: Path | None = typer.Option(None, "--performance-digest", exists=True),
    experiments_dir: Path = typer.Option(default_factory=experiments_dir_default),
):
    """Print orchestrator signals JSON (performance + digest + recent experiment results)."""
    sig = build_orchestrator_signals(
        performance_jsonl=performance_jsonl,
        performance_digest_path=performance_digest,
        experiments_dir=experiments_dir,
    )
    print(json.dumps(sig, ensure_ascii=False, indent=2, default=str))


@orchestrator_app.command("suggest")
def orchestrate_suggest(
    program: Path = typer.Option(default_factory=experiment_program_json_path),
    llm: bool = typer.Option(False, "--llm", help="Use MODEL_ORCHESTRATOR to propose hypotheses."),
    rules: bool = typer.Option(False, "--rules", help="Add one rule-based hypothesis (no API)."),
    append: bool = typer.Option(
        False,
        "--append",
        help="Append to queue; without this, refuse if queue is non-empty (unless --force).",
    ),
    force: bool = typer.Option(False, "--force", help="Replace queue when not using --append (destructive)."),
    mock: bool = typer.Option(False, "--mock", help="Use MockClient for --llm (tests only)."),
):
    """
    Fill or extend the experiment program queue from performance signals.
    With no flags: adds one rule-based hypothesis (same as --rules) if the queue has no queued items.
    """
    from .llm import MockClient

    load_dotenv()
    sig = build_orchestrator_signals()
    st = load_program(program)
    pending = [q for q in st.queue if q.status == "queued"]
    if not llm and not rules:
        rules = True
    if pending and not append and not force:
        print("[yellow]Queue already has queued items. Pass --append or --force.[/yellow]")
        raise typer.Exit(code=1)

    items: list = []
    notes: str | None = None
    if llm:
        cl = MockClient() if mock else None
        llm_items, notes = suggest_with_llm(signals=sig, program=st, client=cl)
        items.extend(llm_items)
    if rules:
        items.append(suggest_rules_based(signals=sig))

    if force and not append:
        st.queue = [q for q in st.queue if q.status != "queued"]
    if notes:
        st.notes = ((st.notes or "").strip() + "\n" + notes).strip() if st.notes else notes
    append_suggestions_to_program(st, items, path=program)
    print(f"[bold green]Updated queue:[/bold green] {program} (+{len(items)} item(s))")
    orchestrate_status(program=program)


@orchestrator_app.command("init")
def orchestrate_init(
    program: Path = typer.Option(default_factory=experiment_program_json_path),
    force: bool = typer.Option(False, "--force", help="Overwrite existing file."),
):
    """Create an empty experiment program JSON."""
    if program.exists() and not force:
        print(f"[yellow]Already exists:[/yellow] {program} (use --force)")
        raise typer.Exit(code=1)
    save_program(load_program(program), path=program)
    print(f"[bold green]Initialized:[/bold green] {program}")


@app.command("build-performance-digest")
def build_performance_digest_cmd(
    performance_jsonl: Path = typer.Option(default_factory=performance_jsonl_path),
    tags_jsonl: Path | None = typer.Option(default_factory=tags_jsonl_path),
    ocr_jsonl: Path | None = typer.Option(default_factory=ocr_slides_jsonl_path),
    tracker_csv: Path | None = typer.Option(
        default_factory=marketing_tracker_csv_path,
        help="Join IG permalinks in performance rows to Asset_ID for tags/OCR.",
    ),
    out: Path = typer.Option(default_factory=performance_digest_json_path),
    top_n: int = typer.Option(3, "--top-n"),
    followers_instagram: int | None = typer.Option(139, "--followers-instagram"),
):
    """
    Build a small JSON digest of past carousel performance (deterministic, no LLM calls).
    Intended to be injected into ideation scoring as weak priors/tie-breakers.
    """
    tc = tracker_csv if tracker_csv and tracker_csv.exists() else None
    digest = build_performance_digest(
        performance_jsonl=performance_jsonl,
        tags_jsonl=tags_jsonl,
        ocr_jsonl=ocr_jsonl,
        tracker_csv=tc,
        top_n=top_n,
        followers_instagram=followers_instagram,
    )
    save_digest(digest, out_path=out)
    print(f"[bold green]Saved performance digest:[/bold green] {out}")


@experiment_app.command("analyze")
def experiment_analyze_cmd(
    experiment_id: str = typer.Option(..., "--experiment-id", help="experiment_id to filter in performance JSONL"),
    performance_jsonl: Path = typer.Option(default_factory=performance_jsonl_path),
    out_dir: Path = typer.Option(default_factory=experiments_dir_default, help="Writes <experiment_id>_result.json here."),
    primary_metric: str = typer.Option(
        "save_rate",
        "--metric",
        help="save_rate | share_rate | profile_visit_rate",
    ),
    min_impressions: int = typer.Option(2000, "--min-impressions", help="Per-arm minimum impressions for meets_min_impressions."),
):
    """Compare arms A vs B for one experiment_id; saves JSON for digest_writer_hints."""
    allowed = {"save_rate", "share_rate", "profile_visit_rate"}
    if primary_metric not in allowed:
        raise typer.BadParameter(f"--metric must be one of: {sorted(allowed)}")
    result = analyze_experiment(
        experiment_id=experiment_id,
        performance_jsonl=performance_jsonl,
        primary_metric=primary_metric,
        min_impressions_per_arm=min_impressions,
    )
    out_path = save_experiment_result(result, out_path=out_dir / f"{experiment_id}_result.json")
    print(f"[bold green]Saved:[/bold green] {out_path}")
    print(f"[bold]winner[/bold]: {result.get('winner')}  [bold]meets_min_impressions[/bold]: {result.get('meets_min_impressions')}")


@experiment_app.command("template")
def experiment_template_cmd(
    out: Path | None = typer.Option(None, "--out", help="Write JSON file; if omitted, print to stdout."),
):
    """Print a starter ExperimentSpec JSON for use with run --experiment-json."""
    spec = ExperimentSpec(
        experiment_id="exp_change_me",
        hypothesis="Describe what you are testing.",
        idea_id="i001",
        treatment_key="hook_style_A_vs_B",
        primary_metric="save_rate",
        min_impressions_per_arm=2000,
        status="planned",
        schedule_arm_order=None,
    )
    text = spec.model_dump_json(indent=2)
    if out is not None:
        out.write_text(text + "\n", encoding="utf-8")
        print(f"[bold green]Saved:[/bold green] {out}")
    else:
        print(text)


@app.command("run-hooks-cta")
def run_hooks_cta(
    from_state: Path = typer.Option(..., "--from-state", exists=True, file_okay=True, dir_okay=False),
    out: Path = typer.Option(..., "--out"),
    export_dir: Path = typer.Option(default_factory=export_dir_default),
    export_tag: str = typer.Option(..., "--export-tag", help="Suffix for export folder + bundle__ copy (e.g. j_gem25_w_g41)."),
    hook_count: int = typer.Option(12, "--hook-count"),
    judge_govern: bool = typer.Option(True, "--judge-govern/--no-judge-govern"),
    hook_policy: bool = typer.Option(
        False,
        "--hook-policy/--no-hook-policy",
        help="Enable hook language policy + Hook Judge (default: off).",
    ),
    hook_rewrite_rounds: int = typer.Option(2, "--hook-rewrite-rounds"),
    mock: bool = typer.Option(False, "--mock"),
    performance_digest: Path | None = typer.Option(
        None,
        "--performance-digest",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    no_performance_digest: bool = typer.Option(False, "--no-performance-digest"),
):
    """
    Re-run hooks + CTA only from a saved run JSON (frozen shortlist).
    MODEL_WRITER required; use --hook-policy for banned-phrase loop + MODEL_HOOK_JUDGE (default: hooks are Writer-only).
    """
    load_dotenv()
    if no_performance_digest:
        perf_digest_obj = None
    elif performance_digest is not None:
        perf_digest_obj = json.loads(performance_digest.read_text(encoding="utf-8"))
    else:
        perf_digest_obj = load_or_build_performance_digest(explicit_path=None, disabled=False)
    state = run_hooks_cta_from_state(
        state_path=from_state,
        out_path=out,
        export_dir=export_dir,
        export_document_id_suffix=export_tag,
        hook_count=hook_count,
        judge_govern=judge_govern,
        hook_policy=hook_policy,
        hook_rewrite_rounds=hook_rewrite_rounds,
        mock=mock,
        performance_digest=perf_digest_obj,
    )
    run_dir = export_dir / state.document.document_id
    bundle = run_dir / "bundle.md"
    if bundle.exists():
        uniq = run_dir / f"bundle__{out.stem}.md"
        uniq.write_text(bundle.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"[bold green]Saved:[/bold green] {out}")
    print(f"[bold]Selected after run:[/bold] {sum(1 for c in state.candidates if c.selected)}")


@app.command("export-bundle")
def export_bundle(
    run_json: Path = typer.Option(..., "--run-json", exists=True, file_okay=True, dir_okay=False),
    export_dir: Path = typer.Option(default_factory=export_dir_default, help="Write clean markdown exports here."),
):
    """
    Generate clean markdown exports (including bundle.md) from an existing run JSON.
    Does not call any models.
    """
    raw = run_json.read_text(encoding="utf-8")
    state = RunState.model_validate_json(raw)
    out_dir = export_selected_markdown(state=state, export_root=export_dir)
    # Also write a uniquely named bundle for easy comparison / avoiding stale editor tabs.
    bundle = out_dir / "bundle.md"
    if bundle.exists():
        uniq = out_dir / f"bundle__{run_json.stem}.md"
        uniq.write_text(bundle.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"[bold green]Exported:[/bold green] {out_dir}")


@app.command("redraft-slides")
def redraft_slides(
    run_json_in: Path = typer.Option(..., "--run-json-in", exists=True, file_okay=True, dir_okay=False),
    run_json_out: Path = typer.Option(..., "--run-json-out"),
    export_dir: Path = typer.Option(default_factory=export_dir_default, help="Write clean markdown exports here."),
    max_slides: int = typer.Option(10, "--max-slides"),
    qa: bool = typer.Option(True, "--qa/--no-qa"),
    judge_govern: bool = typer.Option(True, "--judge-govern/--no-judge-govern"),
    judge_rewrite_rounds: int = typer.Option(1, "--judge-rewrite-rounds"),
    hook_policy: bool = typer.Option(
        False,
        "--hook-policy/--no-hook-policy",
        help="Enable hook language policy + Hook Judge when regenerating hooks (default: off).",
    ),
    hook_rewrite_rounds: int = typer.Option(2, "--hook-rewrite-rounds"),
    performance_digest: Path | None = typer.Option(
        None,
        "--performance-digest",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    no_performance_digest: bool = typer.Option(False, "--no-performance-digest"),
):
    """
    Re-run Writer (and optional Editor QA) for selected ideas only, then re-export.
    Does not re-extract/re-score/re-select.
    """
    load_dotenv()
    if no_performance_digest:
        perf_digest_obj = None
    elif performance_digest is not None:
        perf_digest_obj = json.loads(performance_digest.read_text(encoding="utf-8"))
    else:
        perf_digest_obj = load_or_build_performance_digest(explicit_path=None, disabled=False)
    state = load_run_state(run_json_in)
    state = redraft_selected_slides(
        state=state,
        max_slides=max_slides,
        do_qa=qa,
        judge_govern=judge_govern,
        judge_rewrite_rounds=judge_rewrite_rounds,
        hook_policy=hook_policy,
        hook_rewrite_rounds=hook_rewrite_rounds,
        performance_digest=perf_digest_obj,
    )
    save_run_state(state=state, path=run_json_out)
    out_dir = export_selected_markdown(state=state, export_root=export_dir)
    print(f"[bold green]Saved:[/bold green] {run_json_out}")
    print(f"[bold green]Exported:[/bold green] {out_dir}")


@app.command("ig-download")
def ig_download(
    csv_path: Path = typer.Option(..., "--csv", exists=True, file_okay=True, dir_okay=False),
    out_dir: Path = typer.Option(Path("exports"), "--outdir"),
    username: str | None = typer.Option(None, "--username", envvar="IG_USERNAME"),
    password: str | None = typer.Option(None, "--password", envvar="IG_PASSWORD"),
    session_dir: Path | None = typer.Option(None, "--session-dir"),
    overwrite: bool = typer.Option(False, "--overwrite"),
    limit: int | None = typer.Option(None, "--limit"),
):
    """
    Download Instagram media for each IG permalink in the tracker CSV.
    Saves into {out_dir}/instagram/{Asset_ID}_{shortcode}/...
    """
    load_dotenv()
    index_path = download_from_tracker(
        csv_path=csv_path,
        out_dir=out_dir,
        ig_username=username,
        ig_password=password,
        session_dir=session_dir,
        overwrite=overwrite,
        limit=limit,
    )
    print(f"[bold green]Saved download index:[/bold green] {index_path}")


@app.command("ig-slides-dataset")
def ig_slides_dataset(
    exports_dir: Path = typer.Option(Path("exports"), "--exports-dir"),
    tracker_csv: Path | None = typer.Option(None, "--tracker-csv", exists=True, file_okay=True, dir_okay=False),
    carousel_only: bool = typer.Option(
        True, "--carousel-only/--include-all", help="If true, keep only rows where tracker 'Asset type' == Carousel."
    ),
    out_csv: Path | None = typer.Option(None, "--out-csv"),
    out_jsonl: Path | None = typer.Option(None, "--out-jsonl"),
):
    """
    Build a slide-level dataset by scanning the Instaloader export folders.
    """
    csv_path, jsonl_path = build_slides_dataset(
        exports_dir=exports_dir,
        tracker_csv=tracker_csv,
        carousel_only=carousel_only and tracker_csv is not None,
        out_csv=out_csv,
        out_jsonl=out_jsonl,
    )
    print(f"[bold green]Saved slides dataset:[/bold green] {csv_path}")
    print(f"[bold green]Saved slides JSONL:[/bold green] {jsonl_path}")


@app.command("build-patient-voice-bank")
def build_patient_voice_bank_cmd(
    in_jsonl: Path = typer.Option(Path("threads.patient_only.dedup_clean.jsonl"), "--in", exists=True, file_okay=True, dir_okay=False),
    outdir: Path = typer.Option(Path("exports/instagram/_dataset"), "--outdir"),
    overwrite: bool = typer.Option(True, "--overwrite/--no-overwrite"),
):
    """
    Build a patient voice language bank from cleaned patient-only thread JSONL.
    """
    paths = default_patient_voice_paths(outdir=outdir)
    build_patient_voice_bank(
        in_threads_jsonl=in_jsonl,
        out_messages_jsonl=paths.messages_jsonl,
        out_snippets_jsonl=paths.snippets_jsonl,
        overwrite=overwrite,
    )
    print(f"[bold green]Saved:[/bold green] {paths.messages_jsonl}")
    print(f"[bold green]Saved:[/bold green] {paths.snippets_jsonl}")


@app.command("build-ocr-carousel-bank")
def build_ocr_carousel_bank_cmd(
    in_jsonl: Path = typer.Option(Path("exports/instagram/_dataset/ocr_slides.jsonl"), "--in", exists=True, file_okay=True, dir_okay=False),
    out_jsonl: Path = typer.Option(Path("exports/instagram/_dataset/ocr_carousel_bank.jsonl"), "--out"),
    overwrite: bool = typer.Option(True, "--overwrite/--no-overwrite"),
):
    """
    Group slide-level OCR into a carousel-level OCR bank (one row per asset_id).
    """
    build_ocr_carousel_bank(ocr_slides_jsonl=in_jsonl, out_carousel_bank_jsonl=out_jsonl, overwrite=overwrite)
    print(f"[bold green]Saved:[/bold green] {out_jsonl}")


@app.command("log-performance")
def log_performance(
    post_id: str | None = typer.Option(None, "--post-id", help="Canonical: use tracker Asset_ID (string)."),
    ig_permalink: str | None = typer.Option(None, "--ig-permalink", help="Optional IG permalink; can be resolved to Asset_ID via tracker."),
    tracker_csv: Path | None = typer.Option(
        Path("Marketing - Content - Tracker - Content Tracker (1).csv"),
        "--tracker-csv",
        help="Used to resolve --ig-permalink to Asset_ID when --post-id is omitted.",
    ),
    idea_id: str = typer.Option(..., "--idea-id"),
    pillar: str = typer.Option(..., "--pillar"),
    format_suggestion: str = typer.Option(..., "--format"),
    hook_style: str | None = typer.Option(None, "--hook-style"),
    hook_id: str | None = typer.Option(None, "--hook-id"),
    impressions: int | None = typer.Option(None, "--impressions"),
    saves: int | None = typer.Option(None, "--saves"),
    shares: int | None = typer.Option(None, "--shares"),
    profile_visits: int | None = typer.Option(None, "--profile-visits"),
    likes: int | None = typer.Option(None, "--likes"),
    comments: int | None = typer.Option(None, "--comments"),
    follows: int | None = typer.Option(None, "--follows"),
    dms: int | None = typer.Option(None, "--dms"),
    notes: str | None = typer.Option(None, "--notes"),
    run_id: str | None = typer.Option(None, "--run-id"),
    document_title: str | None = typer.Option(None, "--document-title"),
    experiment_id: str | None = typer.Option(None, "--experiment-id", help="A/B pair grouping (use with --variant)."),
    variant: str | None = typer.Option(None, "--variant", help="A or B (required when --experiment-id is set)."),
    treatment_key: str | None = typer.Option(None, "--treatment-key", help="Stable treatment label (e.g. hook_style_myth_vs_howto)."),
    base_idea_id: str | None = typer.Option(
        None,
        "--base-idea-id",
        help="Pre-split idea id shared by both arms (optional; aids analysis).",
    ),
    path: Path = typer.Option(default_factory=performance_jsonl_path),
    persona_id: str | None = typer.Option(
        None,
        "--persona-id",
        help="Segment id (matches persona card). If omitted, infer from pillar/hook/format via data/persona_rules.json unless --no-infer-persona.",
    ),
    persona_source: str | None = typer.Option(
        None,
        "--persona-source",
        help="explicit | rules | unclassified (default: explicit if --persona-id set, else inferred).",
    ),
    no_infer_persona: bool = typer.Option(
        False,
        "--no-infer-persona",
        help="Do not apply persona_rules.json when --persona-id is omitted.",
    ),
):
    """
    Append one post's performance row to a JSONL log.
    The main goal is to analyse outcomes by hook_style / pillar / format (and persona_id) over time.
    """
    tracker = build_tracker_index(tracker_csv=tracker_csv) if tracker_csv else None
    canonical_post_id = resolve_post_id(post_id=post_id, ig_permalink=ig_permalink, tracker=tracker)
    if not canonical_post_id:
        raise typer.BadParameter("Provide --post-id (Asset_ID) or --ig-permalink resolvable via --tracker-csv.")

    eid = (experiment_id or "").strip() or None
    vraw = (variant or "").strip().upper() or None
    if vraw and vraw not in ("A", "B"):
        raise typer.BadParameter("--variant must be A or B")
    if eid and not vraw:
        raise typer.BadParameter("--variant A|B is required when --experiment-id is set")
    if vraw and not eid:
        raise typer.BadParameter("--experiment-id is required when --variant is set")

    variant_out: Literal["A", "B"] | None = None
    if vraw == "A":
        variant_out = "A"
    elif vraw == "B":
        variant_out = "B"

    PersonaSrc = Literal["explicit", "rules", "unclassified"]
    pid: str | None = (persona_id or "").strip() or None
    psrc: PersonaSrc | None = None
    if pid:
        raw_src = (persona_source or "explicit").strip().lower()
        if raw_src not in ("explicit", "rules", "unclassified"):
            raise typer.BadParameter("--persona-source must be explicit, rules, or unclassified.")
        psrc = cast(PersonaSrc, raw_src)
    elif not no_infer_persona:
        pid, inferred = resolve_persona_proxy(pillar=pillar, hook_style=hook_style, format_suggestion=format_suggestion)
        psrc = cast(PersonaSrc, inferred)
    else:
        pid, psrc = None, None

    observed = PerformanceObserved(
        impressions=impressions,
        saves=saves,
        shares=shares,
        profile_visits=profile_visits,
        likes=likes,
        comments=comments,
        follows=follows,
        dms=dms,
    )
    row = PerformanceLog(
        post_id=canonical_post_id,
        idea_id=idea_id,
        pillar=pillar,
        format_suggestion=format_suggestion,
        hook_style=hook_style,
        hook_id=hook_id,
        run_id=run_id,
        document_title=document_title,
        experiment_id=eid,
        variant=variant_out,
        treatment_key=(treatment_key or "").strip() or None,
        base_idea_id=(base_idea_id or "").strip() or None,
        persona_id=pid,
        persona_source=psrc,
        observed=observed,
        derived=compute_derived(observed),
        notes=notes,
    )
    append_performance_log(path=path, row=row)
    print(f"[bold green]Appended:[/bold green] {path}")


@app.command("link-post")
def link_post(
    caption: str | None = typer.Option(None, "--caption", help="Published caption (IG)."),
    caption_file: Path | None = typer.Option(
        None,
        "--caption-file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="Read caption from a UTF-8 text file.",
    ),
    ocr_text: str | None = typer.Option(None, "--ocr-text", help="Concatenated OCR for all slides (plain text)."),
    ocr_file: Path | None = typer.Option(
        None,
        "--ocr-file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="Read OCR blob from a UTF-8 file.",
    ),
    ocr_jsonl: Path | None = typer.Option(
        None,
        "--ocr-jsonl",
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="Slide OCR JSONL (e.g. from ig-ocr); use with --asset-id.",
    ),
    ocr_carousel_bank: Path | None = typer.Option(
        None,
        "--ocr-carousel-bank",
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="ocr_carousel_bank.jsonl; use with --asset-id.",
    ),
    asset_id: str | None = typer.Option(
        None,
        "--asset-id",
        help="Tracker / IG Asset_ID to pull OCR lines for (with --ocr-jsonl or --ocr-carousel-bank).",
    ),
    run_state: Path | None = typer.Option(
        None,
        "--run-state",
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="One RunState JSON file.",
    ),
    run_dir: Path | None = typer.Option(
        None,
        "--run-dir",
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="Scan recursively for *.json RunState files.",
    ),
    run_glob: str = typer.Option(
        "",
        "--glob",
        help='Glob for RunState JSON (e.g. "outputs/**/*.json").',
    ),
    top: int = typer.Option(15, "--top", min=1, max=200),
    min_score: float = typer.Option(0.0, "--min-score"),
    caption_weight: float = typer.Option(0.4, "--caption-weight"),
    ocr_weight: float = typer.Option(0.6, "--ocr-weight"),
    output_json: Path | None = typer.Option(
        None,
        "--output-json",
        help="Write ranked matches as JSON (list of objects).",
    ),
    append_performance_log: Path | None = typer.Option(
        None,
        "--append-performance-log",
        help="Append the best match as one PerformanceLog row to this JSONL path.",
    ),
    post_id: str | None = typer.Option(
        None,
        "--post-id",
        help="Canonical post id (e.g. tracker Asset_ID); required with --append-performance-log.",
    ),
    document_title: str | None = typer.Option(
        None,
        "--document-title",
        help="Override document_title on the performance row (optional).",
    ),
):
    """
    Link a published post to pipeline output when IG's id differs from RunState ids.

    Provide **caption** and/or **OCR** (raw text, file, or --ocr-jsonl/--ocr-carousel-bank + --asset-id).
    Provide **--run-state**, **--run-dir**, and/or **--glob** to define candidate RunState JSON files.

    Scoring blends caption similarity and OCR-vs-slide-draft similarity (token Jaccard + bigram overlap).
    """
    cap = (caption or "").strip()
    if caption_file is not None:
        cap = caption_file.read_text(encoding="utf-8").strip()
    ocr_blob = (ocr_text or "").strip()
    if ocr_file is not None:
        ocr_blob = ocr_file.read_text(encoding="utf-8").strip()
    aid = (asset_id or "").strip()
    if ocr_jsonl is not None:
        if not aid:
            raise typer.BadParameter("--asset-id is required when using --ocr-jsonl")
        ocr_blob = read_ocr_slides_text_for_asset(ocr_jsonl, aid).strip() or ocr_blob
    if ocr_carousel_bank is not None:
        if not aid:
            raise typer.BadParameter("--asset-id is required when using --ocr-carousel-bank")
        bank_text = read_ocr_carousel_bank_row(ocr_carousel_bank, aid)
        if bank_text:
            ocr_blob = bank_text.strip()

    if not cap and not ocr_blob:
        raise typer.BadParameter("Provide at least one of: --caption / --caption-file / --ocr-text / --ocr-file / OCR via --asset-id.")

    path_inputs: list[Path] = []
    if run_state is not None:
        path_inputs.append(run_state)
    if run_dir is not None:
        path_inputs.append(run_dir)
    globs = [run_glob] if (run_glob or "").strip() else []
    if not path_inputs and not globs:
        raise typer.BadParameter("Provide at least one of: --run-state, --run-dir, or --glob.")

    json_paths = discover_run_json_files(path_inputs, globs)
    if not json_paths:
        raise typer.BadParameter("No RunState JSON files found for the given paths/globs.")

    states: list[tuple[Path | None, RunState]] = []
    for jp in json_paths:
        try:
            states.append((jp, load_run_state_path(jp)))
        except Exception as e:
            print(f"[yellow]Skip[/yellow] {jp}: {e}")

    if not states:
        raise typer.BadParameter("No valid RunState JSON could be loaded.")

    matches = match_post_to_runs(
        states=states,
        caption=cap or None,
        ocr=ocr_blob or None,
        caption_weight=caption_weight,
        ocr_weight=ocr_weight,
        top_n=top,
        min_score=min_score,
    )

    if not matches:
        print("[yellow]No matches above --min-score.[/yellow]")
        return

    from rich.table import Table

    tbl = Table(title="link-post (best first)")
    tbl.add_column("score", justify="right")
    tbl.add_column("idea_id")
    tbl.add_column("document_id")
    tbl.add_column("cap")
    tbl.add_column("ocr")
    tbl.add_column("run")
    for m in matches:
        rp = m.run_path.name if m.run_path else "—"
        tbl.add_row(
            f"{m.score:.3f}",
            m.idea_id,
            m.document_id[:16] + ("…" if len(m.document_id) > 16 else ""),
            f"{m.caption_score:.3f}",
            f"{m.ocr_score:.3f}",
            rp,
        )
    print(tbl)

    if output_json is not None:
        payload = [m.to_dict() for m in matches]
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[bold green]Wrote[/bold green] {output_json}")

    if append_performance_log is not None:
        if not (post_id or "").strip():
            raise typer.BadParameter("--post-id is required with --append-performance-log")
        topm = matches[0]
        notes = (
            f"link-post score={topm.score:.4f} caption_sim={topm.caption_score:.4f} ocr_sim={topm.ocr_score:.4f}"
        )
        row = PerformanceLog(
            post_id=str(post_id).strip(),
            idea_id=topm.idea_id,
            pillar=topm.pillar,
            format_suggestion=topm.format_suggestion,
            hook_style=topm.hook_style,
            hook_id=topm.hook_id,
            run_id=topm.document_id,
            document_title=document_title or topm.document_title,
            observed=PerformanceObserved(),
            derived=compute_derived(PerformanceObserved()),
            notes=notes,
        )
        append_performance_log(path=append_performance_log, row=row)
        print(f"[bold green]Appended PerformanceLog[/bold green] → {append_performance_log}")

    print(
        "\n[dim]Tip: pass metrics with `log-performance` or edit the JSONL row; "
        "this append uses empty observed metrics.[/dim]"
    )


@app.command("normalize-performance-ids")
def normalize_performance_ids(
    path_in: Path = typer.Option(default_factory=performance_jsonl_path),
    path_out: Path = typer.Option(default_factory=performance_normalized_jsonl_path),
    tracker_csv: Path = typer.Option(default_factory=marketing_tracker_csv_path),
):
    """
    Rewrite performance.jsonl so PerformanceLog.post_id uses tracker Asset_ID where possible.
    If a row's post_id matches an IG permalink, it will be mapped to Asset_ID.
    """
    tracker = build_tracker_index(tracker_csv=tracker_csv)
    raw = path_in.read_text(encoding="utf-8").splitlines() if path_in.exists() else []
    out_lines: list[str] = []
    changed = 0
    kept = 0
    for line in raw:
        s = line.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except Exception:
            continue
        old = str(obj.get("post_id") or "").strip()
        new = resolve_post_id(post_id=None, ig_permalink=old, tracker=tracker) if old.startswith("http") else old
        if new and new != old:
            obj["post_id"] = new
            changed += 1
        else:
            kept += 1
        out_lines.append(json.dumps(obj, ensure_ascii=False))
    path_out.parent.mkdir(parents=True, exist_ok=True)
    path_out.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")
    print(f"[bold green]Saved:[/bold green] {path_out}")
    print(f"[bold]changed[/bold]={changed}  [bold]unchanged[/bold]={kept}")


@app.command("import-performance-from-tracker")
def import_performance_from_tracker_cmd(
    tracker_csv: Path = typer.Option(default_factory=marketing_tracker_csv_path),
    tags_jsonl: Path | None = typer.Option(default_factory=tags_jsonl_path),
    out: Path = typer.Option(default_factory=performance_jsonl_path),
    overwrite: bool = typer.Option(False, "--overwrite"),
    carousel_only: bool = typer.Option(True, "--carousel-only/--include-all"),
):
    """
    Import IG performance metrics from the marketing tracker into PerformanceLog JSONL.
    Uses `Asset_ID` as canonical `post_id` to join with tags/OCR.
    """
    written, skipped = import_performance_from_tracker(
        tracker_csv=tracker_csv,
        tags_jsonl=tags_jsonl,
        out_jsonl=out,
        overwrite=overwrite,
        carousel_only=carousel_only,
    )
    print(f"[bold green]Saved:[/bold green] {out}")
    print(f"[bold]written[/bold]={written}  [bold]skipped_existing[/bold]={skipped}")


@metrics_app.command("summary")
def metrics_summary(
    path: Path = typer.Option(default_factory=performance_jsonl_path),
    group_by: str = typer.Option("hook_style", "--group-by", help="hook_style|pillar|format|persona|experiment_id"),
    min_n: int = typer.Option(1, "--min-n"),
):
    """Aggregate PerformanceLog JSONL by tag (canonical reporting)."""
    print_performance_summary_table(path=path, group_by=group_by, min_n=min_n)


@metrics_app.command("carousels")
def metrics_carousels(
    tracker_csv: Path = typer.Option(..., "--csv", exists=True, file_okay=True, dir_okay=False),
    out: Path = typer.Option(Path("outputs/reports/carousel_performance.md"), "--out"),
    ig_followers: int = typer.Option(139, "--ig-followers"),
    tags_jsonl: Path | None = typer.Option(None, "--tags-jsonl", help="Optional tags JSONL for pillar/format/hook_style."),
):
    """Markdown report from tracker CSV (carousel rates + leaderboards)."""
    report_path = write_carousel_tracker_report(
        tracker_csv=tracker_csv,
        out=out,
        ig_followers=ig_followers,
        tags_jsonl=tags_jsonl,
    )
    print(f"[bold green]Saved carousel report:[/bold green] {report_path}")


app.add_typer(orchestrator_app, name="orchestrate")
app.add_typer(experiment_app, name="experiment")
app.add_typer(metrics_app, name="metrics")


@app.command("report-performance")
def report_performance(
    path: Path = typer.Option(default_factory=performance_jsonl_path),
    group_by: str = typer.Option("hook_style", "--group-by", help="hook_style|pillar|format|persona|experiment_id"),
    min_n: int = typer.Option(1, "--min-n"),
):
    """
    Quick aggregation to see what's working by tag.

    Prefer: python -m carousel_agents metrics summary (same behaviour).
    """
    print("[dim]Tip: prefer `metrics summary` — see docs/golden-path.md[/dim]")
    print_performance_summary_table(path=path, group_by=group_by, min_n=min_n)


@app.command("ig-ocr")
def ig_ocr(
    slides_jsonl: Path = typer.Option(Path("exports/instagram/slides_dataset.jsonl"), "--slides-jsonl"),
    out_jsonl: Path = typer.Option(Path("exports/instagram/_dataset/ocr_slides.jsonl"), "--out-jsonl"),
    limit: int | None = typer.Option(None, "--limit"),
    overwrite: bool = typer.Option(False, "--overwrite"),
):
    """
    OCR Instaloader-exported carousel slide images into a reusable text dataset.
    """
    load_dotenv()
    model = get_ocr_model()
    out = ocr_slides_dataset(
        slides_jsonl=slides_jsonl,
        repo_root=Path(".").resolve(),
        out_jsonl=out_jsonl,
        model=model,
        limit=limit,
        overwrite=overwrite,
    )
    print(f"[bold green]Saved OCR JSONL:[/bold green] {out}")


@app.command("report-carousels")
def report_carousels(
    tracker_csv: Path = typer.Option(..., "--csv", exists=True, file_okay=True, dir_okay=False),
    out: Path = typer.Option(Path("outputs/reports/carousel_performance.md"), "--out"),
    ig_followers: int = typer.Option(139, "--ig-followers"),
    tags_jsonl: Path | None = typer.Option(None, "--tags-jsonl", help="Optional tags JSONL to report by pillar/format/hook_style."),
):
    """
    Generate a simple performance report for carousels (rates + leaderboards).

    Prefer: python -m carousel_agents metrics carousels (same behaviour).
    """
    print("[dim]Tip: prefer `metrics carousels` — see docs/golden-path.md[/dim]")
    report_path = write_carousel_tracker_report(
        tracker_csv=tracker_csv,
        out=out,
        ig_followers=ig_followers,
        tags_jsonl=tags_jsonl,
    )
    print(f"[bold green]Saved carousel report:[/bold green] {report_path}")


@app.command("tag-carousels")
def tag_carousels_cmd(
    tracker_csv: Path = typer.Option(..., "--csv", exists=True, file_okay=True, dir_okay=False),
    ocr_jsonl: Path = typer.Option(Path("exports/instagram/_dataset/ocr_slides.jsonl"), "--ocr-jsonl"),
    out_jsonl: Path = typer.Option(Path("exports/instagram/_dataset/carousel_tags.jsonl"), "--out-jsonl"),
    limit: int | None = typer.Option(None, "--limit"),
    overwrite: bool = typer.Option(False, "--overwrite"),
):
    """
    Auto-tag past carousels with pillar/format/hook_style using OCR + captions.
    """
    load_dotenv()
    model = get_tagger_model()
    out = tag_carousels(
        tracker_csv=tracker_csv,
        ocr_jsonl=ocr_jsonl,
        out_jsonl=out_jsonl,
        model=model,
        limit=limit,
        overwrite=overwrite,
    )
    print(f"[bold green]Saved carousel tags:[/bold green] {out}")


@app.command("bundle-to-csv")
def bundle_to_csv(
    bundle: Path = typer.Option(..., "--bundle", exists=True, file_okay=True, dir_okay=False),
    out_dir: Path | None = typer.Option(None, "--out-dir"),
):
    """
    Convert a bundle markdown (bundle.md) into:
    - {bundle}__carousels.csv (one row per carousel)
    - {bundle}__slides.csv (one row per slide)
    """
    car_csv, slide_csv = write_bundle_csvs(bundle_path=bundle, out_dir=out_dir)
    print(f"[bold green]Saved[/bold green] {car_csv}")
    print(f"[bold green]Saved[/bold green] {slide_csv}")


@app.command("compare-runs")
def compare_runs(
    bundles: str = typer.Option(..., "--bundles", help="Comma-separated paths to bundle markdown files."),
    out_dir: Path = typer.Option(Path("outputs/comparisons"), "--out-dir"),
):
    """
    Create two CSVs to compare multiple runs side-by-side:
    - comparison_overview.csv (one row per carousel per run)
    - comparison_detail_slides.csv (one row per slide per run)
    """
    bundle_paths = [Path(p.strip()) for p in (bundles or "").split(",") if p.strip()]
    overview, detail = write_comparison_csvs(bundles=bundle_paths, out_dir=out_dir)
    print(f"[bold green]Saved[/bold green] {overview}")
    print(f"[bold green]Saved[/bold green] {detail}")


def main():
    app()


if __name__ == "__main__":
    main()

