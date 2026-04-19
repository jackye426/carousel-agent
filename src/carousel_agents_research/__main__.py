"""
Research & benchmark CLI (optional). Install the same package; run:

  python -m carousel_agents_research experiment-predict --csv ...
"""

from __future__ import annotations

from pathlib import Path

import typer
from dotenv import load_dotenv
from rich import print

from carousel_agents_research.experiment import run_experiment, run_vision_experiment, run_vision_experiment_tt_only
from carousel_agents_research.judge_benchmark import benchmark_judges
from carousel_agents_research.judge_top_ideas import export_top_ideas_per_model

app = typer.Typer(add_completion=False, help="Benchmarks and metric-prediction experiments.")


@app.command("experiment-predict")
def experiment_predict(
    tracker_csv: Path = typer.Option(..., "--csv", exists=True, file_okay=True, dir_okay=False),
    n: int = typer.Option(6, "--n"),
    models: str = typer.Option(
        "openai/gpt-4.1-mini,openai/gpt-4.1",
        "--models",
        help="Comma-separated model ids for metric prediction (text).",
    ),
    out_dir: Path = typer.Option(Path("outputs/experiments"), "--outdir"),
):
    load_dotenv()
    model_list = [m.strip() for m in (models or "").split(",") if m.strip()]
    report = run_experiment(tracker_csv=tracker_csv, n=n, out_dir=out_dir, models=model_list)
    print(f"[bold green]Saved experiment report:[/bold green] {report}")


@app.command("experiment-vision")
def experiment_vision(
    tracker_csv: Path = typer.Option(..., "--csv", exists=True, file_okay=True, dir_okay=False),
    exports_dir: Path = typer.Option(Path("exports"), "--exports-dir"),
    n: int = typer.Option(4, "--n"),
    max_slides: int = typer.Option(10, "--max-slides"),
    tt_only: bool = typer.Option(False, "--tt-only", help="If set, sample only Carousel rows with TT stats."),
    models: str = typer.Option(
        "openai/gpt-4.1-mini,openai/gpt-4.1",
        "--models",
        help="Comma-separated vision-capable model ids (OpenRouter).",
    ),
    out_dir: Path = typer.Option(Path("outputs/experiments"), "--outdir"),
):
    load_dotenv()
    model_list = [m.strip() for m in (models or "").split(",") if m.strip()]
    if tt_only:
        report = run_vision_experiment_tt_only(
            tracker_csv=tracker_csv,
            exports_dir=exports_dir,
            n=n,
            out_dir=out_dir,
            models=model_list,
            max_slides=max_slides,
        )
    else:
        report = run_vision_experiment(
            tracker_csv=tracker_csv,
            exports_dir=exports_dir,
            n=n,
            out_dir=out_dir,
            models=model_list,
            max_slides=max_slides,
        )
    print(f"[bold green]Saved vision experiment report:[/bold green] {report}")


@app.command("judge-benchmark")
def judge_benchmark_cmd(
    input: Path = typer.Option(..., "--input", exists=True, file_okay=True, dir_okay=False),
    models: str = typer.Option(
        "openai/gpt-4.1-mini",
        "--models",
        help="Comma-separated ideation model ids (default: gpt-4.1 mini only).",
    ),
    runs_per_model: int = typer.Option(2, "--runs-per-model"),
    candidate_count: int = typer.Option(30, "--candidate-count"),
    select_n: int = typer.Option(5, "--select-n"),
    select_pool_k: int = typer.Option(10, "--select-pool-k", help="Shortlist must come from top-K by score."),
    temperature: float = typer.Option(0.1, "--temp"),
    timeout_s: float = typer.Option(300.0, "--timeout", help="Per-request HTTP timeout (seconds)."),
    max_chunks: int | None = typer.Option(
        None,
        "--max-chunks",
        help="Optional cap on source chunks (faster smoke tests; omit for full document).",
    ),
    out_dir: Path = typer.Option(Path("outputs/experiments"), "--out-dir"),
):
    load_dotenv()
    model_list = [m.strip() for m in (models or "").split(",") if m.strip()]
    run_dir = benchmark_judges(
        input_path=input,
        models=model_list,
        runs_per_model=runs_per_model,
        candidate_count=candidate_count,
        select_n=select_n,
        select_pool_k=select_pool_k,
        temperature=temperature,
        timeout_s=timeout_s,
        max_chunks=max_chunks,
        out_dir=out_dir,
    )
    print(f"[bold green]Saved ideation benchmark:[/bold green] {run_dir}")


@app.command("judge-top-ideas")
def judge_top_ideas_cmd(
    input: Path = typer.Option(..., "--input", exists=True, file_okay=True, dir_okay=False),
    models: str = typer.Option(
        "openai/gpt-4.1-mini",
        "--models",
        help="Comma-separated ideation model ids (default: gpt-4.1 mini).",
    ),
    top_n: int = typer.Option(3, "--top-n"),
    candidate_count: int = typer.Option(30, "--candidate-count"),
    temperature: float = typer.Option(0.1, "--temp"),
    timeout_s: float = typer.Option(360.0, "--timeout"),
    max_chunks: int | None = typer.Option(None, "--max-chunks"),
    out_dir: Path = typer.Option(Path("outputs/experiments"), "--out-dir"),
):
    load_dotenv()
    model_list = [m.strip() for m in (models or "").split(",") if m.strip()]
    run_dir = export_top_ideas_per_model(
        input_path=input,
        models=model_list,
        top_n=top_n,
        candidate_count=candidate_count,
        temperature=temperature,
        timeout_s=timeout_s,
        max_chunks=max_chunks,
        out_dir=out_dir,
    )
    print(f"[bold green]Saved top ideas:[/bold green] {run_dir}")


def main():
    app()


if __name__ == "__main__":
    main()
