"""
Central path defaults (override with CAROUSEL_* env vars).

Call these after load_dotenv() so .env is respected when the CLI loads.
"""

from __future__ import annotations

import os
from pathlib import Path


def _path(env_var: str, fallback: str) -> Path:
    return Path(os.environ.get(env_var, fallback)).expanduser()


def performance_jsonl_path() -> Path:
    return _path("CAROUSEL_PERFORMANCE_JSONL", "data/performance.jsonl")


def tags_jsonl_path() -> Path:
    return _path("CAROUSEL_TAGS_JSONL", "exports/instagram/_dataset/carousel_tags.jsonl")


def ocr_slides_jsonl_path() -> Path:
    return _path("CAROUSEL_OCR_SLIDES_JSONL", "exports/instagram/_dataset/ocr_slides.jsonl")


def performance_digest_json_path() -> Path:
    return _path("CAROUSEL_PERFORMANCE_DIGEST_JSON", "outputs/performance_digest.json")


def marketing_tracker_csv_path() -> Path:
    return _path("CAROUSEL_MARKETING_TRACKER_CSV", "Marketing - Content - Tracker - Content Tracker (1).csv")


def export_dir_default() -> Path:
    return _path("CAROUSEL_EXPORT_DIR", "outputs")


def run_json_default() -> Path:
    return _path("CAROUSEL_RUN_JSON", "data/run.json")


def performance_normalized_jsonl_path() -> Path:
    return _path("CAROUSEL_PERFORMANCE_NORMALIZED_JSONL", "data/performance.normalized.jsonl")


def experiments_dir_default() -> Path:
    return _path("CAROUSEL_EXPERIMENTS_DIR", "outputs/experiments")


def experiment_program_json_path() -> Path:
    return _path("CAROUSEL_EXPERIMENT_PROGRAM", "data/experiment_program.json")


def max_parallel_writer_workers() -> int:
    raw = (os.environ.get("CAROUSEL_MAX_PARALLEL") or "1").strip()
    try:
        n = int(raw)
    except ValueError:
        return 1
    return max(1, min(n, 32))
