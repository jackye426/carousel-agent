"""
Structured logging for pipeline stages and LLM calls.

Enable with CAROUSEL_LOG_LEVEL=INFO (default WARNING). Logs JSON lines on logger ``carousel_agents``.
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import time
from typing import Any

_run_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("carousel_run_id", default=None)
_document_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("carousel_document_id", default=None)


def configure_logging() -> None:
    level_name = (os.environ.get("CAROUSEL_LOG_LEVEL") or "WARNING").strip().upper()
    level = getattr(logging, level_name, logging.WARNING)
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=level, format="%(message)s")
    logging.getLogger("carousel_agents").setLevel(level)


def set_run_context(*, run_id: str | None, document_id: str | None = None) -> None:
    _run_id_var.set(run_id)
    _document_id_var.set(document_id)


def log_stage(stage: str, **extra: Any) -> None:
    payload: dict[str, Any] = {
        "event": "stage",
        "stage": stage,
        "run_id": _run_id_var.get(),
        "document_id": _document_id_var.get(),
        **extra,
    }
    logging.getLogger("carousel_agents").info(json.dumps(payload, default=str))


def log_llm_call(
    *,
    stage: str,
    model: str,
    temperature: float,
    latency_ms: float,
    usage: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "event": "llm_call",
        "stage": stage,
        "run_id": _run_id_var.get(),
        "document_id": _document_id_var.get(),
        "model": model,
        "temperature": temperature,
        "latency_ms": round(latency_ms, 2),
    }
    if usage:
        payload["usage"] = usage
    logging.getLogger("carousel_agents").info(json.dumps(payload, default=str))


class StageTimer:
    def __init__(self, stage: str) -> None:
        self.stage = stage
        self._t0 = time.perf_counter()

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._t0) * 1000.0

    def log_done(self, **extra: Any) -> None:
        log_stage(self.stage, duration_ms=round(self.elapsed_ms(), 2), **extra)
