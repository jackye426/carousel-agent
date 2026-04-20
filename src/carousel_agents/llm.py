from __future__ import annotations

import json
import os
import base64
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeVar

import httpx
from pydantic import BaseModel

import instructor
from openai import OpenAI
from httpx import Timeout


TResp = TypeVar("TResp", bound=BaseModel)


Role = Literal["ideation", "writer", "editor", "hook_judge"]


@dataclass(frozen=True)
class ModelConfig:
    role: Role
    model: str
    temperature: float = 0.2


class LLMError(RuntimeError):
    pass


# OpenRouter-style ids (override via MODEL_* env vars).
DEFAULT_MODEL_IDEATION = "mistralai/mistral-large"
# When split ideation is on and MODEL_IDEATION / MODEL_JUDGE are unset, scoring uses this default.
DEFAULT_MODEL_IDEATION_SCORE = "mistralai/mistral-large"
DEFAULT_MODEL_HOOK_JUDGE = "mistralai/mistral-large"
DEFAULT_MODEL_WRITER = "google/gemini-2.5-pro-preview"


def _env(name: str, default: str | None = None) -> str | None:
    v = os.environ.get(name)
    return v if v is not None and v != "" else default


def _ideation_model_id() -> str:
    """Ideation = extract + score + shortlist/carousel governance. MODEL_JUDGE kept as legacy alias."""
    return _env("MODEL_IDEATION") or _env("MODEL_JUDGE", DEFAULT_MODEL_IDEATION)


def _ideation_temperature() -> float:
    t = _env("TEMP_IDEATION")
    if t is not None:
        return float(t)
    return float(_env("TEMP_JUDGE", "0.1"))


def _temp_ideation_extract() -> float:
    t = _env("TEMP_IDEATION_EXTRACT")
    if t is not None:
        return float(t)
    return _ideation_temperature()


def _temp_ideation_score() -> float:
    t = _env("TEMP_IDEATION_SCORE")
    if t is not None:
        return float(t)
    return _ideation_temperature()


def get_ideation_extract_config(*, split_ideation: bool) -> ModelConfig:
    """Extract candidates. When split_ideation=False, same as unified ideation model."""
    if not split_ideation:
        return get_model_config("ideation")
    model = _env("MODEL_IDEATION_EXTRACT") or _env("MODEL_IDEATION") or _env("MODEL_JUDGE", DEFAULT_MODEL_IDEATION)
    return ModelConfig(role="ideation", model=model, temperature=_temp_ideation_extract())


def get_ideation_score_config(*, split_ideation: bool) -> ModelConfig:
    """
    Score + downstream ideation governance (shortlist veto, carousel review).
    When split_ideation=False, same as unified ideation model.
    """
    if not split_ideation:
        return get_model_config("ideation")
    model = _env("MODEL_IDEATION_SCORE") or _env("MODEL_IDEATION") or _env("MODEL_JUDGE", DEFAULT_MODEL_IDEATION_SCORE)
    return ModelConfig(role="ideation", model=model, temperature=_temp_ideation_score())


def get_hook_judge_config() -> ModelConfig:
    """
    Critiques hooks after the Writer produces them (policy revise/veto loop).
    Defaults to Mistral Large when MODEL_HOOK_JUDGE is unset.
    """
    model = _env("MODEL_HOOK_JUDGE") or DEFAULT_MODEL_HOOK_JUDGE
    t_hook = _env("TEMP_HOOK_JUDGE")
    if t_hook is not None:
        temp = float(t_hook)
    else:
        temp = float(_env("TEMP_JUDGE", "0.0"))
    return ModelConfig(role="hook_judge", model=model, temperature=temp)


def get_orchestrator_config() -> ModelConfig:
    """
    Experiment program orchestrator (queue planning). Defaults to MODEL_IDEATION if MODEL_ORCHESTRATOR unset.
    """
    model = _env("MODEL_ORCHESTRATOR") or _ideation_model_id()
    t = _env("TEMP_ORCHESTRATOR")
    temp = float(t) if t is not None else 0.25
    return ModelConfig(role="ideation", model=model, temperature=temp)


def _instructor_mode_for_base_url(base_url: str) -> instructor.Mode:
    b = base_url.lower()
    if "openrouter" in b:
        return instructor.Mode.OPENROUTER_STRUCTURED_OUTPUTS
    return instructor.Mode.TOOLS


def get_model_config(role: Role | str) -> ModelConfig:
    # Accept legacy "judge" alias for ideation.
    r = str(role)
    if r == "judge":
        r = "ideation"
    # Defaults use OpenRouter-style ids (see .env.example). Override via MODEL_* env vars.
    if r == "ideation":
        return ModelConfig(role="ideation", model=_ideation_model_id(), temperature=_ideation_temperature())
    if r == "writer":
        return ModelConfig(role="writer", model=_env("MODEL_WRITER", DEFAULT_MODEL_WRITER), temperature=float(_env("TEMP_WRITER", "0.6")))
    if r == "hook_judge":
        return get_hook_judge_config()
    return ModelConfig(role="editor", model=_env("MODEL_EDITOR", "openai/gpt-4.1-mini"), temperature=float(_env("TEMP_EDITOR", "0.2")))


class OpenAICompatibleClient:
    """
    Minimal OpenAI-compatible Chat Completions client.
    Works with OpenAI and many compatible providers via OPENAI_BASE_URL.
    """

    def __init__(self) -> None:
        # Prefer OpenRouter env naming (repo default), but support OPENAI_API_KEY too.
        self.api_key = _env("OPENROUTER_API_KEY") or _env("openrouter_api_key") or _env("OPENAI_API_KEY")
        if not self.api_key:
            raise LLMError("Set OPENROUTER_API_KEY (preferred) or OPENAI_API_KEY in your environment/.env.")

        # Default base URL to OpenRouter; can be overridden for any OpenAI-compatible provider.
        self.base_url = _env("OPENAI_BASE_URL", _env("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")).rstrip("/")

        # OpenRouter recommended optional headers for analytics/rate limiting context.
        self.site_url = _env("OPENROUTER_SITE_URL")
        self.app_name = _env("OPENROUTER_APP_NAME", "carousel-agents")
        self._instructor_client: instructor.Instructor | None = None

    def _get_instructor(self) -> instructor.Instructor:
        if self._instructor_client is None:
            t = float(_env("LLM_TIMEOUT_S", "300"))
            headers: dict[str, str] = {}
            if self.site_url:
                headers["HTTP-Referer"] = self.site_url
            if self.app_name:
                headers["X-Title"] = self.app_name
            mode = _instructor_mode_for_base_url(self.base_url)
            oa = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                default_headers=headers or None,
                timeout=Timeout(t, connect=60.0),
            )
            self._instructor_client = instructor.from_openai(oa, mode=mode)
        return self._instructor_client

    def chat_structured(
        self,
        *,
        response_model: type[TResp],
        model: str,
        system: str,
        user: str,
        temperature: float,
        max_retries: int | None = None,
        timeout_s: float | None = None,
    ) -> TResp:
        """
        Pydantic-validated completion with Instructor retries on validation failure.
        Uses OpenRouter-structured mode when the base URL targets OpenRouter.
        """
        ic = self._get_instructor()
        retries = max(0, int(max_retries if max_retries is not None else float(_env("INSTRUCTOR_MAX_RETRIES", "3"))))
        t = float(timeout_s) if timeout_s is not None else float(_env("LLM_TIMEOUT_S", "300"))
        t0 = time.perf_counter()
        try:
            out = ic.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                response_model=response_model,
                max_retries=retries,
                timeout=t,
            )
            return out
        finally:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            try:
                from .observability import log_llm_call

                log_llm_call(
                    stage="instructor_structured",
                    model=model,
                    temperature=temperature,
                    latency_ms=latency_ms,
                    usage=None,
                )
            except Exception:
                pass

    def chat_json(
        self,
        *,
        model: str,
        system: str,
        user: str,
        temperature: float,
        json_schema: dict[str, Any] | None = None,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.app_name:
            headers["X-Title"] = self.app_name

        # Prefer strict JSON responses when supported. Many providers ignore/relax this.
        response_format: dict[str, Any] | None = {"type": "json_object"}
        if json_schema:
            response_format = {"type": "json_schema", "json_schema": {"name": "response", "schema": json_schema, "strict": True}}

        payload: dict[str, Any] = {
            "model": model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if response_format:
            payload["response_format"] = response_format

        t = float(timeout_s) if timeout_s is not None else float(_env("LLM_TIMEOUT_S", "300"))
        t0 = time.perf_counter()
        with httpx.Client(timeout=httpx.Timeout(t, connect=60.0)) as client:
            r = client.post(url, headers=headers, json=payload)
            if r.status_code >= 400:
                raise LLMError(f"LLM error {r.status_code}: {r.text[:500]}")
            data = r.json()
        latency_ms = (time.perf_counter() - t0) * 1000.0
        usage: dict[str, Any] | None = None
        if isinstance(data, dict):
            u = data.get("usage")
            if isinstance(u, dict):
                usage = dict(u)
        try:
            from .observability import log_llm_call

            log_llm_call(
                stage="chat_completions_json",
                model=model,
                temperature=temperature,
                latency_ms=latency_ms,
                usage=usage,
            )
        except Exception:
            pass

        # Some providers (including OpenRouter) may return a 200 with an error payload.
        if isinstance(data, dict) and "error" in data:
            err = data.get("error") or {}
            msg = err.get("message") or err.get("error") or str(err)
            code = err.get("code")
            raise LLMError(f"LLM provider error (code={code}): {msg}")

        try:
            content = data["choices"][0]["message"]["content"]
        except Exception as e:
            raise LLMError(f"Unexpected LLM response shape: {data}") from e

        try:
            return json.loads(content)
        except Exception as e:
            raise LLMError(f"Model did not return valid JSON. Content starts: {content[:200]}") from e

    def chat_json_messages(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float,
        json_schema: dict[str, Any] | None = None,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        """
        Like chat_json but allows full message payload (including multimodal content parts).
        """
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.app_name:
            headers["X-Title"] = self.app_name

        response_format: dict[str, Any] | None = {"type": "json_object"}
        if json_schema:
            response_format = {"type": "json_schema", "json_schema": {"name": "response", "schema": json_schema, "strict": True}}

        payload: dict[str, Any] = {
            "model": model,
            "temperature": temperature,
            "messages": messages,
        }
        if response_format:
            payload["response_format"] = response_format

        t = float(timeout_s) if timeout_s is not None else float(_env("LLM_TIMEOUT_S", "300"))
        with httpx.Client(timeout=httpx.Timeout(t, connect=60.0)) as client:
            r = client.post(url, headers=headers, json=payload)
            if r.status_code >= 400:
                raise LLMError(f"LLM error {r.status_code}: {r.text[:500]}")
            data = r.json()

        if isinstance(data, dict) and "error" in data:
            err = data.get("error") or {}
            msg = err.get("message") or err.get("error") or str(err)
            code = err.get("code")
            raise LLMError(f"LLM provider error (code={code}): {msg}")

        try:
            content = data["choices"][0]["message"]["content"]
        except Exception as e:
            raise LLMError(f"Unexpected LLM response shape: {data}") from e

        try:
            return json.loads(content)
        except Exception as e:
            raise LLMError(f"Model did not return valid JSON. Content starts: {content[:200]}") from e


def image_file_to_data_url(path: str) -> str:
    """
    Returns a data URL suitable for OpenAI-compatible image_url parts.
    """
    p = path.lower()
    if p.endswith(".png"):
        mime = "image/png"
    elif p.endswith(".webp"):
        mime = "image/webp"
    else:
        mime = "image/jpeg"
    b = Path(path).read_bytes()  # type: ignore[name-defined]
    enc = base64.b64encode(b).decode("ascii")
    return f"data:{mime};base64,{enc}"


class MockClient:
    """
    Deterministic client for local testing without API keys.
    Returns minimal valid JSON shapes for each pipeline stage.
    """

    def chat_json(
        self,
        *,
        model: str,
        system: str,
        user: str,
        temperature: float,
        json_schema: dict[str, Any] | None = None,
        timeout_s: float = 120.0,
    ) -> dict[str, Any]:
        _ = (model, system, temperature, json_schema, timeout_s)
        if '"candidates"' in user and "Extract carousel-worthy candidate ideas" in user:
            return {
                "editorial_direction": "Mock: focus on practical, low-friction habits grounded in the source text.",
                "candidates": [
                    {
                        "idea_id": "i001",
                        "content_pillar": "habits",
                        "topic": "A 2-minute reset you can do between tasks",
                        "angle": "micro-habit + friction removal",
                        "core_claim": "Small reset rituals can reduce task-switching drag.",
                        "audience_pain": "I feel scattered all day.",
                        "promise": "A simple reset to feel focused again.",
                        "reader_benefit": "You get a fast way to reset attention between tasks without a long routine.",
                        "format_suggestion": "steps",
                        "source_citations": [
                            {"chunk_id": "c001", "excerpt": "TODO: replace with real excerpt", "note": "placeholder"},
                            {"chunk_id": "c001", "excerpt": "TODO: replace with real excerpt", "note": "placeholder"},
                        ],
                        "safety_flags": [],
                    }
                ]
            }
        if '"scores"' in user and "Score each candidate idea" in user:
            return {
                "scores": [
                    {
                        "idea_id": "i001",
                        "hook_strength": 3,
                        "clarity": 3,
                        "save_share": 3,
                        "strategic_fit": 3,
                        "source_support": 1,
                    }
                ]
            }
        if "Select a shortlist" in user:
            return {
                "selected_idea_ids": ["i001"],
                "notes": "Mock selection.",
                "reasons": [{"idea_id": "i001", "selection_reason": "Mock reason."}],
            }
        if "Review the selected shortlist" in user:
            return {
                "reviewed": [
                    {
                        "idea_id": "i001",
                        "decision": {"status": "approve", "reason": "mock", "directives": []},
                    }
                ],
                "notes": "mock shortlist review",
            }
        if "Generate CTA options for this selected idea" in user:
            return {"cta_options": ["Mock CTA one", "Mock CTA two"], "best_cta": "Mock CTA one"}
        if "Write an Instagram caption for this carousel" in user:
            return {"caption": "Mock caption opener.\n\nMock body line.\n\nMock CTA one\n\n#endo #ukhealth"}
        if "Generate hook options" in user:
            return {
                "hooks": [
                    {"hook_id": "h001", "text": "Steal this 2-minute reset between tasks", "style": "how_to", "rationale": "Direct + actionable."},
                    {"hook_id": "h002", "text": "Myth: resets are fluff — here's the evidence", "style": "myth_bust", "rationale": "Contrast frame for A/B."},
                ],
                "best_hook_id": "h001",
                "outline": [{"slide": 1, "purpose": "cover", "key_point": "2-minute reset", "citations": ["c001"]}],
            }
        if "Draft a carousel" in user:
            return {
                "carousel_draft": {
                    "slides": [
                        {"slide_number": 1, "main_text": "The 2-minute reset", "subtext": "Use this between tasks.", "citations": ["c001"]}
                    ],
                    "cta": "Save this for later.",
                    "disclaimer": None,
                }
            }
        if "Critique this drafted carousel" in user:
            return {
                "idea_id": "i001",
                "decision": {"status": "approve", "reason": "mock", "directives": []},
                "issues": [],
            }
        if "Rewrite this carousel draft to address the Ideation pass directives" in user:
            return {
                "carousel_draft": {
                    "slides": [
                        {
                            "slide_number": 1,
                            "main_text": "Rewritten slide",
                            "subtext": "Detail.",
                            "citations": ["c001"],
                        }
                    ],
                    "cta": "Save this for later.",
                    "disclaimer": None,
                }
            }
        if "Rewrite hook options to comply with the hook language policy" in user:
            return {
                "hooks": [
                    {
                        "hook_id": "h001",
                        "text": "Compliant hook text for mock rewrite",
                        "style": "how_to",
                        "rationale": "mock rewrite",
                    }
                ],
                "best_hook_id": "h001",
            }
        if "You are the Writer for DocMap carousel copy" in user and "need_clarification" in user:
            if "You already asked questions" in user:
                return {
                    "need_clarification": False,
                    "questions": [],
                    "assistant_message": "Mock: Thanks — I have what I need to draft hooks.",
                }
            return {
                "need_clarification": True,
                "questions": [
                    "Which NHS vs private angle should we emphasise for this batch?",
                    "Any words or phrases we must avoid for brand safety?",
                ],
                "assistant_message": "Mock: I have two quick questions before drafting hooks.",
            }
        if "QA/edit pass" in user:
            return {
                "carousel_draft": {
                    "slides": [
                        {"slide_number": 1, "main_text": "The 2-minute reset", "subtext": "Use this between tasks.", "citations": ["c001"]}
                    ],
                    "cta": "Save this for later.",
                    "disclaimer": None,
                },
                "qa": {"passed": True, "issues": []},
            }
        if "You are enforcing a hard hook language policy" in user:
            return {"status": "revise", "reason": "mock_policy", "directives": ["Replace banned phrasing with one concrete, patient-grounded detail."]}
        if "experiment program planner" in system.lower():
            return {
                "hypotheses": [
                    {
                        "hypothesis": "Mock: contrast myth_bust vs how_to hooks on access_or_decision.",
                        "treatment_key": "hook_style_contrast",
                        "target_pillar": "access_or_decision",
                        "topic_hint": None,
                        "rationale": "mock orchestrator",
                    }
                ],
                "program_notes": "Mock orchestrator notes.",
            }
        return {}

    def chat_structured(
        self,
        *,
        response_model: type[TResp],
        model: str,
        system: str,
        user: str,
        temperature: float,
        max_retries: int | None = None,
        timeout_s: float | None = None,
    ) -> TResp:
        _ = (max_retries, timeout_s)
        raw = self.chat_json(
            model=model,
            system=system,
            user=user,
            temperature=temperature,
            json_schema=None,
        )
        if not isinstance(raw, dict):
            raw = {}
        return response_model.model_validate(raw)

