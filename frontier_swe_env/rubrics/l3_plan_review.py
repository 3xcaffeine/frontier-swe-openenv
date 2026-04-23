"""L3: LLM-based plan review rubric — scores the agent's proposed subtask plan."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

import openai
from openenv.core.rubrics.base import Rubric

logger = logging.getLogger(__name__)

_DEFAULT_MAX_RETRIES = 3
_DEFAULT_RETRY_BACKOFF = [15, 30, 60]

L3_PROMPT_TEMPLATE = """\
You are evaluating a software engineering plan.

Task: {task_description}

Task instruction (summary):
{instruction_summary}

The agent proposed the following subtask plan:
{plan_json}

Score the following dimensions (integers only):
- coverage (0-10): Does the plan cover the key aspects of the task?
- ordering (0-5): Are subtasks in a logical dependency order?
- granularity (0-5): Are subtasks appropriately sized (not too broad, not too narrow)?
- ambition (0-5): Does the plan aim for meaningful progress given the time constraint?
- time_awareness (0-5): Is the plan realistic for the available time?

Respond ONLY with valid JSON:
{{"coverage": N, "ordering": N, "granularity": N, "ambition": N, "time_awareness": N, "feedback": "..."}}
"""

# L3 dimensions are fixed (task-agnostic plan quality metrics)
_L3_DIMENSIONS = [
    {"name": "coverage", "max": 10},
    {"name": "ordering", "max": 5},
    {"name": "granularity", "max": 5},
    {"name": "ambition", "max": 5},
    {"name": "time_awareness", "max": 5},
]
_L3_MAX_SCORE = sum(d["max"] for d in _L3_DIMENSIONS)


@dataclass
class L3GradingResult:
    """Structured output from L3 plan review."""

    coverage: int = 0
    ordering: int = 0
    granularity: int = 0
    ambition: int = 0
    time_awareness: int = 0
    feedback: str = ""
    normalized: float = 0.0
    metrics: dict[str, float | int] = field(default_factory=dict)


class L3PlanReviewRubric(Rubric):
    """LLM judge that evaluates the quality of an agent's subtask plan.

    Scores five dimensions and normalizes to [0, 1]:
        ``(coverage + ordering + granularity + ambition + time_awareness) / 30``

    Uses the OpenAI-compatible API.
    """

    def __init__(
        self,
        task_description: str = "",
        grader_model: str | None = None,
        api_base_url: str | None = None,
        api_key: str | None = None,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        retry_backoff: list[int] | None = None,
        timeout_seconds: int = 120,
    ):
        super().__init__()
        self.task_description = task_description
        self.grader_model = grader_model
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff or list(_DEFAULT_RETRY_BACKOFF)
        self.timeout_seconds = timeout_seconds

        client_kwargs: dict[str, Any] = {}
        if api_base_url is not None:
            client_kwargs["base_url"] = api_base_url
        if api_key is not None:
            client_kwargs["api_key"] = api_key
        self._client = openai.AsyncOpenAI(**client_kwargs)

    def _build_prompt(self, instruction_summary: str, plan: list[dict]) -> str:
        plan_json = json.dumps(plan, indent=2)
        return L3_PROMPT_TEMPLATE.format(
            task_description=self.task_description or "a software engineering task",
            instruction_summary=instruction_summary,
            plan_json=plan_json,
        )

    async def _call_llm(self, prompt: str) -> str:
        response = await self._client.chat.completions.create(
            model=self.grader_model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""

    def _parse_response(self, text: str) -> L3GradingResult:
        json_match = re.search(r"\{[^}]+\}", text, re.DOTALL)
        if not json_match:
            return L3GradingResult(feedback="Failed to parse JSON from response.")

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError:
            return L3GradingResult(feedback="Invalid JSON in response.")

        coverage = max(0, min(10, int(data.get("coverage", 0))))
        ordering = max(0, min(5, int(data.get("ordering", 0))))
        granularity = max(0, min(5, int(data.get("granularity", 0))))
        ambition = max(0, min(5, int(data.get("ambition", 0))))
        time_awareness = max(0, min(5, int(data.get("time_awareness", 0))))
        feedback = str(data.get("feedback", ""))
        raw_sum = coverage + ordering + granularity + ambition + time_awareness
        normalized = raw_sum / _L3_MAX_SCORE

        return L3GradingResult(
            coverage=coverage,
            ordering=ordering,
            granularity=granularity,
            ambition=ambition,
            time_awareness=time_awareness,
            feedback=feedback,
            normalized=normalized,
        )

    def _backoff(self, attempt: int) -> int:
        idx = min(attempt - 1, len(self.retry_backoff) - 1)
        return self.retry_backoff[idx]

    async def grade(
        self,
        instruction_summary: str,
        plan: list[dict],
    ) -> L3GradingResult:
        """Run the full L3 plan review pipeline."""
        if not plan:
            return L3GradingResult(
                feedback="Empty plan — nothing to evaluate.",
                metrics={"l3/empty_plan": 1},
            )

        prompt = self._build_prompt(instruction_summary, plan)
        t0 = time.perf_counter()

        for attempt in range(1, self.max_retries + 1):
            try:
                response_text = await asyncio.wait_for(
                    self._call_llm(prompt),
                    timeout=self.timeout_seconds,
                )
                result = self._parse_response(response_text)
                result.metrics = {
                    "l3/latency_s": round(time.perf_counter() - t0, 4),
                    "l3/retries": attempt - 1,
                }
                return result

            except openai.RateLimitError:
                logger.warning("L3 rate limited, attempt %d/%d", attempt, self.max_retries)
                if attempt < self.max_retries:
                    await asyncio.sleep(self._backoff(attempt))

            except asyncio.TimeoutError:
                logger.warning("L3 timeout, attempt %d/%d", attempt, self.max_retries)
                if attempt < self.max_retries:
                    await asyncio.sleep(self._backoff(attempt))

            except Exception as exc:
                logger.warning("L3 error: %s, attempt %d/%d", exc, attempt, self.max_retries)
                if attempt < self.max_retries:
                    await asyncio.sleep(self._backoff(attempt))

        return L3GradingResult(
            feedback=f"L3 grading failed after {self.max_retries} attempts.",
            metrics={
                "l3/latency_s": round(time.perf_counter() - t0, 4),
                "l3/all_attempts_failed": 1,
            },
        )

    async def forward(self, action: Any, observation: Any) -> float:
        """Evaluate via LLM judge and return normalized score."""
        instruction = getattr(observation, "instruction_summary", "")
        plan = getattr(observation, "plan", [])
        result = await self.grade(instruction, plan)
        return result.normalized
