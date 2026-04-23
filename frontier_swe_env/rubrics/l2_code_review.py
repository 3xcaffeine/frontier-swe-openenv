"""L2: LLM-based code review rubric — scores a git diff for the current subtask."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any

import openai
from openenv.core.rubrics.base import Rubric

logger = logging.getLogger(__name__)

MAX_DIFF_CHARS = 30_000
_DEFAULT_MAX_RETRIES = 3
_DEFAULT_RETRY_BACKOFF = [15, 30, 60]

L2_PROMPT_TEMPLATE = """\
You are reviewing code for a PostgreSQL wire-compatible adapter written in Zig.
The agent's subtask was: {subtask_description}

Acceptance criteria:
{acceptance_criteria}

Git diff:
```diff
{diff}
```

L1 test results: {l1_summary}

Score the following dimensions (integers only):
- completeness (0-10): Does the diff address the subtask fully?
- correctness (0-10): Is the implementation correct?
- robustness (0-5): Does it handle edge cases?
- forward_compatibility (0-5): Will this work with future subtasks?

Also provide:
- "issues": a list of 1-3 specific, actionable problems the agent should fix
  (e.g. "AuthenticationOk message not sent after StartupMessage")
- "feedback": a one-sentence summary of overall quality

Respond ONLY with valid JSON:
{{"completeness": N, "correctness": N, "robustness": N, "forward_compatibility": N, "issues": ["...", "..."], "feedback": "..."}}
"""


@dataclass
class L2GradingResult:
    """Structured output from L2 code review."""

    completeness: int = 0
    correctness: int = 0
    robustness: int = 0
    forward_compatibility: int = 0
    feedback: str = ""
    normalized: float = 0.0
    metrics: dict[str, float | int] = field(default_factory=dict)


class L2CodeReviewRubric(Rubric):
    """LLM judge that reviews a git diff against a subtask description.

    Scores four dimensions and normalizes to [0, 1]:
        ``(completeness + correctness + robustness + forward_compatibility) / 30``

    Uses the OpenAI-compatible API (works with vLLM, Gemini, etc.).
    """

    def __init__(
        self,
        workspace_dir: str = "/app/postgres-sqlite",
        grader_model: str | None = None,
        api_base_url: str | None = None,
        api_key: str | None = None,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        retry_backoff: list[int] | None = None,
        timeout_seconds: int = 120,
    ):
        super().__init__()
        self.workspace_dir = workspace_dir
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

    def _get_git_diff(self) -> str:
        """Get the git diff from the workspace (local subprocess)."""
        try:
            result = subprocess.run(
                ["git", "-C", self.workspace_dir, "diff", "HEAD"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            diff = result.stdout
            if len(diff) > MAX_DIFF_CHARS:
                diff = diff[:MAX_DIFF_CHARS] + "\n... (diff truncated)"
            return diff
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return ""

    def _build_prompt(
        self,
        diff: str,
        subtask_description: str,
        acceptance_criteria: str,
        l1_summary: str,
    ) -> str:
        return L2_PROMPT_TEMPLATE.format(
            subtask_description=subtask_description,
            acceptance_criteria=acceptance_criteria,
            diff=diff,
            l1_summary=l1_summary,
        )

    async def _call_llm(self, prompt: str) -> str:
        response = await self._client.chat.completions.create(
            model=self.grader_model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""

    def _parse_response(self, text: str) -> L2GradingResult:
        """Parse JSON scores from the LLM response."""
        # Try to find a JSON block in the response.
        # Use a greedy match so nested arrays ("issues": [...]) are captured.
        json_match = re.search(r"\{.+\}", text, re.DOTALL)
        if not json_match:
            return L2GradingResult(feedback="Failed to parse JSON from response.")

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError:
            return L2GradingResult(feedback="Invalid JSON in response.")

        completeness = max(0, min(10, int(data.get("completeness", 0))))
        correctness = max(0, min(10, int(data.get("correctness", 0))))
        robustness = max(0, min(5, int(data.get("robustness", 0))))
        forward_compat = max(0, min(5, int(data.get("forward_compatibility", 0))))
        feedback = str(data.get("feedback", ""))

        # Fold actionable issues into the feedback string so the agent
        # sees them directly in the MCP tool result.
        issues = data.get("issues", [])
        if isinstance(issues, list) and issues:
            issue_lines = "\n".join(f"  - {issue}" for issue in issues)
            feedback = f"{feedback}\nIssues to fix:\n{issue_lines}"

        raw_sum = completeness + correctness + robustness + forward_compat
        normalized = raw_sum / 30.0

        return L2GradingResult(
            completeness=completeness,
            correctness=correctness,
            robustness=robustness,
            forward_compatibility=forward_compat,
            feedback=feedback,
            normalized=normalized,
        )

    def _backoff(self, attempt: int) -> int:
        idx = min(attempt - 1, len(self.retry_backoff) - 1)
        return self.retry_backoff[idx]

    async def grade(
        self,
        subtask_description: str = "",
        acceptance_criteria: str = "",
        l1_summary: str = "",
    ) -> L2GradingResult:
        """Run the full L2 grading pipeline."""
        diff = self._get_git_diff()
        if not diff.strip():
            return L2GradingResult(
                feedback="No git diff found — no code changes to review.",
                metrics={"l2/empty_diff": 1},
            )

        prompt = self._build_prompt(diff, subtask_description, acceptance_criteria, l1_summary)
        t0 = time.perf_counter()

        for attempt in range(1, self.max_retries + 1):
            try:
                response_text = await asyncio.wait_for(
                    self._call_llm(prompt),
                    timeout=self.timeout_seconds,
                )
                result = self._parse_response(response_text)
                result.metrics = {
                    "l2/latency_s": round(time.perf_counter() - t0, 4),
                    "l2/retries": attempt - 1,
                }
                return result

            except openai.RateLimitError:
                logger.warning("L2 rate limited, attempt %d/%d", attempt, self.max_retries)
                if attempt < self.max_retries:
                    await asyncio.sleep(self._backoff(attempt))

            except asyncio.TimeoutError:
                logger.warning("L2 timeout, attempt %d/%d", attempt, self.max_retries)
                if attempt < self.max_retries:
                    await asyncio.sleep(self._backoff(attempt))

            except Exception as exc:
                logger.warning("L2 error: %s, attempt %d/%d", exc, attempt, self.max_retries)
                if attempt < self.max_retries:
                    await asyncio.sleep(self._backoff(attempt))

        return L2GradingResult(
            feedback=f"L2 grading failed after {self.max_retries} attempts.",
            metrics={
                "l2/latency_s": round(time.perf_counter() - t0, 4),
                "l2/all_attempts_failed": 1,
            },
        )

    async def forward(self, action: Any, observation: Any) -> float:
        """Evaluate via LLM judge and return normalized score."""
        subtask_desc = getattr(observation, "subtask_description", "")
        acceptance = getattr(observation, "acceptance_criteria", "")
        l1_summary = getattr(observation, "l1_summary", "")
        result = await self.grade(subtask_desc, acceptance, l1_summary)
        return result.normalized
