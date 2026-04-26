"""Frontier SWE OpenEnv — inference smoke driver.

Drives a real LLM-backed episode against a deployed HF Space and emits a
``[START] / [STEP] / [END]`` log format on stdout.

The Space ships a pi harness behind ``/step`` that holds its own LLM
client and runs a multi-turn loop inside the container. This script keeps
a WebSocket session open, sends a natural-language nudge per outer step,
and reads back the resulting observation. One [STEP] line therefore
corresponds to one outer turn that may have triggered several internal
pi/LLM actions; it is not one LLM tool call per [STEP]. Pi is the agent
we train against in production, so this driver mirrors that path rather
than orchestrating an LLM externally.

A successful [END] line means an LLM ran an episode end-to-end against
the live Space and produced a reward. There are no protocol-only or
state-only fallbacks hidden in this script; the workflow's
``Wait for Space /health`` step is a precondition gate, not a substitute.

Env vars
========
  FSWE_SPACE_URL   (required) live Space URL
  TASK_NAME        log label (default: parsed from FSWE_SPACE_URL)
  BENCHMARK        log label (default: frontier-swe-openenv)
  MAX_STEPS        outer step budget per episode (default: 4)
  TASK_COUNT       episodes per run (default: 1)
  MESSAGE_TIMEOUT  WS recv() timeout, seconds (default: 900)
  MIN/MAX_SUBMISSION_SCORE  open-interval clamps for [END] score
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
import time
import traceback
from typing import Any
from urllib.parse import urlparse

from frontier_swe_env.client import FrontierSweEnv
from frontier_swe_env.models import FrontierSweAction


SPACE_URL = (os.getenv("FSWE_SPACE_URL") or "").rstrip("/")
TASK_NAME = os.getenv("TASK_NAME") or ""
BENCHMARK = os.getenv("BENCHMARK", "frontier-swe-openenv")
MODEL_NAME = os.getenv("FSWE_AGENT_MODEL", "pi-harness")
MAX_STEPS = max(1, int(os.getenv("MAX_STEPS", "4")))
TASK_COUNT = max(1, int(os.getenv("TASK_COUNT", "1")))
MESSAGE_TIMEOUT = float(os.getenv("MESSAGE_TIMEOUT", "900"))
MIN_SUBMISSION_SCORE = float(os.getenv("MIN_SUBMISSION_SCORE", "0.01"))
MAX_SUBMISSION_SCORE = float(os.getenv("MAX_SUBMISSION_SCORE", "0.99"))

# Default per-step nudge — pi reads this and decides what tools to call.
NUDGE = (
    "Make incremental progress on the task. "
    "If you have not submitted a plan yet, call submit_plan with one or two "
    "small subtasks now. Otherwise, call submit_subtask on the current "
    "subtask to record progress. Then call get_status. "
    "Keep responses brief; do not edit large amounts of code."
)


def _single_line(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


def _clamp_open(score: float) -> float:
    """Clamp to the open interval (0, 1) per hackathon submission spec."""
    lo = max(0.01, min(MIN_SUBMISSION_SCORE, MAX_SUBMISSION_SCORE))
    hi = min(0.99, max(MIN_SUBMISSION_SCORE, MAX_SUBMISSION_SCORE))
    if hi <= lo:
        lo, hi = 0.01, 0.99
    return min(max(float(score), lo), hi)


def log_start(task: str, env_label: str, model: str) -> None:
    print(
        f"[START] task={_single_line(task)} env={_single_line(env_label)} "
        f"model={_single_line(model)}",
        flush=True,
    )


def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    err_val = _single_line(error) if error else "null"
    print(
        f"[STEP] step={step} action={_single_line(action)} reward={reward:.2f} "
        f"done={str(done).lower()} error={err_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={_clamp_open(score):.2f} rewards={rewards_str}",
        flush=True,
    )


def _infer_task_label(space_url: str) -> str:
    """Pull the task slug from the Space hostname.

    Matches ``<owner>-frontier-swe-<task>.hf.space`` and returns ``<task>``.
    """
    if TASK_NAME:
        return TASK_NAME
    host = urlparse(space_url).hostname or ""
    m = re.match(r"[^-]+-frontier-swe-(.+)\.hf\.space$", host)
    return m.group(1) if m else host or "unknown"


def _episode_score(obs: Any, frozen_scores: dict[str, float], rewards: list[float]) -> float:
    """Pick the most informative score signal from the final observation.

    Order of preference:
      1. ``observation.episode_reward`` (set on done=True for full episodes)
      2. mean of ``observation.frozen_scores`` values (post-submit_subtask)
      3. last per-step reward
      4. 0.0
    """
    ep = getattr(obs, "episode_reward", None)
    if ep is not None:
        return float(ep)
    if frozen_scores:
        return sum(frozen_scores.values()) / len(frozen_scores)
    if rewards:
        return rewards[-1]
    return 0.0


async def run_episode(env: FrontierSweEnv, episode_idx: int) -> tuple[bool, int, float, list[float]]:
    rewards: list[float] = []
    last_obs: Any = None
    final_done = False

    reset_result = await env.reset()
    last_obs = reset_result.observation

    for step in range(1, MAX_STEPS + 1):
        t0 = time.time()
        result = await env.step(FrontierSweAction(message=NUDGE))
        elapsed = time.time() - t0

        obs = result.observation
        last_obs = obs
        reward = float(result.reward or 0.0)
        rewards.append(reward)

        action_summary = (
            f'phase={obs.phase} '
            f'subtask={getattr(obs, "current_subtask", None)} '
            f'plan_score={getattr(obs, "plan_score", None)} '
            f'elapsed={elapsed:.1f}s'
        )
        log_step(
            step=step,
            action=action_summary,
            reward=reward,
            done=result.done,
            error=None,
        )

        if result.done:
            final_done = True
            break

    frozen = getattr(last_obs, "frozen_scores", {}) or {}
    score = _episode_score(last_obs, frozen, rewards)
    success = score > 0.0 or bool(frozen)
    return success, len(rewards), score, rewards


async def async_main() -> None:
    if not SPACE_URL:
        raise SystemExit("FSWE_SPACE_URL must be set to the live Space URL")

    task_label = _infer_task_label(SPACE_URL)
    print(
        f"[PREFLIGHT] space={SPACE_URL} task={task_label} "
        f"max_steps={MAX_STEPS} task_count={TASK_COUNT} "
        f"message_timeout_s={MESSAGE_TIMEOUT}",
        flush=True,
    )
    caught: Exception | None = None

    try:
        async with FrontierSweEnv(
            base_url=SPACE_URL,
            message_timeout_s=MESSAGE_TIMEOUT,
        ) as env:
            for ep_idx in range(1, TASK_COUNT + 1):
                run_label = f"{task_label}:run{ep_idx}"
                log_start(task=run_label, env_label=BENCHMARK, model=MODEL_NAME)
                success, steps, score, rewards = await run_episode(env, ep_idx)
                log_end(success=success, steps=steps, score=score, rewards=rewards)
    except Exception as exc:
        caught = exc
        print(
            f"[ERROR] type={type(exc).__name__} message={exc}",
            file=sys.stderr,
            flush=True,
        )
        print(f"[ERROR] FSWE_SPACE_URL={SPACE_URL}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)

    if caught is not None:
        raise SystemExit(1) from caught


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
