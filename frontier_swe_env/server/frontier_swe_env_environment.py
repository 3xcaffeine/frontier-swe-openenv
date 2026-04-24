# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Frontier SWE Environment — main environment class.

Runs INSIDE the custom Docker container alongside pi.  Manages:
- Pi as a local subprocess (via PiHarnessAdapter)
- Episode state machine (INIT → PLANNING → EXECUTING → DONE)
- MCP tools (submit_plan, submit_subtask, get_status, advance)
- 3-layer rubric scoring (L1 deterministic + L2/L3 LLM judge)
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import threading
import time
from typing import Any, Optional
from uuid import uuid4

from fastmcp import FastMCP
from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.types import Observation
from openenv.core.harnesses.adapters.pi import PiHarnessAdapter
from openenv.core.harnesses.types import HarnessConfig, HarnessEventType

from ..models import EpisodeState, FrontierSweAction, FrontierSweObservation
from ..rubrics.episode_rubric import EpisodeRubric
from ..rubrics.gate_checks import GateCheckRubric
from ..rubrics.l1_tests import TestOutputRubric
from ..rubrics.l2_code_review import L2CodeReviewRubric
from ..rubrics.l3_plan_review import L3PlanReviewRubric
from ..task_config import TaskConfig
from .mcp_tools import register_mcp_tools

logger = logging.getLogger(__name__)


class FrontierSweEnvironment(MCPEnvironment):
    """OpenEnv environment for FrontierSWE tasks.

    Runs INSIDE the custom Docker container alongside pi.  Manages the
    episode lifecycle, MCP tools, pi subprocess, and rubric scoring.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = False

    def __init__(
        self,
        task_config: Optional[TaskConfig] = None,
        task_name: str = "pg",
        mode: str = "training",
    ) -> None:
        if task_config is not None:
            self.task_config = task_config
        else:
            # D-008: allow task selection via env vars so task images can
            # pick their own config without changing the app wiring.
            import os
            effective_name = os.environ.get("FSWE_TASK_NAME", task_name)
            effective_mode = os.environ.get("FSWE_TASK_MODE", mode)
            from ..tasks import get_task_config
            self.task_config = get_task_config(effective_name, effective_mode)
        self.episode_state = EpisodeState()

        # Build MCP server and register tools
        mcp = FastMCP("frontier-swe-tools")
        register_mcp_tools(mcp, self)
        super().__init__(mcp_server=mcp)

        # Rubric components
        self.gate_rubric = GateCheckRubric(self.task_config.gate_script_path)
        self.test_rubric = TestOutputRubric(
            test_command=self.task_config.visible_test_command,
            output_pattern=self.task_config.l1_output_pattern,
            score_mode=self.task_config.l1_score_mode,
            reward_json_path=self.task_config.reward_json_path,
            timeout_s=int(self.task_config.l1_timeout_s),
        )

        # Resolve grader LLM config.
        # Priority: env vars > TaskConfig fields > hardcoded default.
        #
        # Env vars (all prefixed FSWE_GRADER_*):
        #   FSWE_GRADER_MODEL    — model name for L2/L3 LLM judge
        #   FSWE_GRADER_API_URL  — OpenAI-compatible base URL
        #   FSWE_GRADER_API_KEY  — API key
        import os

        grader_model = (
            os.environ.get("FSWE_GRADER_MODEL")
            or self.task_config.grader_model
        )
        grader_api_base = (
            os.environ.get("FSWE_GRADER_API_URL")
            or self.task_config.grader_api_base_url
        )
        grader_api_key = (
            os.environ.get("FSWE_GRADER_API_KEY")
            or self.task_config.grader_api_key
            or os.environ.get("OPENAI_API_KEY")
        )

        logger.info(
            "Grader LLM config: model=%s, api_base=%s",
            grader_model,
            grader_api_base,
        )

        self.l2_rubric = L2CodeReviewRubric(
            workspace_dir=self.task_config.workspace_dir,
            task_description=self.task_config.task_description,
            dimensions=self.task_config.effective_l2_dimensions,
            grader_model=grader_model,
            api_base_url=grader_api_base,
            api_key=grader_api_key,
        )
        self.l3_rubric = L3PlanReviewRubric(
            task_description=self.task_config.task_description,
            grader_model=grader_model,
            api_base_url=grader_api_base,
            api_key=grader_api_key,
        )
        self.episode_rubric = EpisodeRubric.from_config(self.task_config)

        # Pi harness adapter (created fresh each reset)
        self.adapter: Optional[PiHarnessAdapter] = None
        # Timeout watchdog task
        self._watchdog: Optional[asyncio.Task] = None

        # Dedicated event loop for pi subprocess operations.
        # All async adapter calls (start, send_message, stop) run on this
        # loop so the subprocess is always on the same loop — avoids the
        # "Future attached to a different loop" error.
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        """Return the dedicated event loop, starting one if needed."""
        if self._loop is not None and self._loop.is_running():
            return self._loop
        loop = asyncio.new_event_loop()
        thread = threading.Thread(target=loop.run_forever, daemon=True)
        thread.start()
        self._loop = loop
        self._loop_thread = thread
        return loop

    def _run(self, coro) -> Any:
        """Run *coro* on the dedicated loop from the calling (sync) thread."""
        loop = self._ensure_loop()
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result()

    # Gym API

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> FrontierSweObservation:
        """Start a fresh episode.

        1. Stop any running pi process and cancel watchdog.
        2. Reset workspace to initial git state.
        3. Create PiHarnessAdapter, write .mcp.json, start pi.
        4. Return initial observation immediately (instruction is
           deferred to the first step() call).
        5. Initialise episode state → phase = PLANNING.
        """
        # Cancel previous watchdog
        if self._watchdog is not None and not self._watchdog.done():
            self._watchdog.cancel()
            self._watchdog = None

        # Stop previous pi process
        if self.adapter is not None:
            alive = self._run(self.adapter.is_alive())
            if alive:
                self._run(self.adapter.stop())

        # Reset workspace via git
        self._reset_workspace()

        # Initialise episode state
        self.episode_state = EpisodeState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            phase="PLANNING",
            start_time=time.time(),
            max_subtasks=self.task_config.max_subtasks,
            max_attempts_per_subtask=self.task_config.max_attempts_per_subtask,
            episode_timeout_s=self.task_config.episode_timeout_s,
        )

        # Create pi harness adapter
        #
        # Agent LLM config resolution (env vars > TaskConfig):
        #   FSWE_AGENT_MODEL     — model name pi should use
        #   FSWE_AGENT_PROVIDER  — pi provider (openai, anthropic, google, …)
        #   FSWE_AGENT_API_URL   — OpenAI-compatible base URL
        #   FSWE_AGENT_API_KEY   — API key for the agent endpoint
        import os

        agent_model = (
            os.environ.get("FSWE_AGENT_MODEL")
            or self.task_config.agent_model
        )
        agent_provider = (
            os.environ.get("FSWE_AGENT_PROVIDER")
            or self.task_config.agent_provider
        )
        agent_api_url = (
            os.environ.get("FSWE_AGENT_API_URL")
            or self.task_config.agent_api_base_url
        )
        agent_api_key = (
            os.environ.get("FSWE_AGENT_API_KEY")
            or self.task_config.agent_api_key
            or os.environ.get("OPENAI_API_KEY")
        )

        # Build env vars to pass to the pi subprocess
        pi_env: dict[str, str] = {}
        if agent_api_url:
            pi_env["OPENAI_BASE_URL"] = agent_api_url
        if agent_api_key:
            pi_env["OPENAI_API_KEY"] = agent_api_key

        harness_config = HarnessConfig(
            name="pi",
            command=["pi"],
            working_directory=self.task_config.workspace_dir,
            session_timeout_s=self.task_config.per_turn_timeout_s,
            startup_timeout_s=30.0,
            # pi expects "provider/model" format when using custom providers
            model=f"{agent_provider}/{agent_model}" if agent_provider else agent_model,
            env_vars=pi_env,
        )
        self.adapter = PiHarnessAdapter(
            config=harness_config,
            # Point at /tools/mcp (FastMCP native Streamable HTTP)
            # NOT /mcp (OpenEnv POST-only JSON-RPC which 405s on GET SSE probe)
            mcp_server_url=f"http://localhost:{self.task_config.container_port}/tools/mcp",
            provider=agent_provider,
        )

        logger.info(
            "Agent LLM config: model=%s, provider=%s, api_url=%s",
            agent_model,
            agent_provider,
            agent_api_url,
        )

        # Register this env instance so the shared pi_mcp tools can
        # delegate to our payload handlers (submit_plan, etc.).
        from .app import set_active_env
        set_active_env(self)

        # Inject MCP tools and start pi.
        # We must pass actual tool definitions so PiHarnessAdapter writes
        # .mcp.json — otherwise pi won't discover the OpenEnv MCP tools
        # (submit_plan, submit_subtask, get_status, advance).
        tools = self._get_mcp_tool_definitions()
        self._run(self.adapter.inject_tools(tools))
        self._run(self.adapter.start(self.task_config.workspace_dir))

        # NOTE: We do NOT send the instruction here.  Sending it would
        # block until pi finishes its full autonomous ReAct loop (minutes),
        # violating the Gym contract that reset() returns quickly.
        # Instead, the instruction is prepended to the first step() message
        # (see _step_impl, step_count == 0 branch).

        # Start timeout watchdog
        self._start_watchdog()

        return FrontierSweObservation(
            response=(
                "Environment ready. You are in the PLANNING phase.\n"
                "Send your first message to begin working on the task."
            ),
            phase="PLANNING",
            time_remaining_s=self.task_config.episode_timeout_s,
            done=False,
            reward=0.0,
        )

    def _step_impl(
        self,
        action: FrontierSweAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """Handle non-MCP actions: send a message to pi, get response.

        On the very first step (step_count == 0) the task instruction is
        prepended to the user message so pi receives the full context.
        This keeps reset() fast (~3 s) while ensuring the instruction is
        delivered before the agent begins working.
        """
        message = action.message

        remaining = self._time_remaining()
        if remaining <= 0:
            return self._timeout_observation()

        if self.adapter is None:
            return FrontierSweObservation(
                response="Error: environment not initialised. Call reset() first.",
                phase=self.episode_state.phase,
                done=True,
                reward=0.0,
            )

        # First step: prepend the task instruction so pi gets full context
        if self.episode_state.step_count == 0:
            message = (
                self.task_config.instruction + "\n\n" + message
            )

        response = self._run(self.adapter.send_message(message))
        self.episode_state.step_count += 1

        # Log detailed event summary for observability
        tool_calls = []
        tool_results = []
        errors = []
        for event in response.events:
            if event.type == HarnessEventType.TOOL_CALL:
                self.episode_state.tool_call_count += 1
                name = event.data.get("tool_name") or "?"
                phase = event.data.get("phase", "")
                if phase in ("end", "execution_start"):
                    tool_calls.append(name)
            elif event.type == HarnessEventType.TOOL_RESULT:
                name = event.data.get("tool_name") or "?"
                is_err = event.data.get("is_error", False)
                tool_results.append((name, is_err))
            elif event.type == HarnessEventType.ERROR:
                errors.append(event.data.get("message", str(event.data)))

        # Summarise tool usage
        if tool_calls:
            from collections import Counter
            counts = Counter(tool_calls)
            summary = ", ".join(f"{n}×{c}" for n, c in counts.most_common())
            logger.info(
                "Turn %d tool calls (%d total): %s",
                self.episode_state.step_count, len(tool_calls), summary,
            )
        if errors:
            for err in errors:
                logger.warning("Turn %d error: %s", self.episode_state.step_count, err[:200])

        # Log MCP tool interactions specifically (submit_plan, submit_subtask, etc.)
        for event in response.events:
            if event.type == HarnessEventType.TOOL_CALL and event.data.get("phase") == "end":
                name = event.data.get("tool_name", "")
                if name == "mcp":
                    args = event.data.get("arguments", {})
                    logger.info(
                        "Turn %d MCP tool call: %s",
                        self.episode_state.step_count,
                        json.dumps(args)[:500] if args else "(no args)",
                    )
            elif event.type == HarnessEventType.TOOL_RESULT:
                name = event.data.get("tool_name", "")
                if name == "mcp":
                    result_data = event.data.get("result", "")
                    is_err = event.data.get("is_error", False)
                    logger.info(
                        "Turn %d MCP tool result (error=%s): %s",
                        self.episode_state.step_count, is_err,
                        str(result_data)[:500],
                    )

        # --- Option A: Auto-submit on turn timeout ---
        # If the turn timed out while in EXECUTING phase and the current
        # subtask hasn't exhausted its attempts, auto-submit to get a
        # score signal.
        timed_out = any(
            e.type == HarnessEventType.ERROR
            and "timeout" in str(e.data.get("message", "")).lower()
            for e in response.events
        )
        auto_submit_result = None
        response_text = response.response or ""

        if timed_out and self.episode_state.phase == "EXECUTING":
            current_id = self._current_subtask_id()
            attempts_used = self.episode_state.attempts.get(current_id, 0) if current_id else 999
            max_attempts = self.episode_state.max_attempts_per_subtask
            if current_id and attempts_used < max_attempts:
                logger.info(
                    "Auto-submitting subtask %s on turn timeout", current_id
                )
                try:
                    auto_submit_result = self._run(
                        self.submit_subtask_payload(current_id)
                    )
                    logger.info(
                        "Auto-submit result for %s: score=%.4f best=%.4f",
                        current_id,
                        auto_submit_result.get("score", 0),
                        auto_submit_result.get("best_score", 0),
                    )
                    feedback_str = json.dumps(auto_submit_result)
                    response_text += (
                        f"\n\n[AUTO-SUBMIT on turn timeout] "
                        f"Subtask {current_id} scored: {feedback_str}"
                    )
                except Exception:
                    logger.exception(
                        "Auto-submit failed for subtask %s", current_id
                    )

            # Auto-advance if attempts are now exhausted for the current subtask
            current_id = self._current_subtask_id()
            if current_id:
                attempts_now = self.episode_state.attempts.get(current_id, 0)
                if attempts_now >= max_attempts and self.episode_state.phase == "EXECUTING":
                    logger.info(
                        "Auto-advancing past subtask %s (attempts exhausted)",
                        current_id,
                    )
                    advance_result = self.advance_payload()
                    response_text += (
                        f"\n[AUTO-ADVANCE] Subtask {current_id} attempts exhausted. "
                        f"{json.dumps(advance_result)}"
                    )

        done = response.done or self.episode_state.phase == "DONE"
        reward = self.episode_state.episode_reward if done else 0.0

        return FrontierSweObservation(
            response=response_text,
            phase=self.episode_state.phase,
            current_subtask=self._current_subtask_id(),
            frozen_scores=dict(self.episode_state.frozen_scores),
            time_remaining_s=max(0.0, self._time_remaining()),
            plan_score=self.episode_state.plan_score
            if self.episode_state.plan
            else None,
            subtask_feedback=auto_submit_result,
            done=done,
            reward=reward or 0.0,
        )

    @property
    def state(self) -> EpisodeState:
        return self.episode_state

    def close(self) -> None:
        """Clean up pi process, watchdog, dedicated loop, and MCP resources."""
        if self._watchdog is not None and not self._watchdog.done():
            self._watchdog.cancel()
            self._watchdog = None

        if self.adapter is not None:
            try:
                alive = self._run(self.adapter.is_alive())
                if alive:
                    self._run(self.adapter.stop())
            except Exception:
                logger.warning("Error stopping pi adapter during close", exc_info=True)
            self.adapter = None

        # Shut down the dedicated event loop
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._loop_thread is not None:
                self._loop_thread.join(timeout=5)
            self._loop = None
            self._loop_thread = None

        super().close()

    # MCP tool payload handlers (called from mcp_tools.py)

    async def submit_plan_payload(self, subtasks: list[dict]) -> dict:
        """Handle submit_plan MCP tool call."""
        # Validate phase
        if self.episode_state.phase != "PLANNING":
            return {"error": f"Cannot submit plan in phase {self.episode_state.phase}"}

        # Validate subtask list
        if not subtasks or len(subtasks) > self.episode_state.max_subtasks:
            return {
                "error": f"Plan must have 1-{self.episode_state.max_subtasks} subtasks, "
                f"got {len(subtasks)}",
            }

        required_keys = {"id", "description", "acceptance_criteria"}
        for i, st in enumerate(subtasks):
            missing = required_keys - set(st.keys())
            if missing:
                return {"error": f"Subtask {i} missing keys: {missing}"}

        # Store plan
        self.episode_state.plan = subtasks

        # Run L3 plan review
        l3_result = await self.l3_rubric.grade(
            instruction_summary=self.task_config.instruction[:500],
            plan=subtasks,
        )
        self.episode_state.plan_score = l3_result.normalized

        # Initialise per-subtask tracking
        for st in subtasks:
            self.episode_state.attempts[st["id"]] = 0
            self.episode_state.frozen_scores[st["id"]] = 0.0

        # Transition to EXECUTING
        self.episode_state.phase = "EXECUTING"
        self.episode_state.current_subtask_index = 0
        self.episode_state.tool_call_count += 1

        logger.info(
            "Plan accepted (%d subtasks, L3 score=%.3f)",
            len(subtasks),
            l3_result.normalized,
        )

        return {
            "plan_score": round(l3_result.normalized, 4),
            "feedback": l3_result.feedback,
            "phase": "EXECUTING",
            "current_subtask": subtasks[0]["id"],
        }

    async def submit_subtask_payload(self, subtask_id: str) -> dict:
        """Handle submit_subtask MCP tool call."""
        if self.episode_state.phase != "EXECUTING":
            return {
                "error": f"Cannot submit subtask in phase {self.episode_state.phase}"
            }

        current_id = self._current_subtask_id()
        if subtask_id != current_id:
            return {"error": f"Expected subtask '{current_id}', got '{subtask_id}'"}

        attempts_used = self.episode_state.attempts.get(subtask_id, 0)
        if attempts_used >= self.episode_state.max_attempts_per_subtask:
            return {
                "error": f"No attempts remaining for subtask '{subtask_id}'",
                "attempts_remaining": 0,
            }

        # Increment attempt counter
        self.episode_state.attempts[subtask_id] = attempts_used + 1
        self.episode_state.tool_call_count += 1

        # Get current subtask details
        subtask = self.episode_state.plan[self.episode_state.current_subtask_index]

        # L1 scoring (deterministic, local subprocess)
        gate_score = self.gate_rubric.forward(None, None)
        l1_test_score = 0.0
        if gate_score >= self.task_config.gate_threshold:
            l1_test_score = self.test_rubric.forward(None, None)

        l1_score = (
            self.task_config.gate_weight * gate_score
            + self.task_config.l1_weight * l1_test_score
        )

        l1_extras: dict = {}
        if self.task_config.l1_score_mode == "reward_json":
            reward = getattr(self.test_rubric, "last_reward", None)
            if reward is not None:
                l1_extras = {
                    "status": reward.get("status"),
                    "reason": reward.get("reason"),
                    "geom_mean_ratio": reward.get("geom_mean_ratio"),
                    "compression_score": reward.get("compression_score"),
                    "stage_timings": {
                        "fit_elapsed_sec": reward.get("fit_elapsed_sec"),
                        "compress_elapsed_sec": reward.get("compress_elapsed_sec"),
                        "decompress_elapsed_sec": reward.get("decompress_elapsed_sec"),
                    },
                }
                l1_summary = (
                    f"Gate: {gate_score:.2f} | "
                    f"Verifier: status={reward.get('status')}, "
                    f"geom_mean_ratio={reward.get('geom_mean_ratio')}, "
                    f"reason={reward.get('reason')} | "
                    f"L1 blended: {l1_score:.2f}"
                )
            else:
                l1_summary = (
                    f"Gate: {gate_score:.2f} | Verifier: no reward.json produced | "
                    f"L1 blended: {l1_score:.2f}"
                )
        else:
            l1_summary = (
                f"Gate: {gate_score:.2f}, "
                f"Compat tests: {l1_test_score:.2f}, "
                f"L1 blended: {l1_score:.2f}"
            )

        # L2 scoring (async LLM judge)
        l2_result = await self.l2_rubric.grade(
            subtask_description=subtask.get("description", ""),
            acceptance_criteria=subtask.get("acceptance_criteria", ""),
            l1_summary=l1_summary,
        )
        l2_score = l2_result.normalized

        # Blend L1 and L2
        blended = (
            1.0 - self.task_config.l2_weight
        ) * l1_score + self.task_config.l2_weight * l2_score

        # Track best score
        prev_best = self.episode_state.frozen_scores.get(subtask_id, 0.0)
        self.episode_state.frozen_scores[subtask_id] = max(prev_best, blended)

        attempts_remaining = (
            self.episode_state.max_attempts_per_subtask
            - self.episode_state.attempts[subtask_id]
        )

        logger.info(
            "Subtask %s attempt %d: gate=%.2f l1_test=%.2f l1=%.2f l2=%.2f blended=%.2f (best=%.2f)",
            subtask_id,
            self.episode_state.attempts[subtask_id],
            gate_score,
            l1_test_score,
            l1_score,
            l2_score,
            blended,
            self.episode_state.frozen_scores[subtask_id],
        )

        response = {
            "score": round(blended, 4),
            "l1_score": round(l1_score, 4),
            "l2_score": round(l2_score, 4),
            "gate_score": round(gate_score, 4),
            "test_score": round(l1_test_score, 4),
            "best_score": round(self.episode_state.frozen_scores[subtask_id], 4),
            "feedback": l2_result.feedback,
            "attempts_remaining": attempts_remaining,
        }
        if l1_extras:
            response["l1_extras"] = l1_extras
        return response

    def get_status_payload(self) -> dict:
        """Handle get_status MCP tool call."""
        self.episode_state.tool_call_count += 1
        plan = self.episode_state.plan or []
        plan_count = max(len(plan), 1)
        completion = min(self.episode_state.current_subtask_index / plan_count, 1.0)

        current_id = self._current_subtask_id()
        attempts_used = (
            self.episode_state.attempts.get(current_id, 0) if current_id else 0
        )

        return {
            "phase": self.episode_state.phase,
            "current_subtask": current_id,
            "frozen_scores": dict(self.episode_state.frozen_scores),
            "time_remaining_s": round(max(0.0, self._time_remaining()), 1),
            "completion": round(completion, 4),
            "attempts_used": attempts_used,
            "attempts_remaining": self.episode_state.max_attempts_per_subtask
            - attempts_used,
            "subtasks_total": len(plan),
            "subtasks_completed": self.episode_state.current_subtask_index,
        }

    def advance_payload(self) -> dict:
        """Handle advance MCP tool call."""
        if self.episode_state.phase != "EXECUTING":
            return {"error": f"Cannot advance in phase {self.episode_state.phase}"}

        plan = self.episode_state.plan or []
        if not plan:
            return {"error": "No plan submitted"}

        current_id = self._current_subtask_id()
        frozen_score = self.episode_state.frozen_scores.get(current_id, 0.0)

        self.episode_state.tool_call_count += 1
        self.episode_state.current_subtask_index += 1

        # Check if we've completed all subtasks
        if self.episode_state.current_subtask_index >= len(plan):
            self.episode_state.phase = "DONE"
            self.episode_state.episode_reward = self.episode_rubric.compute(
                self.episode_state
            )
            logger.info(
                "Episode complete. Reward=%.4f", self.episode_state.episode_reward
            )
            return {
                "frozen_score": round(frozen_score, 4),
                "next_subtask_id": None,
                "episode_done": True,
                "episode_reward": round(self.episode_state.episode_reward, 4),
            }

        next_id = self._current_subtask_id()
        logger.info("Advanced from %s (%.2f) to %s", current_id, frozen_score, next_id)

        return {
            "frozen_score": round(frozen_score, 4),
            "next_subtask_id": next_id,
            "episode_done": False,
        }

    # Private helpers

    def _get_mcp_tool_definitions(self) -> list:
        """Extract tool definitions from the shared pi_mcp server.

        We list tools from the module-level ``pi_mcp`` in ``app.py``
        (the FastMCP instance actually served at ``/tools/mcp``),
        because that is where pi-mcp-adapter connects.  The per-env
        FastMCP created in ``__init__`` has the same tools but is
        only used by the OpenEnv ``/mcp`` JSON-RPC endpoint.
        """
        try:
            from fastmcp import Client
            from .app import pi_mcp

            async def _list() -> list:
                async with Client(pi_mcp) as client:
                    return await client.list_tools()

            return self._run(_list())
        except Exception:
            logger.warning("Failed to extract MCP tool definitions", exc_info=True)
            return []

    def _current_subtask_id(self) -> Optional[str]:
        plan = self.episode_state.plan
        idx = self.episode_state.current_subtask_index
        if plan and 0 <= idx < len(plan):
            return plan[idx]["id"]
        return None

    def _time_remaining(self) -> float:
        if self.episode_state.start_time <= 0:
            return 0.0
        elapsed = time.time() - self.episode_state.start_time
        return self.episode_state.episode_timeout_s - elapsed

    def _reset_workspace(self) -> None:
        """Reset the task workspace to its initial git state."""
        ws = self.task_config.workspace_dir
        try:
            subprocess.run(
                ["git", "-C", ws, "checkout", "."],
                capture_output=True,
                timeout=10,
            )
            subprocess.run(
                ["git", "-C", ws, "clean", "-fd"],
                capture_output=True,
                timeout=10,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.warning("Failed to reset workspace at %s", ws)

    def _timeout_observation(self) -> FrontierSweObservation:
        """Handle episode timeout — auto-submit current subtask, freeze, compute reward."""
        if self.episode_state.phase != "DONE":
            # Option B: Auto-submit on episode timeout before computing reward
            if self.episode_state.phase == "EXECUTING":
                current_id = self._current_subtask_id()
                attempts_used = (
                    self.episode_state.attempts.get(current_id, 0)
                    if current_id
                    else 999
                )
                max_attempts = self.episode_state.max_attempts_per_subtask
                if current_id and attempts_used < max_attempts:
                    logger.info(
                        "Episode timeout — auto-submitting subtask %s",
                        current_id,
                    )
                    try:
                        result = self._run(
                            self.submit_subtask_payload(current_id)
                        )
                        logger.info(
                            "Episode timeout auto-submit %s: score=%.4f",
                            current_id,
                            result.get("score", 0),
                        )
                    except Exception:
                        logger.exception(
                            "Episode timeout auto-submit failed for %s",
                            current_id,
                        )

            self.episode_state.phase = "DONE"
            self.episode_state.episode_reward = self.episode_rubric.compute(
                self.episode_state
            )

        return FrontierSweObservation(
            response="Episode timeout. Final reward computed.",
            phase="DONE",
            frozen_scores=dict(self.episode_state.frozen_scores),
            time_remaining_s=0.0,
            episode_reward=self.episode_state.episode_reward,
            done=True,
            reward=self.episode_state.episode_reward or 0.0,
        )

    def _start_watchdog(self) -> None:
        """Start a background task that enforces the episode timeout."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop — watchdog can't be scheduled; timeout is
            # enforced reactively in _step_impl instead.
            return

        async def _watchdog_coro() -> None:
            await asyncio.sleep(self.episode_state.episode_timeout_s)
            if self.episode_state.phase != "DONE":
                logger.info("Watchdog triggered — episode timed out")
                self.episode_state.phase = "DONE"
                self.episode_state.episode_reward = self.episode_rubric.compute(
                    self.episode_state
                )
                # Abort pi
                if self.adapter is not None:
                    try:
                        await self.adapter.stop()
                    except Exception:
                        pass

        self._watchdog = loop.create_task(_watchdog_coro())
