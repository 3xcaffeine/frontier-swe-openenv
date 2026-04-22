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
import logging
import subprocess
import time
from typing import Any, Optional
from uuid import uuid4

from fastmcp import FastMCP
from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.types import Observation
from openenv.core.harnesses.adapters.pi import PiHarnessAdapter
from openenv.core.harnesses.types import HarnessConfig, HarnessEventType
from openenv.core.utils import run_async_safely

from ..models import EpisodeState, FrontierSweAction, FrontierSweObservation
from ..rubrics.episode_rubric import EpisodeRubric
from ..rubrics.gate_checks import GateCheckRubric
from ..rubrics.l1_tests import PGCompatTestRubric
from ..rubrics.l2_code_review import L2CodeReviewRubric
from ..rubrics.l3_plan_review import L3PlanReviewRubric
from ..task_config import TaskConfig, pg_training_config
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
    ) -> None:
        self.task_config = task_config or pg_training_config()
        self.episode_state = EpisodeState()

        # Build MCP server and register tools
        mcp = FastMCP("frontier-swe-tools")
        register_mcp_tools(mcp, self)
        super().__init__(mcp_server=mcp)

        # Rubric components
        self.gate_rubric = GateCheckRubric(self.task_config.gate_script_path)
        self.test_rubric = PGCompatTestRubric(
            test_command=self.task_config.visible_test_command,
        )
        self.l2_rubric = L2CodeReviewRubric(
            workspace_dir=self.task_config.workspace_dir,
        )
        self.l3_rubric = L3PlanReviewRubric()
        self.episode_rubric = EpisodeRubric.from_config(self.task_config)

        # Pi harness adapter (created fresh each reset)
        self.adapter: Optional[PiHarnessAdapter] = None
        # Timeout watchdog task
        self._watchdog: Optional[asyncio.Task] = None

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
        4. Send scoped instruction as first prompt.
        5. Initialise episode state → phase = PLANNING.
        """
        # Cancel previous watchdog
        if self._watchdog is not None and not self._watchdog.done():
            self._watchdog.cancel()
            self._watchdog = None

        # Stop previous pi process
        if self.adapter is not None:
            alive = run_async_safely(self.adapter.is_alive())
            if alive:
                run_async_safely(self.adapter.stop())

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
        harness_config = HarnessConfig(
            name="pi",
            command=["pi"],
            working_directory=self.task_config.workspace_dir,
            session_timeout_s=self.task_config.episode_timeout_s,
            startup_timeout_s=30.0,
        )
        self.adapter = PiHarnessAdapter(
            config=harness_config,
            mcp_server_url=f"http://localhost:{self.task_config.container_port}/mcp",
        )

        # Inject MCP tools and start pi
        run_async_safely(self.adapter.inject_tools([]))
        run_async_safely(self.adapter.start(self.task_config.workspace_dir))

        # Send instruction
        response = run_async_safely(
            self.adapter.send_message(self.task_config.instruction)
        )

        # Start timeout watchdog
        self._start_watchdog()

        return FrontierSweObservation(
            response=response.response,
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
        """Handle non-MCP actions: send a message to pi, get response."""
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

        response = run_async_safely(self.adapter.send_message(message))
        self.episode_state.step_count += 1

        # Count tool calls from events
        for event in response.events:
            if event.type == HarnessEventType.TOOL_CALL:
                self.episode_state.tool_call_count += 1

        done = response.done or self.episode_state.phase == "DONE"
        reward = self.episode_state.episode_reward if done else 0.0

        return FrontierSweObservation(
            response=response.response,
            phase=self.episode_state.phase,
            current_subtask=self._current_subtask_id(),
            frozen_scores=dict(self.episode_state.frozen_scores),
            time_remaining_s=max(0.0, self._time_remaining()),
            plan_score=self.episode_state.plan_score
            if self.episode_state.plan
            else None,
            done=done,
            reward=reward or 0.0,
        )

    @property
    def state(self) -> EpisodeState:
        return self.episode_state

    def close(self) -> None:
        """Clean up pi process, watchdog, and MCP resources."""
        if self._watchdog is not None and not self._watchdog.done():
            self._watchdog.cancel()
            self._watchdog = None

        if self.adapter is not None:
            try:
                alive = run_async_safely(self.adapter.is_alive())
                if alive:
                    run_async_safely(self.adapter.stop())
            except Exception:
                logger.warning("Error stopping pi adapter during close", exc_info=True)
            self.adapter = None

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
        if gate_score >= 0.75:  # At least 3/4 gates pass
            l1_test_score = self.test_rubric.forward(None, None)

        l1_score = (
            self.task_config.gate_weight * gate_score
            + self.task_config.l1_weight * l1_test_score
        )

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

        return {
            "score": round(blended, 4),
            "l1_score": round(l1_score, 4),
            "l2_score": round(l2_score, 4),
            "gate_score": round(gate_score, 4),
            "test_score": round(l1_test_score, 4),
            "best_score": round(self.episode_state.frozen_scores[subtask_id], 4),
            "feedback": l2_result.feedback,
            "attempts_remaining": attempts_remaining,
        }

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
        """Handle episode timeout — freeze everything, compute reward."""
        if self.episode_state.phase != "DONE":
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
