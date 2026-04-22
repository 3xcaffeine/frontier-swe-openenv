# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FastMCP server with 4 episode-management tools for the FrontierSWE environment.

Tools:
    submit_plan  — Propose a subtask plan (PLANNING → EXECUTING)
    submit_subtask — Submit current subtask for scoring
    get_status   — Get current episode status snapshot
    advance      — Freeze subtask score and move to the next one
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastmcp import FastMCP

if TYPE_CHECKING:
    from .frontier_swe_env_environment import FrontierSweEnvironment


def create_mcp_server(env: FrontierSweEnvironment) -> FastMCP:
    """Create a FastMCP server with 4 episode-management tools.

    The ``env`` reference lets tool handlers access episode state and
    trigger scoring.
    """
    mcp = FastMCP("frontier-swe-tools")

    @mcp.tool()
    def submit_plan(subtasks: list[dict]) -> dict:
        """Propose a subtask plan for the episode.

        Args:
            subtasks: List of dicts, each with keys "id", "description",
                and "acceptance_criteria".

        Returns:
            {"plan_score": float, "feedback": str}

        Rules:
            - Can only be called once per episode, during the PLANNING phase.
            - Must have between 1 and max_subtasks subtasks.
            - Transitions the episode from PLANNING to EXECUTING.
        """
        return env._handle_submit_plan(subtasks)

    @mcp.tool()
    def submit_subtask(subtask_id: str) -> dict:
        """Submit current subtask for scoring.

        Args:
            subtask_id: The id of the subtask being submitted.

        Returns:
            {"score": float, "l1_score": float, "l2_score": float,
             "feedback": str, "attempts_remaining": int}
        """
        return env._handle_submit_subtask(subtask_id)

    @mcp.tool()
    def get_status() -> dict:
        """Get current episode status.

        Returns:
            {"phase": str, "current_subtask": str | None,
             "frozen_scores": dict, "time_remaining_s": float,
             "completion": float, "attempts_used": int,
             "attempts_remaining": int}
        """
        return env._handle_get_status()

    @mcp.tool()
    def advance() -> dict:
        """Freeze current subtask score and move to next subtask.

        Returns:
            {"frozen_score": float, "next_subtask_id": str | None}
        """
        return env._handle_advance()

    return mcp
