# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
MCP tool definitions for the Frontier SWE Environment.

Tools are registered on the environment's FastMCP instance inside
FrontierSweEnvironment.__init__:
- submit_plan(subtasks): Propose a subtask plan (PLANNING → EXECUTING)
- submit_subtask(subtask_id): Submit current subtask for L1+L2 scoring
- get_status(): Return episode status snapshot
- advance(): Freeze subtask score and move to next subtask
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastmcp import FastMCP

if TYPE_CHECKING:
    from .frontier_swe_env_environment import FrontierSweEnvironment


def register_mcp_tools(mcp: FastMCP, env: "FrontierSweEnvironment") -> None:
    """Register Frontier-SWE MCP tools on a FastMCP instance."""

    @mcp.tool
    async def submit_plan(subtasks: list[dict]) -> dict:
        """Propose a subtask plan for the episode.

        Each subtask dict must include "id", "description", and
        "acceptance_criteria" keys.  Can only be called once per
        episode, during the PLANNING phase.  Transitions the episode
        from PLANNING to EXECUTING on success.
        """
        return await env.submit_plan_payload(subtasks)

    @mcp.tool
    async def submit_subtask(subtask_id: str) -> dict:
        """Submit the current subtask for L1 (test) + L2 (code-review) scoring."""
        return await env.submit_subtask_payload(subtask_id)

    @mcp.tool
    def get_status() -> dict:
        """Get current episode status snapshot."""
        return env.get_status_payload()

    @mcp.tool
    def advance() -> dict:
        """Freeze current subtask score and move to the next subtask."""
        return env.advance_payload()
