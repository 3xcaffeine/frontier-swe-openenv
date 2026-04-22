# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Data models for the FrontierSWE OpenEnv environment."""

from typing import Any, Dict, List, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


class FrontierSweAction(Action):
    """One conversational turn sent to the pi harness."""

    message: str = Field(..., description="The user message for this turn")


class FrontierSweObservation(Observation):
    """Observation returned after each turn."""

    response: str = Field(default="", description="Pi's text response")
    phase: str = Field(default="INIT", description="INIT | PLANNING | EXECUTING | DONE")
    current_subtask: Optional[str] = Field(
        default=None, description="Current subtask ID"
    )
    frozen_scores: Dict[str, float] = Field(
        default_factory=dict, description="subtask_id → best blended score"
    )
    time_remaining_s: float = Field(
        default=0.0, description="Seconds remaining in episode"
    )
    plan_score: Optional[float] = Field(
        default=None, description="L3 plan score (set after submit_plan)"
    )
    subtask_feedback: Optional[Dict[str, Any]] = Field(
        default=None, description="Latest scoring feedback"
    )
    episode_reward: Optional[float] = Field(
        default=None, description="Final reward (set when done=True)"
    )


class EpisodeState(State):
    """Full internal state for the episode state machine."""

    phase: str = "INIT"
    plan: Optional[List[Dict[str, Any]]] = None
    plan_score: float = 0.0
    current_subtask_index: int = 0
    frozen_scores: Dict[str, float] = Field(default_factory=dict)
    attempts: Dict[str, int] = Field(default_factory=dict)
    tool_call_count: int = 0
    start_time: float = 0.0
    max_subtasks: int = 2
    max_attempts_per_subtask: int = 2
    episode_timeout_s: float = 900.0
    episode_reward: Optional[float] = None
