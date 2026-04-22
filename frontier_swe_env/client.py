# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Frontier SWE Environment Client."""

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from .models import EpisodeState, FrontierSweAction, FrontierSweObservation


class FrontierSweEnv(
    EnvClient[FrontierSweAction, FrontierSweObservation, EpisodeState]
):
    """
    Client for the Frontier SWE Environment.

    Maintains a persistent WebSocket connection to the environment server.
    Each client instance has its own dedicated environment session.

    Example:
        >>> async with FrontierSweEnv(base_url="http://localhost:8000") as client:
        ...     result = await client.reset()
        ...     print(result.observation.phase)  # "PLANNING"
        ...
        ...     result = await client.step(FrontierSweAction(message="Hello"))
        ...     print(result.observation.response)

    Example with Docker:
        >>> client = await FrontierSweEnv.from_docker_image("frontier-swe-pg:latest")
        >>> try:
        ...     result = await client.reset()
        ...     result = await client.step(FrontierSweAction(message="Test"))
        ... finally:
        ...     await client.close()
    """

    def _step_payload(self, action: FrontierSweAction) -> Dict[str, Any]:
        return action.model_dump()

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[FrontierSweObservation]:
        obs_data = payload.get("observation", {})
        observation = FrontierSweObservation(**obs_data)
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> EpisodeState:
        return EpisodeState(**payload)
