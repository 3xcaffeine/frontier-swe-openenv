"""Episode-level reward aggregator.

Combines plan quality, subtask scores, completion, and tool density into
a single episode reward.

    R = plan_weight   * plan_score
      + subtask_weight * mean(frozen_subtask_scores)
      + completion_weight * (attempted / planned)
      + tool_weight   * min(tool_calls / (5 * num_subtasks), 1.0)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import EpisodeState
    from ..task_config import TaskConfig


class EpisodeRubric:
    """Compute the final episode reward from episode state.

    This is not a ``Rubric`` subclass because it operates on
    ``EpisodeState`` directly rather than on action/observation pairs.
    """

    def __init__(
        self,
        plan_weight: float = 0.25,
        subtask_weight: float = 0.60,
        completion_weight: float = 0.10,
        tool_weight: float = 0.05,
    ):
        self.plan_weight = plan_weight
        self.subtask_weight = subtask_weight
        self.completion_weight = completion_weight
        self.tool_weight = tool_weight

    @classmethod
    def from_config(cls, config: TaskConfig) -> EpisodeRubric:
        return cls(
            plan_weight=config.plan_weight,
            subtask_weight=config.subtask_weight,
            completion_weight=config.completion_weight,
            tool_weight=config.tool_weight,
        )

    def compute(self, state: EpisodeState) -> float:
        """Compute the final episode reward.

        Args:
            state: The completed episode state.

        Returns:
            Blended reward in [0, 1].
        """
        plan_count = max(len(state.plan or []), 1)

        # Plan quality (L3 score)
        plan = state.plan_score

        # Mean of frozen subtask scores, padding unscored subtasks with 0
        scores = list(state.frozen_scores.values())
        while len(scores) < plan_count:
            scores.append(0.0)
        subtask_mean = sum(scores) / max(len(scores), 1)

        # Completion ratio: how far through the plan the agent got
        completion = min(state.current_subtask_index / plan_count, 1.0)

        # Tool density: did the agent use MCP tools meaningfully?
        tool_density = min(state.tool_call_count / (5 * plan_count), 1.0)

        reward = (
            self.plan_weight * plan
            + self.subtask_weight * subtask_mean
            + self.completion_weight * completion
            + self.tool_weight * tool_density
        )
        return max(0.0, min(1.0, reward))
