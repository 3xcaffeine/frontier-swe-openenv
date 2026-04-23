"""Task configuration for FrontierSWE environments."""

from __future__ import annotations

from pydantic import BaseModel


# Default L2 scoring dimensions (task-agnostic fallback)
DEFAULT_L2_DIMENSIONS: list[dict] = [
    {"name": "completeness", "max": 10, "description": "Does the diff address the subtask fully?"},
    {"name": "correctness", "max": 10, "description": "Is the implementation correct?"},
    {"name": "robustness", "max": 5, "description": "Does it handle edge cases?"},
    {"name": "forward_compatibility", "max": 5, "description": "Will this work with future subtasks?"},
]


class TaskConfig(BaseModel):
    task_name: str
    docker_image: str
    instruction: str
    workspace_dir: str
    build_command: str
    gate_script_path: str
    visible_test_command: str
    visible_test_total: int
    max_subtasks: int
    max_attempts_per_subtask: int
    episode_timeout_s: float
    per_turn_timeout_s: float = 180.0
    # Task context for L2/L3 rubric prompts
    task_description: str = ""
    task_domain: str = ""
    scoring_context: str = ""
    # L2 scoring dimensions — list of {"name": str, "max": int, "description": str}
    # None uses DEFAULT_L2_DIMENSIONS
    l2_dimensions: list[dict] | None = None
    # L1 test output parsing
    l1_output_pattern: str = r"Total:\s*(\d+)/(\d+)\s*passed"
    l1_score_mode: str = "ratio"  # "ratio" | "speedup" | "compression"
    # Gate threshold: minimum gate score before running L1 tests
    gate_threshold: float = 0.75
    # Scoring weights
    gate_weight: float = 0.30
    l1_weight: float = 0.70
    l2_weight: float = 0.30
    plan_weight: float = 0.25
    subtask_weight: float = 0.60
    completion_weight: float = 0.10
    tool_weight: float = 0.05
    # Agent LLM config (the model pi uses — the one being trained/evaluated)
    agent_model: str | None = None
    agent_provider: str | None = None
    agent_api_base_url: str | None = None
    agent_api_key: str | None = None
    # LLM judge config (L2/L3 rubrics — a separate, typically stronger model)
    grader_model: str | None = None
    grader_api_base_url: str | None = None
    grader_api_key: str | None = None
    # Container config
    container_port: int = 8000
    cpus: int = 8
    memory_mb: int = 32768

    @property
    def effective_l2_dimensions(self) -> list[dict]:
        """Return L2 dimensions, falling back to defaults."""
        return self.l2_dimensions if self.l2_dimensions is not None else list(DEFAULT_L2_DIMENSIONS)


# Backward-compatible re-exports — these now live in tasks/pg.py
from .tasks.pg import PG_TRAINING_INSTRUCTION, pg_demo_config, pg_training_config  # noqa: E402, F401
