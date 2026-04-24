"""Task registry for FrontierSWE environments.

Each task module registers its training and demo config factories.
Use ``get_task_config(name, mode)`` to get a ``TaskConfig`` for a task.

Example::

    from frontier_swe_env.tasks import get_task_config

    config = get_task_config("pg", "training")
"""

from __future__ import annotations

from typing import Callable, Literal

from ..task_config import TaskConfig

_REGISTRY: dict[str, dict[str, Callable[[], TaskConfig]]] = {}


def register_task(
    name: str,
    training_factory: Callable[[], TaskConfig],
    demo_factory: Callable[[], TaskConfig],
) -> None:
    """Register a task's config factories."""
    _REGISTRY[name] = {"training": training_factory, "demo": demo_factory}


def get_task_config(
    name: str, mode: Literal["training", "demo"] = "training"
) -> TaskConfig:
    """Look up a task by name and return its ``TaskConfig``.

    Raises ``ValueError`` if the task name is unknown.
    """
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(f"Unknown task '{name}'. Available: {available}")
    return _REGISTRY[name][mode]()


def list_tasks() -> list[str]:
    """Return the names of all registered tasks."""
    return sorted(_REGISTRY.keys())


# Auto-register tasks on import
from .pg import pg_demo_config, pg_training_config  # noqa: E402

register_task("pg", pg_training_config, pg_demo_config)

from .notebook_compression import notebook_demo_config, notebook_training_config  # noqa: E402

register_task("notebook", notebook_training_config, notebook_demo_config)
register_task("notebook-compression", notebook_training_config, notebook_demo_config)
