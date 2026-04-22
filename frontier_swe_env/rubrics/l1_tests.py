"""L1b: PostgreSQL compatibility test rubric — runs pg_compat_test.sh."""

import os
import re
import subprocess
from typing import Any

from openenv.core.rubrics.base import Rubric


class PGCompatTestRubric(Rubric):
    """Run ``pg_compat_test.sh`` and parse ``Total: X/Y passed`` from stdout.

    The candidate server must already be running on ``PG_PORT`` before
    this rubric is evaluated.
    """

    def __init__(
        self,
        test_command: str = "bash /app/pg_compat_test.sh",
        port: int = 55432,
        host: str = "127.0.0.1",
    ):
        super().__init__()
        self.test_command = test_command
        self.port = port
        self.host = host

    def forward(self, action: Any, observation: Any) -> float:
        env = {**os.environ, "PG_PORT": str(self.port), "PG_HOST": self.host}
        try:
            result = subprocess.run(
                ["bash", "-c", self.test_command],
                capture_output=True,
                text=True,
                timeout=300,
                env=env,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return 0.0

        match = re.search(r"Total:\s*(\d+)/(\d+)\s*passed", result.stdout)
        if match:
            passed = int(match.group(1))
            total = int(match.group(2))
            if total > 0:
                return passed / total
        return 0.0
