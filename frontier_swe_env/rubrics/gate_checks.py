"""L1a: Gate check rubric — runs a bash script and parses GATE_SCORE=N/M."""

import re
import subprocess
from typing import Any

from openenv.core.rubrics.base import Rubric


class GateCheckRubric(Rubric):
    """Run the gate check bash script and parse ``GATE_SCORE=N/M`` from stdout.

    Since the environment runs inside the task container, the gate script
    is executed as a local subprocess (no ``docker exec`` needed).
    """

    def __init__(self, gate_script_path: str = "/app/gate_checks.sh"):
        super().__init__()
        self.gate_script_path = gate_script_path

    def forward(self, action: Any, observation: Any) -> float:
        try:
            result = subprocess.run(
                ["bash", self.gate_script_path],
                capture_output=True,
                text=True,
                timeout=120,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return 0.0

        match = re.search(r"GATE_SCORE=(\d+)/(\d+)", result.stdout)
        if match:
            numerator = int(match.group(1))
            denominator = int(match.group(2))
            if denominator > 0:
                return numerator / denominator
        return 0.0
