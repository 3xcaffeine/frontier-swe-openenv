"""L1b: Test output rubric — runs a test command and parses the score.

Supports multiple score modes:
- "ratio": parse numerator/denominator (e.g. "Total: 6/72 passed")
- "speedup": parse speedup multiplier (e.g. "Speedup: 1.45x")
- "compression": parse compression ratio (e.g. "Ratio: 0.312")
"""

import os
import re
import subprocess
from typing import Any

from openenv.core.rubrics.base import Rubric


class TestOutputRubric(Rubric):
    """Run a test command and parse the output for a score.

    The ``output_pattern`` regex is matched against stdout.  How the
    captured groups are interpreted depends on ``score_mode``:

    - ``"ratio"``: two capture groups → ``int(g1) / int(g2)``
    - ``"speedup"``: one capture group → ``min((float(g1) - 1.0) * 5, 1.0)``
      (maps 1.0x→0, 1.2x→1.0; clamps at 1.0)
    - ``"compression"``: one capture group → ``min((0.5 - float(g1)) / 0.5, 1.0)``
      (maps 0.5→0, 0.0→1.0; clamps at [0, 1])
    """

    def __init__(
        self,
        test_command: str = "bash /app/test.sh",
        output_pattern: str = r"Total:\s*(\d+)/(\d+)\s*passed",
        score_mode: str = "ratio",
        port: int = 0,
        host: str = "127.0.0.1",
        timeout_s: int = 300,
    ):
        super().__init__()
        self.test_command = test_command
        self.output_pattern = output_pattern
        self.score_mode = score_mode
        self.port = port
        self.host = host
        self.timeout_s = timeout_s

    def forward(self, action: Any, observation: Any) -> float:
        env = {**os.environ, "PG_PORT": str(self.port), "PG_HOST": self.host}
        try:
            result = subprocess.run(
                ["bash", "-c", self.test_command],
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                env=env,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return 0.0

        return self._parse_output(result.stdout)

    def _parse_output(self, stdout: str) -> float:
        """Parse the test output according to the configured score mode."""
        match = re.search(self.output_pattern, stdout)
        if not match:
            return 0.0

        if self.score_mode == "ratio":
            return self._parse_ratio(match)
        elif self.score_mode == "speedup":
            return self._parse_speedup(match)
        elif self.score_mode == "compression":
            return self._parse_compression(match)
        else:
            return self._parse_ratio(match)

    @staticmethod
    def _parse_ratio(match: re.Match) -> float:
        """Parse passed/total ratio."""
        try:
            passed = int(match.group(1))
            total = int(match.group(2))
            if total > 0:
                return passed / total
        except (IndexError, ValueError):
            pass
        return 0.0

    @staticmethod
    def _parse_speedup(match: re.Match) -> float:
        """Parse speedup multiplier → normalized to [0, 1]."""
        try:
            speedup = float(match.group(1))
            # 1.0x = no speedup = 0.0, 1.2x+ = 1.0
            return max(0.0, min((speedup - 1.0) * 5.0, 1.0))
        except (IndexError, ValueError):
            pass
        return 0.0

    @staticmethod
    def _parse_compression(match: re.Match) -> float:
        """Parse compression ratio → normalized to [0, 1]."""
        try:
            ratio = float(match.group(1))
            # 0.5 = no compression = 0.0, 0.0 = perfect = 1.0
            return max(0.0, min((0.5 - ratio) / 0.5, 1.0))
        except (IndexError, ValueError):
            pass
        return 0.0


# Backward-compatible alias
PGCompatTestRubric = TestOutputRubric
