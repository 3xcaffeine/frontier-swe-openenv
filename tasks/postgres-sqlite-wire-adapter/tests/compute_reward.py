#!/usr/bin/env python3

import argparse
import json
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute reward for postgres-sqlite-wire-adapter"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--verifier-state", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.verifier_state, encoding="utf-8") as handle:
        state = json.load(handle)

    hard_fail_reasons: list[str] = []
    if not state.get("source_scan_ok", False):
        hard_fail_reasons.append("source_scan_violation")
    if not state.get("zig_project_ok", False):
        hard_fail_reasons.append("zig_project_required")
    if not state.get("disallowed_deps_ok", False):
        hard_fail_reasons.append("disallowed_dependency")
    if not state.get("build_ok", False):
        hard_fail_reasons.append("build_failed")
    if not state.get("has_binary", False):
        hard_fail_reasons.append("binary_missing")
    if not state.get("postgres_source_ok", False):
        hard_fail_reasons.append("hidden_postgres18_tests_missing")
    if not state.get("harness_build_ok", False):
        hard_fail_reasons.append("postgres18_harness_setup_failed")

    # Include graded_compat results (wired into test.sh 2026-04-20)
    graded_compat_passed = int(state.get("graded_compat_passed", 0))
    graded_compat_total = int(state.get("graded_compat_total", 0))

    # Base totals from verifier state (regression + tap), then add graded_compat
    base_passed = int(state.get("tests_passed", 0))
    base_total = int(state.get("tests_total", 0))
    tests_passed = base_passed + graded_compat_passed
    tests_total = base_total + graded_compat_total

    # If regression didn't run (e.g. initdb failed), count those as failed
    # rather than excluded. Expected: 230 regression + 508 TAP + 72 compat = 810.
    EXPECTED_REGRESSION = 230
    EXPECTED_TAP = 508
    EXPECTED_COMPAT = 72
    EXPECTED_TOTAL = EXPECTED_REGRESSION + EXPECTED_TAP + EXPECTED_COMPAT
    if tests_total < EXPECTED_TOTAL and not hard_fail_reasons:
        tests_total = EXPECTED_TOTAL

    pass_rate = tests_passed / max(tests_total, 1)

    reward = 0.0 if hard_fail_reasons else round(pass_rate, 6)

    payload = {
        "reward": reward,
        "score": reward,
        "tests_passed": tests_passed,
        "tests_total": tests_total,
        "test_pass_rate": round(pass_rate, 6),
        "graded_compat_passed": graded_compat_passed,
        "graded_compat_total": graded_compat_total,
        "regression_passed": int(state.get("regression_passed", 0)),
        "regression_total": int(state.get("regression_total", 0)),
        "tap_passed": int(state.get("tap_passed", 0)),
        "tap_total": int(state.get("tap_total", 0)),
        "hard_fail_reasons": hard_fail_reasons,
        "verifier_state": state,
        "subscores": [
            {
                "subtask": "graded_compat",
                "score": round(
                    graded_compat_passed / max(graded_compat_total, 1),
                    6,
                ),
                "stdout": (
                    f"{graded_compat_passed}/{graded_compat_total} "
                    "graded compatibility tests passed"
                ),
                "stderr": "",
            },
            {
                "subtask": "core_regression",
                "score": round(
                    int(state.get("regression_passed", 0))
                    / max(int(state.get("regression_total", 0)), 1),
                    6,
                ),
                "stdout": (
                    f"{state.get('regression_passed', 0)}/"
                    f"{state.get('regression_total', 0)} regression tests passed"
                ),
                "stderr": "",
            },
            {
                "subtask": "tap",
                "score": round(
                    int(state.get("tap_passed", 0))
                    / max(int(state.get("tap_total", 0)), 1),
                    6,
                ),
                "stdout": (
                    f"{state.get('tap_passed', 0)}/"
                    f"{state.get('tap_total', 0)} TAP tests passed"
                ),
                "stderr": "",
            },
        ],
        "reason": (
            f"HARD FAIL: {hard_fail_reasons}"
            if hard_fail_reasons
            else (
                f"{tests_passed}/{tests_total} hidden tests passed "
                f"({pass_rate:.1%})"
            )
        ),
    }

    reward_json = os.path.join(args.output_dir, "reward.json")
    reward_txt = os.path.join(args.output_dir, "reward.txt")
    with open(reward_json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    with open(reward_txt, "w", encoding="utf-8") as handle:
        handle.write(str(reward))

    print(payload["reason"])
    print(f"Reward: {reward:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
