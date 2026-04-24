#!/usr/bin/env python3
"""
Backfill episode_reward for trajectories that ended without one.

The server only computes episode_reward when the episode transitions to DONE
(via advance past last subtask, or watchdog timeout). Episodes that ended
because the client hit max_turns while the server was still in EXECUTING or
PLANNING phase have reward=null.

This script recomputes the reward offline using the same EpisodeRubric formula:

  R = 0.25 × plan_score
    + 0.60 × mean(frozen_subtask_scores, padded to plan_count)
    + 0.10 × completion (scored_subtasks / plan_count)
    + 0.05 × tool_density (min(tool_calls / (5 × plan_count), 1.0))

Usage:
    python scripts/backfill_rewards.py                         # default: trajectories/
    python scripts/backfill_rewards.py --dir trajectories/     # explicit dir
    python scripts/backfill_rewards.py --dry-run               # show what would change
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def compute_reward(result: dict) -> float | None:
    """Compute episode reward from result.json data.

    Returns None if there's not enough data (no plan submitted).
    """
    plan = result.get("plan")
    plan_score = result.get("plan_score", 0.0) or 0.0
    frozen_scores = result.get("frozen_scores", {}) or {}
    tool_call_count = result.get("tool_call_count", 0) or 0

    # Can't compute without a plan
    if not plan:
        # No plan → only tool_weight contributes, rest is 0
        # But we still return a reward so the trajectory is usable
        plan_count = 1
    else:
        plan_count = max(len(plan), 1)

    # Weights (must match EpisodeRubric defaults / pg_training_config)
    plan_weight = 0.25
    subtask_weight = 0.60
    completion_weight = 0.10
    tool_weight = 0.05

    # Mean of frozen subtask scores, padding unscored subtasks with 0
    scores = list(frozen_scores.values())
    while len(scores) < plan_count:
        scores.append(0.0)
    subtask_mean = sum(scores) / max(len(scores), 1)

    # Completion: how many subtasks were scored (have non-zero or were attempted)
    # We infer current_subtask_index from the number of scored subtasks
    scored_count = len(frozen_scores)
    completion = min(scored_count / plan_count, 1.0)

    # Tool density
    tool_density = min(tool_call_count / (5 * plan_count), 1.0)

    reward = (
        plan_weight * plan_score
        + subtask_weight * subtask_mean
        + completion_weight * completion
        + tool_weight * tool_density
    )
    return max(0.0, min(1.0, reward))


def main():
    parser = argparse.ArgumentParser(description="Backfill missing episode rewards")
    parser.add_argument("--dir", default="trajectories", help="Trajectories directory")
    parser.add_argument("--dry-run", action="store_true", help="Show changes without writing")
    args = parser.parse_args()

    traj_dir = Path(args.dir)
    if not traj_dir.exists():
        print(f"Directory not found: {traj_dir}")
        return

    updated = 0
    skipped = 0
    total = 0

    for ep_dir in sorted(traj_dir.glob("episode_*")):
        result_path = ep_dir / "result.json"
        if not result_path.exists():
            continue

        total += 1
        result = json.loads(result_path.read_text())
        ep_id = result.get("episode_id", ep_dir.name)

        existing_reward = result.get("episode_reward")
        if existing_reward is not None:
            skipped += 1
            print(f"  {ep_id}: already has reward={existing_reward:.4f} — skipped")
            continue

        reward = compute_reward(result)

        phase = result.get("phase", "?")
        plan_score = result.get("plan_score", 0)
        scores = result.get("frozen_scores", {})
        scores_str = " ".join(f"{k}={v:.3f}" for k, v in scores.items()) if scores else "none"

        print(f"  {ep_id}: phase={phase} plan={plan_score:.3f} scores=[{scores_str}] → reward={reward:.4f}")

        if not args.dry_run:
            result["episode_reward"] = reward
            result["_reward_backfilled"] = True
            result_path.write_text(json.dumps(result, indent=2))
            updated += 1

    print()
    print(f"Total: {total} episodes")
    print(f"Skipped (already had reward): {skipped}")
    print(f"{'Would update' if args.dry_run else 'Updated'}: {total - skipped}")

    # Print reward distribution
    if not args.dry_run:
        rewards = []
        for ep_dir in sorted(traj_dir.glob("episode_*")):
            result_path = ep_dir / "result.json"
            if result_path.exists():
                r = json.loads(result_path.read_text())
                if r.get("episode_reward") is not None:
                    rewards.append((r.get("episode_id", "?"), r["episode_reward"]))

        if rewards:
            rewards.sort(key=lambda x: x[1])
            print()
            print("Reward distribution (sorted):")
            for ep_id, reward in rewards:
                bar = "█" * int(reward * 40)
                print(f"  ep {ep_id:>3}: {reward:.4f} {bar}")
            vals = [r for _, r in rewards]
            print(f"\n  min={min(vals):.4f}  max={max(vals):.4f}  "
                  f"mean={sum(vals)/len(vals):.4f}  median={vals[len(vals)//2]:.4f}")


if __name__ == "__main__":
    main()
