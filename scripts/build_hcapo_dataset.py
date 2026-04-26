#!/usr/bin/env python3
"""Build an HCAPO step-weighted SFT dataset from trajectories + hindsight scores.

Combines trajectory-level GRPO advantages with step-level hindsight Q_H values
to produce per-step HCAPO advantages (Eq. 8 from paper 2603.08754).

Input:
    trajectories/episode_NNN/  — result.json + pi_session.jsonl + hindsight_scores.json

Output:
    datasets/hcapo_train.jsonl — one row per episode with step-level advantages

Usage:
    uv run python scripts/build_hcapo_dataset.py --min-reward 0.2 --omega 1.0
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
from build_training_dataset import load_episode

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_hcapo")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_hindsight_scores(episode_dir: Path) -> dict | None:
    path = episode_dir / "hindsight_scores.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def load_episodes_with_scores(
    input_dir: Path, min_reward: float,
) -> list[dict]:
    """Load episodes that have both valid rewards and hindsight scores."""
    episodes = []
    for ep_dir in sorted(input_dir.glob("episode_*")):
        ep = load_episode(ep_dir, include_thinking=True, max_tool_result_chars=4000)
        if ep is None:
            continue
        if ep["reward"] < min_reward:
            logger.info("  Episode %s: reward=%.4f < %.4f, skipped", ep["episode_id"], ep["reward"], min_reward)
            continue

        scores = load_hindsight_scores(ep_dir)
        if scores is None:
            logger.warning("  Episode %s: no hindsight_scores.json, skipped", ep["episode_id"])
            continue

        ep["_hindsight"] = scores
        ep["_dir"] = str(ep_dir)
        episodes.append(ep)
        logger.info(
            "  Episode %s: reward=%.4f, %d steps, %d messages",
            ep["episode_id"], ep["reward"],
            len(scores.get("steps", [])), len(ep["messages"]),
        )

    return episodes


# ---------------------------------------------------------------------------
# Advantage computation (Eq. 3, 5, 8)
# ---------------------------------------------------------------------------

def compute_grpo_advantages(episodes: list[dict]) -> list[float]:
    """Trajectory-level GRPO advantages: A_i = (R_i - mu) / sigma  (Eq. 3)."""
    rewards = [ep["reward"] for ep in episodes]
    mu = sum(rewards) / len(rewards)
    variance = sum((r - mu) ** 2 for r in rewards) / len(rewards)
    sigma = math.sqrt(variance) if variance > 0 else 1.0
    return [(r - mu) / sigma for r in rewards]


def compute_hcapo_advantages(
    episodes: list[dict],
    omega: float = 1.0,
    use_smoothed: bool = True,
) -> list[list[float]]:
    """Multi-scale HCAPO advantages per step (Eq. 8).

    Returns a list of step-advantage lists, one per episode.
    """
    grpo_advs = compute_grpo_advantages(episodes)

    # Collect all Q_H values for global normalization
    all_qh: list[float] = []
    for ep in episodes:
        for step in ep["_hindsight"]["steps"]:
            key = "q_h_smoothed" if use_smoothed else "q_h"
            all_qh.append(step.get(key, step.get("q_h", 0.0)))

    mu_h = sum(all_qh) / len(all_qh) if all_qh else 0.0
    var_h = sum((q - mu_h) ** 2 for q in all_qh) / len(all_qh) if all_qh else 1.0
    sigma_h = math.sqrt(var_h) if var_h > 0 else 1.0

    logger.info(
        "GRPO advantages: min=%.3f max=%.3f | Q_H stats: mu=%.4f sigma=%.4f",
        min(grpo_advs), max(grpo_advs), mu_h, sigma_h,
    )

    per_episode_advantages: list[list[float]] = []
    for ep_idx, ep in enumerate(episodes):
        a_grpo = grpo_advs[ep_idx]
        steps = ep["_hindsight"]["steps"]
        key = "q_h_smoothed" if use_smoothed else "q_h"

        step_advs: list[float] = []
        for step in steps:
            qh = step.get(key, step.get("q_h", 0.0))
            a_micro = (qh - mu_h) / sigma_h

            # Do-no-harm mask: for successful trajectories, clip negative micro advantages
            if a_grpo > 0:
                a_micro = max(a_micro, 0.0)

            a_hcapo = a_grpo + omega * a_micro
            step_advs.append(a_hcapo)

        per_episode_advantages.append(step_advs)

    return per_episode_advantages


def normalize_advantages(
    per_episode_advantages: list[list[float]],
) -> list[list[float]]:
    """Clip to non-negative, then normalize so non-zero weights have mean 1.0."""
    all_positive: list[float] = []
    for advs in per_episode_advantages:
        for a in advs:
            clamped = max(a, 0.0)
            if clamped > 0:
                all_positive.append(clamped)

    mean_pos = sum(all_positive) / len(all_positive) if all_positive else 1.0

    normalized: list[list[float]] = []
    for advs in per_episode_advantages:
        normalized.append([max(a, 0.0) / mean_pos for a in advs])

    return normalized


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def identify_assistant_indices(messages: list[dict]) -> list[int]:
    return [i for i, m in enumerate(messages) if m.get("role") == "assistant"]


def build_hcapo_dataset(
    episodes: list[dict],
    per_episode_advantages: list[list[float]],
) -> list[dict]:
    dataset: list[dict] = []

    for ep, advantages in zip(episodes, per_episode_advantages):
        messages = ep["messages"]
        assistant_indices = identify_assistant_indices(messages)

        if len(advantages) != len(assistant_indices):
            logger.warning(
                "Episode %s: %d advantages vs %d assistant messages — truncating to min",
                ep["episode_id"], len(advantages), len(assistant_indices),
            )
            n = min(len(advantages), len(assistant_indices))
            advantages = advantages[:n]
            assistant_indices = assistant_indices[:n]

        # Skip episodes where all advantages are 0 (below-average trajectories)
        if all(a == 0 for a in advantages):
            logger.info("  Episode %s: all advantages are 0, excluded", ep["episode_id"])
            continue

        grpo_advs = compute_grpo_advantages(episodes)
        ep_idx = episodes.index(ep)

        dataset.append({
            "messages": messages,
            "step_advantages": [round(a, 6) for a in advantages],
            "step_message_indices": assistant_indices,
            "_episode_id": ep["episode_id"],
            "_reward": ep["reward"],
            "_grpo_advantage": round(grpo_advs[ep_idx], 6),
            "_num_steps": len(advantages),
        })

    return dataset


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_jsonl(data: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info("Wrote %d examples to %s (%.1f KB)", len(data), path, path.stat().st_size / 1024)


def write_summary(
    data: list[dict],
    episodes: list[dict],
    args: argparse.Namespace,
    path: Path,
) -> None:
    all_advs = []
    for row in data:
        all_advs.extend(row["step_advantages"])

    nonzero = [a for a in all_advs if a > 0]
    summary = {
        "total_episodes_loaded": len(episodes),
        "episodes_in_dataset": len(data),
        "total_steps": len(all_advs),
        "nonzero_steps": len(nonzero),
        "advantage_stats": {
            "min": round(min(all_advs), 4) if all_advs else 0,
            "max": round(max(all_advs), 4) if all_advs else 0,
            "mean": round(sum(all_advs) / len(all_advs), 4) if all_advs else 0,
            "nonzero_mean": round(sum(nonzero) / len(nonzero), 4) if nonzero else 0,
        },
        "config": {
            "omega": args.omega,
            "min_reward": args.min_reward,
            "use_smoothed": not args.no_smooth,
        },
    }
    path.write_text(json.dumps(summary, indent=2))
    logger.info("Summary → %s", path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build HCAPO step-weighted SFT dataset")
    parser.add_argument("--input-dir", default="trajectories")
    parser.add_argument("--output-dir", default="datasets")
    parser.add_argument("--min-reward", type=float, default=0.2, help="Min episode reward to include")
    parser.add_argument("--omega", type=float, default=1.0, help="Hindsight weighting coefficient (Eq. 8)")
    parser.add_argument("--no-smooth", action="store_true", help="Use raw Q_H instead of smoothed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    logger.info("Loading episodes from %s...", input_dir)
    episodes = load_episodes_with_scores(input_dir, min_reward=args.min_reward)

    if not episodes:
        logger.error("No valid episodes found! Run compute_hindsight_scores.py first.")
        sys.exit(1)

    logger.info(
        "Loaded %d episodes (rewards: %.4f — %.4f)",
        len(episodes),
        min(ep["reward"] for ep in episodes),
        max(ep["reward"] for ep in episodes),
    )

    logger.info("Computing HCAPO advantages (omega=%.2f)...", args.omega)
    raw_advantages = compute_hcapo_advantages(
        episodes, omega=args.omega, use_smoothed=not args.no_smooth,
    )

    logger.info("Normalizing advantages...")
    advantages = normalize_advantages(raw_advantages)

    logger.info("Building dataset...")
    dataset = build_hcapo_dataset(episodes, advantages)

    if not dataset:
        logger.error("No usable episodes after advantage computation!")
        sys.exit(1)

    write_jsonl(dataset, output_dir / "hcapo_train.jsonl")
    write_summary(dataset, episodes, args, output_dir / "hcapo_summary.json")

    logger.info(
        "Done — %d episodes, %d total steps in dataset.",
        len(dataset),
        sum(row["_num_steps"] for row in dataset),
    )


if __name__ == "__main__":
    main()
