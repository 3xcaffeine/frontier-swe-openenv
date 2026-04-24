#!/usr/bin/env python3
"""Convert Pi session JSONL trajectories into datasets for offline RL training.

Produces three output formats from the same trajectory data:

1. DPO (Direct Preference Optimization) — for TRL DPOTrainer / Unsloth
    Pairs trajectories by reward to create chosen/rejected preference pairs.
    Supports episode-level and turn-level pairing.
    Format: {prompt: [...], chosen: [...], rejected: [...]}.

2. SFT (Supervised Fine-Tuning) — for TRL SFTTrainer / Unsloth
    Top-K episodes as multi-turn conversations for imitation learning.
    Supports conversational or text-column outputs.
    Format: {messages: [...]} or {text: "..."}.

3. GRPO (Group Relative Policy Optimization) — for TRL GRPOTrainer
    Prompt-only dataset for online generation.
    Supports first-turn-only or all-user-turn prompts.
    Format: {prompt: [...]}.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_dataset")


# JSONL → conversation messages


def _serialize_tool_arguments(args: Any) -> str:
    """Serialize tool call arguments to a JSON string when possible."""
    if isinstance(args, str):
        return args
    return json.dumps(args if args is not None else {}, ensure_ascii=False)


def _content_text(content: Any) -> str:
    """Extract text content from Pi content blocks or scalar content."""
    if isinstance(content, list):
        return "\n".join(
            c.get("text", "")
            for c in content
            if isinstance(c, dict) and c.get("type") == "text"
        ).strip()
    return str(content).strip()


def _assistant_from_pi_message(msg: dict, include_thinking: bool) -> dict | None:
    """Convert a Pi assistant message into assistant + structured tool_calls."""
    content = msg.get("content", [])
    if not isinstance(content, list):
        text = str(content).strip()
        return {"role": "assistant", "content": text} if text else None

    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []

    for item in content:
        if not isinstance(item, dict):
            continue
        t = item.get("type", "")
        if t == "thinking" and include_thinking:
            text = item.get("text", "").strip()
            if text:
                text_parts.append(f"<think>\n{text}\n</think>")
        elif t == "text":
            text = item.get("text", "").strip()
            if text:
                text_parts.append(text)
        elif t == "toolCall":
            name = item.get("name", "unknown")
            args = item.get("arguments", item.get("input", {}))
            call_id = item.get("id", "")
            tool_calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": _serialize_tool_arguments(args),
                    },
                }
            )

    assistant_msg: dict[str, Any] = {
        "role": "assistant",
        "content": "\n\n".join(text_parts).strip(),
    }
    if tool_calls:
        assistant_msg["tool_calls"] = tool_calls

    # Keep assistant messages with tool calls even when textual content is empty.
    if assistant_msg["content"] or assistant_msg.get("tool_calls"):
        return assistant_msg
    return None


def _tool_from_pi_message(msg: dict, max_tool_result_chars: int) -> dict | None:
    """Convert a Pi toolResult message into a structured tool message."""
    text = _content_text(msg.get("content", []))
    if len(text) > max_tool_result_chars:
        text = (
            text[:max_tool_result_chars] + f"\n... [truncated, {len(text)} chars total]"
        )

    tool_name = msg.get("toolName", "unknown")
    tool_call_id = msg.get("toolCallId", "")
    is_error = msg.get("isError", False)
    if is_error:
        text = f"[tool_error]\n{text}" if text else "[tool_error]"

    return {
        "role": "tool",
        "name": tool_name,
        "tool_call_id": tool_call_id,
        "content": text,
    }


def _validate_messages(messages: list[dict]) -> list[dict]:
    """Drop malformed messages and enforce basic role/content constraints."""
    valid_roles = {"user", "assistant", "tool"}
    cleaned: list[dict] = []
    for msg in messages:
        role = msg.get("role")
        if role not in valid_roles:
            continue
        if role == "assistant":
            content = msg.get("content", "")
            if not isinstance(content, str):
                continue
            if not content and not msg.get("tool_calls"):
                continue
        elif role in {"user", "tool"}:
            content = msg.get("content", "")
            if not isinstance(content, str):
                continue
            if role == "user" and not content:
                continue
        cleaned.append(msg)
    return cleaned


def load_trajectory(
    session_path: Path,
    include_thinking: bool = True,
    max_tool_result_chars: int = 4000,
) -> list[dict]:
    """Load a Pi session JSONL and convert to OpenAI-style conversation messages.

    Returns messages with roles user, assistant and tool.
    Assistant tool calls are preserved as assistant.tool_calls.
    """
    raw_entries = []
    with open(session_path) as f:
        for line in f:
            line = line.strip()
            if line:
                raw_entries.append(json.loads(line))

    msg_entries = []
    for entry in raw_entries:
        if entry.get("type") == "compaction":
            # Keep full original transcript and skip compaction summaries.
            continue
        if entry.get("type") == "message":
            msg_entries.append(entry["message"])

    messages = []
    for msg in msg_entries:
        role = msg.get("role", "")
        content = msg.get("content", [])

        if role == "user":
            text = _content_text(content)
            if text:
                messages.append({"role": "user", "content": text})
        elif role == "assistant":
            assistant_msg = _assistant_from_pi_message(
                msg, include_thinking=include_thinking
            )
            if assistant_msg:
                messages.append(assistant_msg)
        elif role == "toolResult":
            tool_msg = _tool_from_pi_message(
                msg, max_tool_result_chars=max_tool_result_chars
            )
            if tool_msg:
                messages.append(tool_msg)

    return _validate_messages(messages)


def load_episode(
    episode_dir: Path,
    include_thinking: bool = True,
    max_tool_result_chars: int = 4000,
) -> dict | None:
    """Load a complete episode: messages + metadata."""
    result_path = episode_dir / "result.json"
    session_path = episode_dir / "pi_session.jsonl"

    if not result_path.exists() or not session_path.exists():
        return None

    result = json.loads(result_path.read_text())
    reward = result.get("episode_reward")
    if reward is None:
        return None

    messages = load_trajectory(
        session_path,
        include_thinking=include_thinking,
        max_tool_result_chars=max_tool_result_chars,
    )
    if len(messages) < 2:  # Need at least user + assistant
        return None

    return {
        "episode_id": result.get("episode_id"),
        "messages": messages,
        "reward": reward,
        "plan_score": result.get("plan_score", 0),
        "frozen_scores": result.get("frozen_scores", {}),
        "turns": result.get("turns", 0),
        "phase": result.get("phase", ""),
        "tool_call_count": result.get("tool_call_count", 0),
    }


# Dataset builders


def split_user_turns(messages: list[dict]) -> list[dict]:
    """Split conversation into user-turn slices.

    Each turn contains:
      - prompt: messages up to and including a user message
      - completion: assistant/tool messages until the next user message
    """
    turns: list[dict] = []
    user_indices = [i for i, msg in enumerate(messages) if msg.get("role") == "user"]

    for t_idx, u_idx in enumerate(user_indices):
        next_u_idx = (
            user_indices[t_idx + 1] if t_idx + 1 < len(user_indices) else len(messages)
        )
        prompt = messages[: u_idx + 1]
        completion = messages[u_idx + 1 : next_u_idx]

        if any(m.get("role") == "assistant" for m in completion):
            turns.append(
                {
                    "turn_index": t_idx,
                    "prompt": prompt,
                    "completion": completion,
                }
            )

    return turns


def render_messages_as_text(messages: list[dict]) -> str:
    """Render conversation into text-only transcript for SFT text datasets."""
    lines: list[str] = []
    for msg in messages:
        role = msg.get("role", "")
        if role == "assistant" and msg.get("tool_calls"):
            if msg.get("content", ""):
                lines.append(f"assistant: {msg['content']}")
            tool_calls = json.dumps(msg.get("tool_calls"), ensure_ascii=False)
            lines.append(f"assistant_tool_calls: {tool_calls}")
        elif role == "tool":
            name = msg.get("name", "tool")
            lines.append(f"tool[{name}]: {msg.get('content', '')}")
        else:
            lines.append(f"{role}: {msg.get('content', '')}")
    return "\n".join(lines).strip()


def _user_prompt_key(prompt: list[dict]) -> str:
    """Build a stable key for aligning/deduping prompt turns."""
    if not prompt:
        return ""
    last = prompt[-1]
    if last.get("role") == "user":
        return last.get("content", "")
    return json.dumps(prompt, ensure_ascii=False, sort_keys=True)


def build_dpo_pairs(
    episodes: list[dict],
    reward_gap: float = 0.03,
    max_pairs: int = 500,
    mode: str = "turn",
) -> list[dict]:
    """Build DPO preference pairs from episode trajectories.

    Mode "episode" uses one pair per episode pair.
    Mode "turn" aligns turns by index and user prompt content,
    yielding denser multi-turn preference supervision.

    We pair episodes where reward_chosen - reward_rejected >= reward_gap.
    All combinations are generated up to max_pairs.

    TRL DPOTrainer conversational format:
      {
        "prompt": [{"role": "user", "content": "..."}],
        "chosen": [{"role": "assistant", "content": "..."}, ...],
        "rejected": [{"role": "assistant", "content": "..."}, ...]
      }
    """
    # Sort by reward descending
    episodes_sorted = sorted(episodes, key=lambda x: x["reward"], reverse=True)

    pairs = []
    for i, chosen_ep in enumerate(episodes_sorted):
        for j in range(i + 1, len(episodes_sorted)):
            rejected_ep = episodes_sorted[j]
            gap = chosen_ep["reward"] - rejected_ep["reward"]

            if gap < reward_gap:
                continue

            chosen_msgs = chosen_ep["messages"]
            rejected_msgs = rejected_ep["messages"]

            if mode == "episode":
                first_user_idx_c = next(
                    (k for k, m in enumerate(chosen_msgs) if m["role"] == "user"), -1
                )
                first_user_idx_r = next(
                    (k for k, m in enumerate(rejected_msgs) if m["role"] == "user"), -1
                )
                if first_user_idx_c < 0 or first_user_idx_r < 0:
                    continue

                prompt_msgs = chosen_msgs[: first_user_idx_c + 1]
                chosen_response = chosen_msgs[first_user_idx_c + 1 :]
                rejected_response = rejected_msgs[first_user_idx_r + 1 :]
                if not prompt_msgs or not chosen_response or not rejected_response:
                    continue

                pairs.append(
                    {
                        "prompt": prompt_msgs,
                        "chosen": chosen_response,
                        "rejected": rejected_response,
                        "_chosen_reward": chosen_ep["reward"],
                        "_rejected_reward": rejected_ep["reward"],
                        "_chosen_ep": chosen_ep["episode_id"],
                        "_rejected_ep": rejected_ep["episode_id"],
                        "_reward_gap": round(gap, 4),
                        "_pair_mode": "episode",
                    }
                )
            else:
                chosen_turns = split_user_turns(chosen_msgs)
                rejected_turns = split_user_turns(rejected_msgs)
                turn_count = min(len(chosen_turns), len(rejected_turns))

                for t_idx in range(turn_count):
                    c_turn = chosen_turns[t_idx]
                    r_turn = rejected_turns[t_idx]

                    # Avoid pairing semantically different turns.
                    if _user_prompt_key(c_turn["prompt"]) != _user_prompt_key(
                        r_turn["prompt"]
                    ):
                        continue

                    if not c_turn["completion"] or not r_turn["completion"]:
                        continue

                    pairs.append(
                        {
                            "prompt": c_turn["prompt"],
                            "chosen": c_turn["completion"],
                            "rejected": r_turn["completion"],
                            "_chosen_reward": chosen_ep["reward"],
                            "_rejected_reward": rejected_ep["reward"],
                            "_chosen_ep": chosen_ep["episode_id"],
                            "_rejected_ep": rejected_ep["episode_id"],
                            "_reward_gap": round(gap, 4),
                            "_pair_mode": "turn",
                            "_turn_index": t_idx,
                        }
                    )

                    if len(pairs) >= max_pairs:
                        return pairs

            if len(pairs) >= max_pairs:
                return pairs

    return pairs


def build_sft_dataset(
    episodes: list[dict],
    top_k: int = 5,
    min_reward: float = 0.0,
    output_format: str = "conversational",
) -> list[dict]:
    """Build SFT dataset from top-K episodes.

    Output formats:
      - conversational: {"messages": [...]} for chat-template SFT
      - text: {"text": "..."} for text-column SFT
    """
    # Filter and sort
    valid = [e for e in episodes if e["reward"] >= min_reward]
    valid.sort(key=lambda x: x["reward"], reverse=True)
    selected = valid[:top_k]

    dataset = []
    for ep in selected:
        if output_format == "text":
            dataset.append(
                {
                    "text": render_messages_as_text(ep["messages"]),
                    "_episode_id": ep["episode_id"],
                    "_reward": ep["reward"],
                }
            )
        else:
            dataset.append(
                {
                    "messages": ep["messages"],
                    "_episode_id": ep["episode_id"],
                    "_reward": ep["reward"],
                }
            )

    return dataset


def build_grpo_dataset(
    episodes: list[dict],
    prompt_mode: str = "all-user-turns",
) -> list[dict]:
    """Build GRPO prompt-only dataset.

    GRPO generates its own completions online, so we only need prompts.

    TRL GRPOTrainer format:
      {"prompt": [{"role": "user", "content": "..."}]}
    """
    if not episodes:
        return []

    if prompt_mode == "first":
        first_user = None
        for m in episodes[0]["messages"]:
            if m["role"] == "user":
                first_user = m
                break
        if not first_user:
            return []
        return [{"prompt": [first_user], "_prompt_mode": "first"}]

    prompts: list[dict] = []
    seen: set[str] = set()
    for ep in episodes:
        for turn in split_user_turns(ep["messages"]):
            key = json.dumps(turn["prompt"], ensure_ascii=False, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            prompts.append(
                {
                    "prompt": turn["prompt"],
                    "_prompt_mode": "all-user-turns",
                    "_source_episode": ep["episode_id"],
                    "_turn_index": turn["turn_index"],
                }
            )

    return prompts


# Output writers


def write_jsonl(data: list[dict], path: Path, strip_meta: bool = False) -> None:
    """Write dataset as JSONL (one JSON object per line)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for item in data:
            if strip_meta:
                item = {k: v for k, v in item.items() if not k.startswith("_")}
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info(
        "Wrote %d examples to %s (%.1f KB)", len(data), path, path.stat().st_size / 1024
    )


def write_hf_dataset(data: list[dict], path: Path, strip_meta: bool = False) -> None:
    """Write as a HuggingFace-compatible dataset directory.

    Saves a Dataset directory that can be loaded with:
        from datasets import load_from_disk
        ds = load_from_disk(str(path))
    """
    try:
        from datasets import Dataset

        if strip_meta:
            data = [
                {k: v for k, v in item.items() if not k.startswith("_")}
                for item in data
            ]
        ds = Dataset.from_list(data)
        ds.save_to_disk(str(path))
        logger.info("Wrote HF dataset (%d rows) to %s", len(ds), path)
    except ImportError:
        logger.warning(
            "datasets library not installed, skipping HF format. "
            "Install with: pip install datasets"
        )


# Main


def main():
    parser = argparse.ArgumentParser(
        description="Convert trajectories to training datasets",
    )
    parser.add_argument(
        "--input-dir",
        default="trajectories",
        help="Directory containing episode_NNN/ subdirectories",
    )
    parser.add_argument(
        "--output-dir",
        default="datasets",
        help="Output directory for generated datasets",
    )
    parser.add_argument(
        "--format",
        choices=["all", "dpo", "sft", "grpo"],
        default="all",
        help="Which dataset format(s) to generate",
    )
    parser.add_argument(
        "--dpo-mode",
        choices=["episode", "turn"],
        default="turn",
        help="DPO pairing mode: whole-episode or per-turn pairs (default: turn)",
    )
    parser.add_argument(
        "--sft-format",
        choices=["conversational", "text"],
        default="conversational",
        help="SFT output schema (default: conversational)",
    )
    parser.add_argument(
        "--grpo-prompts",
        choices=["first", "all-user-turns"],
        default="all-user-turns",
        help="GRPO prompt extraction mode (default: all-user-turns)",
    )
    parser.add_argument(
        "--reward-gap",
        type=float,
        default=0.03,
        help="Minimum reward gap between chosen/rejected for DPO pairs (default: 0.03)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top episodes for SFT dataset (default: 5)",
    )
    parser.add_argument(
        "--min-reward",
        type=float,
        default=0.0,
        help="Minimum reward threshold for SFT episodes (default: 0.0)",
    )
    parser.add_argument(
        "--min-usable-reward",
        type=float,
        default=0.0,
        help="Minimum reward threshold for usable episodes in SFT/GRPO (default: 0.0)",
    )
    parser.add_argument(
        "--max-tool-result-chars",
        type=int,
        default=4000,
        help="Max chars to keep from each tool result (default: 4000)",
    )
    parser.add_argument(
        "--no-thinking",
        action="store_true",
        help="Drop assistant thinking blocks from converted messages",
    )
    parser.add_argument(
        "--hf-format",
        action="store_true",
        help="Also save as HuggingFace Dataset (requires 'datasets' library)",
    )
    parser.add_argument(
        "--strip-meta",
        action="store_true",
        help="Remove _prefixed metadata fields from output (cleaner for training)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Load all episodes
    logger.info("Loading episodes from %s...", input_dir)
    episodes = []
    for ep_dir in sorted(input_dir.glob("episode_*")):
        ep = load_episode(
            ep_dir,
            include_thinking=not args.no_thinking,
            max_tool_result_chars=args.max_tool_result_chars,
        )
        if ep:
            episodes.append(ep)
            logger.info(
                "  Episode %s: reward=%.4f msgs=%d phase=%s",
                ep["episode_id"],
                ep["reward"],
                len(ep["messages"]),
                ep["phase"],
            )
        else:
            logger.warning("  %s: skipped (missing data or no reward)", ep_dir.name)

    if not episodes:
        logger.error("No valid episodes found!")
        return

    logger.info(
        "Loaded %d episodes (rewards: %.4f - %.4f, mean %.4f)",
        len(episodes),
        min(e["reward"] for e in episodes),
        max(e["reward"] for e in episodes),
        sum(e["reward"] for e in episodes) / len(episodes),
    )

    usable = [e for e in episodes if e["reward"] >= args.min_usable_reward]
    logger.info(
        "Usable episodes (reward >= %.4f): %d", args.min_usable_reward, len(usable)
    )

    # Build datasets
    formats = args.format
    pairs: list[dict] = []
    sft_data: list[dict] = []
    grpo_data: list[dict] = []

    if formats in ("all", "dpo"):
        logger.info("\n=== Building DPO dataset ===")
        pairs = build_dpo_pairs(
            episodes,
            reward_gap=args.reward_gap,
            mode=args.dpo_mode,
        )
        logger.info(
            "Generated %d DPO pairs (min gap=%.4f)",
            len(pairs),
            min(p["_reward_gap"] for p in pairs) if pairs else 0,
        )

        if pairs:
            # Show distribution
            gaps = [p["_reward_gap"] for p in pairs]
            logger.info(
                "  Gap distribution: min=%.4f max=%.4f mean=%.4f",
                min(gaps),
                max(gaps),
                sum(gaps) / len(gaps),
            )

            write_jsonl(
                pairs, output_dir / "dpo_train.jsonl", strip_meta=args.strip_meta
            )
            if args.hf_format:
                write_hf_dataset(
                    pairs, output_dir / "dpo_hf", strip_meta=args.strip_meta
                )
        else:
            logger.warning("No DPO pairs generated! Try lowering --reward-gap")

    if formats in ("all", "sft"):
        logger.info("\n=== Building SFT dataset ===")
        sft_data = build_sft_dataset(
            usable,
            top_k=args.top_k,
            min_reward=args.min_reward,
            output_format=args.sft_format,
        )
        logger.info("Generated %d SFT examples", len(sft_data))

        if sft_data:
            for item in sft_data:
                if args.sft_format == "text":
                    logger.info(
                        "  Episode %s: reward=%.4f text_chars=%d",
                        item["_episode_id"],
                        item["_reward"],
                        len(item.get("text", "")),
                    )
                else:
                    logger.info(
                        "  Episode %s: reward=%.4f msgs=%d",
                        item["_episode_id"],
                        item["_reward"],
                        len(item.get("messages", [])),
                    )
            write_jsonl(
                sft_data, output_dir / "sft_train.jsonl", strip_meta=args.strip_meta
            )
            if args.hf_format:
                write_hf_dataset(
                    sft_data, output_dir / "sft_hf", strip_meta=args.strip_meta
                )

    if formats in ("all", "grpo"):
        logger.info("\n=== Building GRPO dataset ===")
        grpo_data = build_grpo_dataset(usable, prompt_mode=args.grpo_prompts)
        logger.info("Generated %d GRPO prompts", len(grpo_data))

        if grpo_data:
            write_jsonl(
                grpo_data, output_dir / "grpo_train.jsonl", strip_meta=args.strip_meta
            )
            if args.hf_format:
                write_hf_dataset(
                    grpo_data, output_dir / "grpo_hf", strip_meta=args.strip_meta
                )

    # Write a summary
    summary = {
        "input_dir": str(input_dir),
        "total_episodes": len(episodes),
        "usable_episodes": len(usable),
        "conversion": {
            "include_thinking": not args.no_thinking,
            "max_tool_result_chars": args.max_tool_result_chars,
            "dpo_mode": args.dpo_mode,
            "sft_format": args.sft_format,
            "grpo_prompts": args.grpo_prompts,
            "min_usable_reward": args.min_usable_reward,
        },
        "reward_stats": {
            "min": round(min(e["reward"] for e in episodes), 4),
            "max": round(max(e["reward"] for e in episodes), 4),
            "mean": round(sum(e["reward"] for e in episodes) / len(episodes), 4),
        },
        "formats_generated": [],
    }

    if formats in ("all", "dpo"):
        summary["formats_generated"].append(
            {
                "format": "dpo",
                "file": "dpo_train.jsonl",
                "pairs": len(pairs),
                "pair_mode": args.dpo_mode,
                "reward_gap": args.reward_gap,
                "usage": (
                    "from datasets import load_dataset\n"
                    "from trl import DPOTrainer, DPOConfig\n\n"
                    "ds = load_dataset('json', data_files='datasets/dpo_train.jsonl', split='train')\n"
                    "trainer = DPOTrainer(\n"
                    "    model='your-model',\n"
                    "    train_dataset=ds,\n"
                    "    args=DPOConfig(output_dir='dpo_output'),\n"
                    ")\n"
                    "trainer.train()"
                ),
            }
        )

    if formats in ("all", "sft"):
        summary["formats_generated"].append(
            {
                "format": "sft",
                "file": "sft_train.jsonl",
                "examples": len(sft_data),
                "top_k": args.top_k,
                "sft_format": args.sft_format,
                "usage": (
                    "from datasets import load_dataset\n"
                    "from trl import SFTTrainer, SFTConfig\n\n"
                    "ds = load_dataset('json', data_files='datasets/sft_train.jsonl', split='train')\n"
                    "trainer = SFTTrainer(\n"
                    "    model='your-model',\n"
                    "    train_dataset=ds,\n"
                    "    args=SFTConfig(output_dir='sft_output'),\n"
                    ")\n"
                    "trainer.train()"
                ),
                "unsloth_usage": (
                    "from datasets import load_dataset\n"
                    "from unsloth import FastLanguageModel\n"
                    "from trl import SFTTrainer, SFTConfig\n\n"
                    "ds = load_dataset('json', data_files='datasets/sft_train.jsonl', split='train')\n"
                    "# If using sft-format=text, set dataset_text_field='text' in SFTConfig.\n"
                    "model, tokenizer = FastLanguageModel.from_pretrained(model_name='Qwen/Qwen3.6-27B')\n"
                    "trainer = SFTTrainer(model=model, train_dataset=ds, tokenizer=tokenizer, args=SFTConfig(output_dir='sft_output'))\n"
                    "trainer.train()"
                ),
            }
        )

    if formats in ("all", "grpo"):
        summary["formats_generated"].append(
            {
                "format": "grpo",
                "file": "grpo_train.jsonl",
                "prompts": len(grpo_data),
                "prompt_mode": args.grpo_prompts,
                "usage": (
                    "from datasets import load_dataset\n"
                    "from trl import GRPOTrainer, GRPOConfig\n\n"
                    "ds = load_dataset('json', data_files='datasets/grpo_train.jsonl', split='train')\n"
                    "trainer = GRPOTrainer(\n"
                    "    model='your-model',\n"
                    "    reward_funcs=your_reward_fn,\n"
                    "    train_dataset=ds,\n"
                    "    args=GRPOConfig(output_dir='grpo_output'),\n"
                    ")\n"
                    "trainer.train()"
                ),
            }
        )

    summary_path = output_dir / "dataset_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("\nSummary written to %s", summary_path)


if __name__ == "__main__":
    main()
