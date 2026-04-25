#!/usr/bin/env python3
"""Train with HCAPO step-level advantages using Unsloth + TRL.

Implements offline HCAPO training: each assistant message in a multi-turn
conversation gets a per-step advantage weight derived from hindsight credit
assignment (paper 2603.08754, Eq. 8).

Expected dataset format (produced by build_hcapo_dataset.py):
  {
    "messages": [... multi-turn conversation ...],
    "step_advantages": [1.23, 0.87, 1.45, ...],
    "step_message_indices": [1, 4, 7, ...],
    "_episode_id": 12,
    "_reward": 0.4058
  }

Usage:
    uv run python scripts/train_hcapo.py --config training/hcapo_config.json --max-steps 1  # smoke test
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import random
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_hcapo")


# Helpers


def _seed_everything(seed: int, torch_module: Any) -> None:
    random.seed(seed)
    torch_module.manual_seed(seed)
    torch_module.cuda.manual_seed_all(seed)


def _normalize_tool_arguments(arguments: Any) -> dict[str, Any]:
    if arguments is None:
        return {"arguments": "{}"}
    if isinstance(arguments, str):
        text = arguments.strip()
        if not text:
            return {"arguments": "{}"}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {"arguments": arguments}
        return {"arguments": json.dumps(parsed, ensure_ascii=False)}
    return {"arguments": json.dumps(arguments, ensure_ascii=False)}


def _normalize_chat_message(message: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(message)
    tool_calls = normalized.get("tool_calls")
    if not isinstance(tool_calls, list):
        return normalized
    out_calls: list[Any] = []
    for tc in tool_calls:
        if not isinstance(tc, dict):
            out_calls.append(tc)
            continue
        call = dict(tc)
        fn = call.get("function")
        if isinstance(fn, dict):
            fn = dict(fn)
            fn["arguments"] = _normalize_tool_arguments(fn.get("arguments"))
            call["function"] = fn
        elif "arguments" in call:
            call["arguments"] = _normalize_tool_arguments(call.get("arguments"))
        out_calls.append(call)
    normalized["tool_calls"] = out_calls
    return normalized


def _normalize_messages(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [_normalize_chat_message(m) for m in value if isinstance(m, dict)]


# Dataset preparation


def _normalize_hcapo_example(example: dict[str, Any]) -> dict[str, Any]:
    return {
        "messages": _normalize_messages(example.get("messages")),
        "step_advantages": example.get("step_advantages", []),
        "step_message_indices": example.get("step_message_indices", []),
        "reward": example.get("_reward") or example.get("reward") or 0.0,
        "episode_id": example.get("_episode_id") or example.get("episode_id") or -1,
    }


def _has_assistant_message(messages: list[dict]) -> bool:
    return any(m.get("role") == "assistant" for m in messages)


def _load_and_prepare_dataset(args: argparse.Namespace) -> Any:
    from datasets import load_dataset

    data_files = args.dataset
    if args.dataset_id:
        from huggingface_hub import hf_hub_download

        logger.info(
            "Downloading HCAPO dataset %s/%s",
            args.dataset_id,
            args.dataset_filename,
        )
        data_files = hf_hub_download(
            repo_id=args.dataset_id,
            repo_type="dataset",
            filename=args.dataset_filename,
        )

    logger.info("Loading HCAPO dataset from %s", data_files)
    ds = load_dataset("json", data_files=data_files, split="train")
    logger.info("Loaded %d raw rows", len(ds))

    if len(ds) == 0:
        raise ValueError("Dataset is empty")

    ds = ds.map(_normalize_hcapo_example, num_proc=args.num_proc)
    keep_cols = {
        "messages",
        "step_advantages",
        "step_message_indices",
        "reward",
        "episode_id",
    }
    drop_cols = [c for c in ds.column_names if c not in keep_cols]
    if drop_cols:
        ds = ds.remove_columns(drop_cols)

    ds = ds.filter(
        lambda row: (
            len(row.get("messages") or []) > 0
            and _has_assistant_message(row.get("messages") or [])
            and len(row.get("step_advantages") or []) > 0
        ),
        num_proc=args.num_proc,
    )

    if len(ds) == 0:
        raise ValueError("No usable rows after filtering")

    total_steps = sum(len(row["step_advantages"]) for row in ds)
    logger.info("Prepared %d episodes, %d total steps", len(ds), total_steps)
    return ds


# Custom HCAPO Trainer + Data Collator


def _find_label_spans(labels: list[int]) -> list[tuple[int, int]]:
    """Find contiguous non-(-100) spans in labels.

    Each span corresponds to one assistant message's trainable tokens.
    """
    spans: list[tuple[int, int]] = []
    in_span = False
    start = 0
    for i, label in enumerate(labels):
        if label != -100:
            if not in_span:
                start = i
                in_span = True
        else:
            if in_span:
                spans.append((start, i))
                in_span = False
    if in_span:
        spans.append((start, len(labels)))
    return spans


def _build_hcapo_data_collator(
    processing_class: Any,
    sft_args: Any,
    data_collator_cls: type,
) -> Any:
    pad_token = (
        sft_args.pad_token or processing_class.pad_token or processing_class.eos_token
    )
    pad_token_id = processing_class.convert_tokens_to_ids(pad_token)
    if pad_token_id is None:
        raise ValueError(f"Pad token ({pad_token!r}) not in vocabulary")

    base_collator = data_collator_cls(
        pad_token_id=pad_token_id,
        completion_only_loss=False,
        padding_free=sft_args.padding_free,
        return_position_ids=False,
        pad_to_multiple_of=sft_args.pad_to_multiple_of,
    )

    class HCAPODataCollator:
        """Collator that preserves step_advantages and builds per-token step_weights."""

        def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
            import torch

            all_step_advs = []
            for ex in examples:
                all_step_advs.append(ex.pop("step_advantages", []))
                ex.pop("step_message_indices", None)
                ex.pop("reward", None)
                ex.pop("episode_id", None)

            batch = base_collator(examples)

            labels = batch["labels"]
            batch_size, seq_len = labels.shape
            step_weights = torch.ones(batch_size, seq_len, dtype=torch.float32)

            for b in range(batch_size):
                row_labels = labels[b].tolist()
                spans = _find_label_spans(row_labels)
                advs = all_step_advs[b] if b < len(all_step_advs) else []

                for span_idx, (start, end) in enumerate(spans):
                    weight = advs[span_idx] if span_idx < len(advs) else 1.0
                    step_weights[b, start:end] = max(weight, 0.0)

            batch["step_weights"] = step_weights
            return batch

    return HCAPODataCollator()


def _build_hcapo_trainer_cls(sft_trainer_cls: type) -> type:
    """Build a Trainer subclass that weights loss by per-step HCAPO advantages."""

    class HCAPOTrainer(sft_trainer_cls):
        def compute_loss(
            self,
            model: Any,
            inputs: dict[str, Any],
            return_outputs: bool = False,
            **kwargs: Any,
        ) -> Any:
            import torch

            inputs = dict(inputs)
            step_weights = inputs.pop("step_weights", None)

            labels = inputs.pop("labels", None)
            if labels is None:
                raise ValueError("HCAPO training requires labels")

            outputs = model(**inputs)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            token_loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="none",
                ignore_index=-100,
            ).view_as(shift_labels)

            label_mask = shift_labels.ne(-100)

            if step_weights is not None:
                shift_weights = step_weights[..., 1:].contiguous()
                weighted_loss = token_loss * shift_weights
                denom = (label_mask.float() * shift_weights).sum().clamp_min(1.0)
                loss = weighted_loss.sum() / denom
            else:
                loss = token_loss.sum() / label_mask.sum().clamp_min(1)

            return (loss, outputs) if return_outputs else loss

    return HCAPOTrainer


def _as_token_list(value: Any) -> list[int]:
    """Normalize tokenizer output that may be either a flat or batched list."""
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and value and isinstance(value[0], list):
        value = value[0]
    return list(value or [])


def _ensure_generation_chat_template(processing_class: Any) -> None:
    """Add generation blocks to Qwen-style templates for assistant masks.

    Transformers only returns `assistant_masks` when the chat template marks
    assistant output with `{% generation %}` / `{% endgeneration %}`. Qwen 3.5's
    template currently lacks those markers, so patch only the assistant branch
    in memory before tokenizing.
    """
    template = getattr(processing_class, "chat_template", None)
    if not template:
        raise RuntimeError("Tokenizer has no chat_template")
    if "{% generation %}" in template:
        return

    lines = template.splitlines()
    assistant_idx = next(
        (
            idx
            for idx, line in enumerate(lines)
            if line.strip() == '{%- elif message.role == "assistant" %}'
        ),
        None,
    )
    if assistant_idx is None:
        raise RuntimeError("Could not locate assistant branch in chat_template")

    end_idx = next(
        (
            idx
            for idx in range(assistant_idx + 1, len(lines))
            if lines[idx].strip() == "{{- '<|im_end|>\\n' }}"
        ),
        None,
    )
    if end_idx is None:
        raise RuntimeError(
            "Could not locate assistant branch terminator in chat_template"
        )

    lines.insert(assistant_idx + 1, "        {% generation %}")
    lines.insert(end_idx + 2, "        {% endgeneration %}")
    processing_class.chat_template = "\n".join(lines)
    logger.info("Patched tokenizer chat_template with assistant generation markers")


def _tokenize_hcapo_dataset(
    dataset: Any, processing_class: Any, args: argparse.Namespace
) -> Any:
    """Pre-tokenize chat examples so Unsloth skips its formatting_func path.

    The current Unsloth SFTTrainer wrapper requires a formatting_func whenever
    the dataset lacks a plain `text` column, even though TRL can handle
    conversational `messages` directly. The patched template emits
    `assistant_masks`, which our collator uses for assistant-only labels.
    """
    _ensure_generation_chat_template(processing_class)

    def tokenize_example(example: dict[str, Any]) -> dict[str, Any]:
        messages = example.get("messages") or []
        processed = processing_class.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_assistant_tokens_mask=True,
            truncation=True,
            max_length=args.max_seq_length,
        )

        input_ids = _as_token_list(processed.get("input_ids"))
        assistant_masks = _as_token_list(processed.get("assistant_masks"))
        if len(input_ids) != len(assistant_masks):
            raise RuntimeError(
                f"assistant_masks length mismatch: {len(assistant_masks)} vs {len(input_ids)} input_ids"
            )
        if 1 not in assistant_masks:
            raise RuntimeError(
                "Tokenized example has no assistant tokens within max_seq_length"
            )

        return {
            "input_ids": input_ids,
            "assistant_masks": assistant_masks,
        }

    logger.info("Tokenizing chat dataset with assistant masks...")
    tokenized = dataset.map(
        tokenize_example,
        remove_columns=["messages"],
        num_proc=args.num_proc,
        desc="Tokenizing HCAPO chats",
    )
    logger.info("Tokenized %d HCAPO examples", len(tokenized))
    return tokenized


# Model + SFT config helpers


def _remove_qwen_vision_mappings() -> dict[str, str]:
    from transformers.models.auto.modeling_auto import (
        MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES,
    )

    popped: dict[str, str] = {}
    for key in list(MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES.keys()):
        if "qwen" in key.lower():
            popped[key] = MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES.pop(key)
    return popped


def _restore_qwen_vision_mappings(popped: dict[str, str]) -> None:
    from transformers.models.auto.modeling_auto import (
        MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES,
    )

    MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES.update(popped)


def _make_sft_config(
    sft_config_cls: type, args: argparse.Namespace, output_dir: Path
) -> Any:
    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "max_steps": args.max_steps,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "warmup_steps": args.warmup_steps,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "lr_scheduler_type": "cosine",
        "optim": "adamw_8bit",
        "weight_decay": args.weight_decay,
        "bf16": args.bf16,
        "fp16": False,
        "report_to": args.report_to,
        "remove_unused_columns": False,
    }

    params = inspect.signature(sft_config_cls.__init__).parameters
    if "max_length" in params:
        kwargs["max_length"] = args.max_seq_length
    elif "max_seq_length" in params:
        kwargs["max_seq_length"] = args.max_seq_length

    if "assistant_only_loss" in params:
        # We pre-tokenize HCAPO chats before constructing SFTTrainer so Unsloth
        # skips its formatting_func path. At that point the dataset is no longer
        # "conversational" to TRL/Unsloth, so assistant_only_loss=True would be
        # rejected. Assistant-only labels are still enforced by assistant_masks
        # in the custom HCAPO data collator.
        kwargs["assistant_only_loss"] = False
    else:
        raise ValueError("Installed TRL SFTConfig does not support assistant_only_loss")

    if "run_name" in params and args.run_name:
        kwargs["run_name"] = args.run_name

    return sft_config_cls(**kwargs)


def _make_trainer(
    trainer_cls: type,
    model: Any,
    sft_args: Any,
    dataset: Any,
    raw_tokenizer: Any,
    data_collator: Any,
) -> Any:
    kwargs: dict[str, Any] = {
        "model": model,
        "args": sft_args,
        "train_dataset": dataset,
        "data_collator": data_collator,
    }
    params = inspect.signature(trainer_cls.__init__).parameters
    if "processing_class" in params:
        kwargs["processing_class"] = raw_tokenizer
    elif "tokenizer" in params:
        kwargs["tokenizer"] = raw_tokenizer
    return trainer_cls(**kwargs)


def _validate_tokenized_loss_masks(dataset: Any) -> None:
    column_names = set(getattr(dataset, "column_names", []) or [])
    if "assistant_masks" in column_names:
        total = len(dataset)
        zero_rows = sum(
            1 for row in dataset if not any(row.get("assistant_masks") or [])
        )
        if zero_rows == total:
            raise ValueError(
                "All examples have empty assistant masks - nothing trainable"
            )
        if zero_rows:
            logger.warning(
                "%d/%d examples have empty assistant masks", zero_rows, total
            )
        else:
            logger.info("Validated: all %d examples have assistant masks", total)
        return

    if "labels" not in column_names:
        logger.warning("No labels column to validate")
        return
    total = len(dataset)
    zero_rows = sum(
        1 for row in dataset if not any(l != -100 for l in (row.get("labels") or []))
    )
    if zero_rows == total:
        raise ValueError("All examples have fully masked labels — nothing trainable")
    if zero_rows:
        logger.warning("%d/%d examples have fully masked labels", zero_rows, total)
    else:
        logger.info("Validated: all %d examples have trainable tokens", total)


# CLI


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train HCAPO step-weighted SFT with Unsloth + TRL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Smoke test
  uv run python scripts/train_hcapo.py --config training/hcapo_config.json --max-steps 1

  # Full run
  uv run python scripts/train_hcapo.py --config training/hcapo_config.json
""",
    )
    p.add_argument("--config", default=None, help="JSON config file with CLI defaults")
    p.add_argument("--dataset", default="datasets/hcapo_train.jsonl")
    p.add_argument("--dataset-id", default=None, help="HF dataset repo containing hcapo_train.jsonl")
    p.add_argument("--dataset-filename", default="hcapo_train.jsonl")
    p.add_argument("--output-dir", default="outputs/hcapo")
    p.add_argument("--model-name", default="Qwen/Qwen3.5-4B")
    p.add_argument("--max-seq-length", type=int, default=16384)
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--num-proc", type=int, default=1)
    p.add_argument("--prepare-dataset-only", action="store_true")
    p.add_argument("--report-to", nargs="+", default=[])
    p.add_argument("--run-name", default=None)
    p.add_argument("--trackio-space", default=None)
    p.add_argument("--trackio-project", default=None)

    g = p.add_argument_group("LoRA")
    g.add_argument("--lora-r", type=int, default=32)
    g.add_argument("--lora-alpha", type=int, default=32)
    g.add_argument("--lora-dropout", type=float, default=0.0)
    g.add_argument(
        "--target-modules",
        nargs="+",
        default=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    g = p.add_argument_group("Optimisation")
    g.add_argument("--learning-rate", type=float, default=5e-6)
    g.add_argument("--weight-decay", type=float, default=0.01)
    g.add_argument("--num-train-epochs", type=float, default=1.0)
    g.add_argument("--max-steps", type=int, default=-1)
    g.add_argument("--per-device-train-batch-size", type=int, default=1)
    g.add_argument("--gradient-accumulation-steps", type=int, default=8)
    g.add_argument("--warmup-steps", type=int, default=5)
    g.add_argument("--logging-steps", type=int, default=1)
    g.add_argument("--save-steps", type=int, default=100)
    g.add_argument("--save-total-limit", type=int, default=2)

    g = p.add_argument_group("Export")
    g.add_argument("--save-merged-16bit", action="store_true")
    g.add_argument("--merged-output-dir", default="outputs/hcapo_merged_16bit")
    g.add_argument("--push-to-hub", action="store_true")
    g.add_argument("--output-repo", default=None, help="HF model repo for adapter upload")
    g.add_argument("--hub-private", action="store_true")

    return p


def _load_config_defaults(config_path: str | None) -> dict[str, Any]:
    if not config_path:
        return {}
    cfg = json.loads(Path(config_path).read_text())
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a JSON object: {config_path}")
    return cfg


def _parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    pre_args, _ = pre.parse_known_args()
    parser = _build_arg_parser()
    defaults = _load_config_defaults(pre_args.config)
    if defaults:
        parser.set_defaults(**defaults)
    return parser.parse_args()


# Main


def main() -> None:
    args = _parse_args()

    if args.prepare_dataset_only:
        ds = _load_and_prepare_dataset(args)
        logger.info("Dataset preparation complete: %d examples", len(ds))
        return

    import unsloth  # noqa: F401
    from unsloth import FastLanguageModel, is_bfloat16_supported

    import torch
    from trl import SFTConfig, SFTTrainer
    from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

    if not is_bfloat16_supported():
        raise ValueError("bf16 is required but not supported on this hardware")
    args.bf16 = True

    _seed_everything(args.seed, torch)
    if args.config:
        logger.info("Config: %s", args.config)
    if args.trackio_space:
        os.environ["TRACKIO_SPACE_ID"] = args.trackio_space
    if args.trackio_project:
        os.environ["TRACKIO_PROJECT_NAME"] = args.trackio_project

    dataset = _load_and_prepare_dataset(args)

    logger.info("Loading model: %s", args.model_name)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        max_seq_length=args.max_seq_length,
        use_rslora=False,
        loftq_config=None,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sft_args = _make_sft_config(SFTConfig, args, output_dir)
    logger.info(
        "HCAPO training: max_seq_length=%d, assistant masks handled by HCAPO collator",
        args.max_seq_length,
    )

    popped_vision = _remove_qwen_vision_mappings()
    if popped_vision:
        logger.info(
            "Removed vision mappings for text-only training: %s", list(popped_vision)
        )

    raw_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    dataset = _tokenize_hcapo_dataset(dataset, raw_tokenizer, args)
    trainer_cls = _build_hcapo_trainer_cls(SFTTrainer)
    data_collator = _build_hcapo_data_collator(
        processing_class=raw_tokenizer,
        sft_args=sft_args,
        data_collator_cls=DataCollatorForLanguageModeling,
    )

    logger.info("Initialising HCAPO trainer with %d examples...", len(dataset))
    try:
        trainer = _make_trainer(
            trainer_cls=trainer_cls,
            model=model,
            sft_args=sft_args,
            dataset=dataset,
            raw_tokenizer=raw_tokenizer,
            data_collator=data_collator,
        )
        # Unsloth replaces the collator for pre-tokenized datasets during
        # initialization; restore the HCAPO collator so step weights are used.
        trainer.data_collator = data_collator
    finally:
        _restore_qwen_vision_mappings(popped_vision)

    _validate_tokenized_loss_masks(trainer.train_dataset)

    train_result = trainer.train()
    logger.info("Training finished: %s", train_result.metrics)

    logger.info("Saving adapter → %s", output_dir)
    trainer.save_model(str(output_dir))
    raw_tokenizer.save_pretrained(str(output_dir))

    (output_dir / "train_metrics.json").write_text(
        json.dumps(train_result.metrics, indent=2)
    )
    (output_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2))
    (output_dir / "sft_config.json").write_text(
        json.dumps(sft_args.to_dict(), indent=2, default=str)
    )

    if args.save_merged_16bit:
        merged_dir = Path(args.merged_output_dir)
        merged_dir.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Saving merged 16-bit → %s", merged_dir)
        model.save_pretrained_merged(
            str(merged_dir), tokenizer, save_method="merged_16bit"
        )

    if args.push_to_hub:
        if not args.output_repo:
            raise ValueError("--push-to-hub requires --output-repo")
        from huggingface_hub import HfApi, create_repo

        logger.info("Uploading adapter output to https://huggingface.co/%s", args.output_repo)
        create_repo(
            args.output_repo,
            repo_type="model",
            private=args.hub_private,
            exist_ok=True,
        )
        HfApi().upload_folder(
            folder_path=str(output_dir),
            repo_id=args.output_repo,
            repo_type="model",
            commit_message="Upload HCAPO adapter",
        )

    logger.info("Done")


if __name__ == "__main__":
    main()
