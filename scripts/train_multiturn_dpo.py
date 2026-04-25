#!/usr/bin/env python3
"""Train multi-turn DPO with Unsloth + Hugging Face TRL.

This script expects a dataset produced by `scripts/build_training_dataset.py`
in DPO format:
  {
    "prompt": [... multi-turn messages ...],
    "chosen": [... assistant/tool continuation ...],
    "rejected": [... assistant/tool continuation ...]
  }

The script converts message arrays into text columns (`prompt`, `chosen`,
`rejected`) and then trains with TRL's DPOTrainer on an Unsloth LoRA model.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any

import unsloth  # Must be imported before trl/transformers for all optimizations.
from unsloth import FastLanguageModel, is_bfloat16_supported

try:
    from unsloth import PatchDPOTrainer
    PatchDPOTrainer()
except ImportError:
    pass

import torch
from datasets import Dataset, load_dataset
from trl import DPOConfig, DPOTrainer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_multiturn_dpo")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _normalize_dpo_field(value: Any) -> Any:
    """Keep conversational fields structured so TRL can apply chat templates."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        if all(isinstance(item, dict) for item in value):
            return value
        return "\n\n".join(str(item) for item in value)
    return str(value or "")


def _has_nonempty_content(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return len(value) > 0
    return bool(str(value or "").strip())


def _normalize_dpo_example(example: dict[str, Any]) -> dict[str, Any]:
    """Normalize prompt/chosen/rejected while preserving conversational schema."""
    return {
        "prompt": _normalize_dpo_field(example.get("prompt")),
        "chosen": _normalize_dpo_field(example.get("chosen")),
        "rejected": _normalize_dpo_field(example.get("rejected")),
    }


def _load_and_prepare_dataset(path: str, num_proc: int) -> Dataset:
    logger.info("Loading DPO dataset from %s", path)
    ds = load_dataset("json", data_files=path, split="train")
    logger.info("Loaded %d raw rows", len(ds))

    if len(ds) == 0:
        raise ValueError("Dataset is empty")

    ds = ds.map(_normalize_dpo_example, num_proc=num_proc)
    keep_cols = {"prompt", "chosen", "rejected"}
    drop_cols = [col for col in ds.column_names if col not in keep_cols]
    if drop_cols:
        ds = ds.remove_columns(drop_cols)
    ds = ds.filter(
        lambda row: (
            _has_nonempty_content(row.get("prompt"))
            and _has_nonempty_content(row.get("chosen"))
            and _has_nonempty_content(row.get("rejected"))
        ),
        num_proc=num_proc,
    )
    logger.info("Prepared %d rows after filtering", len(ds))

    if len(ds) == 0:
        raise ValueError("No usable rows after preprocessing")

    return ds


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train multi-turn DPO using Unsloth + TRL")
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Optional JSON config file containing any CLI argument names "
            "(e.g. max_seq_length, model_name, output_dir). CLI flags override config values."
        ),
    )
    parser.add_argument("--dataset", default="datasets/dpo_train.jsonl", help="Path to DPO JSONL")
    parser.add_argument("--output-dir", default="outputs/dpo_unsloth", help="Output directory")
    parser.add_argument("--model-name", default="Qwen/Qwen3.6-27B", help="Base model name/path")
    parser.add_argument("--max-seq-length", type=int, default=16384, help="Training max sequence length")
    parser.add_argument(
        "--max-prompt-length",
        type=int,
        default=None,
        help=(
            "Max prompt tokens for DPO packing. "
            "Defaults to 75%% of --max-seq-length when unset."
        ),
    )
    parser.add_argument("--load-in-4bit", action="store_true", help="Load base model in 4-bit")
    parser.add_argument(
        "--allow-qwen-4bit",
        action="store_true",
        help=(
            "Override safety check and allow --load-in-4bit for Qwen3.6 models. "
            "Not recommended due to higher quantization error."
        ),
    )
    parser.add_argument("--bf16", action="store_true", help="Enable bf16 training")
    parser.add_argument("--fp16", action="store_true", help="Enable fp16 training")
    parser.add_argument("--seed", type=int, default=3407, help="Random seed")
    parser.add_argument("--num-proc", type=int, default=1, help="Dataset map/filter workers")

    # LoRA / PEFT
    parser.add_argument("--lora-r", type=int, default=64, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=64, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.0, help="LoRA dropout")
    parser.add_argument(
        "--target-modules",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "out_proj"],
        help="LoRA target modules",
    )

    # DPO / optimization
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta")
    parser.add_argument("--learning-rate", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--num-train-epochs", type=float, default=1.0, help="Training epochs")
    parser.add_argument("--max-steps", type=int, default=-1, help="Max update steps (-1 disables)")
    parser.add_argument("--per-device-train-batch-size", type=int, default=1, help="Per-device batch size")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8, help="Grad accumulation")
    parser.add_argument("--warmup-steps", type=int, default=5, help="Warmup steps")
    parser.add_argument("--logging-steps", type=int, default=10, help="Logging frequency")
    parser.add_argument("--save-steps", type=int, default=100, help="Checkpoint frequency")
    parser.add_argument("--save-total-limit", type=int, default=3, help="Max checkpoints to keep")

    # Export
    parser.add_argument("--save-merged-16bit", action="store_true", help="Export merged 16-bit model")
    parser.add_argument(
        "--merged-output-dir",
        default="outputs/dpo_unsloth_merged_16bit",
        help="Directory for merged 16-bit output",
    )

    return parser


def _load_config_defaults(config_path: str | None) -> dict[str, Any]:
    """Load JSON defaults for argparse from a config file."""
    if not config_path:
        return {}
    config = json.loads(Path(config_path).read_text())
    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a JSON object: {config_path}")
    return config


def _parse_args() -> argparse.Namespace:
    """Parse arguments with optional config defaults.

    We parse --config first, set parser defaults from JSON, then parse all args so
    direct CLI flags take precedence over config values.
    """
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=None)
    pre_args, _ = pre_parser.parse_known_args()

    parser = _build_arg_parser()
    config_defaults = _load_config_defaults(pre_args.config)
    if config_defaults:
        parser.set_defaults(**config_defaults)

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.bf16 and args.fp16:
        raise ValueError("Choose only one of --bf16 or --fp16")
    if not args.bf16 and not args.fp16:
        args.bf16 = is_bfloat16_supported()
        args.fp16 = not args.bf16
        logger.info("Auto-detected precision: bf16=%s fp16=%s", args.bf16, args.fp16)

    model_name_lower = args.model_name.lower()
    is_qwen36 = "qwen3.6" in model_name_lower or "qwen-3.6" in model_name_lower
    if args.load_in_4bit and is_qwen36 and not args.allow_qwen_4bit:
        raise ValueError(
            "QLoRA/4-bit is not recommended for Qwen3.6 due to higher quantization "
            "differences. Remove --load-in-4bit for bf16 LoRA, or pass "
            "--allow-qwen-4bit to override."
        )

    _seed_everything(args.seed)
    if args.config:
        logger.info("Loaded run config from %s", args.config)

    dataset = _load_and_prepare_dataset(args.dataset, num_proc=args.num_proc)

    if args.max_prompt_length is None:
        max_prompt_length = max(1024, int(args.max_seq_length * 0.75))
    else:
        max_prompt_length = args.max_prompt_length

    if max_prompt_length >= args.max_seq_length:
        raise ValueError("--max-prompt-length must be smaller than --max-seq-length")

    logger.info("Loading base model %s", args.model_name)
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

    dpo_args = DPOConfig(
        output_dir=str(output_dir),
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        weight_decay=args.weight_decay,
        bf16=args.bf16,
        fp16=args.fp16,
        max_length=args.max_seq_length,
        max_prompt_length=max_prompt_length,
        beta=args.beta,
        report_to=[],
        remove_unused_columns=False,
    )

    logger.info(
        "Effective DPO lengths: max_length=%d, max_prompt_length=%d",
        args.max_seq_length,
        max_prompt_length,
    )

    # Qwen3.5/3.6 is a unified VLM so Unsloth/TRL auto-detect it as a vision
    # model and route through process_row which expects pixel_values.  For
    # text-only DPO we must remove the model from the vision mapping *before*
    # DPOTrainer.__init__ runs _prepare_dataset.
    from transformers.models.auto.modeling_auto import MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES
    _popped_vision_keys: dict[str, str] = {}
    for key in list(MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES.keys()):
        if "qwen" in key.lower():
            _popped_vision_keys[key] = MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES.pop(key)
    if _popped_vision_keys:
        logger.info("Removed vision mapping keys for text-only DPO: %s", list(_popped_vision_keys))

    # For Qwen3.5/3.6 unified VLMs, FastLanguageModel returns a processor
    # (Qwen3VLProcessor) not a plain tokenizer.  DPOTrainer must receive the
    # raw tokenizer so it doesn't try to run image processing on text.
    raw_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)

    logger.info("Starting DPO training with %d samples", len(dataset))
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_args,
        train_dataset=dataset,
        processing_class=raw_tokenizer,
    )

    # Restore vision mapping so downstream code isn't affected.
    MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES.update(_popped_vision_keys)
    train_result = trainer.train()
    logger.info("Training finished: %s", train_result.metrics)

    logger.info("Saving LoRA adapter and tokenizer to %s", output_dir)
    trainer.save_model(str(output_dir))
    raw_tokenizer.save_pretrained(str(output_dir))

    metrics_path = output_dir / "train_metrics.json"
    metrics_path.write_text(json.dumps(train_result.metrics, indent=2))
    config_path = output_dir / "run_config.json"
    config_path.write_text(json.dumps(vars(args), indent=2))
    dpo_cfg_path = output_dir / "dpo_config.json"
    dpo_cfg_path.write_text(json.dumps(dpo_args.to_dict(), indent=2, default=str))

    if args.save_merged_16bit:
        merged_dir = Path(args.merged_output_dir)
        merged_dir.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Saving merged 16-bit model to %s", merged_dir)
        model.save_pretrained_merged(
            str(merged_dir),
            tokenizer,
            save_method="merged_16bit",
        )

    logger.info("Done")


if __name__ == "__main__":
    main()
