# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "unsloth",
#     "unsloth_zoo",
#     "trl",
#     "datasets",
#     "torch>=2.10.0",
#     "bitsandbytes>=0.49.2",
#     "accelerate",
#     "peft",
#     "transformers>=5",
#     "huggingface_hub",
#     "trackio>=0.25.0",
# ]
# ///
"""Self-contained multi-turn DPO training script for HF Jobs / Spaces.

Designed to run via HF Jobs:
    hf jobs uv run scripts/train_dpo_hfjob.py --flavor a100-large --timeout 4h \
        --secrets HF_TOKEN --env HF_ENDPOINT=https://hf-mirror.com

Or inside an HF Space Docker container where config is passed via env vars
(DATASET_ID, MODEL_NAME, OUTPUT_REPO, MAX_SEQ_LENGTH, etc.).
CLI args always override env vars.

The script:
  1. Downloads the DPO dataset from a HF Hub dataset repo
  2. Converts message-list columns to text
  3. Trains bf16 LoRA DPO with Unsloth + TRL
  4. Pushes the trained adapter to HF Hub
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import unsloth
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
logger = logging.getLogger("train_dpo_hfjob")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _normalize_dpo_field(value: Any) -> Any:
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
    return {
        "prompt": _normalize_dpo_field(example.get("prompt")),
        "chosen": _normalize_dpo_field(example.get("chosen")),
        "rejected": _normalize_dpo_field(example.get("rejected")),
    }


def _load_and_prepare_dataset(dataset_id: str, split: str = "train") -> Dataset:
    logger.info("Loading dataset %s (split=%s) from Hub", dataset_id, split)
    ds = load_dataset(dataset_id, split=split)
    logger.info("Loaded %d raw rows", len(ds))

    ds = ds.map(_normalize_dpo_example, num_proc=1)
    keep = {"prompt", "chosen", "rejected"}
    drop = [c for c in ds.column_names if c not in keep]
    if drop:
        ds = ds.remove_columns(drop)
    ds = ds.filter(
        lambda row: (
            _has_nonempty_content(row.get("prompt"))
            and _has_nonempty_content(row.get("chosen"))
            and _has_nonempty_content(row.get("rejected"))
        ),
    )
    logger.info("Prepared %d rows after filtering", len(ds))
    return ds


def _env(name: str, default: str | None = None) -> str | None:
    """Read from environment, returning *default* if unset/empty."""
    val = os.environ.get(name, "")
    return val if val else default


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-id", default=_env("DATASET_ID"), help="HF Hub dataset repo")
    p.add_argument("--dataset-split", default=_env("DATASET_SPLIT", "train"))
    p.add_argument("--model-name", default=_env("MODEL_NAME", "Qwen/Qwen3.6-27B"))
    p.add_argument("--output-repo", default=_env("OUTPUT_REPO"), help="HF Hub repo to push adapter")
    p.add_argument("--output-dir", default=_env("OUTPUT_DIR", "/tmp/dpo_output"))
    p.add_argument("--max-seq-length", type=int, default=int(_env("MAX_SEQ_LENGTH", "16384")))
    p.add_argument("--max-prompt-length", type=int, default=None)
    p.add_argument("--lora-r", type=int, default=int(_env("LORA_R", "64")))
    p.add_argument("--lora-alpha", type=int, default=int(_env("LORA_ALPHA", "64")))
    p.add_argument("--beta", type=float, default=float(_env("BETA", "0.1")))
    p.add_argument("--learning-rate", type=float, default=float(_env("LEARNING_RATE", "5e-6")))
    p.add_argument("--num-train-epochs", type=float, default=float(_env("NUM_TRAIN_EPOCHS", "1.0")))
    p.add_argument("--per-device-train-batch-size", type=int, default=int(_env("PER_DEVICE_TRAIN_BATCH_SIZE", "1")))
    p.add_argument("--gradient-accumulation-steps", type=int, default=int(_env("GRADIENT_ACCUMULATION_STEPS", "8")))
    p.add_argument("--warmup-steps", type=int, default=int(_env("WARMUP_STEPS", "5")))
    p.add_argument("--logging-steps", type=int, default=int(_env("LOGGING_STEPS", "1")))
    p.add_argument("--save-merged-16bit", action="store_true",
                   default=_env("SAVE_MERGED_16BIT", "").lower() in ("1", "true"))
    p.add_argument("--trackio-space", default=_env("TRACKIO_SPACE"), help="HF Space ID for Trackio dashboard")
    p.add_argument("--run-name", default=_env("RUN_NAME", "dpo-qwen36-27b"), help="Trackio run name")
    p.add_argument("--seed", type=int, default=int(_env("SEED", "3407")))
    args = p.parse_args()

    if not args.dataset_id:
        p.error("--dataset-id is required (or set DATASET_ID env var)")

    _seed_everything(args.seed)

    if args.trackio_space:
        os.environ["TRACKIO_SPACE_ID"] = args.trackio_space
        logger.info("Trackio dashboard: https://huggingface.co/spaces/%s", args.trackio_space)

    dataset = _load_and_prepare_dataset(args.dataset_id, args.dataset_split)

    max_prompt_length = args.max_prompt_length or max(1024, int(args.max_seq_length * 0.75))

    logger.info("Loading model %s", args.model_name)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=False,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj", "out_proj"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        max_seq_length=args.max_seq_length,
        use_rslora=False,
        loftq_config=None,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    push_to_hub = bool(args.output_repo)

    dpo_args = DPOConfig(
        output_dir=str(output_dir),
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=9999,
        save_total_limit=1,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        weight_decay=0.01,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        max_length=args.max_seq_length,
        max_prompt_length=max_prompt_length,
        beta=args.beta,
        report_to=["trackio"] if args.trackio_space else [],
        run_name=args.run_name,
        remove_unused_columns=False,
        push_to_hub=push_to_hub,
        hub_model_id=args.output_repo if push_to_hub else None,
    )

    from transformers.models.auto.modeling_auto import MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES
    _popped: dict[str, str] = {}
    for key in list(MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES.keys()):
        if "qwen" in key.lower():
            _popped[key] = MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES.pop(key)
    if _popped:
        logger.info("Removed vision mapping keys for text-only DPO: %s", list(_popped))

    raw_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)

    logger.info("Starting DPO training — %d samples, max_length=%d", len(dataset), args.max_seq_length)
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_args,
        train_dataset=dataset,
        processing_class=raw_tokenizer,
    )

    MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES.update(_popped)

    result = trainer.train()
    logger.info("Training finished: %s", result.metrics)

    trainer.save_model(str(output_dir))
    raw_tokenizer.save_pretrained(str(output_dir))

    metrics_path = output_dir / "train_metrics.json"
    metrics_path.write_text(json.dumps(result.metrics, indent=2))

    if push_to_hub:
        logger.info("Pushing adapter to %s", args.output_repo)
        trainer.push_to_hub()

    if args.save_merged_16bit:
        merged_dir = output_dir / "merged_16bit"
        logger.info("Saving merged 16-bit model to %s", merged_dir)
        model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")
        if push_to_hub:
            from huggingface_hub import HfApi
            api = HfApi()
            api.upload_folder(
                folder_path=str(merged_dir),
                repo_id=f"{args.output_repo}-merged",
                repo_type="model",
                create_pr=False,
            )

    logger.info("Done")


if __name__ == "__main__":
    main()
