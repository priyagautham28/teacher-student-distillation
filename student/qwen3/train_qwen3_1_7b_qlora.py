from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
from trl import SFTConfig, SFTTrainer


MODEL_ID = "Qwen/Qwen3-1.7B"
SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QLoRA distillation of Qwen3-1.7B on teacher-generated GSM8K SFT data."
    )
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--val-file", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/qwen3_1_7b_gsm8k_qlora"),
    )
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Use a small positive value such as 20 for a smoke test; -1 runs all epochs.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help='Checkpoint path, or "latest" to resume from the newest checkpoint.',
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc

            messages = row.get("messages")
            if (
                not isinstance(messages, list)
                or len(messages) < 2
                or messages[-1].get("role") != "assistant"
            ):
                raise ValueError(
                    f"Line {line_number} of {path} does not contain a valid "
                    "conversation ending in an assistant message."
                )

            # Conversational prompt-completion format:
            # loss is computed only on the teacher completion.
            rows.append(
                {
                    "problem_id": row.get("problem_id"),
                    "prompt": messages[:-1],
                    "completion": [messages[-1]],
                }
            )

    if not rows:
        raise ValueError(f"No usable records found in {path}")
    return rows


def newest_checkpoint(output_dir: Path) -> str | None:
    checkpoints = []
    for path in output_dir.glob("checkpoint-*"):
        try:
            step = int(path.name.split("-")[-1])
        except ValueError:
            continue
        checkpoints.append((step, path))
    return str(max(checkpoints)[1]) if checkpoints else None


def main() -> None:
    args = parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    random.seed(SEED)
    np.random.seed(SEED)
    set_seed(SEED)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this QLoRA training script.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = load_jsonl(args.train_file)
    val_rows = load_jsonl(args.val_file)
    train_dataset = Dataset.from_list(train_rows)
    val_dataset = Dataset.from_list(val_rows)

    print(f"Train examples: {len(train_dataset)}")
    print(f"Validation examples: {len(val_dataset)}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    compute_dtype = (
        torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    )
    print(f"4-bit compute dtype: {compute_dtype}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config,
        device_map={"": 0},
        torch_dtype=compute_dtype,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )

    training_args = SFTConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=compute_dtype == torch.bfloat16,
        fp16=compute_dtype == torch.float16,
        max_length=args.max_length,
        completion_only_loss=True,
        packing=False,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=3,
        logging_strategy="steps",
        logging_steps=5,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        weight_decay=0.0,
        optim="adamw_torch",
        report_to="none",
        seed=SEED,
        data_seed=SEED,
        remove_unused_columns=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    resume: str | bool | None = None
    if args.resume_from_checkpoint == "latest":
        resume = newest_checkpoint(args.output_dir)
        if resume is None:
            print("No checkpoint found; starting from the beginning.")
    elif args.resume_from_checkpoint:
        resume = args.resume_from_checkpoint

    train_result = trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    manifest = {
        "base_model": MODEL_ID,
        "train_file": str(args.train_file),
        "val_file": str(args.val_file),
        "train_examples": len(train_dataset),
        "val_examples": len(val_dataset),
        "seed": SEED,
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": "all-linear",
        "quantization": "4-bit NF4 double quantization",
        "compute_dtype": str(compute_dtype),
    }
    with (args.output_dir / "training_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")

    print(f"\nTraining complete. Adapter and tokenizer saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
