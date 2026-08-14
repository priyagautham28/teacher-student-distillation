"""
QLoRA training script for the Llama-3.2-1B student track.
Logs every run to MLflow: hyperparameters, per-epoch metrics, and the
final trained adapter as an artifact, so every config change and its
result is tracked and comparable later.

Expects teacher SFT JSONL from cot_faithfulness (fields: question,
teacher_final_answer / gold_answer, student_target, messages).

Usage:
    python train_v3.py --variant answer_only \\
      --train_file ../cot_faithfulness/data_final_v2/teacher_gsm8k_train_*_full_sft.jsonl \\
      --val_file ../cot_faithfulness/data_final_v2/teacher_gsm8k_val_*_full_sft.jsonl

    python train_v3.py --variant reasoning \\
      --train_file data/train_sft.jsonl --val_file data/val_sft.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from functools import partial

import mlflow
import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    TrainerCallback,
)
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
MLFLOW_EXPERIMENT = "llama-3.2-1b-distillation"

# Must match evaluate.py SYSTEM_PROMPT / USER_TEMPLATE exactly. Teacher JSONL
# messages often use a shorter system text and a raw question; we rebuild
# prompts here so train + gen-eval + shared eval see the same distribution.
STUDENT_SYSTEM_PROMPT = (
    "Solve the mathematical problem using step-by-step reasoning, with at most "
    "one arithmetic operation per step. Return the reasoning inside <reasoning> "
    "tags and the numerical answer inside <final_answer> tags. The "
    "<final_answer> tag must contain only the normalized numerical answer "
    "without units, currency symbols, commas, percent signs, or explanatory "
    "text."
)
USER_TEMPLATE = "Problem:\n{question}"
ANSWER_ONLY_SYSTEM_PROMPT = (
    "Solve the mathematical problem. "
    "Return only the numerical answer inside <final_answer> tags."
)

_REASONING_RE = re.compile(
    r"<reasoning>(.*?)</reasoning>", flags=re.DOTALL | re.IGNORECASE
)
_ANSWER_RE = re.compile(
    r"<final_answer>\s*(.*?)\s*</final_answer>", flags=re.DOTALL | re.IGNORECASE
)


def _assistant_target(example: dict) -> str:
    target = example.get("student_target") or ""
    if target:
        return target
    for message in reversed(example.get("messages") or []):
        if message.get("role") == "assistant" and message.get("content"):
            return message["content"]
    return ""


def extract_fields(example: dict) -> tuple[str, str, str]:
    """Map teacher SFT / legacy flat rows to (question, reasoning, answer)."""
    question = (example.get("question") or "").strip()
    target = _assistant_target(example)

    reasoning = ""
    answer = (
        example.get("teacher_final_answer")
        or example.get("gold_answer")
        or example.get("answer")
        or ""
    )

    if target:
        match = _REASONING_RE.search(target)
        if match:
            reasoning = match.group(1).strip()
        match = _ANSWER_RE.search(target)
        if match:
            answer = match.group(1).strip()

    if not reasoning:
        reasoning = (
            example.get("reasoning") or example.get("teacher_reasoning") or ""
        ).strip()
    if not answer:
        answer = (example.get("answer") or "").strip()

    return question, reasoning, str(answer).strip()


def gold_of(example: dict) -> str:
    """Grading target for the generation eval: GSM8K gold when available, so
    accuracy is measured against the dataset rather than against the teacher.
    """
    gold = example.get("gold_answer")
    if gold in (None, ""):
        gold = extract_fields(example)[2]
    return str(gold).strip()


def build_messages(example: dict, variant: str) -> list[dict[str, str]]:
    """Build chat messages for answer-only or full tagged reasoning SFT.

    Always rebuild system/user to match evaluate.py. Keep the teacher
    assistant target (student_target / messages assistant) for supervision.
    """
    question, reasoning, answer = extract_fields(example)
    user_content = USER_TEMPLATE.format(question=question)

    if variant == "answer_only":
        return [
            {"role": "system", "content": ANSWER_ONLY_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {
                "role": "assistant",
                "content": f"<final_answer>{answer}</final_answer>",
            },
        ]

    target = _assistant_target(example)
    if not target:
        target = (
            f"<reasoning>\n{reasoning}\n</reasoning>\n"
            f"<final_answer>{answer}</final_answer>"
        )
    return [
        {"role": "system", "content": STUDENT_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": target},
    ]


def format_example(example: dict, variant: str, tokenizer) -> dict[str, str]:
    messages = build_messages(example, variant)
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def build_model_and_tokenizer(lora_r, lora_alpha, lora_dropout):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
    )

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        # Attention (q/k/v/o) AND the MLP/feed-forward projections. Attention-only
        # LoRA is the common default but leaves the SwiGLU feed-forward block --
        # which does a large share of the actual computation -- untouched.
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()  # required for gradient checkpointing + PEFT
    model.print_trainable_parameters()  # sanity check: confirms LoRA is only training a small %
    return model, tokenizer


def get_response_template(tokenizer) -> str:
    """Return the literal string the chat template inserts right before the
    assistant's turn, so DataCollatorForCompletionOnlyLM can mask loss on
    everything before it (system prompt + question). Derived from the
    tokenizer at runtime rather than hardcoded, so this doesn't silently
    break if the chat template ever changes.
    """
    probe = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "x"},
            {"role": "user", "content": "y"},
            {"role": "assistant", "content": "z"},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )
    # Llama 3.x chat templates mark each turn with
    # "<|start_header_id|>ROLE<|end_header_id|>\n\n" -- take the assistant
    # header itself, which is the standard response_template shape TRL expects.
    match = re.search(r"(<\|start_header_id\|>assistant<\|end_header_id\|>\n\n)", probe)
    if not match:
        raise ValueError(
            "Could not locate an assistant header in the chat template output: "
            f"{probe!r}. Inspect this string and set response_template manually."
        )
    return match.group(1)


def score_generation(text: str, gold: str, variant: str) -> tuple[bool, bool]:
    """Return (answer_correct, format_valid) for one generated completion."""
    has_final_tag = "<final_answer>" in text and "</final_answer>" in text
    has_reasoning_tag = variant == "answer_only" or (
        "<reasoning>" in text and "</reasoning>" in text
    )
    format_valid = has_final_tag and has_reasoning_tag

    match = _ANSWER_RE.search(text)
    if not match:
        return False, format_valid
    predicted = match.group(1).strip().replace(",", "").replace("$", "")
    try:
        correct = abs(float(predicted) - float(gold.replace(",", ""))) < 1e-6
    except ValueError:
        correct = False
    return correct, format_valid


class GenerationEvalCallback(TrainerCallback):
    """Generates on a fixed validation sample at every evaluation and injects
    exact-match and tag-format validity into the Trainer's metrics dict.

    Writing into that dict is the whole point: eval_loss going down does not
    guarantee the model closes its tags or gets the right answer, and unless
    these numbers reach the Trainer they cannot drive best-checkpoint restore
    or early stopping.
    """

    def __init__(
        self,
        val_raw_dataset,
        tokenizer,
        variant,
        num_samples=100,
        batch_size=16,
        max_new_tokens=512,
        seed=42,
    ):
        rng = random.Random(seed)
        indices = rng.sample(
            range(len(val_raw_dataset)), min(num_samples, len(val_raw_dataset))
        )
        samples = [val_raw_dataset[i] for i in indices]
        self.tokenizer = tokenizer
        self.variant = variant
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.system_prompt = (
            ANSWER_ONLY_SYSTEM_PROMPT if variant == "answer_only" else STUDENT_SYSTEM_PROMPT
        )
        self.questions = [extract_fields(e)[0] for e in samples]
        self.golds = [gold_of(e) for e in samples]

    def _prompt(self, question: str) -> str:
        return self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": USER_TEMPLATE.format(question=question)},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

    def _generate(self, model) -> list[str]:
        tok = self.tokenizer
        prompts = [self._prompt(q) for q in self.questions]
        # Decoder-only generation needs left padding, otherwise the batched
        # continuations start after a run of pad tokens.
        old_side, tok.padding_side = tok.padding_side, "left"
        texts = []
        try:
            for i in range(0, len(prompts), self.batch_size):
                batch = tok(
                    prompts[i:i + self.batch_size],
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,  # the chat template already adds BOS
                ).to(model.device)
                with torch.inference_mode():
                    output = model.generate(
                        **batch,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                        use_cache=True,  # gradient checkpointing sets config.use_cache=False
                        pad_token_id=tok.pad_token_id,
                    )
                texts += tok.batch_decode(
                    output[:, batch["input_ids"].shape[-1]:], skip_special_tokens=True
                )
        finally:
            tok.padding_side = old_side
        return texts

    def on_evaluate(self, args, state, control, metrics=None, model=None, **kwargs):
        if model is None:
            return
        model.eval()
        texts = self._generate(model)
        model.train()

        scored = [
            score_generation(t, g, self.variant) for t, g in zip(texts, self.golds)
        ]
        n = len(scored) or 1
        exact_match = sum(c for c, _ in scored) / n
        format_valid = sum(f for _, f in scored) / n

        print(
            f"[epoch {state.epoch:.2f}] gen_eval exact_match={exact_match:.2%} "
            f"format_valid={format_valid:.2%} (n={n})"
        )
        # Mutating this dict is what makes the metric usable for model
        # selection: Trainer.evaluate hands the same object to the callbacks
        # and then returns it to _determine_best_metric.
        if metrics is not None:
            metrics["eval_gen_exact_match"] = exact_match
            metrics["eval_gen_format_valid_rate"] = format_valid
        mlflow.log_metric("gen_eval_exact_match", exact_match, step=state.global_step)
        mlflow.log_metric("gen_eval_format_valid_rate", format_valid, step=state.global_step)


def run_training(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name=f"{args.variant}_lr{args.lr}_r{args.r}"):
        mlflow.log_params(
            {
                "variant": args.variant,
                "learning_rate": args.lr,
                "lora_r": args.r,
                "lora_alpha": args.alpha,
                "lora_dropout": args.dropout,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "grad_accum_steps": args.grad_accum,
                "max_seq_length": args.max_seq_length,
                "neftune_alpha": args.neftune_alpha,
                "early_stopping_patience": args.early_stopping_patience,
                "eval_samples_for_generation": args.eval_samples_for_generation,
                "metric_for_best_model": args.metric_for_best_model,
                "seed": args.seed,
                "train_file": args.train_file,
                "val_file": args.val_file,
                "model_name": MODEL_NAME,
            }
        )

        model, tokenizer = build_model_and_tokenizer(args.r, args.alpha, args.dropout)

        train_dataset = load_dataset("json", data_files=args.train_file, split="train")
        val_dataset = load_dataset("json", data_files=args.val_file, split="train")

        # Keep a reference to the raw (unformatted) validation rows before
        # .map() strips the original columns -- GenerationEvalCallback needs
        # question/gold_answer, not the flattened "text" field.
        raw_val_for_gen_eval = val_dataset

        formatter = partial(format_example, variant=args.variant, tokenizer=tokenizer)
        train_dataset = train_dataset.map(formatter, remove_columns=train_dataset.column_names)
        val_dataset = val_dataset.map(formatter, remove_columns=val_dataset.column_names)

        response_template = get_response_template(tokenizer)
        print(f"Using response_template: {response_template!r}")
        collator = DataCollatorForCompletionOnlyLM(
            response_template=response_template,
            tokenizer=tokenizer,
        )

        greater_is_better = args.metric_for_best_model != "eval_loss"
        training_args = SFTConfig(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model=args.metric_for_best_model,
            greater_is_better=greater_is_better,
            bf16=True,
            gradient_checkpointing=True,
            neftune_noise_alpha=args.neftune_alpha,
            seed=args.seed,
            max_seq_length=args.max_seq_length,
            dataset_text_field="text",
            packing=False,
            report_to=[],
        )

        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            args=training_args,
            data_collator=collator,
        )

        class MLflowCallback(TrainerCallback):
            def on_log(self, args_, state, control, logs=None, **kwargs):
                if logs:
                    for k, v in logs.items():
                        if isinstance(v, (int, float)):
                            mlflow.log_metric(k, v, step=state.global_step)

        trainer.add_callback(MLflowCallback())
        # Order matters: EarlyStoppingCallback reads the metrics dict in its own
        # on_evaluate, so the generation eval has to populate it first.
        trainer.add_callback(
            GenerationEvalCallback(
                raw_val_for_gen_eval,
                tokenizer,
                args.variant,
                num_samples=args.eval_samples_for_generation,
                batch_size=args.gen_eval_batch_size,
                max_new_tokens=args.gen_eval_max_new_tokens,
                seed=args.seed,
            )
        )
        trainer.add_callback(
            EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)
        )

        print(
            f"\nStarting training: variant={args.variant}, "
            f"lr={args.lr}, r={args.r}, "
            f"train_rows={len(train_dataset)}, val_rows={len(val_dataset)}, "
            f"selecting on {args.metric_for_best_model}"
        )
        trainer.train()

        adapter_dir = os.path.join(args.output_dir, "final_adapter")
        trainer.save_model(adapter_dir)
        mlflow.log_artifacts(adapter_dir, artifact_path="adapter")

        # Runs after best-checkpoint restore, so this also re-scores the
        # generation eval on the checkpoint actually being kept.
        final_metrics = trainer.evaluate()
        mlflow.log_metrics(
            {
                f"final_{k}": v
                for k, v in final_metrics.items()
                if isinstance(v, (int, float))
            }
        )

        print(f"\nDone. Adapter saved to {adapter_dir} and logged to MLflow run.")
        print(f"Final eval metrics: {json.dumps(final_metrics, indent=2, default=str)}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["answer_only", "reasoning"], required=True)
    p.add_argument("--train_file", required=True)
    p.add_argument("--val_file", required=True)
    p.add_argument("--output_dir", default="./llama-1b-output")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--r", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.05)
    # Selection is on exact-match now, so overshooting is recoverable: the best
    # checkpoint is restored rather than whichever epoch happened to be last.
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=4)
    # Teacher tagged CoTs are longer than answer-only; 1024 is safer than 512.
    # Measured max on the v4 teacher data is ~610 tokens, so nothing truncates.
    p.add_argument("--max_seq_length", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    # NEFTune: adds small random noise to input embeddings during training only.
    # Near-zero cost, reliably improves instruction-following quality.
    p.add_argument("--neftune_alpha", type=float, default=5.0)
    # Stops training if the selected metric hasn't improved for this many eval
    # rounds (paired with load_best_model_at_end so the best is restored).
    p.add_argument("--early_stopping_patience", type=int, default=2)
    # What to select checkpoints on. eval_gen_exact_match is the task metric;
    # eval_loss is available for comparison against older runs.
    p.add_argument(
        "--metric_for_best_model",
        default="eval_gen_exact_match",
        choices=["eval_gen_exact_match", "eval_gen_format_valid_rate", "eval_loss"],
    )
    # How many validation questions to generate on at each eval. At n=20 the
    # standard error is ~11 points, which is too noisy to select on; 100 puts it
    # near 5 and makes one sample worth 1 point.
    p.add_argument("--eval_samples_for_generation", type=int, default=100)
    p.add_argument("--gen_eval_batch_size", type=int, default=16)
    p.add_argument("--gen_eval_max_new_tokens", type=int, default=512)
    return p.parse_args()


if __name__ == "__main__":
    run_training(parse_args())
