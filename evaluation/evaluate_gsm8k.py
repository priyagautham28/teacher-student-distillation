# -*- coding: utf-8 -*-
"""Evaluate Qwen/Llama/Gemma-style causal language models on the official GSM8K test set.

This evaluator is model-family-agnostic: any instruction-tuned Hugging Face
causal LM whose tokenizer exposes a chat template works via --model, including
Qwen, Llama, and Gemma checkpoints. --disable-qwen-thinking is the one
Qwen-specific option (it toggles Qwen3's hidden "thinking" mode); it is
silently ignored for model families that don't support it, such as Llama and
Gemma.

Purpose
-------
This script is a shared evaluator across the teacher/student pipeline:
1. Teacher (e.g. Qwen)
2. Base student before SFT (e.g. Gemma, Llama, or Qwen)
3. Student after QLoRA/SFT (same student family as above)

It does not train or update model weights. It generates one answer per GSM8K test
question, saves a detailed JSONL prediction record, and writes aggregate metrics.
Dataset (openai/gsm8k test split), prompt, and decoding settings (seed 42,
temperature 0.0, top_p 1.0) are fixed so every model is scored under identical
conditions.

Primary metric
--------------
- exact_match_accuracy

Main secondary metric
---------------------
- correct_and_valid_rate

Supported inference backends
----------------------------
- transformers: Hugging Face model or local checkpoint, optionally with a PEFT adapter
- openai: OpenAI-compatible endpoint such as a local vLLM server

Examples
--------
# Student smoke test, base model, 20 questions (Gemma shown below; same pattern for Qwen/Llama):
python evaluate_gsm8k.py --backend transformers --model google/gemma-3-1b-it \
  --stage before_sft --limit 20 
python evaluate_gsm8k.py --backend transformers --model meta-llama/Llama-3.2-1B-Instruct \
  --stage before_sft --limit 20 
python evaluate_gsm8k.py --backend transformers --model Qwen/Qwen3-1.7B \
  --stage before_sft --limit 20 

# Full QLoRA evaluation (same pattern for a Llama student):
python evaluate_gsm8k.py --backend transformers --model google/gemma-3-1b-it \
  --adapter-path outputs/gemma3_1b_gsm8k_qlora --stage after_sft 
python evaluate_gsm8k.py --backend transformers --model meta-llama/Llama-3.2-1B-Instruct \
  --adapter-path outputs/llama3_1b_gsm8k_qlora --stage after_sft 
python evaluate_gsm8k.py --backend transformers --model Qwen/Qwen3-1.7B \
  --adapter-path outputs/qwen3_1_7b_gsm8k_qlora --stage after_sft 

# Automatic before/after comparison in one run:
python evaluate_gsm8k.py --backend transformers --model google/gemma-3-1b-it \
  --adapter-path outputs/gemma3_1b_gsm8k_qlora --compare-base 
python evaluate_gsm8k.py --backend transformers --model meta-llama/Llama-3.2-1B-Instruct \
  --adapter-path outputs/llama3_1b_gsm8k_qlora --compare-base 
python evaluate_gsm8k.py --backend transformers --model Qwen/Qwen3-1.7B \
  --adapter-path outputs/qwen3_1_7b_gsm8k_qlora --compare-base 

# Qwen teacher served by vLLM/OpenAI-compatible API:
python evaluate_gsm8k.py --backend openai --model Qwen/Qwen3-14B-AWQ \
  --base-url http://localhost:8000/v1 --api-key EMPTY --stage teacher
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import statistics
import time
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Any, Optional, Protocol


# =============================================================================
# 1. FIXED EVALUATION SETTINGS
# =============================================================================
# Dataset, prompt, and decoding settings are fixed constants rather than CLI
# flags: every teacher/student run in this pipeline must see the same
# questions, the same instructions, and deterministic decoding for the
# comparison between conditions to be meaningful.

DATASET = "openai/gsm8k"
DATASET_CONFIG = "main"
# Pinned so every run loads the exact same snapshot even if the dataset repo
# is ever updated. Current as of 2026-03-23; check
# https://huggingface.co/api/datasets/openai/gsm8k for the latest sha if
# this ever needs bumping (GSM8K itself is a frozen benchmark, so it
# shouldn't need to).
DATASET_REVISION = "740312add88f781978c0658806c59bc2815b9866"
DATASET_SPLIT = "test"

SYSTEM_PROMPT = (
    "Solve the mathematical problem using concise step-by-step reasoning. "
    "Return the reasoning inside <reasoning> tags and the numerical answer "
    "inside <final_answer> tags. The <final_answer> tag must contain only the "
    "normalized numerical answer without units, currency symbols, commas, "
    "percent signs, or explanatory text."
)
USER_TEMPLATE = "Problem:\n{question}"

SEED = 42
TEMPERATURE = 0.0
TOP_P = 1.0

FORMAT_FAILURES = {
    "missing_reasoning_tag",
    "missing_final_answer_tag",
    "malformed_reasoning_tag",
    "malformed_final_answer_tag",
    "empty_reasoning",
    "non_numeric_final_answer",
    "ambiguous_final_answer",
    "truncated_output",
}

ALL_FAILURE_REASONS = (
    "wrong_answer",
    "missing_reasoning_tag",
    "missing_final_answer_tag",
    "malformed_reasoning_tag",
    "empty_reasoning",
    "malformed_final_answer_tag",
    "non_numeric_final_answer",
    "ambiguous_final_answer",
    "no_extractable_answer",
    "truncated_output",
    "generation_error",
)


# =============================================================================
# 2. CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class EvalConfig:
    backend: str
    model: str
    stage: str
    adapter_path: Optional[str]
    base_url: Optional[str]
    limit: Optional[int]
    max_new_tokens: int
    max_input_tokens: int
    load_in_4bit: bool
    dtype: str
    trust_remote_code: bool
    disable_qwen_thinking: bool  # Qwen-only; ignored for Gemma/Llama/other students.

    def config_hash(self) -> str:
        """Short hash of the settings that affect prediction content (not
        stage, limit, or transport details like base_url), so two runs that
        differ only in adapter, quantization, token limits, etc. never
        collide on the same predictions file and get misread as resumed."""
        payload = json.dumps(
            {
                "backend": self.backend,
                "model": self.model,
                "adapter_path": self.adapter_path,
                "max_new_tokens": self.max_new_tokens,
                "max_input_tokens": self.max_input_tokens,
                "load_in_4bit": self.load_in_4bit,
                "dtype": self.dtype,
                "trust_remote_code": self.trust_remote_code,
                "disable_qwen_thinking": self.disable_qwen_thinking,
                "dataset": DATASET,
                "dataset_config": DATASET_CONFIG,
                "dataset_split": DATASET_SPLIT,
                "dataset_revision": DATASET_REVISION,
                "system_prompt": SYSTEM_PROMPT,
                "user_template": USER_TEMPLATE,
                "seed": SEED,
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
            },
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:8]


@dataclass
class GenerationResult:
    text: str
    finish_reason: Optional[str]
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_tokens: Optional[int]
    elapsed_seconds: float
    peak_cuda_memory_bytes: Optional[int] = None


class GenerationBackend(Protocol):
    model_metadata: dict[str, Any]

    def generate(self, messages: list[dict[str, str]]) -> GenerationResult:
        ...


# =============================================================================
# 3. GENERAL HELPERS
# =============================================================================


def slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return value.strip("_") or "model"


def directory_size_gb(path: Optional[str]) -> Optional[float]:
    """Total size in GB of a file or directory tree, e.g. a saved PEFT
    adapter. Useful for demonstrating how much smaller a QLoRA adapter is
    than the full base-model checkpoint it modifies."""
    if not path:
        return None
    target = Path(path)
    if not target.exists():
        return None
    if target.is_file():
        return target.stat().st_size / (1024**3)
    return sum(f.stat().st_size for f in target.rglob("*") if f.is_file()) / (1024**3)


def _file_ends_without_newline(path: Path) -> bool:
    """True if path is non-empty and its last byte is not a newline."""
    if not path.exists() or path.stat().st_size == 0:
        return False
    with path.open("rb") as handle:
        handle.seek(-1, os.SEEK_END)
        return handle.read(1) != b"\n"


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append one record. "a" mode never inserts a newline on its own: if the
    file's last existing record is missing its trailing "\\n" (e.g. a crash
    that landed exactly on that byte), a plain append glues the new record
    onto the same line and corrupts both. Guard against that here.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_leading_newline = _file_ends_without_newline(path)
    with path.open("a", encoding="utf-8") as handle:
        if needs_leading_newline:
            handle.write("\n")
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL rows.

    A malformed *final* line is tolerated and dropped, since that is the
    expected shape of a process being killed mid-write (append_jsonl's
    fsync makes that the realistic failure mode, not a torn write earlier
    in the file). A malformed line anywhere else means the file is
    corrupted in a way that cannot be silently reconciled, so this raises.
    """
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8") as handle:
        lines = handle.readlines()

    nonblank_line_numbers = [
        line_number for line_number, line in enumerate(lines, start=1) if line.strip()
    ]
    last_nonblank_line_number = nonblank_line_numbers[-1] if nonblank_line_numbers else -1

    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            records.append(json.loads(stripped))
        except json.JSONDecodeError as exc:
            if line_number == last_nonblank_line_number:
                print(
                    f"Warning: dropping partially written final JSONL line "
                    f"{line_number} in {path}"
                )
            else:
                raise ValueError(
                    f"Invalid JSONL in {path} at line {line_number}: {exc}. "
                    "This is not the final line, so it cannot be safely "
                    "auto-repaired; restore from backup before continuing."
                ) from exc
    return records


def repair_trailing_jsonl_record(path: Path) -> bool:
    """Rewrite path if it has a partially written trailing line or is
    missing its trailing newline (both would otherwise corrupt the next
    append_jsonl() call by gluing onto it). Returns True if a repair was
    made. Raises ValueError (via read_jsonl) if corruption is found before
    the final line, since that cannot be safely auto-repaired.
    """
    if not path.exists():
        return False

    with path.open("r", encoding="utf-8") as handle:
        nonblank_line_count = sum(1 for line in handle if line.strip())

    records = read_jsonl(path)
    if len(records) == nonblank_line_count and not _file_ends_without_newline(path):
        return False

    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)
    print(f"Repaired {path}: normalized a partially written or unterminated final line.")
    return True


def extract_gsm8k_gold(answer: str) -> str:
    """Extract the official final answer that follows GSM8K's #### marker."""
    if "####" in answer:
        return answer.split("####")[-1].strip().replace(",", "")
    return answer.strip().split()[-1].replace(",", "")


def build_messages(question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(question=question)},
    ]


# =============================================================================
# 4. TAG PARSING AND NUMERIC EQUIVALENCE
# =============================================================================


def extract_tag(text: str, tag_name: str) -> Optional[str]:
    if not text:
        return None
    pattern = rf"<{tag_name}>\s*(.*?)\s*</{tag_name}>"
    match = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else None


def _tag_positions(text: str, tag_name: str) -> tuple[list[int], list[int]]:
    opens = [
        match.start()
        for match in re.finditer(rf"<{tag_name}>", text, re.IGNORECASE)
    ]
    closes = [
        match.start()
        for match in re.finditer(rf"</{tag_name}>", text, re.IGNORECASE)
    ]
    return opens, closes


def analyze_tag_structure(raw_output: str) -> dict[str, bool]:
    """Identify missing, duplicate, incomplete, reversed, or nested tags."""
    text = raw_output or ""
    reasoning_opens, reasoning_closes = _tag_positions(text, "reasoning")
    final_opens, final_closes = _tag_positions(text, "final_answer")

    reasoning_missing = not reasoning_opens and not reasoning_closes
    final_missing = not final_opens and not final_closes

    reasoning_malformed = (
        not reasoning_missing
        and (
            len(reasoning_opens) != 1
            or len(reasoning_closes) != 1
            or reasoning_opens[0] > reasoning_closes[0]
        )
    )
    final_malformed = (
        not final_missing
        and (
            len(final_opens) != 1
            or len(final_closes) != 1
            or final_opens[0] > final_closes[0]
        )
    )

    both_clean = (
        not reasoning_missing
        and not final_missing
        and not reasoning_malformed
        and not final_malformed
    )
    if both_clean:
        reasoning_open, reasoning_close = reasoning_opens[0], reasoning_closes[0]
        final_open, final_close = final_opens[0], final_closes[0]

        spans_overlap = (
            reasoning_open < final_open < reasoning_close
            or final_open < reasoning_open < final_close
        )
        wrong_order = final_open < reasoning_open
        if spans_overlap or wrong_order:
            reasoning_malformed = True
            final_malformed = True

    return {
        "reasoning_missing": reasoning_missing,
        "final_missing": final_missing,
        "reasoning_malformed": reasoning_malformed,
        "final_malformed": final_malformed,
    }


def normalize_number_text(text: str) -> str:
    """Remove display formatting for answer extraction and numeric comparison."""
    return (
        text.strip()
        .replace(",", "")
        .replace("$", "")
        .replace("€", "")
        .replace("£", "")
        .replace("₹", "")
        .replace("\\boxed{", "")
        .replace("}", "")
        .strip()
    )


def is_strict_numeric_text(text: str) -> bool:
    """True only when the original field is a normalized number and nothing else.

    Unlike normalize_number_text(), this intentionally does not remove commas,
    currency symbols, percent signs, units, or words. This lets evaluation separate
    mathematical correctness from strict output-format compliance.
    """
    cleaned = text.strip()
    pattern = (
        r"[+-]?\d+\s+\d+\s*/\s*\d+"
        r"|[+-]?\d+\s*/\s*[+-]?\d+"
        r"|[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
    )
    return re.fullmatch(pattern, cleaned) is not None


_NUMBER_CANDIDATE_PATTERN = re.compile(
    r"[+-]?\d+\s+\d+\s*/\s*\d+%?"
    r"|[+-]?\d+\s*/\s*[+-]?\d+%?"
    r"|[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?%?"
)


def find_number_candidates(text: str) -> list[str]:
    """All independent number-like substrings in text (mixed numbers and
    fractions count as one candidate each). Used both to extract a single
    answer and to detect an ambiguous multi-candidate final answer such as
    "8 or 10", where more than one candidate appears.
    """
    if not text:
        return []
    return _NUMBER_CANDIDATE_PATTERN.findall(normalize_number_text(text))


def extract_number(text: str, *, prefer_last: bool = False) -> Optional[str]:
    """Extract an integer, decimal, fraction, mixed number, scientific notation, or percent."""
    matches = find_number_candidates(text)
    if not matches:
        return None

    selected = matches[-1] if prefer_last else matches[0]
    selected = re.sub(r"\s+", " ", selected).strip()
    return re.sub(r"\s*/\s*", "/", selected)


def extract_fallback_answer(text: str) -> tuple[Optional[str], bool]:
    """Prefer an explicit answer phrase; otherwise use the final number.

    Returns (answer, is_ambiguous). This branch only runs when there is no
    complete <final_answer> tag pair, but the same risk applies as in the
    tagged case: an answer phrase or bare sentence naming more than one
    number (e.g. "Answer is 8 or 10") means the model never committed to a
    single value, and must not be scored as correct just because gold
    happens to match whichever candidate comes first.
    """
    if not text:
        return None, False

    match = re.search(
        r"(?:final\s+answer|answer)\s*(?:is|:|=)\s*([^\n<]+)",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        candidates = find_number_candidates(match.group(1))
        if len(candidates) > 1:
            return None, True
        if len(candidates) == 1:
            return extract_number(match.group(1)), False

    number_pattern = _NUMBER_CANDIDATE_PATTERN.pattern
    if re.search(
        rf"(?:{number_pattern})\s+or\s+(?:{number_pattern})", text, flags=re.IGNORECASE
    ):
        return None, True

    return extract_number(text, prefer_last=True), False


def convert_to_fraction(value: Optional[str]) -> Optional[Fraction]:
    if value is None:
        return None

    normalized = normalize_number_text(value)
    is_percent = normalized.endswith("%")
    if is_percent:
        normalized = normalized[:-1]

    try:
        mixed_match = re.fullmatch(r"(-?\d+)\s+(\d+)\s*/\s*(\d+)", normalized)
        if mixed_match:
            whole = int(mixed_match.group(1))
            numerator = int(mixed_match.group(2))
            denominator = int(mixed_match.group(3))
            fraction_part = Fraction(numerator, denominator)
            result = whole - fraction_part if whole < 0 else whole + fraction_part
        elif "/" in normalized:
            result = Fraction(normalized)
        else:
            result = Fraction(Decimal(normalized))
        return result / 100 if is_percent else result
    except (ValueError, ZeroDivisionError, InvalidOperation):
        return None


def answers_match(predicted_answer: Optional[str], gold_answer: str) -> bool:
    predicted_value = convert_to_fraction(predicted_answer)
    gold_value = convert_to_fraction(gold_answer)
    return (
        predicted_value is not None
        and gold_value is not None
        and predicted_value == gold_value
    )


# =============================================================================
# 5. OUTPUT EVALUATION
# =============================================================================


def evaluate_output(
    raw_output: str,
    gold_answer: str,
    *,
    is_truncated: bool,
    generation_error: Optional[str] = None,
) -> dict[str, Any]:
    """Evaluate one model output without changing model weights."""
    raw_output = raw_output or ""
    failure_reasons: list[str] = []

    if generation_error is not None:
        failure_reasons.append("generation_error")

    tag_state = analyze_tag_structure(raw_output)
    if tag_state["reasoning_missing"]:
        failure_reasons.append("missing_reasoning_tag")
    if tag_state["final_missing"]:
        failure_reasons.append("missing_final_answer_tag")
    if tag_state["reasoning_malformed"]:
        failure_reasons.append("malformed_reasoning_tag")
    if tag_state["final_malformed"]:
        failure_reasons.append("malformed_final_answer_tag")

    tagged_reasoning = extract_tag(raw_output, "reasoning")
    tagged_final = extract_tag(raw_output, "final_answer")
    reasoning = tagged_reasoning.strip() if tagged_reasoning else ""

    # A <reasoning> tag that matched but captured nothing (or only whitespace)
    # is a distinct defect from a missing tag: the model emitted the required
    # structure without doing any of the required work inside it.
    if tagged_reasoning is not None and not tagged_reasoning.strip():
        failure_reasons.append("empty_reasoning")

    is_ambiguous_final_answer = False
    if tagged_final is not None:
        answer_source = "final_answer_tag"
        # More than one independent number inside <final_answer> (e.g.
        # "8 or 10") means the model never committed to a single answer.
        # Silently picking the first candidate would let it get lucky
        # whenever that first number happens to match gold.
        if len(find_number_candidates(tagged_final)) > 1:
            is_ambiguous_final_answer = True
            predicted_answer = None
            failure_reasons.append("ambiguous_final_answer")
        else:
            predicted_answer = extract_number(tagged_final, prefer_last=False)
        if not is_strict_numeric_text(tagged_final):
            failure_reasons.append("non_numeric_final_answer")
    else:
        predicted_answer, fallback_ambiguous = extract_fallback_answer(raw_output)
        if fallback_ambiguous:
            is_ambiguous_final_answer = True
            failure_reasons.append("ambiguous_final_answer")
        answer_source = "fallback" if predicted_answer is not None else "missing"

    answer_extracted = predicted_answer is not None
    if not answer_extracted and not is_ambiguous_final_answer:
        failure_reasons.append("no_extractable_answer")

    is_correct = answers_match(predicted_answer, gold_answer)
    if answer_extracted and not is_correct:
        failure_reasons.append("wrong_answer")

    if is_truncated:
        failure_reasons.append("truncated_output")

    # Preserve stable ordering while removing accidental duplicates.
    ordered_failures = [
        reason for reason in ALL_FAILURE_REASONS if reason in set(failure_reasons)
    ]

    is_valid_format = not any(reason in FORMAT_FAILURES for reason in ordered_failures)
    is_correct_and_valid = is_correct and is_valid_format

    return {
        "reasoning": reasoning,
        "predicted_answer": predicted_answer,
        "answer_source": answer_source,
        "answer_extracted": answer_extracted,
        "is_correct": is_correct,
        "is_valid_format": is_valid_format,
        "is_correct_and_valid": is_correct_and_valid,
        "is_truncated": is_truncated,
        "validation_errors": ordered_failures,
    }


# =============================================================================
# 6. INFERENCE BACKENDS
# =============================================================================


class OpenAICompatibleBackend:
    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str,
        timeout_seconds: float,
        max_new_tokens: int,
        disable_qwen_thinking: bool,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The openai package is required for --backend openai. "
                "Install it with: pip install -U openai"
            ) from exc

        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout_seconds,
            max_retries=0,
        )
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.disable_qwen_thinking = disable_qwen_thinking
        self.model_metadata = {
            "backend": "openai",
            "model": model,
            "base_url": base_url,
            "parameter_count": None,
        }

    def generate(self, messages: list[dict[str, str]]) -> GenerationResult:
        request: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_new_tokens,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "seed": SEED,
        }
        if self.disable_qwen_thinking:
            # Qwen-specific chat_template_kwargs. Serving stacks (e.g. vLLM) for
            # Gemma/Llama/other non-Qwen students generally ignore unknown
            # chat_template_kwargs, so this is safe to leave on by default, but
            # verify against your specific server if it errors instead.
            request["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": False}
            }

        start = time.perf_counter()
        response = self.client.chat.completions.create(**request)
        elapsed = time.perf_counter() - start

        choice = response.choices[0]
        content = choice.message.content or ""
        usage = getattr(response, "usage", None)

        return GenerationResult(
            text=content,
            finish_reason=getattr(choice, "finish_reason", None),
            prompt_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
            completion_tokens=(
                getattr(usage, "completion_tokens", None) if usage else None
            ),
            total_tokens=getattr(usage, "total_tokens", None) if usage else None,
            elapsed_seconds=elapsed,
            peak_cuda_memory_bytes=None,
        )


class TransformersBackend:
    def __init__(
        self,
        *,
        model_name_or_path: str,
        adapter_path: Optional[str],
        max_new_tokens: int,
        max_input_tokens: int = 1536,
        load_in_4bit: bool,
        dtype_name: str,
        trust_remote_code: bool,
        disable_qwen_thinking: bool,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "torch and transformers are required for --backend transformers. "
                "Install them with: pip install -U torch transformers accelerate"
            ) from exc

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.max_input_tokens = max_input_tokens
        self.disable_qwen_thinking = disable_qwen_thinking

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        dtype = self._resolve_dtype(dtype_name)
        model_kwargs: dict[str, Any] = {
            "device_map": "auto",
            "trust_remote_code": trust_remote_code,
        }
        if dtype is not None:
            model_kwargs["dtype"] = dtype

        if load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as exc:
                raise RuntimeError(
                    "bitsandbytes support is required for --load-in-4bit. "
                    "Install it with: pip install -U bitsandbytes"
                ) from exc

            compute_dtype = dtype
            if compute_dtype is None:
                compute_dtype = (
                    torch.bfloat16
                    if torch.cuda.is_available()
                    and torch.cuda.is_bf16_supported()
                    else torch.float16
                )
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype,
            )

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                **model_kwargs,
            )
        except TypeError as exc:
            # Compatibility fallback for Transformers versions that still expect
            # torch_dtype instead of dtype.
            if "dtype" not in model_kwargs:
                raise
            model_kwargs["torch_dtype"] = model_kwargs.pop("dtype")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                **model_kwargs,
            )

        if adapter_path:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError(
                    "peft is required when --adapter-path is supplied. "
                    "Install it with: pip install -U peft"
                ) from exc
            self.model = PeftModel.from_pretrained(
                self.model,
                adapter_path,
                is_trainable=False,
            )

        self.model.eval()
        self.input_device = self._resolve_input_device()
        parameter_count = sum(parameter.numel() for parameter in self.model.parameters())
        self.model_metadata = {
            "backend": "transformers",
            "model": model_name_or_path,
            "adapter_path": adapter_path,
            "parameter_count": parameter_count,
            "load_in_4bit": load_in_4bit,
            "dtype": dtype_name,
            "max_input_tokens": max_input_tokens,
            # None when there's no adapter; lets a report show how much
            # smaller the saved QLoRA adapter is than the full checkpoint.
            "adapter_storage_gb": directory_size_gb(adapter_path),
        }

    def _resolve_dtype(self, dtype_name: str) -> Any:
        torch = self.torch
        if dtype_name == "auto":
            if torch.cuda.is_available():
                return (
                    torch.bfloat16
                    if torch.cuda.is_bf16_supported()
                    else torch.float16
                )
            return torch.float32
        mapping = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        if dtype_name not in mapping:
            raise ValueError(f"Unsupported dtype: {dtype_name}")
        return mapping[dtype_name]

    def _resolve_input_device(self) -> Any:
        """Prefer the input-embedding layer's device over an arbitrary
        parameter's. next(self.model.parameters()).device returns whichever
        parameter happens to be first in iteration order, which is not
        guaranteed to be where the embedding table lives once
        device_map="auto" spreads a model across multiple GPUs or offloads
        part of it to CPU. Since embedding lookup is the first thing the
        forward pass does with input_ids, placing inputs on any other
        device risks a "tensors on different devices" error.
        """
        try:
            input_embeddings = self.model.get_input_embeddings()
            if input_embeddings is not None:
                return input_embeddings.weight.device
        except (AttributeError, NotImplementedError):
            pass
        return next(self.model.parameters()).device

    def _format_inputs(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        template_kwargs: dict[str, Any] = {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_tensors": "pt",
            "return_dict": True,
        }
        if self.disable_qwen_thinking:
            template_kwargs["enable_thinking"] = False

        try:
            inputs = self.tokenizer.apply_chat_template(messages, **template_kwargs)
        except TypeError:
            # Gemma, Llama, and other non-Qwen chat templates (as well as some
            # older tokenizer implementations) do not accept enable_thinking.
            # Retry without that Qwen-specific option.
            template_kwargs.pop("enable_thinking", None)
            inputs = self.tokenizer.apply_chat_template(messages, **template_kwargs)

        # With return_dict=True, apply_chat_template normally returns a
        # BatchEncoding -- which subclasses UserDict, not dict, but already
        # supports .items() like any mapping. Only a bare tensor (from a
        # template/tokenizer that doesn't honor return_dict) needs wrapping.
        # Checking hasattr(inputs, "to") to decide is not a valid test: a
        # BatchEncoding also has .to(), so that check wrapped the *normal*
        # case, replacing "input_ids" with the whole BatchEncoding object
        # and silently dropping "attention_mask".
        if isinstance(inputs, self.torch.Tensor):
            inputs = {"input_ids": inputs}

        # Deliberately not using the tokenizer's truncation=True/max_length
        # here: silently truncating would risk cutting off part of the
        # question itself and scoring the model as "wrong" on a question it
        # never fully saw. Raising instead surfaces that loudly -- and
        # safely, since evaluate_dataset() already wraps backend.generate()
        # in a try/except and records this one example as a generation_error
        # without killing the rest of the run.
        prompt_length = int(inputs["input_ids"].shape[-1])
        if prompt_length > self.max_input_tokens:
            raise ValueError(
                f"Prompt is {prompt_length} tokens, exceeding --max-input-tokens "
                f"={self.max_input_tokens}. Raise the limit or inspect this "
                "example instead of silently truncating the question."
            )

        return {key: value.to(self.input_device) for key, value in inputs.items()}

    def _reset_cuda_peak_memory(self) -> None:
        torch = self.torch
        if not torch.cuda.is_available():
            return
        for device_index in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(device_index)

    def _peak_cuda_memory(self) -> Optional[int]:
        torch = self.torch
        if not torch.cuda.is_available():
            return None
        return sum(
            int(torch.cuda.max_memory_allocated(device_index))
            for device_index in range(torch.cuda.device_count())
        )

    def generate(self, messages: list[dict[str, str]]) -> GenerationResult:
        torch = self.torch
        inputs = self._format_inputs(messages)
        input_length = int(inputs["input_ids"].shape[-1])

        self._reset_cuda_peak_memory()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": TEMPERATURE > 0,
            "return_dict_in_generate": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "use_cache": True,
        }
        if TEMPERATURE > 0:
            generation_kwargs["temperature"] = TEMPERATURE
            generation_kwargs["top_p"] = TOP_P

        with torch.inference_mode():
            output = self.model.generate(**inputs, **generation_kwargs)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        sequence = output.sequences[0]
        generated_ids = sequence[input_length:]
        completion_tokens = int(generated_ids.shape[-1])
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        eos_ids = self.model.generation_config.eos_token_id
        if eos_ids is None:
            eos_set: set[int] = set()
        elif isinstance(eos_ids, int):
            eos_set = {eos_ids}
        else:
            eos_set = {int(token_id) for token_id in eos_ids}

        last_token = int(generated_ids[-1].item()) if completion_tokens else None
        ended_with_eos = last_token in eos_set if last_token is not None else False
        reached_limit = completion_tokens >= self.max_new_tokens
        finish_reason = "length" if reached_limit and not ended_with_eos else "stop"

        prompt_tokens = input_length
        return GenerationResult(
            text=text,
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            elapsed_seconds=elapsed,
            peak_cuda_memory_bytes=self._peak_cuda_memory(),
        )


# =============================================================================
# 7. METRIC AGGREGATION
# =============================================================================


def _safe_rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _mean(values: list[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def percentile(values: list[float], fraction: float) -> Optional[float]:
    """Linear-interpolation percentile (fraction in [0, 1], e.g. 0.95 for p95).
    Useful alongside mean/median for comparing inference latency before and
    after SFT: a heavy tail can raise p95 well above the mean/median even
    when typical-case latency looks fine.
    """
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]
    return ordered[lower_index] * (upper_index - position) + ordered[upper_index] * (
        position - lower_index
    )

def bootstrap_accuracy_ci(
    records: list[dict[str, Any]],
    *,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = SEED,
) -> dict[str, Optional[float]]:
    """Bootstrap confidence interval on exact-match accuracy: resample the
    per-example correctness outcomes with replacement n_resamples times and
    take the empirical percentile interval. Cheap and distribution-free,
    unlike assuming a normal approximation on a possibly small test set.
    """
    outcomes = [1 if record.get("is_correct") else 0 for record in records]
    n = len(outcomes)
    if n == 0:
        return {"ci_lower": None, "ci_upper": None}

    rng = random.Random(seed)
    resampled_means = []
    for _ in range(n_resamples):
        sample_sum = sum(outcomes[rng.randrange(n)] for _ in range(n))
        resampled_means.append(sample_sum / n)

    resampled_means.sort()
    alpha = (1 - confidence) / 2
    lower_index = int(alpha * n_resamples)
    upper_index = int((1 - alpha) * n_resamples) - 1
    return {
        "ci_lower": resampled_means[lower_index],
        "ci_upper": resampled_means[upper_index],
    }

def _sum_optional_int(records: list[dict[str, Any]], key: str) -> Optional[int]:
    values = [record.get(key) for record in records]
    valid_values = [value for value in values if isinstance(value, int)]
    return sum(valid_values) if valid_values else None


def aggregate_metrics(
    records: list[dict[str, Any]],
    *,
    config: EvalConfig,
    model_metadata: dict[str, Any],
    prediction_path: Path,
) -> dict[str, Any]:
    total = len(records)
    correct = sum(bool(record.get("is_correct")) for record in records)
    valid = sum(bool(record.get("is_valid_format")) for record in records)
    extracted = sum(bool(record.get("answer_extracted")) for record in records)
    correct_and_valid = sum(
        bool(record.get("is_correct_and_valid")) for record in records
    )
    truncated = sum(bool(record.get("is_truncated")) for record in records)
    generation_errors = sum(
        "generation_error" in record.get("validation_errors", [])
        for record in records
    )

    completion_token_values = [
        int(record["completion_tokens"])
        for record in records
        if isinstance(record.get("completion_tokens"), int)
    ]
    elapsed_values = [
        float(record["elapsed_seconds"])
        for record in records
        if isinstance(record.get("elapsed_seconds"), (int, float))
        and record.get("elapsed_seconds", 0) >= 0
        and "generation_error" not in record.get("validation_errors", [])
    ]
    peak_memory_values = [
        int(record["peak_cuda_memory_bytes"])
        for record in records
        if isinstance(record.get("peak_cuda_memory_bytes"), int)
    ]
    accuracy_ci = bootstrap_accuracy_ci(records)

    failure_counter: Counter[str] = Counter()
    for record in records:
        failure_counter.update(record.get("validation_errors", []))

    total_completion_tokens = sum(completion_token_values)
    total_inference_seconds = sum(elapsed_values)

    return {
        "evaluation_type": "gsm8k_shared_model_evaluation",
        "primary_metric": "exact_match_accuracy",
        "main_secondary_metric": "correct_and_valid_rate",
        "model": config.model,
        "stage": config.stage,
        "backend": config.backend,
        "adapter_path": config.adapter_path,
        "config_hash": config.config_hash(),
        "dataset": DATASET,
        "dataset_revision": DATASET_REVISION,
        "requested_limit": config.limit,
        "total_examples": total,
        "exact_match_accuracy": _safe_rate(correct, total),
        "accuracy_ci_lower": accuracy_ci["ci_lower"],
        "accuracy_ci_upper": accuracy_ci["ci_upper"],
        "valid_format_rate": _safe_rate(valid, total),
        "answer_extraction_success_rate": _safe_rate(extracted, total),
        "correct_and_valid_rate": _safe_rate(correct_and_valid, total),
        "truncation_rate": _safe_rate(truncated, total),
        "generation_error_rate": _safe_rate(generation_errors, total),
        "average_completion_tokens": _mean(
            [float(value) for value in completion_token_values]
        ),
        "average_inference_seconds": _mean(elapsed_values),
        "median_inference_seconds": (
            statistics.median(elapsed_values) if elapsed_values else None
        ),
        "p95_inference_seconds": percentile(elapsed_values, 0.95),
        "completion_tokens_per_second": (
            total_completion_tokens / total_inference_seconds
            if total_inference_seconds > 0
            else None
        ),
        "total_prompt_tokens": _sum_optional_int(records, "prompt_tokens"),
        "total_completion_tokens": (
            total_completion_tokens if completion_token_values else None
        ),
        "total_inference_seconds": (
            total_inference_seconds if elapsed_values else None
        ),
        "peak_cuda_memory_bytes": max(peak_memory_values)
        if peak_memory_values
        else None,
        "failure_reason_counts": {
            reason: failure_counter.get(reason, 0) for reason in ALL_FAILURE_REASONS
        },
        "model_metadata": model_metadata,
        "generation_config": {
            "max_new_tokens": config.max_new_tokens,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "seed": SEED,
            "disable_qwen_thinking": config.disable_qwen_thinking,
        },
        "prompt": {
            "system_prompt": SYSTEM_PROMPT,
            "user_template": USER_TEMPLATE,
        },
        "prediction_file": str(prediction_path),
    }


# =============================================================================
# 7b. COMPARISON AND CSV SUMMARY REPORTING
# =============================================================================


def compute_comparison_metrics(
    *,
    before_accuracy: float,
    after_accuracy: float,
    teacher_accuracy: Optional[float] = None,
) -> dict[str, Optional[float]]:
    """Absolute/relative accuracy improvement from SFT, and the fine-tuned
    student's remaining gap to the teacher.

    Example: before=0.30, after=0.45 -> absolute=0.15, relative=0.50 (50%).
    gap_to_teacher_accuracy is measured against `after_accuracy` (the
    fine-tuned student), matching how a distillation report typically frames
    "how close did SFT get the student to the teacher".
    """
    absolute_improvement = after_accuracy - before_accuracy
    relative_improvement = (
        absolute_improvement / before_accuracy if before_accuracy > 0 else None
    )
    gap_to_teacher_accuracy = (
        teacher_accuracy - after_accuracy if teacher_accuracy is not None else None
    )
    return {
        "absolute_improvement": absolute_improvement,
        "relative_improvement": relative_improvement,
        "gap_to_teacher_accuracy": gap_to_teacher_accuracy,
    }


SUMMARY_CSV_FIELDS = (
    "condition",
    "model",
    "adapter_path",
    "exact_match_accuracy",
    "accuracy_ci_lower",
    "accuracy_ci_upper",
    "valid_format_rate",
    "correct_and_valid_rate",
    "mean_inference_seconds",
    "median_inference_seconds",
    "p95_inference_seconds",
    "completion_tokens_per_second",
    "peak_cuda_memory_gb",
    "model_parameter_count",
    "adapter_storage_gb",
    "absolute_improvement_over_before",
    "relative_improvement_over_before",
    "gap_to_teacher_accuracy",
)


def summary_row_from_metrics(
    metrics: dict[str, Any], *, condition: str, **extra: Any
) -> dict[str, Any]:
    """Flatten one aggregate_metrics() result (or a hand-built dict with at
    least exact_match_accuracy, e.g. a bare --teacher-accuracy) into a single
    CSV-ready row. Extra comparison-only fields (improvement/gap) are passed
    in by the caller since they depend on more than one condition's metrics.
    """
    peak_bytes = metrics.get("peak_cuda_memory_bytes")
    model_metadata = metrics.get("model_metadata") or {}
    row: dict[str, Any] = {
        "condition": condition,
        "model": metrics.get("model"),
        "adapter_path": metrics.get("adapter_path"),
        "exact_match_accuracy": metrics.get("exact_match_accuracy"),
        "accuracy_ci_lower": metrics.get("accuracy_ci_lower"),
        "accuracy_ci_upper": metrics.get("accuracy_ci_upper"),
        "valid_format_rate": metrics.get("valid_format_rate"),
        "correct_and_valid_rate": metrics.get("correct_and_valid_rate"),
        "mean_inference_seconds": metrics.get("average_inference_seconds"),
        "median_inference_seconds": metrics.get("median_inference_seconds"),
        "p95_inference_seconds": metrics.get("p95_inference_seconds"),
        "completion_tokens_per_second": metrics.get("completion_tokens_per_second"),
        "peak_cuda_memory_gb": (
            peak_bytes / (1024**3) if isinstance(peak_bytes, (int, float)) else None
        ),
        "model_parameter_count": model_metadata.get("parameter_count"),
        "adapter_storage_gb": model_metadata.get("adapter_storage_gb"),
    }
    row.update(extra)
    return row


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write the human-friendly comparison table (Condition, Accuracy, Valid
    format, latency stats, tokens/sec, GPU memory, ...) as CSV, easier to
    open in Excel/Sheets for a report than the full JSON metrics files.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in SUMMARY_CSV_FIELDS})


# =============================================================================
# 8. DATASET AND EVALUATION LOOP
# =============================================================================


def load_gsm8k_examples(limit: Optional[int]) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "datasets is required. Install it with: pip install -U datasets"
        ) from exc

    split = load_dataset(
    DATASET, DATASET_CONFIG, split=DATASET_SPLIT, revision=DATASET_REVISION)
    stop = len(split) if limit is None else min(len(split), limit)

    examples: list[dict[str, Any]] = []
    for source_index in range(stop):
        row = split[source_index]
        examples.append(
            {
                "problem_id": f"gsm8k_test_{source_index:04d}",
                "source_index": source_index,
                "question": str(row["question"]),
                "gold_answer": extract_gsm8k_gold(str(row["answer"])),
            }
        )
    return examples


def create_backend(
    config: EvalConfig, *, api_key: str, timeout_seconds: float
) -> GenerationBackend:
    """Build the backend for one EvalConfig. Takes an EvalConfig rather than
    the raw argparse.Namespace so --compare-base can build two backends (base
    and base+adapter) from two different configs sharing the same CLI args.
    api_key/timeout_seconds live outside EvalConfig (they don't affect
    predictions, only how the request is transported) so they're passed
    separately.
    """
    if config.backend == "openai":
        if not config.base_url:
            raise ValueError("--base-url is required for --backend openai")
        return OpenAICompatibleBackend(
            model=config.model,
            base_url=config.base_url,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            max_new_tokens=config.max_new_tokens,
            disable_qwen_thinking=config.disable_qwen_thinking,
        )

    return TransformersBackend(
        model_name_or_path=config.model,
        adapter_path=config.adapter_path,
        max_new_tokens=config.max_new_tokens,
        max_input_tokens=config.max_input_tokens,
        load_in_4bit=config.load_in_4bit,
        dtype_name=config.dtype,
        trust_remote_code=config.trust_remote_code,
        disable_qwen_thinking=config.disable_qwen_thinking,
    )


def _rewrite_predictions_clean(
    predictions_path: Path,
    examples: list[dict[str, Any]],
    records_by_id: dict[str, dict[str, Any]],
    existing_record_count: int,
) -> None:
    """Rewrite predictions.jsonl with at most one row per example, in dataset
    order, using only the latest record for each id currently in
    records_by_id.

    append_jsonl() during the loop still appends every attempt (including a
    prior generation_error) so nothing is lost if the process is killed
    mid-run; retrying a generation_error therefore leaves an old error row
    behind a new one for the same problem_id. This keeps the file matching
    its "one detailed record per example" contract instead of silently
    accumulating superseded rows across repeated resumes.

    existing_record_count is the number of rows the caller already read from
    predictions_path, passed in to avoid reading the file a second time here.
    """
    ordered = [
        records_by_id[example["problem_id"]]
        for example in examples
        if example["problem_id"] in records_by_id
    ]
    if len(ordered) == existing_record_count:
        return  # Already one row per known record; nothing to compact.

    temp_path = predictions_path.with_suffix(predictions_path.suffix + ".tmp")
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    with temp_path.open("w", encoding="utf-8") as handle:
        for record in ordered:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, predictions_path)


def evaluate_dataset(
    *,
    config: EvalConfig,
    backend: GenerationBackend,
    examples: list[dict[str, Any]],
    predictions_path: Path,
    metrics_path: Path,
) -> dict[str, Any]:
    # Repair any partially written trailing line left by a killed prior run
    # before treating the predictions file as the source of truth for resume.
    repair_trailing_jsonl_record(predictions_path)

    existing_records = read_jsonl(predictions_path)
    existing_by_id = {
        str(record["problem_id"]): record
        for record in existing_records
        if record.get("problem_id") is not None
    }

    target_ids = {example["problem_id"] for example in examples}
    # Ignore unrelated rows if a user manually combined files.
    records_by_id = {
        problem_id: record
        for problem_id, record in existing_by_id.items()
        if problem_id in target_ids
    }
    # A record saved with a generation_error (timeout, GPU error, etc.) is a
    # technical failure, not a scored prediction -- it must be retried on
    # resume, not skipped forever. Only error-free records count as done.
    completed_ids = {
        problem_id
        for problem_id, record in records_by_id.items()
        if record.get("generation_error") is None
    }
    # Normalize any duplicate rows a previous retried run may have appended
    # for the same problem_id (append_jsonl always adds a new row; only the
    # latest one per id is kept in records_by_id) before starting new work.
    _rewrite_predictions_clean(
        predictions_path, examples, records_by_id, len(existing_records)
    )

    rows_appended_this_run = 0
    total_examples = len(examples)
    already_done = len(completed_ids)
    retryable = len(records_by_id) - already_done
    print(
        f"Evaluating {config.model} [{config.stage}] on {total_examples} examples. "
        f"Resuming with {already_done} already completed"
        + (f", {retryable} to retry after a prior generation error." if retryable else ".")
    )

    for position, example in enumerate(examples, start=1):
        problem_id = example["problem_id"]
        if problem_id in completed_ids:
            continue

        messages = build_messages(example["question"])
        generation_error: Optional[str] = None
        result: Optional[GenerationResult] = None

        try:
            result = backend.generate(messages)
            raw_output = result.text
            is_truncated = result.finish_reason == "length"
        except Exception as exc:  # Keep the run resumable and count technical failures.
            generation_error = f"{type(exc).__name__}: {exc}"
            raw_output = ""
            is_truncated = False

        evaluation = evaluate_output(
            raw_output,
            example["gold_answer"],
            is_truncated=is_truncated,
            generation_error=generation_error,
        )

        record: dict[str, Any] = {
            "problem_id": problem_id,
            "source_index": example["source_index"],
            "question": example["question"],
            "gold_answer": example["gold_answer"],
            "raw_output": raw_output,
            "finish_reason": result.finish_reason if result else None,
            "generation_error": generation_error,
            "prompt_tokens": result.prompt_tokens if result else None,
            "completion_tokens": result.completion_tokens if result else None,
            "total_tokens": result.total_tokens if result else None,
            "elapsed_seconds": result.elapsed_seconds if result else None,
            "peak_cuda_memory_bytes": (
                result.peak_cuda_memory_bytes if result else None
            ),
            **evaluation,
        }

        append_jsonl(predictions_path, record)
        records_by_id[problem_id] = record
        rows_appended_this_run += 1

        status = "correct" if record["is_correct"] else "wrong"
        print(
            f"[{position}/{total_examples}] {problem_id}: {status}; "
            f"valid_format={record['is_valid_format']}; "
            f"errors={record['validation_errors']}"
        )

        # Refresh partial metrics after every example. This makes interrupted runs
        # immediately inspectable and safe to resume.
        ordered_partial_records = [
            records_by_id[item["problem_id"]]
            for item in examples
            if item["problem_id"] in records_by_id
        ]
        partial_metrics = aggregate_metrics(
            ordered_partial_records,
            config=config,
            model_metadata=backend.model_metadata,
            prediction_path=predictions_path,
        )
        write_json_atomic(metrics_path, partial_metrics)

    ordered_records = [records_by_id[example["problem_id"]] for example in examples]
    _rewrite_predictions_clean(
        predictions_path,
        examples,
        records_by_id,
        len(existing_records) + rows_appended_this_run,
    )
    final_metrics = aggregate_metrics(
        ordered_records,
        config=config,
        model_metadata=backend.model_metadata,
        prediction_path=predictions_path,
    )
    write_json_atomic(metrics_path, final_metrics)
    return final_metrics


# =============================================================================
# 9. COMMAND-LINE INTERFACE
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a teacher or student causal LM on GSM8K."
    )
    parser.add_argument(
        "--backend",
        choices=("transformers", "openai"),
        default="transformers",
    )
    parser.add_argument(
        "--model",
        default="google/gemma-3-1b-it",
        help=(
            "Hugging Face repo id or local checkpoint path. Any instruction-tuned "
            "causal LM with a chat template works, e.g. a Qwen teacher or a "
            "Gemma/Llama/Qwen student."
        ),
    )
    parser.add_argument(
        "--stage",
        required=False,
        default="before_sft",
        help="Examples: teacher, before_sft, after_sft",
    )
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument(
        "--compare-base",
        action="store_true",
        help=(
            "Automatically evaluate --model twice in one run: once with no "
            "adapter (before_sft) and once with --adapter-path applied "
            "(after_sft), then report absolute/relative accuracy improvement "
            "and write a combined summary.csv. Requires --adapter-path and "
            "--backend transformers."
        ),
    )
    parser.add_argument(
        "--teacher-accuracy",
        type=float,
        default=None,
        help=(
            "Teacher's exact_match_accuracy (0-1), used to report the "
            "student's gap_to_teacher_accuracy. Ignored if "
            "--teacher-metrics-path is also given."
        ),
    )
    parser.add_argument(
        "--teacher-metrics-path",
        type=Path,
        default=None,
        help=(
            "Path to a metrics.json from a prior teacher evaluation run. "
            "When given, the full teacher row (accuracy, latency, etc.) is "
            "included in summary.csv instead of just a bare accuracy number."
        ),
    )

    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--timeout-seconds", type=float, default=180.0)

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Use a small value only for smoke tests. Omit for the full test set.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=1536,
        help=(
            "Truncates the tokenized prompt to this many tokens (transformers "
            "backend only) so an unusually long question can't silently "
            "exceed the model's context window."
        ),
    )

    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--disable-qwen-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Disable Qwen3 hidden-thinking mode when the serving/template stack "
            "supports it. Qwen-specific: has no effect for non-Qwen students "
            "such as Gemma or Llama, whose chat templates don't accept it."
        ),
    )

    parser.add_argument("--output-dir", type=Path, default=Path("eval_outputs"))
    parser.add_argument(
        "--run-tag",
        default=None,
        help=(
            "Optional label (e.g. v1, v2, final) folded into output filenames. "
            "Leave unset for smoke tests, where rerunning the same config "
            "should resume/reuse the same files; set it for full runs you "
            "want kept as a permanent, separately named version instead of "
            "resuming into an earlier run's files."
        ),
    )
    return parser.parse_args()


def _build_config(
    args: argparse.Namespace, *, adapter_path: Optional[str], stage: str
) -> EvalConfig:
    """Build an EvalConfig from CLI args, overriding adapter_path/stage.
    --compare-base calls this twice (adapter_path=None/stage="before_sft",
    then adapter_path=args.adapter_path/stage="after_sft") to evaluate the
    same base model with and without the adapter in one invocation.
    """
    return EvalConfig(
        backend=args.backend,
        model=args.model,
        stage=stage,
        adapter_path=adapter_path,
        base_url=args.base_url,
        limit=args.limit,
        max_new_tokens=args.max_new_tokens,
        max_input_tokens=args.max_input_tokens,
        load_in_4bit=args.load_in_4bit,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        disable_qwen_thinking=args.disable_qwen_thinking,
    )


def _run_slug(config: EvalConfig, run_tag: Optional[str] = None) -> str:
    """Build the base filename for one run: model + stage + a short hash.

    The hash changes whenever something that affects the answers changes
    (adapter, quantization, token limits, ...), so two different setups
    never share a file by accident. If run_tag is given (e.g. "v1"), it's
    added to the end so that run gets its own permanent files instead of
    resuming/overwriting an earlier one. Leave run_tag unset for quick
    smoke tests, where resuming/reusing the same files is fine.
    """
    parts = [slugify(config.model), slugify(config.stage), config.config_hash()]
    if run_tag:
        parts.append(slugify(run_tag))
    return "_".join(parts)


def run_condition(
    args: argparse.Namespace, config: EvalConfig, examples: list[dict[str, Any]]
) -> tuple[dict[str, Any], Path, Path]:
    """Run one evaluation condition end to end: build its backend, run
    evaluate_dataset, and return (metrics, predictions_path, metrics_path).
    """
    run_slug = _run_slug(config, args.run_tag)
    predictions_path = args.output_dir / f"{run_slug}_predictions.jsonl"
    metrics_path = args.output_dir / f"{run_slug}_metrics.json"

    backend = create_backend(config, api_key=args.api_key, timeout_seconds=args.timeout_seconds)
    metrics = evaluate_dataset(
        config=config,
        backend=backend,
        examples=examples,
        predictions_path=predictions_path,
        metrics_path=metrics_path,
    )
    return metrics, predictions_path, metrics_path


def validate_args(args: argparse.Namespace) -> None:
    """Cross-field CLI validation, split out from main() so it's directly
    unit-testable without needing a real dataset/model/network access."""
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be greater than zero")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be greater than zero")
    if args.max_input_tokens <= 0:
        raise ValueError("--max-input-tokens must be greater than zero")
    if args.teacher_accuracy is not None and not 0 <= args.teacher_accuracy <= 1:
        raise ValueError("--teacher-accuracy must be between 0 and 1")
    if args.compare_base:
        if not args.adapter_path:
            raise ValueError(
                "--compare-base requires --adapter-path (the adapter to "
                "compare against the unmodified base model)"
            )
        if args.backend != "transformers":
            raise ValueError(
                "--compare-base requires --backend transformers, since it "
                "loads the base checkpoint and the adapter locally to "
                "compare them in one run"
            )


def main() -> None:
    args = parse_args()
    validate_args(args)

    random.seed(SEED)
    try:
        import torch

        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)
    except ImportError:
        pass

    args.output_dir.mkdir(parents=True, exist_ok=True)

    teacher_metrics: Optional[dict[str, Any]] = None
    teacher_accuracy: Optional[float] = args.teacher_accuracy
    if args.teacher_metrics_path:
        teacher_metrics = json.loads(
            Path(args.teacher_metrics_path).read_text(encoding="utf-8")
        )
        teacher_accuracy = teacher_metrics.get("exact_match_accuracy", teacher_accuracy)

    summary_rows: list[dict[str, Any]] = []
    if teacher_metrics is not None:
        summary_rows.append(summary_row_from_metrics(teacher_metrics, condition="teacher"))
    elif teacher_accuracy is not None:
        summary_rows.append({"condition": "teacher", "exact_match_accuracy": teacher_accuracy})

    if args.compare_base:
        before_config = _build_config(args, adapter_path=None, stage="before_sft")
        after_config = _build_config(args, adapter_path=args.adapter_path, stage="after_sft")
        examples = load_gsm8k_examples(args.limit)

        print(f"\n=== Condition: before_sft ({args.model}, no adapter) ===")
        before_metrics, _, _ = run_condition(args, before_config, examples)
        summary_rows.append(summary_row_from_metrics(before_metrics, condition="before_sft"))

        print(f"\n=== Condition: after_sft ({args.model} + {args.adapter_path}) ===")
        after_metrics, _, _ = run_condition(args, after_config, examples)

        comparison = compute_comparison_metrics(
            before_accuracy=before_metrics["exact_match_accuracy"],
            after_accuracy=after_metrics["exact_match_accuracy"],
            teacher_accuracy=teacher_accuracy,
        )
        summary_rows.append(
            summary_row_from_metrics(
                after_metrics,
                condition="after_sft",
                absolute_improvement_over_before=comparison["absolute_improvement"],
                relative_improvement_over_before=comparison["relative_improvement"],
                gap_to_teacher_accuracy=comparison["gap_to_teacher_accuracy"],
            )
        )

        print("\nComparison summary")
        print(f"  Before-SFT accuracy:  {before_metrics['exact_match_accuracy']:.4f}")
        print(f"  After-SFT accuracy:   {after_metrics['exact_match_accuracy']:.4f}")
        print(f"  Absolute improvement: {comparison['absolute_improvement']:.4f}")
        if comparison["relative_improvement"] is not None:
            print(f"  Relative improvement: {comparison['relative_improvement']:.2%}")
        if comparison["gap_to_teacher_accuracy"] is not None:
            print(f"  Gap to teacher:       {comparison['gap_to_teacher_accuracy']:.4f}")

        # Per-model filename: --compare-base runs are typically repeated once
        # per student (Gemma, Llama, Qwen, ...) against the same --output-dir,
        # and a shared "summary.csv" would let each student's run silently
        # overwrite the previous one's. The adapter's config_hash is also
        # included so comparing two different adapters for the same base
        # model doesn't overwrite the same summary either, and --run-tag
        # (e.g. "v1") is appended when given so a deliberate rerun of the
        # same comparison is kept as its own permanent version.
        compare_summary_parts = [slugify(args.model), after_config.config_hash()]
        if args.run_tag:
            compare_summary_parts.append(slugify(args.run_tag))
        summary_path = args.output_dir / ("_".join(compare_summary_parts) + "_summary.csv")
    else:
        config = _build_config(args, adapter_path=args.adapter_path, stage=args.stage)
        examples = load_gsm8k_examples(args.limit)
        metrics, predictions_path, metrics_path = run_condition(args, config, examples)

        extra: dict[str, Any] = {}
        if teacher_accuracy is not None:
            extra["gap_to_teacher_accuracy"] = teacher_accuracy - metrics["exact_match_accuracy"]
        summary_rows.append(
            summary_row_from_metrics(metrics, condition=config.stage, **extra)
        )

        print("\nEvaluation complete")
        print(f"Predictions: {predictions_path}")
        print(f"Metrics:     {metrics_path}")
        print(
            f"Exact-match accuracy: {metrics['exact_match_accuracy']:.4f}\n"
            f"Correct-and-valid:    {metrics['correct_and_valid_rate']:.4f}"
        )

        summary_path = args.output_dir / f"{_run_slug(config, args.run_tag)}_summary.csv"

    write_summary_csv(summary_path, summary_rows)
    print(f"Summary CSV: {summary_path}")


if __name__ == "__main__":
    main()
