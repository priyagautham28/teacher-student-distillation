# -*- coding: utf-8 -*-
"""Generate a validated GSM8K distillation dataset with a local Qwen teacher.

The teacher is not trained here. It generates verified reasoning examples for
student-model supervised fine-tuning (SFT).

Recommended workflow:
1. Keep PILOT_MODE=True and inspect the 10/5 pilot outputs.
2. Set PILOT_MODE=False only after tagged reasoning is confirmed.
3. Fine-tune students using only the generated *_sft.jsonl files.
4. Keep the official GSM8K test split untouched for final evaluation.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
import uuid
from collections import Counter
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Any, Optional    

from datasets import load_dataset
from openai import OpenAI


# =============================================================================
# 1. CONFIGURATION
# =============================================================================

SEED = 42
N_TRAIN = 2000
N_VAL = 500
SOURCE_DATASET = "openai/gsm8k"
SOURCE_CONFIG = "main"
SOURCE_SPLIT = "train"
# Pin to a specific HF dataset git revision (commit SHA or tag) for full
# reproducibility. None resolves to whatever `datasets` treats as current,
# which means the upstream repo could change under the same name/config.
SOURCE_DATASET_REVISION: Optional[str] = None

MODEL = "Qwen/Qwen3-14B-AWQ"
MODEL_SLUG = "qwen3_14b_awq"
BASE_URL = "http://localhost:8000/v1"
REQUEST_TIMEOUT_SECONDS = 180.0

MAX_TOKENS = 2048
TEMPERATURE = 0.6
TOP_P = 0.95
MAX_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 2
SNAPSHOT_EVERY_N = 50

# Run a small pilot first. Pilot and full runs use different output files.
PILOT_MODE = False
PILOT_N_TRAIN = 10
PILOT_N_VAL = 5

# Accepted records are always skipped. Rejected records can be regenerated.
RETRY_REJECTED_ON_RESUME = True

# Final SFT quality policy: tagged <reasoning>/<final_answer> output is always
# required, in pilot and full runs alike. Raw reasoning_content from the model
# is never promoted into training targets; it is retained only in audit
# records for inspection.
ENABLE_THINKING = False
REJECT_SIGNIFICANT_TEXT_OUTSIDE_TAGS = True
OUTSIDE_TEXT_WARNING_CHAR_LIMIT = 20

# Output-quality thresholds.
MIN_REASONING_CHARACTERS = 20
MIN_REASONING_WORDS = 8
REQUIRE_NUMERIC_CALCULATION_IN_REASONING = True
MAX_REASONING_CHARACTERS = 4_000
MAX_REASONING_LINES = 30
MAX_REPEATED_LINE_COUNT = 3
MAX_REPEATED_SENTENCE_COUNT = 3
MIN_RECOMMENDED_STEPS = 2
MAX_RECOMMENDED_STEPS = 8

PROMPT_VERSION = "gsm8k_teacher_v3"
DATA_DIR = Path("data_final")


# =============================================================================
# 2. PROMPTS
# =============================================================================

SYSTEM_PROMPT = """
You are a mathematical reasoning teacher creating high-quality supervised
training examples for a smaller language model.

Solve the given mathematical word problem carefully.

Rules:
1. Provide concise and logically complete reasoning.
2. Use 2 to 8 clear reasoning steps when practical.
3. Include every calculation needed to understand the solution.
4. Verify the arithmetic before producing the final answer.
5. Use only one solution method.
6. Do not mention being an AI, training data, GSM8K, or the teacher model.
7. Do not include Markdown, code fences, headings, or text outside the tags.
8. Return both tags and close both tags.
9. The <final_answer> tag must contain only the normalized numerical answer.
10. Do not include commas, units, currency symbols, percentages, or explanatory
    text inside <final_answer>.

Required output format:

<reasoning>
Step 1: ...
Step 2: ...
</reasoning>
<final_answer>number only</final_answer>
""".strip()

STUDENT_SYSTEM_PROMPT = (
    "Solve the mathematical problem using concise step-by-step reasoning. "
    "Return the reasoning inside <reasoning> tags and the numerical answer "
    "inside <final_answer> tags."
)


def build_config_payload() -> dict[str, Any]:
    """Return every setting that can materially change generated data."""
    return {
        "teacher_model": MODEL,
        "base_url": BASE_URL,
        "prompt_version": PROMPT_VERSION,
        "system_prompt": SYSTEM_PROMPT,
        "student_system_prompt": STUDENT_SYSTEM_PROMPT,
        "seed": SEED,
        "source_dataset": SOURCE_DATASET,
        "source_config": SOURCE_CONFIG,
        "source_split": SOURCE_SPLIT,
        "source_dataset_revision": SOURCE_DATASET_REVISION,
        "n_train": N_TRAIN,
        "n_val": N_VAL,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_attempts": MAX_ATTEMPTS,
        "retry_delay_seconds": RETRY_DELAY_SECONDS,
        "snapshot_every_n": SNAPSHOT_EVERY_N,
        "enable_thinking": ENABLE_THINKING,
        "reject_significant_text_outside_tags": REJECT_SIGNIFICANT_TEXT_OUTSIDE_TAGS,
        "outside_text_warning_char_limit": OUTSIDE_TEXT_WARNING_CHAR_LIMIT,
        "min_reasoning_characters": MIN_REASONING_CHARACTERS,
        "min_reasoning_words": MIN_REASONING_WORDS,
        "require_numeric_calculation_in_reasoning": REQUIRE_NUMERIC_CALCULATION_IN_REASONING,
        "max_reasoning_characters": MAX_REASONING_CHARACTERS,
        "max_reasoning_lines": MAX_REASONING_LINES,
        "max_repeated_line_count": MAX_REPEATED_LINE_COUNT,
        "max_repeated_sentence_count": MAX_REPEATED_SENTENCE_COUNT,
        "min_recommended_steps": MIN_RECOMMENDED_STEPS,
        "max_recommended_steps": MAX_RECOMMENDED_STEPS,
    }


CONFIG_PAYLOAD = build_config_payload()
CONFIG_HASH = hashlib.sha256(
    json.dumps(CONFIG_PAYLOAD, sort_keys=True).encode("utf-8")
).hexdigest()[:10]
RUN_MODE = "pilot" if PILOT_MODE else "full"
RUN_SLUG = f"{MODEL_SLUG}_{PROMPT_VERSION}_{CONFIG_HASH}_{RUN_MODE}"

# The train/val split only depends on these fields, not on the rest of
# CONFIG_PAYLOAD (model, temperature, prompt version, ...), so it gets its
# own identity hash: unrelated generation-setting changes must keep reusing
# the same cached split, but a different SOURCE_DATASET_REVISION must not
# silently reuse indices that were chosen against a different dataset build.
SPLIT_IDENTITY_PAYLOAD: dict[str, Any] = {
    "seed": SEED,
    "source_dataset": SOURCE_DATASET,
    "source_config": SOURCE_CONFIG,
    "source_split": SOURCE_SPLIT,
    "source_dataset_revision": SOURCE_DATASET_REVISION,
    "n_train": N_TRAIN,
    "n_val": N_VAL,
}
SPLIT_IDENTITY_HASH = hashlib.sha256(
    json.dumps(SPLIT_IDENTITY_PAYLOAD, sort_keys=True).encode("utf-8")
).hexdigest()[:10]

TRAIN_EVENTS_OUT = DATA_DIR / f"teacher_gsm8k_train_{RUN_SLUG}_events.jsonl"
VAL_EVENTS_OUT = DATA_DIR / f"teacher_gsm8k_val_{RUN_SLUG}_events.jsonl"
TRAIN_ACCEPTED_OUT = DATA_DIR / f"teacher_gsm8k_train_{RUN_SLUG}_accepted_audit.jsonl"
VAL_ACCEPTED_OUT = DATA_DIR / f"teacher_gsm8k_val_{RUN_SLUG}_accepted_audit.jsonl"
TRAIN_REJECTED_OUT = DATA_DIR / f"teacher_gsm8k_train_{RUN_SLUG}_rejected_audit.jsonl"
VAL_REJECTED_OUT = DATA_DIR / f"teacher_gsm8k_val_{RUN_SLUG}_rejected_audit.jsonl"
TRAIN_MINIMAL_OUT = DATA_DIR / f"teacher_gsm8k_train_{RUN_SLUG}_sft.jsonl"
VAL_MINIMAL_OUT = DATA_DIR / f"teacher_gsm8k_val_{RUN_SLUG}_sft.jsonl"
TRAIN_METRICS_OUT = DATA_DIR / f"teacher_gsm8k_train_{RUN_SLUG}_metrics.json"
VAL_METRICS_OUT = DATA_DIR / f"teacher_gsm8k_val_{RUN_SLUG}_metrics.json"
SUBSET_INDICES_OUT = DATA_DIR / f"gsm8k_subset_indices_{SPLIT_IDENTITY_HASH}.json"
RUN_MANIFEST_OUT = DATA_DIR / f"run_manifest_{RUN_SLUG}.json"
RUN_METRICS_OUT = DATA_DIR / f"run_metrics_{RUN_SLUG}.json"

client = OpenAI(
    base_url=BASE_URL,
    api_key="EMPTY",
    timeout=REQUEST_TIMEOUT_SECONDS,
    max_retries=0,  # This script manages retries so every attempt is recorded.
)


# =============================================================================
# 3. FILE HELPERS
# =============================================================================

def file_ends_without_newline(path: Path) -> bool:
    """True if path is non-empty and its last byte is not a newline."""
    if not path.exists() or path.stat().st_size == 0:
        return False
    with path.open("rb") as file:
        file.seek(-1, os.SEEK_END)
        return file.read(1) != b"\n"


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append one record. Used only where duplicate keys are not possible.

    "a" mode never inserts a newline on its own: if the file's last existing
    record is missing its trailing "\\n" (e.g. an older file, or a crash that
    landed exactly on that byte), a plain append glues the new record onto
    the same line and corrupts both. Guard against that here so the fix
    lives at the one place every write goes through.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_leading_newline = file_ends_without_newline(path)
    with path.open("a", encoding="utf-8") as file:
        if needs_leading_newline:
            file.write("\n")
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
        file.flush()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL rows.

    A malformed *final* line is tolerated and dropped, since that is the
    expected shape of a process being killed mid-write. A malformed line
    anywhere else means the file is corrupted in a way that cannot be
    silently reconciled, so this raises instead of quietly losing data.
    """
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows

    with path.open("r", encoding="utf-8") as file:
        lines = file.readlines()

    nonblank_line_numbers = [
        line_number for line_number, line in enumerate(lines, start=1) if line.strip()
    ]
    last_nonblank_line_number = nonblank_line_numbers[-1] if nonblank_line_numbers else -1

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if line_number == last_nonblank_line_number:
                print(
                    f"Warning: dropping partially written final JSONL line "
                    f"{line_number} in {path}"
                )
            else:
                raise RuntimeError(
                    f"Malformed JSONL line {line_number} in {path} is not the "
                    "final line. The file is corrupted; run "
                    "repair_trailing_jsonl_record() or restore from backup "
                    "before continuing."
                )
    return rows


def repair_trailing_jsonl_record(path: Path) -> bool:
    """Rewrite path if it has a partially written trailing line or is missing
    its trailing newline.

    A record that parses fine but lacks a trailing "\\n" looks clean by row
    count alone, yet a later append_jsonl() call would glue onto it. Treat
    that as a repair case too, not just a genuinely truncated final line.

    Returns True if a repair was made, False if the file was already clean.
    Raises RuntimeError (via load_jsonl) if corruption is found before the
    final line, since that cannot be safely auto-repaired.
    """
    if not path.exists():
        return False

    with path.open("r", encoding="utf-8") as file:
        nonblank_line_count = sum(1 for line in file if line.strip())

    rows = load_jsonl(path)
    if len(rows) == nonblank_line_count and not file_ends_without_newline(path):
        return False

    atomic_write_jsonl(path, rows)
    print(f"Repaired {path}: normalized a partially written or unterminated final line.")
    return True


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Atomically replace a JSONL file to avoid partial rewrite corruption."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
        file.flush()
    temporary.replace(path)


def load_latest_state(events_path: Path) -> dict[str, dict[str, Any]]:
    """Reconstruct the latest record for each problem from the append-only log."""
    latest: dict[str, dict[str, Any]] = {}
    for record in load_jsonl(events_path):
        problem_id = record.get("problem_id")
        if problem_id:
            latest[str(problem_id)] = record
    return latest


def create_minimal_sft_record(record: dict[str, Any]) -> dict[str, Any]:
    """Keep only fields required for student SFT and traceability."""
    return {
        "problem_id": record["problem_id"],
        "question": record["question"],
        "gold_answer": record["gold_answer"],
        "teacher_final_answer": record["teacher_final_answer"],
        "student_target": record["student_target"],
        "messages": record["messages"],
    }


def compact_state_files(
    events_path: Path,
    accepted_path: Path,
    rejected_path: Path,
    minimal_path: Path,
) -> dict[str, dict[str, Any]]:
    """Build deduplicated snapshots from the latest append-only state events."""
    latest = load_latest_state(events_path)
    accepted = sorted(
        (row for row in latest.values() if row.get("status") == "accepted"),
        key=lambda row: str(row.get("problem_id", "")),
    )
    rejected = sorted(
        (row for row in latest.values() if row.get("status") == "rejected"),
        key=lambda row: str(row.get("problem_id", "")),
    )
    minimal = [create_minimal_sft_record(row) for row in accepted]

    atomic_write_jsonl(accepted_path, accepted)
    atomic_write_jsonl(rejected_path, rejected)
    atomic_write_jsonl(minimal_path, minimal)
    return latest


def count_unique_ids(path: Path) -> int:
    return len({
        str(row["problem_id"])
        for row in load_jsonl(path)
        if row.get("problem_id") is not None
    })


def compute_source_dataset_fingerprint(dataset: Any) -> str:
    """Hash the raw source dataset content (question+answer, in row order).

    SOURCE_DATASET_REVISION pins the HF repo revision used to load the
    dataset, but that alone doesn't prove the content is what a prior run
    saw (a moved tag, a re-uploaded revision, or a local cache mismatch all
    slip past it). This fingerprint is a direct content check on top of that.
    """
    hasher = hashlib.sha256()
    for item in dataset:
        hasher.update(
            json.dumps(
                {"question": item["question"], "answer": item["answer"]},
                sort_keys=True,
            ).encode("utf-8")
        )
    return hasher.hexdigest()


def write_or_verify_manifest(source_dataset_fingerprint: str) -> None:
    """Refuse to continue if an existing run manifest does not match settings."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_slug": RUN_SLUG,
        "run_mode": RUN_MODE,
        "config_hash": CONFIG_HASH,
        "config": CONFIG_PAYLOAD,
        "source_dataset_fingerprint": source_dataset_fingerprint,
        "pilot_n_train": PILOT_N_TRAIN,
        "pilot_n_val": PILOT_N_VAL,
        "retry_rejected_on_resume": RETRY_REJECTED_ON_RESUME,
    }

    if RUN_MANIFEST_OUT.exists():
        with RUN_MANIFEST_OUT.open("r", encoding="utf-8") as file:
            existing = json.load(file)
        if existing != manifest:
            raise RuntimeError(
                f"Existing manifest {RUN_MANIFEST_OUT} does not match current settings. "
                "Use a new prompt version or remove the old run files."
            )
        return

    with RUN_MANIFEST_OUT.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)


# =============================================================================
# 4. ANSWER PARSING
# =============================================================================

def extract_gsm8k_gold(answer: str) -> str:
    if "####" in answer:
        return answer.split("####")[-1].strip().replace(",", "")
    return answer.strip().split()[-1].replace(",", "")


def extract_tag(text: str, tag_name: str) -> Optional[str]:
    if not text:
        return None
    pattern = rf"<{tag_name}>\s*(.*?)\s*</{tag_name}>"
    match = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else None


def _tag_positions(text: str, tag_name: str) -> tuple[list[int], list[int]]:
    opens = [m.start() for m in re.finditer(rf"<{tag_name}>", text, re.IGNORECASE)]
    closes = [m.start() for m in re.finditer(rf"</{tag_name}>", text, re.IGNORECASE)]
    return opens, closes


def _tag_is_malformed(opens: list[int], closes: list[int]) -> bool:
    """A tag that never appears at all is "missing", handled separately by
    extract_tag()'s callers; it is not "malformed". Anything else that isn't
    exactly one clean open/close pair in the right order is malformed."""
    if not opens and not closes:
        return False
    if len(opens) != 1 or len(closes) != 1:
        return True
    return opens[0] > closes[0]


def validate_tag_structure(raw_content: str) -> list[str]:
    """Reject duplicate, incomplete, reversed, or nested reasoning/answer tags.

    extract_tag() only ever returns the first <tag>...</tag> match, so a
    response with duplicate or malformed tags would otherwise silently look
    fine. This walks tag positions directly so those cases are caught.
    """
    errors: list[str] = []
    text = raw_content or ""

    reasoning_opens, reasoning_closes = _tag_positions(text, "reasoning")
    final_opens, final_closes = _tag_positions(text, "final_answer")

    reasoning_bad = _tag_is_malformed(reasoning_opens, reasoning_closes)
    final_bad = _tag_is_malformed(final_opens, final_closes)

    # Nesting/reversed order can only be evaluated once both tags actually
    # appear exactly once each (not malformed, and not simply absent); when
    # that's true but the two spans cross or reverse, both tags are implicated.
    both_present_and_clean = (
        not reasoning_bad
        and not final_bad
        and len(reasoning_opens) == 1
        and len(final_opens) == 1
    )
    if both_present_and_clean:
        ro, rc = reasoning_opens[0], reasoning_closes[0]
        fo, fc = final_opens[0], final_closes[0]
        if ro < fo < rc or fo < ro < fc or fo < ro:
            reasoning_bad = True
            final_bad = True

    if reasoning_bad:
        errors.append("duplicate_or_malformed_reasoning_tags")
    if final_bad:
        errors.append("duplicate_or_malformed_final_answer_tags")

    return errors


def normalize_number_text(text: str) -> str:
    """Remove formatting while preserving a trailing percent sign."""
    return (
        text.strip()
        .replace(",", "")
        .replace("$", "")
        .replace("€", "")
        .replace("£", "")
        .replace("\\boxed{", "")
        .replace("}", "")
        .strip()
    )


def is_strict_numeric_text(text: str) -> bool:
    """Return True only when the entire field is one normalized number."""
    cleaned = normalize_number_text(text)
    pattern = (
        r"[+-]?\d+\s+\d+\s*/\s*\d+"
        r"|[+-]?\d+\s*/\s*[+-]?\d+"
        r"|[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
    )
    return re.fullmatch(pattern, cleaned) is not None


def extract_number(text: str, *, prefer_last: bool = False) -> Optional[str]:
    """Extract integers, decimals, fractions, scientific notation, or percents."""
    if not text:
        return None

    cleaned = normalize_number_text(text)
    pattern = (
        r"[+-]?\d+\s+\d+\s*/\s*\d+%?"
        r"|[+-]?\d+\s*/\s*[+-]?\d+%?"
        r"|[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?%?"
    )
    matches = re.findall(pattern, cleaned)
    if not matches:
        return None

    selected = matches[-1] if prefer_last else matches[0]
    # Collapse whitespace instead of stripping it outright: a mixed number like
    # "1 1/2" needs the space between the whole part and the fraction to
    # survive, or it silently becomes "11/2".
    selected = re.sub(r"\s+", " ", selected).strip()
    return re.sub(r"\s*/\s*", "/", selected)


def extract_fallback_answer(text: str) -> Optional[str]:
    """Prefer explicit answer phrases; otherwise use the final number."""
    if not text:
        return None

    patterns = [
        r"<final_answer>\s*(.*?)\s*</final_answer>",
        r"(?:final\s+answer|answer)\s*(?:is|:|=)\s*([^\n<]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            value = extract_number(match.group(1), prefer_last=False)
            if value is not None:
                return value

    return extract_number(text, prefer_last=True)


def convert_to_fraction(value: Optional[str]) -> Optional[Fraction]:
    if value is None:
        return None

    normalized = value.strip()
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


def answers_match(teacher_answer: Optional[str], gold_answer: str) -> bool:
    teacher_value = convert_to_fraction(teacher_answer)
    gold_value = convert_to_fraction(gold_answer)
    return (
        teacher_value is not None
        and gold_value is not None
        and teacher_value == gold_value
    )


# =============================================================================
# 5. OUTPUT VALIDATION
# =============================================================================

def normalized_nonempty_lines(text: str) -> list[str]:
    return [
        re.sub(r"\s+", " ", line.strip().lower())
        for line in text.splitlines()
        if line.strip()
    ]


def has_excessive_line_repetition(reasoning: str) -> bool:
    counts = Counter(normalized_nonempty_lines(reasoning))
    return any(count > MAX_REPEATED_LINE_COUNT for count in counts.values())


def has_excessive_sentence_repetition(reasoning: str) -> bool:
    sentences = [
        re.sub(r"\s+", " ", sentence.strip().lower())
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", reasoning)
        if len(sentence.strip()) >= 12
    ]
    counts = Counter(sentences)
    return any(count > MAX_REPEATED_SENTENCE_COUNT for count in counts.values())


def count_numbered_steps(reasoning: str) -> int:
    return len(re.findall(r"(?im)^\s*(?:step\s*)?\d+\s*[:.)-]", reasoning))


# Requires an operator or "=" directly between two numbers (e.g. "5 + 3 = 8"),
# not just any digit. A restated total like "the answer is 8 apples" has
# digits but no evidence of an actual calculation, and should not pass.
CALCULATION_EVIDENCE_PATTERN = re.compile(r"\d\s*[+\-*/=×÷]\s*\d")


def text_outside_required_tags(raw_content: str) -> str:
    cleaned = re.sub(
        r"<reasoning>.*?</reasoning>",
        "",
        raw_content,
        flags=re.DOTALL | re.IGNORECASE,
    )
    cleaned = re.sub(
        r"<final_answer>.*?</final_answer>",
        "",
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return cleaned.strip()


def normalize_reasoning_for_hash(reasoning: str) -> str:
    return re.sub(r"\s+", " ", reasoning.strip().lower())


def reasoning_hash(reasoning: str) -> str:
    return hashlib.sha256(
        normalize_reasoning_for_hash(reasoning).encode("utf-8")
    ).hexdigest()


def validate_teacher_output(
    raw_content: str,
    raw_reasoning_content: str,
    finish_reason: Optional[str],
    gold_answer: str,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    raw_content = raw_content or ""
    raw_reasoning_content = raw_reasoning_content or ""

    if finish_reason == "length":
        errors.append("truncated_output")

    errors.extend(validate_tag_structure(raw_content))

    # A <final_answer> tag is always required, mirroring the reasoning tag
    # requirement below. extract_fallback_answer() is still run when the tag
    # is missing so the audit record shows what a loose parse would have
    # found, but it never makes an untagged response valid.
    tagged_answer = extract_tag(raw_content, "final_answer")
    if tagged_answer is not None:
        teacher_final_answer = extract_number(tagged_answer)
        answer_source = "final_answer_tag"
        if not is_strict_numeric_text(tagged_answer):
            errors.append("non_numeric_text_inside_final_answer_tag")
    else:
        errors.append("missing_tagged_final_answer")
        teacher_final_answer = extract_fallback_answer(raw_content)
        answer_source = "fallback"

    if teacher_final_answer is None:
        errors.append("missing_or_invalid_numeric_answer")

    # Tagged reasoning is always required. raw_reasoning_content (the model's
    # internal thinking) is never promoted into teacher_reasoning: it is kept
    # only in the raw audit fields, never in the SFT training target.
    tagged_reasoning = extract_tag(raw_content, "reasoning")
    if tagged_reasoning:
        teacher_reasoning = tagged_reasoning.strip()
        reasoning_source = "reasoning_tag"
    else:
        teacher_reasoning = ""
        reasoning_source = "missing"
        errors.append("missing_tagged_reasoning")
        if raw_reasoning_content.strip():
            warnings.append("raw_reasoning_content_present_but_not_used")

    if len(teacher_reasoning) > MAX_REASONING_CHARACTERS:
        errors.append("reasoning_too_long")

    word_count = len(teacher_reasoning.split())
    if teacher_reasoning and (
        len(teacher_reasoning) < MIN_REASONING_CHARACTERS
        or word_count < MIN_REASONING_WORDS
    ):
        errors.append("reasoning_too_short")

    if (
        teacher_reasoning
        and REQUIRE_NUMERIC_CALCULATION_IN_REASONING
        and not CALCULATION_EVIDENCE_PATTERN.search(teacher_reasoning)
    ):
        errors.append("reasoning_lacks_calculation_detail")

    line_count = len(
        [line for line in teacher_reasoning.splitlines() if line.strip()]
    )
    if line_count > MAX_REASONING_LINES:
        errors.append("too_many_reasoning_lines")

    if teacher_reasoning and has_excessive_line_repetition(teacher_reasoning):
        errors.append("repetitive_reasoning_lines")
    if teacher_reasoning and has_excessive_sentence_repetition(teacher_reasoning):
        errors.append("repetitive_reasoning_sentences")

    step_count = count_numbered_steps(teacher_reasoning)
    if step_count and not (
        MIN_RECOMMENDED_STEPS <= step_count <= MAX_RECOMMENDED_STEPS
    ):
        warnings.append("step_count_outside_recommended_range")
    elif step_count == 0:
        warnings.append("no_numbered_steps_detected")

    outside_text = text_outside_required_tags(raw_content)
    if outside_text:
        warnings.append("text_outside_required_tags")
        if (
            REJECT_SIGNIFICANT_TEXT_OUTSIDE_TAGS
            and tagged_reasoning is not None
            and tagged_answer is not None
            and len(outside_text) > OUTSIDE_TEXT_WARNING_CHAR_LIMIT
        ):
            errors.append("significant_text_outside_required_tags")

    is_correct = answers_match(teacher_final_answer, gold_answer)
    if teacher_final_answer is not None and not is_correct:
        errors.append("wrong_answer")

    return {
        "is_valid": not errors,
        "validation_errors": errors,
        "format_warnings": warnings,
        "teacher_reasoning": teacher_reasoning,
        "teacher_final_answer": teacher_final_answer,
        "is_correct": is_correct,
        "answer_source": answer_source,
        "reasoning_source": reasoning_source,
        "reasoning_character_count": len(teacher_reasoning),
        "reasoning_word_count": word_count,
        "reasoning_line_count": line_count,
        "numbered_step_count": step_count,
        "reasoning_hash": reasoning_hash(teacher_reasoning) if teacher_reasoning else None,
    }


# =============================================================================
# 6. RECORD CREATION AND METRICS
# =============================================================================

def create_student_target(reasoning: str, final_answer: str) -> str:
    return (
        f"<reasoning>\n{reasoning.strip()}\n</reasoning>\n"
        f"<final_answer>{final_answer}</final_answer>"
    )


def get_usage(response: Any) -> dict[str, Optional[int]]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
        }
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def aggregate_attempt_usage(
    attempt_history: list[dict[str, Any]],
) -> dict[str, Optional[int]]:
    """Sum token usage across all attempts, not only the final attempt."""
    aggregate: dict[str, Optional[int]] = {}
    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        values = [
            attempt[field]
            for attempt in attempt_history
            if isinstance(attempt.get(field), int)
        ]
        aggregate[field] = sum(values) if values else None
    return aggregate


def base_metadata(
    problem_id: str,
    role: str,
    source_index: int,
    question: str,
    gold: str,
) -> dict[str, Any]:
    return {
        "problem_id": problem_id,
        "source_dataset": SOURCE_DATASET,
        "source_config": SOURCE_CONFIG,
        "source_split": SOURCE_SPLIT,
        "subset": role,
        "source_index": source_index,
        "question": question,
        "gold_answer": gold,
        "teacher_model": MODEL,
        "prompt_version": PROMPT_VERSION,
        "config_hash": CONFIG_HASH,
        "run_mode": RUN_MODE,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "thinking_enabled": ENABLE_THINKING,
    }


def create_accepted_record(
    *,
    problem_id: str,
    role: str,
    source_index: int,
    question: str,
    gold_answer: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    validation = result["validation"]
    reasoning = validation["teacher_reasoning"]
    final_answer = validation["teacher_final_answer"]
    student_target = create_student_target(reasoning, final_answer)

    record = base_metadata(problem_id, role, source_index, question, gold_answer)
    record.update(
        {
            "teacher_reasoning": reasoning,
            "teacher_final_answer": final_answer,
            "is_correct": True,
            "student_target": student_target,
            "messages": [
                {"role": "system", "content": STUDENT_SYSTEM_PROMPT},
                {"role": "user", "content": question},
                {"role": "assistant", "content": student_target},
            ],
            # Raw fields are audit-only. Students train from the minimal SFT file.
            "raw_teacher_content": result["raw_content"],
            "raw_reasoning_content": result["raw_reasoning_content"],
            "finish_reason": result["finish_reason"],
            "status": "accepted",
            "validation_errors": validation["validation_errors"],
            "format_warnings": validation["format_warnings"],
            "answer_source": validation["answer_source"],
            "reasoning_source": validation["reasoning_source"],
            "reasoning_character_count": validation["reasoning_character_count"],
            "reasoning_word_count": validation["reasoning_word_count"],
            "reasoning_line_count": validation["reasoning_line_count"],
            "numbered_step_count": validation["numbered_step_count"],
            "reasoning_hash": validation["reasoning_hash"],
            "attempts": result["attempt"],
            "attempt_history": result["attempt_history"],
            "elapsed_seconds": round(result["total_elapsed_seconds"], 3),
            **result["usage"],
        }
    )
    return record


def create_rejected_record(
    *,
    problem_id: str,
    role: str,
    source_index: int,
    question: str,
    gold_answer: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    validation = result["validation"]
    record = base_metadata(problem_id, role, source_index, question, gold_answer)
    record.update(
        {
            "teacher_reasoning": validation.get("teacher_reasoning", ""),
            "teacher_final_answer": validation.get("teacher_final_answer"),
            "is_correct": validation.get("is_correct", False),
            "raw_teacher_content": result["raw_content"],
            "raw_reasoning_content": result["raw_reasoning_content"],
            "finish_reason": result["finish_reason"],
            "status": "rejected",
            "validation_errors": validation.get(
                "validation_errors", ["generation_failed"]
            ),
            "format_warnings": validation.get("format_warnings", []),
            "answer_source": validation.get("answer_source"),
            "reasoning_source": validation.get("reasoning_source"),
            "reasoning_hash": validation.get("reasoning_hash"),
            "error_message": result["error_message"],
            "attempts": result["attempt"],
            "attempt_history": result["attempt_history"],
            "elapsed_seconds": round(result["total_elapsed_seconds"], 3),
            **result["usage"],
        }
    )
    return record


def compute_generation_round(prior_record: Optional[dict[str, Any]]) -> int:
    """1 for a first attempt at a question; incremented each time a prior
    rejected record is regenerated on resume."""
    if prior_record and prior_record.get("status") == "rejected":
        return int(prior_record.get("generation_round", 1)) + 1
    return 1


def add_lifetime_metrics(
    record: dict[str, Any],
    prior_record: Optional[dict[str, Any]],
) -> dict[str, Any]:
    """Carry cumulative retry/time/token totals across interrupted reruns."""
    prior = prior_record if prior_record and prior_record.get("status") == "rejected" else None

    def previous_total(field: str, fallback_field: str) -> int:
        if not prior:
            return 0
        value = prior.get(field, prior.get(fallback_field, 0))
        return int(value) if isinstance(value, int) else 0

    def current_total(field: str) -> int:
        value = record.get(field)
        return int(value) if isinstance(value, int) else 0

    prior_elapsed = 0.0
    if prior:
        value = prior.get("lifetime_elapsed_seconds", prior.get("elapsed_seconds", 0.0))
        if isinstance(value, (int, float)):
            prior_elapsed = float(value)

    record["generation_round"] = compute_generation_round(prior_record)
    record["lifetime_attempts"] = (
        previous_total("lifetime_attempts", "attempts")
        + current_total("attempts")
    )
    record["lifetime_elapsed_seconds"] = round(
        prior_elapsed + float(record.get("elapsed_seconds", 0.0)), 3
    )

    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        lifetime_field = f"lifetime_{field}"
        total = previous_total(lifetime_field, field) + current_total(field)
        record[lifetime_field] = total if total > 0 else None

    return record


def write_subset_metrics(
    latest_state: dict[str, dict[str, Any]],
    path: Path,
    candidate_count: int,
) -> dict[str, Any]:
    """Aggregate lifetime totals for one subset (train/val) and write them as JSON.

    Takes the in-memory latest-state dict directly (problem_id -> record),
    the same shape generate_subset() already builds, so no extra file re-read
    is needed. Uses the lifetime_* fields (see add_lifetime_metrics), not the
    plain attempts/tokens/elapsed_seconds fields, since the latter only
    reflect the most recent generation round: an item resumed and
    regenerated after a rejection would otherwise have its earlier rounds'
    cost silently dropped.
    """
    rows = list(latest_state.values())
    accepted = [row for row in rows if row.get("status") == "accepted"]
    rejected = [row for row in rows if row.get("status") == "rejected"]

    def total(field: str) -> float:
        return sum(
            row[field] for row in rows if isinstance(row.get(field), (int, float))
        )

    total_attempts = total("lifetime_attempts")
    total_elapsed_seconds = total("lifetime_elapsed_seconds")
    total_prompt_tokens = total("lifetime_prompt_tokens")
    total_completion_tokens = total("lifetime_completion_tokens")

    metrics = {
        "accepted": len(accepted),
        "rejected": len(rejected),
        "total_attempts": total_attempts,
        "total_elapsed_seconds": round(total_elapsed_seconds, 3),
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_prompt_tokens + total_completion_tokens,
        "completion_tokens_per_second": (
            round(total_completion_tokens / total_elapsed_seconds, 3)
            if total_elapsed_seconds
            else None
        ),
        "acceptance_rate": (
            round(len(accepted) / candidate_count, 3) if candidate_count else None
        ),
        "top_rejection_reasons": Counter(
            error
            for row in rejected
            for error in row.get("validation_errors", [])
        ).most_common(10),
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    temporary.replace(path)

    return metrics


def compute_dataset_fingerprint(*minimal_paths: Path) -> str:
    """Hash the released SFT content so any change to the dataset is detectable."""
    hasher = hashlib.sha256()
    for path in minimal_paths:
        rows = sorted(load_jsonl(path), key=lambda row: str(row.get("problem_id", "")))
        for row in rows:
            hasher.update(
                json.dumps(
                    {
                        "problem_id": row.get("problem_id"),
                        "question": row.get("question"),
                        "student_target": row.get("student_target"),
                    },
                    sort_keys=True,
                ).encode("utf-8")
            )
    return hasher.hexdigest()


def write_run_metrics(
    path: Path,
    *,
    train_metrics: dict[str, Any],
    val_metrics: dict[str, Any],
    dataset_fingerprint: str,
) -> None:
    """Write a single dedicated metrics/fingerprint JSON file for the run."""
    payload = {
        "run_slug": RUN_SLUG,
        "run_mode": RUN_MODE,
        "config_hash": CONFIG_HASH,
        "dataset_version": RUN_SLUG,
        "dataset_fingerprint": dataset_fingerprint,
        "generated_at": time.time(),
        "train": train_metrics,
        "val": val_metrics,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    temporary.replace(path)


# =============================================================================
# 7. SERVER CHECK AND GENERATION
# =============================================================================

def check_server() -> None:
    print(f"Checking local model server at {BASE_URL} ...")
    models = client.models.list()
    model_ids = [item.id for item in models.data]
    print(f"Available models: {model_ids}")

    if MODEL not in model_ids:
        raise RuntimeError(
            f"Configured model '{MODEL}' is not available from /v1/models. "
            f"Available models: {model_ids}. Set MODEL to the exact served-model name."
        )


def load_reasoning_hashes(path: Path) -> set[str]:
    return {
        str(row["reasoning_hash"])
        for row in load_jsonl(path)
        if row.get("reasoning_hash")
    }


def derive_seed(source_index: int, attempt: int) -> int:
    """Deterministic per-question, per-attempt seed so reruns are reproducible.

    attempt stays within MAX_ATTEMPTS (a small single-digit count), so
    multiplying source_index by 100 keeps every question's attempts in their
    own block with no risk of collision with a neighboring source_index.
    """
    return SEED + source_index * 100 + attempt


def generate_one_example(
    question: str,
    gold_answer: str,
    forbidden_reasoning_hashes: set[str],
    source_index: int,
) -> dict[str, Any]:
    last_validation: dict[str, Any] = {
        "is_valid": False,
        "validation_errors": ["not_generated"],
        "format_warnings": [],
        "teacher_reasoning": "",
        "teacher_final_answer": None,
        "is_correct": False,
        "reasoning_hash": None,
    }
    last_raw_content = ""
    last_raw_reasoning = ""
    last_finish_reason: Optional[str] = None
    last_error: Optional[str] = None
    attempt_history: list[dict[str, Any]] = []
    total_start = time.time()

    for attempt in range(1, MAX_ATTEMPTS + 1):
        attempt_start = time.time()
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Question:\n{question}"},
                ],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                seed=derive_seed(source_index, attempt),
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": ENABLE_THINKING}
                },
            )

            # A successful response supersedes any error from an earlier
            # attempt; otherwise a stale error_message from attempt 1 could
            # linger on a record built from a later, successful attempt.
            last_error = None

            choice = response.choices[0]
            message = choice.message
            last_raw_content = message.content or ""
            last_raw_reasoning = (
                getattr(message, "reasoning_content", None) or ""
            )
            last_finish_reason = getattr(choice, "finish_reason", None)
            current_usage = get_usage(response)
            last_validation = validate_teacher_output(
                last_raw_content,
                last_raw_reasoning,
                last_finish_reason,
                gold_answer,
            )

            current_hash = last_validation.get("reasoning_hash")
            if (
                last_validation["is_valid"]
                and current_hash
                and current_hash in forbidden_reasoning_hashes
            ):
                last_validation["validation_errors"].append(
                    "duplicate_reasoning_output"
                )
                last_validation["is_valid"] = False

            attempt_elapsed = time.time() - attempt_start
            attempt_history.append(
                {
                    "attempt": attempt,
                    "elapsed_seconds": round(attempt_elapsed, 3),
                    "finish_reason": last_finish_reason,
                    "validation_errors": list(
                        last_validation["validation_errors"]
                    ),
                    "format_warnings": list(
                        last_validation["format_warnings"]
                    ),
                    "teacher_final_answer": last_validation.get(
                        "teacher_final_answer"
                    ),
                    "answer_source": last_validation.get("answer_source"),
                    "reasoning_source": last_validation.get("reasoning_source"),
                    "reasoning_character_count": last_validation.get(
                        "reasoning_character_count"
                    ),
                    "reasoning_hash": last_validation.get("reasoning_hash"),
                    **current_usage,
                }
            )

            if last_validation["is_valid"]:
                return {
                    "accepted": True,
                    "attempt": attempt,
                    "validation": last_validation,
                    "raw_content": last_raw_content,
                    "raw_reasoning_content": last_raw_reasoning,
                    "finish_reason": last_finish_reason,
                    "error_message": None,
                    "total_elapsed_seconds": time.time() - total_start,
                    "usage": aggregate_attempt_usage(attempt_history),
                    "attempt_history": attempt_history,
                }

            print(
                f"    Attempt {attempt} rejected: "
                f"{last_validation['validation_errors']}"
            )

        except Exception as error:  # Keep a long run alive after isolated failures.
            last_error = str(error)
            attempt_elapsed = time.time() - attempt_start
            # This attempt produced no response at all, so any output kept
            # from an earlier attempt must not be reported as if it belonged
            # to this (failed) attempt.
            last_raw_content = ""
            last_raw_reasoning = ""
            last_finish_reason = None
            last_validation = {
                "is_valid": False,
                "validation_errors": ["server_or_client_error"],
                "format_warnings": [],
                "teacher_reasoning": "",
                "teacher_final_answer": None,
                "is_correct": False,
                "reasoning_hash": None,
            }
            attempt_history.append(
                {
                    "attempt": attempt,
                    "elapsed_seconds": round(attempt_elapsed, 3),
                    "finish_reason": None,
                    "validation_errors": ["server_or_client_error"],
                    "format_warnings": [],
                    "error_message": last_error,
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "total_tokens": None,
                }
            )
            print(f"    Attempt {attempt} failed: {error}")

        if attempt < MAX_ATTEMPTS:
            time.sleep(RETRY_DELAY_SECONDS * attempt)

    return {
        "accepted": False,
        "attempt": MAX_ATTEMPTS,
        "validation": last_validation,
        "raw_content": last_raw_content,
        "raw_reasoning_content": last_raw_reasoning,
        "finish_reason": last_finish_reason,
        "error_message": last_error,
        "total_elapsed_seconds": time.time() - total_start,
        "usage": aggregate_attempt_usage(attempt_history),
        "attempt_history": attempt_history,
    }


# =============================================================================
# 8. SUBSET PROCESSING
# =============================================================================

def generate_subset(
    dataset: Any,
    indices: list[int],
    role: str,
    events_out: Path,
    accepted_out: Path,
    rejected_out: Path,
    minimal_out: Path,
    metrics_out: Path,
    external_forbidden_reasoning_hashes: Optional[set[str]] = None,
) -> tuple[set[str], dict[str, Any]]:
    # The append-only event log is the durable source of truth. Snapshots are
    # rebuilt from the latest event for each problem ID.
    latest_state = compact_state_files(
        events_out, accepted_out, rejected_out, minimal_out
    )
    accepted_ids = {
        problem_id
        for problem_id, row in latest_state.items()
        if row.get("status") == "accepted"
    }
    rejected_ids = {
        problem_id
        for problem_id, row in latest_state.items()
        if row.get("status") == "rejected"
    }
    completed_ids = (
        accepted_ids
        if RETRY_REJECTED_ON_RESUME
        else accepted_ids | rejected_ids
    )
    accepted_reasoning_hashes = {
        str(row["reasoning_hash"])
        for row in latest_state.values()
        if row.get("status") == "accepted" and row.get("reasoning_hash")
    }
    if external_forbidden_reasoning_hashes:
        accepted_reasoning_hashes.update(external_forbidden_reasoning_hashes)

    print("\n" + "=" * 80)
    print(
        f"Subset: {role}; candidates={len(indices)}; "
        f"accepted_existing={len(accepted_ids)}; "
        f"rejected_existing={len(rejected_ids)}"
    )
    print(f"State events:   {events_out}")
    print(f"Accepted audit: {accepted_out}")
    print(f"Rejected audit: {rejected_out}")
    print(f"Minimal SFT:    {minimal_out}")
    print("=" * 80)

    start_time = time.time()
    processed_new = accepted_new = rejected_new = 0

    for position, source_index in enumerate(indices, start=1):
        problem_id = f"gsm8k_{role}_{source_index}"
        if problem_id in completed_ids:
            continue

        prior_record = latest_state.get(problem_id)
        question = dataset[source_index]["question"]
        gold_answer = extract_gsm8k_gold(dataset[source_index]["answer"])
        print(f"\n[{role} {position}/{len(indices)}] {problem_id}")

        result = generate_one_example(
            question,
            gold_answer,
            accepted_reasoning_hashes,
            source_index,
        )

        if result["accepted"]:
            record = create_accepted_record(
                problem_id=problem_id,
                role=role,
                source_index=source_index,
                question=question,
                gold_answer=gold_answer,
                result=result,
            )
            record = add_lifetime_metrics(record, prior_record)
            accepted_reasoning_hashes.add(record["reasoning_hash"])
            accepted_new += 1
            print(
                f"    Accepted answer={record['teacher_final_answer']}; "
                f"attempt={record['attempts']}"
            )
        else:
            record = create_rejected_record(
                problem_id=problem_id,
                role=role,
                source_index=source_index,
                question=question,
                gold_answer=gold_answer,
                result=result,
            )
            record = add_lifetime_metrics(record, prior_record)
            rejected_new += 1
            print(f"    Rejected: {record['validation_errors']}")

        # Append immediately so a disconnect loses at most the active request.
        append_jsonl(events_out, record)
        latest_state[problem_id] = record
        processed_new += 1

        # Periodic compact snapshots make audit/SFT files usable during a run.
        if processed_new % SNAPSHOT_EVERY_N == 0:
            compact_state_files(
                events_out, accepted_out, rejected_out, minimal_out
            )
            average = (time.time() - start_time) / processed_new
            print(
                f"Progress new={processed_new}, accepted={accepted_new}, "
                f"rejected={rejected_new}, avg={average:.1f}s/example"
            )

    latest_state = compact_state_files(
        events_out, accepted_out, rejected_out, minimal_out
    )
    final_accepted = sum(
        row.get("status") == "accepted" for row in latest_state.values()
    )
    final_rejected = sum(
        row.get("status") == "rejected" for row in latest_state.values()
    )

    print("\n" + "-" * 80)
    print(f"Completed {role}")
    print(
        f"New processed: {processed_new}; accepted: {accepted_new}; "
        f"rejected: {rejected_new}"
    )
    print(f"Unique accepted: {final_accepted}")
    print(f"Unique final rejected: {final_rejected}")
    print(f"Minimal SFT rows: {count_unique_ids(minimal_out)}")
    print(f"Elapsed: {(time.time() - start_time) / 3600:.2f} hours")
    print("-" * 80)

    metrics = write_subset_metrics(latest_state, metrics_out, len(indices))
    print(f"Metrics: {metrics_out}")
    return accepted_reasoning_hashes, metrics


# =============================================================================
# 9. REPRODUCIBLE SPLITS
# =============================================================================

def normalized_question(question: str) -> str:
    return re.sub(r"\W+", " ", question.lower()).strip()


def expected_split_metadata(source_dataset_fingerprint: str) -> dict[str, Any]:
    # SPLIT_IDENTITY_PAYLOAD already governs the cache filename (see its
    # module-level definition), so a mismatched revision can't silently load
    # someone else's file. The content fingerprint is a second, independent
    # check: SOURCE_DATASET_REVISION can be left unpinned (None), in which
    # case the filename alone can't detect the underlying content changing
    # between runs.
    return {
        **SPLIT_IDENTITY_PAYLOAD,
        "source_dataset_fingerprint": source_dataset_fingerprint,
    }


def validate_indices(
    dataset: Any,
    train_indices: list[int],
    val_indices: list[int],
) -> None:
    """Raise if the split is unsafe: bad index types/ranges, duplicates,
    train/val overlap, or the same normalized question appearing in both
    splits (which would let a memorized answer leak from train into val)."""
    dataset_size = len(dataset)

    for name, indices in (("train", train_indices), ("val", val_indices)):
        for value in indices:
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(
                    f"Non-integer index {value!r} found in {name}_indices."
                )
            if value < 0 or value >= dataset_size:
                raise ValueError(
                    f"Index {value} in {name}_indices is out of range for a "
                    f"dataset of size {dataset_size}."
                )

        duplicates = sorted(
            {value for value, count in Counter(indices).items() if count > 1}
        )
        if duplicates:
            raise ValueError(
                f"Duplicate indices found within {name}_indices: {duplicates}"
            )

    overlap = sorted(set(train_indices) & set(val_indices))
    if overlap:
        raise ValueError(f"train_indices and val_indices overlap: {overlap}")

    # Track every (split, index) that produced each normalized question, not
    # just which split names touched it: two different indices landing in the
    # same split with the same normalized question are duplicates too, and a
    # set keyed only by split name would collapse them into "one split" and
    # miss it.
    normalized_occurrences: dict[str, list[tuple[str, int]]] = {}
    for name, indices in (("train", train_indices), ("val", val_indices)):
        for index in indices:
            normalized = normalized_question(dataset[index]["question"])
            normalized_occurrences.setdefault(normalized, []).append((name, index))

    duplicated = sorted(
        normalized
        for normalized, occurrences in normalized_occurrences.items()
        if len(occurrences) > 1
    )
    if duplicated:
        raise ValueError(
            "Duplicate normalized questions found (within a split or across "
            f"train/val) (showing up to 5 of {len(duplicated)}): {duplicated[:5]}"
        )


def create_or_load_indices(
    dataset: Any, source_dataset_fingerprint: str
) -> dict[str, list[int]]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    expected = expected_split_metadata(source_dataset_fingerprint)

    if SUBSET_INDICES_OUT.exists():
        with SUBSET_INDICES_OUT.open("r", encoding="utf-8") as file:
            saved = json.load(file)

        for key, expected_value in expected.items():
            if saved.get(key) != expected_value:
                raise ValueError(
                    f"Saved split metadata mismatch for '{key}': "
                    f"found {saved.get(key)!r}, expected {expected_value!r}. "
                    f"Rename or delete {SUBSET_INDICES_OUT} to create a new split."
                )

        train_indices = saved.get("train_indices", [])
        val_indices = saved.get("val_indices", [])
        if len(train_indices) != N_TRAIN or len(val_indices) != N_VAL:
            raise ValueError(
                f"Saved index counts do not match N_TRAIN/N_VAL in "
                f"{SUBSET_INDICES_OUT}."
            )
        validate_indices(dataset, train_indices, val_indices)
        return {
            "train_indices": train_indices,
            "val_indices": val_indices,
        }

    required = N_TRAIN + N_VAL
    if required > len(dataset):
        raise ValueError(
            f"Requested {required} examples but dataset contains {len(dataset)}"
        )

    # Exact normalized duplicates are grouped so they cannot cross train/val.
    groups: dict[str, list[int]] = {}
    for index, item in enumerate(dataset):
        groups.setdefault(
            normalized_question(item["question"]), []
        ).append(index)

    group_values = list(groups.values())
    rng = random.Random(SEED)
    rng.shuffle(group_values)
    selected: list[int] = []

    for group in group_values:
        selected.append(group[0])
        if len(selected) == required:
            break

    if len(selected) < required:
        raise ValueError(
            "Not enough unique normalized questions for the requested split"
        )

    train_indices = selected[:N_TRAIN]
    val_indices = selected[N_TRAIN:]
    validate_indices(dataset, train_indices, val_indices)
    saved = {
        **expected,
        "train_indices": train_indices,
        "val_indices": val_indices,
    }
    with SUBSET_INDICES_OUT.open("w", encoding="utf-8") as file:
        json.dump(saved, file, indent=2)

    return {
        "train_indices": train_indices,
        "val_indices": val_indices,
    }


# =============================================================================
# 10. MAIN
# =============================================================================

def main() -> None:
    check_server()

    print("Loading GSM8K...")
    dataset = load_dataset(
        SOURCE_DATASET,
        SOURCE_CONFIG,
        split=SOURCE_SPLIT,
        revision=SOURCE_DATASET_REVISION,
    )
    print(f"Loaded {len(dataset)} examples.")

    # Computed once and reused for both the run manifest and the split-index
    # cache, so a mismatch (changed settings, or an upstream dataset change
    # under an unpinned revision) is caught immediately in either place.
    source_dataset_fingerprint = compute_source_dataset_fingerprint(dataset)
    write_or_verify_manifest(source_dataset_fingerprint)

    # Repair any partially written trailing line left by a killed prior run
    # before treating the event logs as the source of truth for resume.
    repair_trailing_jsonl_record(TRAIN_EVENTS_OUT)
    repair_trailing_jsonl_record(VAL_EVENTS_OUT)

    indices = create_or_load_indices(dataset, source_dataset_fingerprint)
    train_indices = indices["train_indices"]
    val_indices = indices["val_indices"]

    if PILOT_MODE:
        train_indices = train_indices[:PILOT_N_TRAIN]
        val_indices = val_indices[:PILOT_N_VAL]
        print(
            f"PILOT MODE: train={len(train_indices)}, "
            f"val={len(val_indices)}"
        )

    train_reasoning_hashes, train_metrics = generate_subset(
        dataset,
        train_indices,
        "train",
        TRAIN_EVENTS_OUT,
        TRAIN_ACCEPTED_OUT,
        TRAIN_REJECTED_OUT,
        TRAIN_MINIMAL_OUT,
        TRAIN_METRICS_OUT,
    )
    _, val_metrics = generate_subset(
        dataset,
        val_indices,
        "val",
        VAL_EVENTS_OUT,
        VAL_ACCEPTED_OUT,
        VAL_REJECTED_OUT,
        VAL_MINIMAL_OUT,
        VAL_METRICS_OUT,
        external_forbidden_reasoning_hashes=train_reasoning_hashes,
    )

    dataset_fingerprint = compute_dataset_fingerprint(TRAIN_MINIMAL_OUT, VAL_MINIMAL_OUT)
    write_run_metrics(
        RUN_METRICS_OUT,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        dataset_fingerprint=dataset_fingerprint,
    )

    print("\nGeneration complete.")
    print(f"Train state events:   {TRAIN_EVENTS_OUT}")
    print(f"Accepted train audit: {TRAIN_ACCEPTED_OUT}")
    print(f"Student SFT train:    {TRAIN_MINIMAL_OUT}")
    print(f"Val state events:     {VAL_EVENTS_OUT}")
    print(f"Accepted val audit:   {VAL_ACCEPTED_OUT}")
    print(f"Student SFT val:      {VAL_MINIMAL_OUT}")
    print(f"Run metrics:          {RUN_METRICS_OUT}")
    print(f"Dataset fingerprint:  {dataset_fingerprint}")
    print(
        "The official GSM8K test split remains untouched for final "
        "before/after student evaluation."
    )


if __name__ == "__main__":
    main()