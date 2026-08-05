"""Rescue $ false-rejects into SFT after re-running validate_teacher_output.

Updates for each split (train/val):
  - *_events.jsonl          (append accepted rescue event)
  - *_accepted_audit.jsonl  (rebuild)
  - *_rejected_audit.jsonl  (rebuild)
  - *_sft.jsonl             (rebuild)
  - *_metrics.json          (rebuild)
Then:
  - run_metrics_*.json      (rebuild + new fingerprint)
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path
import generate__teacher_gsm8k as gen

DATA_DIR = Path("data_final")
SLUG = "qwen3_14b_awq_gsm8k_teacher_v3_9cbf703286_full"

# Proposed looser calc check (allows $ between operator and digit).
# STRICT is taken from gen.CALCULATION_EVIDENCE_PATTERN at runtime.
LOOSE_DOLLAR = re.compile(r"\d\s*[+\-*/=×÷]\s*\$?\s*\d")

def is_dollar_false_reject(row: dict, strict: re.Pattern) -> bool:
    errs = set(row.get("validation_errors") or [])
    text = row.get("teacher_reasoning") or ""
    return (
        bool(row.get("is_correct"))
        and "reasoning_lacks_calculation_detail" in errs
        and "wrong_answer" not in errs
        and not strict.search(text)
        and bool(LOOSE_DOLLAR.search(text))
    )


def build_accepted_from_reject(gen, row: dict, validation: dict) -> dict:
    """Promote a rejected audit row to a full accepted record (with SFT fields)."""
    reasoning = validation["teacher_reasoning"]
    final_answer = validation["teacher_final_answer"]
    student_target = gen.create_student_target(reasoning, final_answer)

    accepted = dict(row)  # keep lifetime/usage/attempt history
    accepted.update(
        {
            "status": "accepted",
            "validation_errors": validation["validation_errors"],
            "format_warnings": validation["format_warnings"],
            "teacher_reasoning": reasoning,
            "teacher_final_answer": final_answer,
            "is_correct": True,
            "student_target": student_target,
            "messages": [
                {"role": "system", "content": gen.STUDENT_SYSTEM_PROMPT},
                {"role": "user", "content": row["question"]},
                {"role": "assistant", "content": student_target},
            ],
            "answer_source": validation["answer_source"],
            "reasoning_source": validation["reasoning_source"],
            "reasoning_character_count": validation["reasoning_character_count"],
            "reasoning_word_count": validation["reasoning_word_count"],
            "reasoning_line_count": validation["reasoning_line_count"],
            "numbered_step_count": validation["numbered_step_count"],
            "reasoning_hash": validation["reasoning_hash"],
            "error_message": None,
            "rescued_from": "dollar_calc_regex",
            "rescued_at": time.time(),
        }
    )
    return accepted


def revalidate(gen, row: dict, pattern: re.Pattern) -> dict:
    """Run the generator's validator with a temporary calc pattern."""
    old = gen.CALCULATION_EVIDENCE_PATTERN
    gen.CALCULATION_EVIDENCE_PATTERN = pattern
    try:
        return gen.validate_teacher_output(
            raw_content=row.get("raw_teacher_content") or "",
            raw_reasoning_content=row.get("raw_reasoning_content") or "",
            finish_reason=row.get("finish_reason"),
            gold_answer=str(row.get("gold_answer")),
        )
    finally:
        gen.CALCULATION_EVIDENCE_PATTERN = old


def paths_for(role: str) -> dict[str, Path]:
    base = f"teacher_gsm8k_{role}_{SLUG}"
    return {
        "events": DATA_DIR / f"{base}_events.jsonl",
        "accepted": DATA_DIR / f"{base}_accepted_audit.jsonl",
        "rejected": DATA_DIR / f"{base}_rejected_audit.jsonl",
        "sft": DATA_DIR / f"{base}_sft.jsonl",
        "metrics": DATA_DIR / f"{base}_metrics.json",
    }


def backup_file(path: Path, backup_dir: Path) -> None:
    if path.exists():
        shutil.copy2(path, backup_dir / path.name)


def rescue_role(gen, role: str, dry_run: bool) -> dict:
    p = paths_for(role)
    strict = gen.CALCULATION_EVIDENCE_PATTERN
    rejected_rows = gen.load_jsonl(p["rejected"])
    candidates = [r for r in rejected_rows if is_dollar_false_reject(r, strict)]

    rescued = []
    skipped = []
    for row in candidates:
        # Must pass the real validator with $? pattern
        cur = revalidate(gen, row, strict)
        loose = revalidate(gen, row, LOOSE_DOLLAR)
        if cur["is_valid"]:
            skipped.append((row["problem_id"], "already_valid_under_strict"))
            continue
        if not loose["is_valid"]:
            skipped.append((row["problem_id"], loose["validation_errors"]))
            continue
        rescued.append(build_accepted_from_reject(gen, row, loose))

    print(
        f"\n[{role}] dollar candidates={len(candidates)} "
        f"pass_validate_loose={len(rescued)} skipped={len(skipped)}"
    )
    for pid, why in skipped[:5]:
        print(f"  skip {pid}: {why}")
    for row in rescued[:3]:
        print(f"  rescue {row['problem_id']} answer={row['teacher_final_answer']}")

    if dry_run or not rescued:
        return {
            "role": role,
            "rescued": len(rescued),
            "candidates": len(candidates),
            "metrics": None,
        }

    # Backup then append accepted events + rebuild snapshots/metrics
    backup_dir = DATA_DIR / f"backup_before_dollar_rescue_{int(time.time())}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    for path in p.values():
        backup_file(path, backup_dir)
    print(f"[{role}] backup -> {backup_dir}")

    for row in rescued:
        gen.append_jsonl(p["events"], row)

    latest = gen.compact_state_files(
        p["events"], p["accepted"], p["rejected"], p["sft"]
    )
    n_candidates = gen.N_TRAIN if role == "train" else gen.N_VAL
    metrics = gen.write_subset_metrics(latest, p["metrics"], n_candidates)

    print(
        f"[{role}] new accepted={metrics['accepted']} "
        f"rejected={metrics['rejected']} "
        f"sft_rows={gen.count_unique_ids(p['sft'])}"
    )
    return {
        "role": role,
        "rescued": len(rescued),
        "candidates": len(candidates),
        "metrics": metrics,
        "backup_dir": str(backup_dir),
    }


def rewrite_run_metrics(gen, train_metrics, val_metrics) -> None:
    train_sft = paths_for("train")["sft"]
    val_sft = paths_for("val")["sft"]
    fingerprint = gen.compute_dataset_fingerprint(train_sft, val_sft)
    out = DATA_DIR / f"run_metrics_{SLUG}.json"
    # backup
    if out.exists():
        shutil.copy2(out, out.with_suffix(out.suffix + ".pre_rescue.bak"))
    gen.write_run_metrics(
        out,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        dataset_fingerprint=fingerprint,
    )
    print(f"Updated {out}")
    print(f"New dataset_fingerprint: {fingerprint}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only validate/count, do not write files",
    )
    args = parser.parse_args()

    print(f"Generator pattern (STRICT): {gen.CALCULATION_EVIDENCE_PATTERN.pattern}")
    print(f"Rescue pattern (LOOSE):     {LOOSE_DOLLAR.pattern}")
    print(f"Dry run: {args.dry_run}")

    results = []
    for role in ("train", "val"):
        results.append(rescue_role(gen, role, dry_run=args.dry_run))

    total = sum(r["rescued"] for r in results)
    print("\n" + "=" * 60)
    print(f"TOTAL rescued (pass validate_teacher_output with $?): {total}")

    if args.dry_run:
        print("Dry run only — no files changed.")
        print("Re-run without --dry-run to apply.")
        return

    if total > 0:
        train_m = results[0]["metrics"]
        val_m = results[1]["metrics"]
        if train_m is None:
            train_m = json.loads(paths_for("train")["metrics"].read_text())
        if val_m is None:
            val_m = json.loads(paths_for("val")["metrics"].read_text())
        rewrite_run_metrics(gen, train_m, val_m)
        print("Done. Share updated *_sft.jsonl with teammates.")
    else:
        print("Nothing rescued — run_metrics left untouched.")


if __name__ == "__main__":
    main()
