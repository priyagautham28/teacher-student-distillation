# -*- coding: utf-8 -*-
"""Post-hoc statistical analysis and reporting, run AFTER predictions exist.

This file is deliberately separate from evaluate_gsm8k.py. Everything here
is "poster polish": it doesn't generate any new predictions and doesn't
touch model weights, it only reads *_predictions.jsonl and *_summary.csv
files that evaluate_gsm8k.py already produced, and turns them into stronger,
more defensible claims for the report/poster.

Two independent things live in this file:

1. McNemar's test (mcnemar_test / run_mcnemar_from_paths)
   Answers: "is this accuracy difference between two conditions on the
   SAME test set real, or could it just be noise?" This is the correct
   test here specifically because every model in this project is
   evaluated on the identical, fixed GSM8K test set (see
   evaluate_gsm8k.py's DATASET_SPLIT="test" and its deterministic
   load_gsm8k_examples()) -- that makes the two sets of outcomes PAIRED,
   not independent samples, and McNemar's test is the standard test for
   paired binary (correct/wrong) outcomes. A plain two-sample test would
   be the wrong tool here.

   Two use cases, same underlying function:
     - base_llama_predictions.jsonl  vs  after_sft_llama_predictions.jsonl
       ("did SFT actually help, or is the improvement within noise?")
     - teacher_predictions.jsonl     vs  after_sft_llama_predictions.jsonl
       ("is the student's remaining gap to the teacher a real gap, or
       could it be explained by test-set noise?")

2. Team summary merge (merge_team_summaries)
   Combines the three teammates' individual summary.csv files (Qwen
   student, Llama student, Gemma/other student -- each produced by their
   own evaluate_gsm8k.py --compare-base run) into one team-wide comparison
   table for the poster. Handles the one real gotcha: if every teammate's
   --compare-base run was given --teacher-metrics-path, the teacher's row
   would otherwise appear THREE times when the files are combined -- this
   keeps it once.

Usage
-----
# McNemar: did SFT help this student?
python poster_analysis.py compare \
    --predictions-a eval_outputs/llama_before_sft_predictions.jsonl \
    --predictions-b eval_outputs/llama_after_sft_predictions.jsonl \
    --label-a before_sft --label-b after_sft

# McNemar: is the student's gap to the teacher real?
python poster_analysis.py compare \
    --predictions-a eval_outputs/llama_after_sft_predictions.jsonl \
    --predictions-b eval_outputs/qwen_teacher_predictions.jsonl \
    --label-a after_sft --label-b teacher

# Merge all three teammates' summary.csv into one poster table
python poster_analysis.py merge \
    --summaries eval_outputs/qwen_summary.csv eval_outputs/llama_summary.csv eval_outputs/gemma_summary.csv \
    --output eval_outputs/team_comparison_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from math import comb, erfc, sqrt
from pathlib import Path
from typing import Any, Optional

# Reused directly from the shared evaluator instead of reimplemented, so
# this file can never silently drift out of sync with how predictions are
# actually written or how the CSV schema is defined.
from evaluate_gsm8k import SUMMARY_CSV_FIELDS, read_jsonl


# =============================================================================
# 1. McNEMAR'S TEST
# =============================================================================


def _binomial_two_sided_p(k: int, n: int, p: float = 0.5) -> float:
    """Exact two-sided binomial test p-value for k successes out of n trials
    under the null hypothesis that the true success probability is p.

    Used here as the exact form of McNemar's test: under the null hypothesis
    that condition A and condition B are equally likely to be the one that's
    "right when the other is wrong", each discordant pair is like an
    independent coin flip with p=0.5. This computes, across every possible
    outcome count i from 0 to n, the two-sided p-value as the total
    probability mass of all outcomes at least as extreme as the one actually
    observed (i.e. every outcome whose probability is <= the observed
    outcome's probability, which is the standard symmetric definition of a
    two-sided exact test).

    Preferred over the chi-square approximation used below when n = b + c
    is small, since the chi-square approximation is known to be unreliable
    once the discordant-pair count drops below roughly 25.
    """
    if n == 0:
        return 1.0

    probabilities = [
        comb(n, i) * (p**i) * ((1 - p) ** (n - i)) for i in range(n + 1)
    ]
    observed_probability = probabilities[k]
    # Small float tolerance so the observed outcome's own probability mass
    # is reliably included in the sum despite floating-point rounding.
    return min(
        1.0,
        sum(
            probability
            for probability in probabilities
            if probability <= observed_probability + 1e-12
        ),
    )


def mcnemar_test(
    records_a: list[dict[str, Any]],
    records_b: list[dict[str, Any]],
    *,
    label_a: str = "condition_a",
    label_b: str = "condition_b",
) -> dict[str, Any]:
    """McNemar's test on paired correct/incorrect outcomes from two
    evaluation runs over the SAME test set.

    Why this test and not a simpler comparison: records_a and records_b are
    not two independent samples, they are two different models' answers to
    the identical set of questions (matched by problem_id). That pairing is
    exactly what McNemar's test is designed for -- it only looks at the
    examples where the two conditions DISAGREE (one got it right, the other
    didn't); examples where both conditions agree (both right, or both
    wrong) carry no information about which condition is actually better,
    so they are correctly excluded from the test statistic.

    Parameters
    ----------
    records_a, records_b:
        Lists of prediction records as read from a *_predictions.jsonl file
        (via read_jsonl). Each record must have "problem_id" and
        "is_correct".
    label_a, label_b:
        Human-readable names for the two conditions, carried through into
        the returned dict purely for readability in printed/saved output.

    Returns
    -------
    A dict containing:
        - shared_n: how many problem_ids were present in both inputs and
          therefore actually usable for the test
        - only_a_correct: count where A was correct and B was wrong
        - only_b_correct: count where B was correct and A was wrong
        - both_correct / both_wrong: agreement counts (not used in the
          test itself, included for transparency)
        - method: "exact_binomial" or "chi_square_continuity_corrected",
          whichever was actually used
        - statistic: the chi-square statistic (only meaningful when
          method is chi_square_continuity_corrected; None otherwise)
        - p_value: two-sided p-value under the null hypothesis that A and B
          are equally likely to be the one that's correct on a discordant
          pair
        - significant_at_0_05: convenience boolean, p_value < 0.05
    """
    correctness_a = {
        str(record["problem_id"]): bool(record.get("is_correct"))
        for record in records_a
        if record.get("problem_id") is not None
    }
    correctness_b = {
        str(record["problem_id"]): bool(record.get("is_correct"))
        for record in records_b
        if record.get("problem_id") is not None
    }
    shared_ids = set(correctness_a) & set(correctness_b)

    both_correct = sum(
        1 for pid in shared_ids if correctness_a[pid] and correctness_b[pid]
    )
    both_wrong = sum(
        1 for pid in shared_ids if not correctness_a[pid] and not correctness_b[pid]
    )
    only_a_correct = sum(
        1 for pid in shared_ids if correctness_a[pid] and not correctness_b[pid]
    )
    only_b_correct = sum(
        1 for pid in shared_ids if not correctness_a[pid] and correctness_b[pid]
    )

    discordant_total = only_a_correct + only_b_correct

    # Rule of thumb from the McNemar's test literature: below ~25 discordant
    # pairs, the chi-square approximation's sampling distribution is not
    # reliably close enough to the true distribution to trust; use the
    # exact binomial form instead. At or above 25, the chi-square form is
    # standard practice and slightly easier for a reader to recognize.
    if discordant_total == 0:
        method = "exact_binomial"
        statistic = None
        p_value = 1.0
    elif discordant_total < 25:
        method = "exact_binomial"
        statistic = None
        p_value = _binomial_two_sided_p(
            k=min(only_a_correct, only_b_correct), n=discordant_total
        )
    else:
        method = "chi_square_continuity_corrected"
        # Yates' continuity correction: subtracting 1 from |b - c| before
        # squaring compensates for approximating a discrete distribution
        # (the binomial) with a continuous one (chi-square).
        statistic = (abs(only_a_correct - only_b_correct) - 1) ** 2 / discordant_total
        # Survival function of a chi-square distribution with 1 degree of
        # freedom, computed directly from the standard normal complementary
        # error function -- avoids needing scipy for a one-line formula.
        p_value = erfc(sqrt(statistic / 2))

    return {
        "label_a": label_a,
        "label_b": label_b,
        "shared_n": len(shared_ids),
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "only_a_correct": only_a_correct,
        "only_b_correct": only_b_correct,
        "discordant_total": discordant_total,
        "method": method,
        "statistic": statistic,
        "p_value": p_value,
        "significant_at_0_05": p_value < 0.05,
    }


def run_mcnemar_from_paths(
    predictions_path_a: Path,
    predictions_path_b: Path,
    *,
    label_a: str,
    label_b: str,
    output_path: Optional[Path] = None,
) -> dict[str, Any]:
    """Load two predictions.jsonl files (via the shared read_jsonl helper,
    so parsing behaves identically to how evaluate_gsm8k.py itself reads
    them) and run mcnemar_test on them. Prints a plain-language summary and,
    if output_path is given, also saves the full result as JSON.
    """
    records_a = read_jsonl(predictions_path_a)
    records_b = read_jsonl(predictions_path_b)

    if not records_a:
        raise ValueError(f"No records found in {predictions_path_a}")
    if not records_b:
        raise ValueError(f"No records found in {predictions_path_b}")

    result = mcnemar_test(records_a, records_b, label_a=label_a, label_b=label_b)

    print(f"\nMcNemar's test: {label_a} vs. {label_b}")
    print(f"  Shared test examples:        {result['shared_n']}")
    print(f"  Both correct:                {result['both_correct']}")
    print(f"  Both wrong:                  {result['both_wrong']}")
    print(f"  Only {label_a} correct:      {result['only_a_correct']}")
    print(f"  Only {label_b} correct:      {result['only_b_correct']}")
    print(f"  Method:                      {result['method']}")
    print(f"  p-value:                     {result['p_value']:.4f}")
    if result["significant_at_0_05"]:
        print(
            f"  -> Significant at p < 0.05: the difference between "
            f"{label_a} and {label_b} is unlikely to be due to chance."
        )
    else:
        print(
            f"  -> Not significant at p < 0.05: the difference between "
            f"{label_a} and {label_b} could plausibly be due to test-set "
            "noise rather than a real effect."
        )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"  Saved: {output_path}")

    return result


# =============================================================================
# 2. TEAM SUMMARY MERGE
# =============================================================================


def merge_team_summaries(paths: list[Path], output_path: Path) -> None:
    """Combine multiple teammates' summary.csv files (all produced by the
    same evaluate_gsm8k.py, so they share the same SUMMARY_CSV_FIELDS
    schema) into one team-wide comparison table for the poster.

    The one real gotcha this handles: if every teammate's --compare-base run
    was given --teacher-metrics-path (as this project's workflow expects),
    each of their individual summary.csv files independently embeds a copy
    of the teacher's row. Concatenating three such files naively would give
    the merged table three duplicate teacher rows instead of one. This scans
    every input file's "condition" column and keeps only the FIRST row
    where condition == "teacher", discarding the rest, regardless of which
    teammate's file it came from.

    Every kept row gets a new "source_file" column so it's still possible
    to trace which teammate's evaluation a given non-teacher row came from.
    """
    seen_teacher_row = False
    combined_rows: list[dict[str, Any]] = []

    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if row.get("condition") == "teacher":
                    if seen_teacher_row:
                        continue  # Skip every teacher row after the first one seen.
                    seen_teacher_row = True
                row["source_file"] = path.stem
                combined_rows.append(row)

    fieldnames = ["source_file"] + list(SUMMARY_CSV_FIELDS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in combined_rows:
            writer.writerow({field: row.get(field) for field in fieldnames})

    print(f"Merged {len(paths)} summary file(s) into {output_path}")
    print(f"  Total rows: {len(combined_rows)} (teacher row deduplicated to 1 if present)")


# =============================================================================
# 3. COMMAND-LINE INTERFACE
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Post-hoc statistical analysis for the reasoning-distillation "
            "project: McNemar significance testing and team summary merging. "
            "Run this AFTER evaluate_gsm8k.py has produced predictions."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    compare_parser = subparsers.add_parser(
        "compare",
        help=(
            "Run McNemar's test between two predictions.jsonl files from "
            "the same test set (base vs. after-SFT, or student vs. teacher)."
        ),
    )
    compare_parser.add_argument(
        "--predictions-a", type=Path, required=True, help="Path to the first predictions.jsonl"
    )
    compare_parser.add_argument(
        "--predictions-b", type=Path, required=True, help="Path to the second predictions.jsonl"
    )
    compare_parser.add_argument(
        "--label-a", default="condition_a", help="Readable name for the first condition"
    )
    compare_parser.add_argument(
        "--label-b", default="condition_b", help="Readable name for the second condition"
    )
    compare_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save the full result as JSON",
    )

    merge_parser = subparsers.add_parser(
        "merge",
        help="Merge multiple teammates' summary.csv files into one poster comparison table.",
    )
    merge_parser.add_argument(
        "--summaries",
        type=Path,
        nargs="+",
        required=True,
        help="Paths to each teammate's summary.csv",
    )
    merge_parser.add_argument(
        "--output", type=Path, required=True, help="Path to write the merged comparison CSV"
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "compare":
        run_mcnemar_from_paths(
            args.predictions_a,
            args.predictions_b,
            label_a=args.label_a,
            label_b=args.label_b,
            output_path=args.output,
        )
    elif args.command == "merge":
        merge_team_summaries(args.summaries, args.output)


if __name__ == "__main__":
    main()
