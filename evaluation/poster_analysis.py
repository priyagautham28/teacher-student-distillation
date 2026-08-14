#!/usr/bin/env python3
"""Post-hoc statistical analysis for poster/report claims (CPU-only).

Reads existing *_predictions.jsonl / *_metrics.json / *_summary.csv from
shared evaluate.py runs. Does not load models or use the GPU.

Commands
--------
1. compare  — McNemar's test on paired correct/wrong outcomes
2. table    — build a before / after / teacher summary CSV from metrics.json
3. merge    — combine teammates' summary.csv files (dedupe teacher row)

Examples (paths match this machine's Llama track)
-------------------------------------------------
# Did SFT help?
python poster_analysis.py compare \\
  --predictions-a eval_outputs/llama_before_sft_rerun/meta-llama_Llama-3.2-1B-Instruct_before_sft_4a8500c9_rerun1_predictions.jsonl \\
  --predictions-b eval_outputs/meta-llama_Llama-3.2-1B-Instruct_after_sft_10a84857_predictions.jsonl \\
  --label-a before_sft --label-b after_sft \\
  --output eval_outputs/mcnemar_before_vs_after.json

# Is the gap to the teacher real?
python poster_analysis.py compare \\
  --predictions-a eval_outputs/meta-llama_Llama-3.2-1B-Instruct_after_sft_10a84857_predictions.jsonl \\
  --predictions-b ../cot_faithfulness/eval_outputs_v2/Qwen_Qwen3-14B-AWQ_teacher_3cb9a5c9_predictions.jsonl \\
  --label-a after_sft --label-b teacher \\
  --output eval_outputs/mcnemar_student_vs_teacher.json

# Poster numbers table from existing metrics (no re-eval)
python poster_analysis.py table \\
  --before-metrics eval_outputs/llama_before_sft_rerun/meta-llama_Llama-3.2-1B-Instruct_before_sft_4a8500c9_rerun1_metrics.json \\
  --after-metrics eval_outputs/meta-llama_Llama-3.2-1B-Instruct_after_sft_10a84857_metrics.json \\
  --teacher-metrics ../cot_faithfulness/eval_outputs_v2/Qwen_Qwen3-14B-AWQ_teacher_3cb9a5c9_metrics.json \\
  --output eval_outputs/llama_compare_summary.csv

# UW purple/gold accuracy bar chart for the poster
python poster_analysis.py plot \\
  --before-metrics eval_outputs/llama_before_sft_rerun/meta-llama_Llama-3.2-1B-Instruct_before_sft_4a8500c9_rerun1_metrics.json \\
  --after-metrics eval_outputs/meta-llama_Llama-3.2-1B-Instruct_after_sft_10a84857_metrics.json \\
  --teacher-metrics ../cot_faithfulness/eval_outputs_v2/Qwen_Qwen3-14B-AWQ_teacher_3cb9a5c9_metrics.json \\
  --output eval_outputs/llama_accuracy_bars_purple_gold.png

# Merge teammate summary CSVs
python poster_analysis.py merge \\
  --summaries path/to/qwen_summary.csv path/to/llama_summary.csv path/to/gemma_summary.csv \\
  --output eval_outputs/team_comparison_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from math import comb, erfc, sqrt
from pathlib import Path
from typing import Any, Optional

# Shared with evaluate.py so CSV schema / JSONL parsing cannot drift.
from evaluate import SUMMARY_CSV_FIELDS, read_jsonl, summary_row_from_metrics

# UW brand — purple + mid gold only (same hexes as plot_v3 / plot_wrong_ans)
UW_PURPLE = "#4B2E83"
UW_HUSKY_PURPLE = "#32006E"
UW_GOLD = "#E3BF42"
UW_HERITAGE_GOLD = "#85754D"
UW_GRAY = "#666666"


# =============================================================================
# 1. McNEMAR'S TEST
# =============================================================================


def _binomial_two_sided_p(k: int, n: int, p: float = 0.5) -> float:
    """Exact two-sided binomial p-value (symmetric: sum mass with P(i) <= P(k))."""
    if n == 0:
        return 1.0

    probabilities = [
        comb(n, i) * (p**i) * ((1 - p) ** (n - i)) for i in range(n + 1)
    ]
    observed_probability = probabilities[k]
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
    """McNemar's test on paired correct/incorrect outcomes (same problem_ids)."""
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
    n_shared = len(shared_ids)
    accuracy_a = (
        (both_correct + only_a_correct) / n_shared if n_shared else None
    )
    accuracy_b = (
        (both_correct + only_b_correct) / n_shared if n_shared else None
    )
    absolute_difference = (
        None
        if accuracy_a is None or accuracy_b is None
        else accuracy_b - accuracy_a
    )

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
        statistic = (abs(only_a_correct - only_b_correct) - 1) ** 2 / discordant_total
        p_value = erfc(sqrt(statistic / 2))

    return {
        "label_a": label_a,
        "label_b": label_b,
        "n_records_a": len(correctness_a),
        "n_records_b": len(correctness_b),
        "shared_n": n_shared,
        "accuracy_a": accuracy_a,
        "accuracy_b": accuracy_b,
        "absolute_difference_b_minus_a": absolute_difference,
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
    records_a = read_jsonl(predictions_path_a)
    records_b = read_jsonl(predictions_path_b)

    if not records_a:
        raise ValueError(f"No records found in {predictions_path_a}")
    if not records_b:
        raise ValueError(f"No records found in {predictions_path_b}")

    result = mcnemar_test(records_a, records_b, label_a=label_a, label_b=label_b)
    result["predictions_a"] = str(predictions_path_a)
    result["predictions_b"] = str(predictions_path_b)

    if result["shared_n"] < min(result["n_records_a"], result["n_records_b"]):
        print(
            "WARNING: not all problem_ids overlap; McNemar uses the intersection only "
            f"(shared={result['shared_n']}, "
            f"a={result['n_records_a']}, b={result['n_records_b']})."
        )

    print(f"\nMcNemar's test: {label_a} vs. {label_b}")
    print(f"  Shared test examples:        {result['shared_n']}")
    if result["accuracy_a"] is not None and result["accuracy_b"] is not None:
        print(f"  Accuracy {label_a}:            {result['accuracy_a']:.4f}")
        print(f"  Accuracy {label_b}:            {result['accuracy_b']:.4f}")
        print(
            f"  Difference ({label_b} - {label_a}): "
            f"{result['absolute_difference_b_minus_a']:+.4f}"
        )
    print(f"  Both correct:                {result['both_correct']}")
    print(f"  Both wrong:                  {result['both_wrong']}")
    print(f"  Only {label_a} correct:      {result['only_a_correct']}")
    print(f"  Only {label_b} correct:      {result['only_b_correct']}")
    print(f"  Method:                      {result['method']}")
    if result["statistic"] is not None:
        print(f"  Chi-square statistic:        {result['statistic']:.4f}")
    print(f"  p-value:                     {result['p_value']:.6g}")
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
# 2. METRICS TABLE (no re-eval)
# =============================================================================


def _load_metrics(path: Path) -> dict[str, Any]:
    metrics = json.loads(path.read_text(encoding="utf-8"))
    if "exact_match_accuracy" not in metrics:
        raise ValueError(f"{path} missing exact_match_accuracy")
    return metrics


def write_compare_table(
    *,
    before_metrics_path: Path,
    after_metrics_path: Path,
    teacher_metrics_path: Optional[Path],
    output_path: Path,
) -> None:
    """Build a compare-base-style summary.csv from existing metrics.json files."""
    before = _load_metrics(before_metrics_path)
    after = _load_metrics(after_metrics_path)
    teacher = (
        _load_metrics(teacher_metrics_path) if teacher_metrics_path is not None else None
    )

    before_acc = float(before["exact_match_accuracy"])
    after_acc = float(after["exact_match_accuracy"])
    teacher_acc = (
        float(teacher["exact_match_accuracy"]) if teacher is not None else None
    )

    absolute_improvement = after_acc - before_acc
    relative_improvement = (
        absolute_improvement / before_acc if before_acc > 0 else None
    )
    gap_to_teacher = (
        teacher_acc - after_acc if teacher_acc is not None else None
    )

    rows: list[dict[str, Any]] = []
    if teacher is not None:
        rows.append(summary_row_from_metrics(teacher, condition="teacher"))
    rows.append(summary_row_from_metrics(before, condition="before_sft"))
    rows.append(
        summary_row_from_metrics(
            after,
            condition="after_sft",
            absolute_improvement_over_before=absolute_improvement,
            relative_improvement_over_before=relative_improvement,
            gap_to_teacher_accuracy=gap_to_teacher,
        )
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SUMMARY_CSV_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in SUMMARY_CSV_FIELDS})

    print(f"\nComparison table -> {output_path}")
    print(f"  before_sft:  {before_acc:.4f}")
    print(f"  after_sft:   {after_acc:.4f}")
    print(f"  improvement: {absolute_improvement:+.4f}")
    if relative_improvement is not None:
        print(f"  relative:    {relative_improvement:.2%}")
    if teacher_acc is not None and gap_to_teacher is not None:
        print(f"  teacher:     {teacher_acc:.4f}")
        print(f"  gap:         {gap_to_teacher:.4f}")


# =============================================================================
# 3. TEAM SUMMARY MERGE
# =============================================================================


def merge_team_summaries(paths: list[Path], output_path: Path) -> None:
    """Merge teammate summary.csv files; keep only the first teacher row."""
    seen_teacher_row = False
    combined_rows: list[dict[str, Any]] = []

    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if row.get("condition") == "teacher":
                    if seen_teacher_row:
                        continue
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
    print(
        f"  Total rows: {len(combined_rows)} "
        f"(teacher row deduplicated to 1 if present)"
    )


# =============================================================================
# 4. POSTER BAR CHART (UW purple / gold)
# =============================================================================


def plot_accuracy_bars(
    *,
    before_metrics_path: Path,
    after_metrics_path: Path,
    teacher_metrics_path: Optional[Path],
    output_path: Path,
    title: str = "GSM8K exact-match accuracy",
) -> None:
    """Vertical bar chart: before / after / teacher — same purple/gold as plot_v3."""
    import matplotlib.pyplot as plt

    before = _load_metrics(before_metrics_path)
    after = _load_metrics(after_metrics_path)
    labels = ["Before SFT", "After SFT"]
    values = [
        float(before["exact_match_accuracy"]),
        float(after["exact_match_accuracy"]),
    ]
    # Same purple as plot_v3 train/exact-match; gold only for teacher (like val)
    colors = [UW_PURPLE, UW_PURPLE]

    if teacher_metrics_path is not None:
        teacher = _load_metrics(teacher_metrics_path)
        labels.append("Teacher")
        values.append(float(teacher["exact_match_accuracy"]))
        colors.append(UW_GOLD)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    bars = ax.bar(labels, values, color=colors, width=0.65)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Exact-match accuracy", color=UW_GRAY)
    ax.set_title(title, color=UW_PURPLE, fontsize=13, pad=12)
    ax.tick_params(colors=UW_GRAY)
    ax.yaxis.grid(True, alpha=0.25, color=UW_GRAY)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(UW_GOLD)
    ax.bar_label(bars, labels=[f"{v:.1%}" for v in values], padding=4, color=UW_GRAY)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, facecolor="white")
    print(f"Saved accuracy chart -> {output_path}")


# =============================================================================
# 5. CLI
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Post-hoc analysis for reasoning-distillation poster/report: "
            "McNemar tests, metrics tables, UW-branded charts, and team merge. "
            "Run AFTER evaluate.py has produced predictions/metrics."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    compare_parser = subparsers.add_parser(
        "compare",
        help="McNemar test between two predictions.jsonl files (same test set).",
    )
    compare_parser.add_argument("--predictions-a", type=Path, required=True)
    compare_parser.add_argument("--predictions-b", type=Path, required=True)
    compare_parser.add_argument("--label-a", default="condition_a")
    compare_parser.add_argument("--label-b", default="condition_b")
    compare_parser.add_argument("--output", type=Path, default=None)

    table_parser = subparsers.add_parser(
        "table",
        help="Build before/after/teacher summary.csv from existing metrics.json files.",
    )
    table_parser.add_argument("--before-metrics", type=Path, required=True)
    table_parser.add_argument("--after-metrics", type=Path, required=True)
    table_parser.add_argument("--teacher-metrics", type=Path, default=None)
    table_parser.add_argument("--output", type=Path, required=True)

    plot_parser = subparsers.add_parser(
        "plot",
        help="Bar chart of before/after/teacher accuracy (UW purple/gold).",
    )
    plot_parser.add_argument("--before-metrics", type=Path, required=True)
    plot_parser.add_argument("--after-metrics", type=Path, required=True)
    plot_parser.add_argument("--teacher-metrics", type=Path, default=None)
    plot_parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval_outputs/llama_accuracy_bars_purple_gold_v2.png"),
    )
    plot_parser.add_argument(
        "--title",
        default="Llama-3.2-1B GSM8K exact-match accuracy",
    )

    merge_parser = subparsers.add_parser(
        "merge",
        help="Merge multiple teammates' summary.csv files into one table.",
    )
    merge_parser.add_argument("--summaries", type=Path, nargs="+", required=True)
    merge_parser.add_argument("--output", type=Path, required=True)

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
    elif args.command == "table":
        write_compare_table(
            before_metrics_path=args.before_metrics,
            after_metrics_path=args.after_metrics,
            teacher_metrics_path=args.teacher_metrics,
            output_path=args.output,
        )
    elif args.command == "plot":
        plot_accuracy_bars(
            before_metrics_path=args.before_metrics,
            after_metrics_path=args.after_metrics,
            teacher_metrics_path=args.teacher_metrics,
            output_path=args.output,
            title=args.title,
        )
    elif args.command == "merge":
        merge_team_summaries(args.summaries, args.output)


if __name__ == "__main__":
    main()
