#!/usr/bin/env python3
"""Classify wrong-but-valid GSM8K predictions into error categories.

Wrong-but-valid = is_correct=False AND is_valid_format=True
(excludes pure truncation/format failures).

Usage:
  cd ~/reasoning-distillation
  source .venv/bin/activate
  python classify_wrong_valid.py \
    --predictions eval_outputs/meta-llama_Llama-3.2-1B-Instruct_after_sft_10a84857_predictions.jsonl \
    --out-dir eval_outputs/error_analysis_10a84857
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

OP_EQ_RE = re.compile(
    r"(-?\d[\d,]*(?:\.\d+)?)\s*([+\-*/x×])\s*(-?\d[\d,]*(?:\.\d+)?)\s*=\s*(-?\d[\d,]*(?:\.\d+)?)"
)
NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
STEP_RE = re.compile(r"Step\s+(\d+):\s*(.*)", re.I)


CATEGORIES = [
    "arithmetic_mistake",
    "wrong_operation",
    "dropped_intermediate_step",
    "misunderstood_question",
    "copied_irrelevant_teacher_pattern",  # needs teacher file; else rare/unknown
    "premature_answer",
    "correct_reasoning_wrong_extraction",
    "hallucinated_reasoning",
    "off_by_one",
    "unit_error",
    "other",
]


def to_float(x) -> float | None:
    if x is None:
        return None
    try:
        return float(str(x).replace(",", "").replace("$", "").strip())
    except Exception:
        return None


def eval_op(a: float, op: str, b: float) -> float | None:
    op = op.replace("x", "*").replace("×", "*")
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    if op == "/":
        return None if b == 0 else a / b
    return None


def find_arith_mistakes(reasoning: str) -> list[str]:
    bad = []
    for a, op, b, c in OP_EQ_RE.findall(reasoning or ""):
        af, bf, cf = to_float(a), to_float(b), to_float(c)
        if af is None or bf is None or cf is None:
            continue
        true = eval_op(af, op, bf)
        if true is None:
            continue
        if abs(true - cf) > 1e-3 and abs(true - cf) / max(abs(true), 1.0) > 0.01:
            bad.append(f"{a}{op}{b}={c} (true={true:g})")
    return bad


def question_numbers(q: str) -> list[float]:
    return [float(x.replace(",", "")) for x in NUM_RE.findall(q or "")]


def classify(row: dict, teacher_target: str | None = None) -> tuple[str, str]:
    """Return (label, evidence). First matching rule wins."""
    q = row.get("question") or ""
    reason = row.get("reasoning") or ""
    raw = row.get("raw_output") or ""
    gold = to_float(row.get("gold_answer"))
    pred = to_float(row.get("predicted_answer"))
    steps = STEP_RE.findall(reason)
    n_steps = len(steps)
    q_nums = question_numbers(q)
    ql = q.lower()

    # 1) extraction bug: tagged answer != last computed number, and last number == gold
    eqs = OP_EQ_RE.findall(reason)
    if eqs and gold is not None:
        last_rhs = to_float(eqs[-1][3])
        if last_rhs is not None and abs(last_rhs - gold) < 1e-6:
            if pred is None or abs(pred - gold) > 1e-6:
                return (
                    "correct_reasoning_wrong_extraction",
                    f"last eq rhs={last_rhs} matches gold; pred={row.get('predicted_answer')}",
                )

    # 2) off-by-one
    if gold is not None and pred is not None and abs(abs(pred - gold) - 1) < 1e-6:
        return "off_by_one", f"pred={pred} gold={gold}"

    # 3) unit / scale (10x, 100x, 0.01x, percent slip)
    if gold not in (None, 0) and pred is not None:
        ratio = pred / gold
        for s in (10, 100, 1000, 0.1, 0.01, 0.001):
            if abs(ratio - s) < 0.02:
                return "unit_error", f"pred/gold≈{ratio:.4g}"

    # 4) arithmetic mistake in written equations
    bad = find_arith_mistakes(reason)
    if bad:
        return "arithmetic_mistake", "; ".join(bad[:3])

    # 5) premature answer: very few steps
    if n_steps <= 2 and gold is not None and pred is not None:
        return "premature_answer", f"only {n_steps} steps"

    # 6) misunderstood question heuristics
    misread_hints = []
    if "tip" in ql and "total" in ql and pred is not None and gold is not None:
        # tip-only instead of total is common
        if pred < gold and any(abs(pred - n) < 1e-6 for n in q_nums):
            misread_hints.append("possible tip-only vs total")
    if re.search(r"\beach\b", ql) and gold is not None and pred is not None:
        if abs(pred - 2 * gold) < 1e-6 or (gold != 0 and abs(pred / gold - 2) < 0.05):
            misread_hints.append("each vs total factor~2")
    if any(w in ql for w in ["left", "remain", "remaining", "still need", "how much more"]):
        if gold is not None and pred is not None and gold != 0:
            if abs(pred - gold) / abs(gold) > 0.5:
                misread_hints.append("remainder/difference style miss")
    # unused question numbers (weak)
    reason_nums = set(NUM_RE.findall(reason))
    q_num_strs = set(NUM_RE.findall(q))
    unused = [n for n in q_num_strs if n not in reason_nums and n not in {"1", "2"}]
    if len(unused) >= 2:
        misread_hints.append(f"unused question nums={unused[:5]}")
    if misread_hints:
        return "misunderstood_question", "; ".join(misread_hints[:3])

    # 7) wrong operation: factor 2 / half vs gold, with no local arith bug
    if gold not in (None, 0) and pred is not None:
        if abs(pred / gold - 2) < 0.05 or abs(pred / gold - 0.5) < 0.05:
            return "wrong_operation", f"pred/gold≈{pred/gold:.3g}"

    # 8) hallucinated reasoning: numbers not in question appear a lot
    if q_nums:
        qset = set(round(x, 6) for x in q_nums)
        invented = []
        for n in NUM_RE.findall(reason):
            v = to_float(n)
            if v is None:
                continue
            if all(abs(v - qv) > 1e-6 for qv in qset) and v not in (0, 1):
                # allow results of ops roughly; flag large invented constants
                if abs(v) >= 10:
                    invented.append(n)
        if len(set(invented)) >= 3 and n_steps >= 4:
            return "hallucinated_reasoning", f"invented nums≈{sorted(set(invented))[:6]}"

    # 9) dropped intermediate step: asks multi-part but few eqs
    multi = len(re.findall(r"\b(then|after|next|finally|remaining|left)\b", ql)) >= 2
    if multi and len(eqs) <= 1 and n_steps <= 3:
        return "dropped_intermediate_step", f"multi-clause q but eqs={len(eqs)} steps={n_steps}"

    # 10) copied irrelevant teacher pattern (optional): target diverges a lot
    if teacher_target:
        # if student reasoning shares little lexical overlap with teacher target
        t = teacher_target.lower()
        s = reason.lower()
        t_toks = set(re.findall(r"[a-z]{4,}", t))
        s_toks = set(re.findall(r"[a-z]{4,}", s))
        if t_toks:
            overlap = len(t_toks & s_toks) / len(t_toks)
            if overlap < 0.05 and n_steps >= 6:
                return "copied_irrelevant_teacher_pattern", f"teacher lexical overlap={overlap:.2f}"

    return "other", "no strong heuristic match"


def load_teacher_map(path: Path | None) -> dict[str, str]:
    """Map question -> student_target/reasoning if a teacher SFT jsonl is given."""
    if path is None or not path.exists():
        return {}
    out = {}
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            q = r.get("question")
            tgt = r.get("student_target") or r.get("reasoning") or ""
            if q:
                out[q] = tgt
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--teacher-file",
        type=Path,
        default=None,
        help="Optional teacher SFT jsonl to enable teacher-pattern checks",
    )
    ap.add_argument("--examples-per-class", type=int, default=5)
    args = ap.parse_args()

    teacher = load_teacher_map(args.teacher_file)
    rows = [json.loads(l) for l in args.predictions.open() if l.strip()]
    wrong_valid = [
        r
        for r in rows
        if (not r.get("is_correct")) and r.get("is_valid_format") and (not r.get("is_truncated"))
    ]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    labeled = []
    counts = Counter()
    examples = defaultdict(list)

    for r in wrong_valid:
        label, evidence = classify(r, teacher.get(r.get("question") or ""))
        counts[label] += 1
        item = {
            "problem_id": r.get("problem_id"),
            "label": label,
            "evidence": evidence,
            "gold_answer": r.get("gold_answer"),
            "predicted_answer": r.get("predicted_answer"),
            "question": r.get("question"),
            "reasoning": r.get("reasoning"),
            "completion_tokens": r.get("completion_tokens"),
        }
        labeled.append(item)
        if len(examples[label]) < args.examples_per_class:
            examples[label].append(item)

    # write outputs
    summary = {
        "source_predictions": str(args.predictions),
        "total_predictions": len(rows),
        "wrong_but_valid": len(wrong_valid),
        "counts": {k: counts.get(k, 0) for k in CATEGORIES},
        "note": (
            "Heuristic labels are noisy. 'other' is expected to be large. "
            "Use examples JSON for manual spot-check / report quotes."
        ),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.out_dir / "examples_by_class.json").write_text(json.dumps(examples, indent=2))

    with (args.out_dir / "labeled.csv").open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "problem_id",
                "label",
                "evidence",
                "gold_answer",
                "predicted_answer",
                "completion_tokens",
                "question",
            ],
        )
        w.writeheader()
        for item in labeled:
            w.writerow({k: item.get(k) for k in w.fieldnames})

    with (args.out_dir / "labeled.jsonl").open("w") as f:
        for item in labeled:
            f.write(json.dumps(item) + "\n")

    print(f"wrong-but-valid: {len(wrong_valid)} / {len(rows)}")
    print("counts:")
    for k in CATEGORIES:
        print(f"  {counts.get(k, 0):4d}  {k}")
    print(f"\nwrote: {args.out_dir}/summary.json")
    print(f"wrote: {args.out_dir}/labeled.csv")
    print(f"wrote: {args.out_dir}/examples_by_class.json")


if __name__ == "__main__":
    main()