"""Plot error-category counts from analyze_wrong_ans.py output.

Usage:
  python plot_wrong_ans.py \
    --summary eval_outputs/error_analysis_10a84857/summary.json \
    --out eval_outputs/error_analysis_10a84857/error_categories_purple_gold.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

# UW brand — purple + mid gold only (same hexes as plot_v3 / poster_analysis)
UW_PURPLE = "#4B2E83"
UW_GOLD = "#E3BF42"
UW_GRAY = "#666666"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--title",
        default="",
        help="Optional chart title; leave empty for poster (caption in PPT instead)",
    )
    args = p.parse_args()

    summary = json.loads(args.summary.read_text())
    counts = summary["counts"]
    # drop zeros for a cleaner chart
    items = [(k, v) for k, v in counts.items() if v > 0]
    items.sort(key=lambda kv: kv[1], reverse=True)

    labels = [k.replace("_", " ") for k, _ in items]
    values = [v for _, v in items]

    # Poster-readable sizes (small 4-col slot; put section title in PPT)
    plt.rcParams.update({"font.size": 14})
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.barh(labels[::-1], values[::-1], color=UW_PURPLE)
    ax.set_xlabel("Count", fontsize=16, color=UW_GRAY)
    if args.title.strip():
        ax.set_title(args.title.strip(), color=UW_PURPLE, fontsize=18, pad=10)
    ax.bar_label(bars, padding=4, fontsize=14, color=UW_GRAY)
    ax.set_xlim(0, max(values) * 1.15)
    ax.tick_params(axis="both", labelsize=14, colors=UW_GRAY)
    for spine in ax.spines.values():
        spine.set_color(UW_GOLD)
    fig.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300, facecolor="white", bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
