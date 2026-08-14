#!/usr/bin/env python3
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
    p.add_argument("--title", default="Llama after-SFT wrong-but-valid error categories")
    args = p.parse_args()

    summary = json.loads(args.summary.read_text())
    counts = summary["counts"]
    # drop zeros for a cleaner chart
    items = [(k, v) for k, v in counts.items() if v > 0]
    items.sort(key=lambda kv: kv[1], reverse=True)

    labels = [k.replace("_", " ") for k, _ in items]
    values = [v for _, v in items]
    total = summary.get("wrong_but_valid") or sum(values)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    # Solid purple bars + gold frame (first purple/gold style)
    bars = ax.barh(labels[::-1], values[::-1], color=UW_PURPLE)
    ax.set_xlabel("Count", color=UW_GRAY)
    ax.set_title(f"{args.title}\n(n={total} wrong-but-valid)", color=UW_PURPLE)
    ax.bar_label(bars, padding=3, fontsize=9, color=UW_GRAY)
    ax.set_xlim(0, max(values) * 1.15)
    ax.tick_params(colors=UW_GRAY)
    for spine in ax.spines.values():
        spine.set_color(UW_GOLD)
    fig.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, facecolor="white")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
