#!/usr/bin/env python3
"""Plot train/val curves for a QLoRA run.

Defaults to the most recent MLflow run. Pass a run dir to pick one explicitly,
and --compare <run> to overlay an earlier run on the same axes.

    python plot_run.py
    python plot_run.py mlruns/407488069768834968/36808d7c... \
      --compare mlruns/407488069768834968/d88f72f3...
"""

import sys
from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt

# UW brand — purple + mid gold only (shared with poster_analysis / plot_wrong_ans)
UW_PURPLE = "#4B2E83"
UW_HUSKY_PURPLE = "#32006E"
UW_GOLD = "#E3BF42"
UW_HERITAGE_GOLD = "#85754D"
UW_GRAY = "#666666"

SEL_TO_METRIC = {
    "eval_gen_exact_match": "gen_eval_exact_match",
    "eval_gen_format_valid_rate": "gen_eval_format_valid_rate",
    "eval_loss": "eval_loss",
}


def latest_run(root=Path("mlruns")):
    runs = [p for p in root.glob("*/*") if (p / "metrics").is_dir()]
    if not runs:
        sys.exit("No MLflow runs found under mlruns/")
    return max(runs, key=lambda p: p.stat().st_mtime)


def series(run, name):
    """MLflow metric files are 'timestamp value step'. Keep the first value per
    step: the post-restore trainer.evaluate() re-logs at the final step and
    would otherwise fold back into the curve as a spike."""
    path = run / "metrics" / name
    if not path.exists():
        return [], []
    first = {}
    for line in path.read_text().split("\n"):
        parts = line.split()
        if len(parts) >= 3:
            first.setdefault(int(float(parts[2])), float(parts[1]))
    items = sorted(first.items())
    return [s for s, _ in items], [v for _, v in items]


def restored(run, name):
    """Last entry for a metric, i.e. the post-restore measurement."""
    path = run / "metrics" / name
    lines = [l for l in path.read_text().split("\n") if l.strip()] if path.exists() else []
    return float(lines[-1].split()[1]) if len(lines) > 1 else None


def param(run, name, default=None):
    path = run / "params" / name
    return path.read_text().strip() if path.exists() else default


def best_step(run):
    """Step of the checkpoint the Trainer restored, per metric_for_best_model."""
    sel = param(run, "metric_for_best_model", "eval_loss")
    steps, vals = series(run, SEL_TO_METRIC.get(sel, sel))
    if not steps:
        return None
    pick = min if "loss" in sel else max
    return pick(zip(vals, steps))[1]


def plot(run, axes, label, alpha=1.0, *, train_color=UW_PURPLE, val_color=UW_GOLD):
    n = int(param(run, "eval_samples_for_generation", 20))
    lo, hi = axes

    ax = lo[0]
    ax.plot(
        *series(run, "loss"),
        marker="o",
        ms=2,
        alpha=alpha,
        color=train_color,
        label=f"{label} train",
    )
    ax.plot(
        *series(run, "eval_loss"),
        marker="s",
        ms=6,
        alpha=alpha,
        color=val_color,
        label=f"{label} val",
    )

    ax = lo[1]
    ax.plot(
        *series(run, "mean_token_accuracy"),
        marker="o",
        ms=2,
        alpha=alpha,
        color=train_color,
        label=f"{label} train",
    )
    ax.plot(
        *series(run, "eval_mean_token_accuracy"),
        marker="s",
        ms=6,
        alpha=alpha,
        color=val_color,
        label=f"{label} val",
    )

    ax = hi[0]
    steps, vals = series(run, "gen_eval_exact_match")
    ax.plot(
        steps,
        vals,
        marker="o",
        alpha=alpha,
        color=train_color,
        label=f"{label} exact-match (n={n})",
    )
    # +/- 1 standard error: at n=100 this band is ~5 points wide, so overlapping
    # bands between epochs are not a real difference.
    se = [sqrt(v * (1 - v) / n) for v in vals]
    ax.fill_between(
        steps,
        [v - s for v, s in zip(vals, se)],
        [v + s for v, s in zip(vals, se)],
        color=train_color,
        alpha=0.15 * alpha,
    )
    ax.plot(
        *series(run, "gen_eval_format_valid_rate"),
        marker="s",
        ls="--",
        alpha=alpha,
        color=UW_GOLD,
        label=f"{label} format valid",
    )

    hi[1].plot(
        *series(run, "learning_rate"),
        alpha=alpha,
        color=UW_HUSKY_PURPLE,
        label=label,
    )

    step = best_step(run)
    if step is not None:
        for ax in (*lo, *hi):
            ax.axvline(step, color=UW_GOLD, ls=":", lw=1.2, alpha=alpha)


args = [a for a in sys.argv[1:] if not a.startswith("--")]
run = Path(args[0]) if args else latest_run()
compare = None
if "--compare" in sys.argv:
    compare = Path(sys.argv[sys.argv.index("--compare") + 1])

fig, ax = plt.subplots(2, 2, figsize=(13, 8))
axes = (ax[0], ax[1])

if compare:
    # Faded husky purple / heritage gold so the primary run stays dominant
    plot(
        compare,
        axes,
        compare.name[:8],
        alpha=0.4,
        train_color=UW_HUSKY_PURPLE,
        val_color=UW_HERITAGE_GOLD,
    )
plot(run, axes, run.name[:8])

fig.suptitle(
    f"Llama-3.2-1B {param(run, 'variant')} QLoRA — "
    f"lr={param(run, 'learning_rate')} r={param(run, 'lora_r')} "
    f"epochs={param(run, 'epochs')} | selecting on {param(run, 'metric_for_best_model')} "
    f"| gold dotted = restored checkpoint (step {best_step(run)})",
    fontsize=11,
    color=UW_PURPLE,
)
for a, title, ylab in (
    (ax[0][0], "Loss", "loss"),
    (ax[0][1], "Token accuracy", "accuracy"),
    (ax[1][0], "Generation eval (shaded = ±1 SE)", "rate"),
    (ax[1][1], "Learning rate", "lr"),
):
    a.set_title(title, color=UW_PURPLE)
    a.set_xlabel("step", color=UW_GRAY)
    a.set_ylabel(ylab, color=UW_GRAY)
    a.tick_params(colors=UW_GRAY)
    a.grid(True, alpha=0.25, color=UW_GRAY)
    a.legend(fontsize=8)
ax[1][0].set_ylim(0, 1.05)

if "--out" in sys.argv:
    out = Path(sys.argv[sys.argv.index("--out") + 1])
else:
    out = Path("eval_outputs") / f"curves_{run.name[:8]}.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.tight_layout()
fig.savefig(out, dpi=150, facecolor="white")
print(f"Saved {out}")