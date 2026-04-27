"""
visualize.py — All plotting for the GRPO experiment

For local analysis 

Usage:
    python src/visualize.py --history results/training_history.json
    python src/visualize.py --compare results/grpo_history.json results/ablation_history.json
"""

import json
import argparse
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


# ---------------------------------------------------------------------------
# STYLE — consistent look across all figures
# ---------------------------------------------------------------------------

STYLE = {
    "grpo":     {"color": "#7C6BFF", "label": "GRPO (with reward)", "lw": 2.0},
    "ablation": {"color": "#FF6B9D", "label": "Ablation (no reward)", "lw": 1.5, "ls": "--"},
}

plt.rcParams.update({
    "font.family":      "monospace",
    "font.size":        11,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.25,
    "grid.linestyle":   "--",
    "figure.dpi":       150,
})


def load_history(path: str) -> list:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# PLOT 1 — Main results: 3-panel figure (the money shot)
# ---------------------------------------------------------------------------

def plot_main_results(grpo_history: list, ablation_history: list = None,
                      save_path: str = "results/grpo_main_results.png"):
    """
    3-panel figure:
      Left:   Mean reward / accuracy over training steps
      Middle: Average reasoning trace length over training steps
      Right:  Scatter — does longer reasoning → higher accuracy?

    This is the figure that goes in your README.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig = plt.figure(figsize=(15, 5))
    fig.suptitle(
        "GRPO Training: Does RL Teach a Model to Reason?",
        fontsize=14, fontweight="bold", y=1.02
    )
    gs = gridspec.GridSpec(1, 3, wspace=0.35)
    ax1, ax2, ax3 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])

    def extract(history, key):
        return (
            [h["step"] for h in history if key in h],
            [h[key]   for h in history if key in h]
        )

    # ── Panel 1: Reward / accuracy ────────────────────────────────────────
    sx, sy = extract(grpo_history, "mean_reward")
    ax1.plot(sx, sy, **{k: v for k, v in STYLE["grpo"].items() if k != "label"},
             label=STYLE["grpo"]["label"])

    if ablation_history:
        ax, ay = extract(ablation_history, "mean_reward")
        ax1.plot(ax, ay, **{k: v for k, v in STYLE["ablation"].items() if k != "label"},
                 label=STYLE["ablation"]["label"])

    ax1.set_title("Accuracy over training")
    ax1.set_xlabel("Training steps")
    ax1.set_ylabel("Mean reward (proxy for accuracy)")
    ax1.set_ylim(bottom=0)
    ax1.legend(fontsize=9)

    # Annotate start and end
    if sy:
        ax1.annotate(f"start: {sy[0]:.2f}", xy=(sx[0], sy[0]),
                     xytext=(sx[0] + max(sx)*0.05, sy[0] + 0.02),
                     fontsize=8, color=STYLE["grpo"]["color"])
        ax1.annotate(f"end: {sy[-1]:.2f}", xy=(sx[-1], sy[-1]),
                     xytext=(sx[-1] - max(sx)*0.15, sy[-1] + 0.02),
                     fontsize=8, color=STYLE["grpo"]["color"])

    # ── Panel 2: Trace length ─────────────────────────────────────────────
    tx, ty = extract(grpo_history, "avg_trace_length")
    ax2.plot(tx, ty, **{k: v for k, v in STYLE["grpo"].items() if k != "label"},
             label=STYLE["grpo"]["label"])

    if ablation_history:
        atx, aty = extract(ablation_history, "avg_trace_length")
        ax2.plot(atx, aty, **{k: v for k, v in STYLE["ablation"].items() if k != "label"},
                 label=STYLE["ablation"]["label"])

    ax2.set_title("Reasoning trace length")
    ax2.set_xlabel("Training steps")
    ax2.set_ylabel("Avg completion length (words)")
    ax2.legend(fontsize=9)
    ax2.set_ylim(bottom=0)

    # ── Panel 3: Scatter — trace length vs accuracy ───────────────────────
    if tx and ty and sx and sy:
        # Align on common steps
        grpo_df = {h["step"]: h for h in grpo_history if "avg_trace_length" in h and "mean_reward" in h}
        steps   = sorted(grpo_df.keys())
        lengths = [grpo_df[s]["avg_trace_length"] for s in steps]
        rewards = [grpo_df[s]["mean_reward"]       for s in steps]

        # Color points by training progress (early=light, late=dark)
        colors = plt.cm.Purples(np.linspace(0.3, 0.9, len(steps)))
        for i, (l, r) in enumerate(zip(lengths, rewards)):
            ax3.scatter(l, r, color=colors[i], s=30, zorder=3)

        # Trend line
        if len(lengths) > 2:
            z = np.polyfit(lengths, rewards, 1)
            p = np.poly1d(z)
            xs = np.linspace(min(lengths), max(lengths), 100)
            ax3.plot(xs, p(xs), color=STYLE["grpo"]["color"],
                     linewidth=1, linestyle="--", alpha=0.6, label="trend")

            # Correlation coefficient
            corr = np.corrcoef(lengths, rewards)[0, 1]
            ax3.text(0.05, 0.92, f"r = {corr:.2f}", transform=ax3.transAxes,
                     fontsize=10, color=STYLE["grpo"]["color"])

    ax3.set_title("Longer reasoning → better accuracy?")
    ax3.set_xlabel("Avg trace length (words)")
    ax3.set_ylabel("Mean reward")

    # Colorbar to show training progress
    sm = plt.cm.ScalarMappable(cmap="Purples",
                                norm=plt.Normalize(vmin=0, vmax=max(steps) if steps else 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax3, shrink=0.8)
    cbar.set_label("Training step", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    print(f"✓ Saved main results figure → {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# PLOT 2 — Ablation comparison (standalone)
# ---------------------------------------------------------------------------

def plot_ablation_comparison(grpo_history: list, ablation_history: list,
                              save_path: str = "results/ablation_comparison.png"):
    """
    Side-by-side comparison of GRPO vs ablation.
    The gap between these curves is your headline finding.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("GRPO vs Ablation — Does the Reward Signal Matter?",
                 fontsize=13, fontweight="bold")

    def plot_both(ax, key, ylabel, title):
        gx, gy = zip(*[(h["step"], h[key]) for h in grpo_history if key in h]) if grpo_history else ([], [])
        ax_x, ay = zip(*[(h["step"], h[key]) for h in ablation_history if key in h]) if ablation_history else ([], [])

        ax.plot(gx, gy, color=STYLE["grpo"]["color"],
                lw=STYLE["grpo"]["lw"], label=STYLE["grpo"]["label"])
        ax.plot(ax_x, ay, color=STYLE["ablation"]["color"],
                lw=STYLE["ablation"]["lw"], ls=STYLE["ablation"]["ls"],
                label=STYLE["ablation"]["label"])

        # Shade the gap
        if gx and ax_x:
            common = set(gx) & set(ax_x)
            if common:
                cs = sorted(common)
                g_map  = {h["step"]: h[key] for h in grpo_history if key in h}
                ab_map = {h["step"]: h[key] for h in ablation_history if key in h}
                gvals  = [g_map[s]  for s in cs if s in g_map  and s in ab_map]
                abvals = [ab_map[s] for s in cs if s in g_map  and s in ab_map]
                cs_filt = [s for s in cs if s in g_map and s in ab_map]
                ax.fill_between(cs_filt, abvals, gvals,
                                alpha=0.12, color=STYLE["grpo"]["color"],
                                label="Gap (reward effect)")

        ax.set_title(title)
        ax.set_xlabel("Training steps")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=9)
        ax.set_ylim(bottom=0)

    plot_both(ax1, "mean_reward",     "Mean reward",          "Accuracy: GRPO vs Ablation")
    plot_both(ax2, "avg_trace_length","Avg trace length (words)", "Trace length: GRPO vs Ablation")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    print(f"✓ Saved ablation comparison → {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# PLOT 3 — Training loss curve
# ---------------------------------------------------------------------------

def plot_loss_curve(history: list, save_path: str = "results/loss_curve.png"):
    """Simple loss over steps — useful for diagnosing instability."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    steps  = [h["step"] for h in history if "loss" in h]
    losses = [h["loss"]  for h in history if "loss" in h]

    if not steps:
        print("No loss values in history — skipping loss curve")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, losses, color=STYLE["grpo"]["color"], lw=1.5, alpha=0.8)

    # Smooth overlay
    if len(losses) > 10:
        window = max(3, len(losses) // 20)
        smooth = np.convolve(losses, np.ones(window)/window, mode='valid')
        ax.plot(steps[window-1:], smooth, color=STYLE["grpo"]["color"],
                lw=2.5, label=f"smoothed (w={window})")

    ax.set_title("Training loss over steps")
    ax.set_xlabel("Step")
    ax.set_ylabel("GRPO loss")
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    print(f"✓ Saved loss curve → {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# PLOT 4 — Format compliance over time
# ---------------------------------------------------------------------------

def plot_format_compliance(history: list, save_path: str = "results/format_compliance.png"):
    """
    Track what % of completions use <think> tags over training.
    Should rise early (format reward kicks in) then plateau.
    If it never rises: format reward isn't working.
    If it rises but accuracy doesn't: model learned formatting but not reasoning.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    steps  = [h["step"]        for h in history if "format_rate" in h]
    rates  = [h["format_rate"] for h in history if "format_rate" in h]
    accs   = [h["mean_reward"] for h in history if "mean_reward" in h and "format_rate" in h]

    if not steps:
        print("No format_rate in history — skipping")
        return

    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax2 = ax1.twinx()

    ax1.plot(steps, rates, color="#6BFFB8", lw=2, label="Format compliance (<think> usage)")
    ax2.plot(steps, accs,  color=STYLE["grpo"]["color"], lw=2,
             linestyle="--", label="Mean reward (accuracy)")

    ax1.set_xlabel("Training steps")
    ax1.set_ylabel("Format compliance rate", color="#6BFFB8")
    ax2.set_ylabel("Mean reward",            color=STYLE["grpo"]["color"])
    ax1.set_ylim(0, 1.05)
    ax2.set_ylim(0)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper left")
    ax1.set_title("Format compliance vs accuracy over training")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    print(f"✓ Saved format compliance plot → {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot GRPO experiment results")
    parser.add_argument("--history",  type=str, help="Path to training_history.json")
    parser.add_argument("--ablation", type=str, help="Path to ablation_history.json")
    parser.add_argument("--all",      action="store_true", help="Generate all plots")
    args = parser.parse_args()

    grpo_history     = load_history(args.history)  if args.history  else []
    ablation_history = load_history(args.ablation) if args.ablation else []

    if grpo_history:
        plot_main_results(grpo_history, ablation_history)
        plot_loss_curve(grpo_history)
        plot_format_compliance(grpo_history)

    if grpo_history and ablation_history:
        plot_ablation_comparison(grpo_history, ablation_history)

    if not grpo_history:
        print("Pass --history path/to/training_history.json to generate plots")


if __name__ == "__main__":
    main()