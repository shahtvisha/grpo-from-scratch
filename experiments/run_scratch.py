"""
experiments/run_scratch.py — Run GRPO using your from-scratch implementation

Usage (Colab):
    !python experiments/run_scratch.py
    !python experiments/run_scratch.py --model Qwen/Qwen2.5-0.5B-Instruct --steps 100
"""

import sys
import os
import json
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.dataset import (
    generate_splits, reward_function,
    validate_reward, load_dataset
)
from src.model import load_model, get_device, make_reference_model
from src.grpo import GRPOConfig, train
from src.visualize import plot_main_results, plot_loss_curve, plot_format_compliance


# ---------------------------------------------------------------------------
# ARGUMENT PARSING
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",   default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--steps",   type=int,   default=300)
    p.add_argument("--G",       type=int,   default=8)
    p.add_argument("--lr",      type=float, default=5e-6)
    p.add_argument("--beta",    type=float, default=0.1,
                   help="KL penalty weight")
    p.add_argument("--output",  default="results")
    p.add_argument("--reuse-data", action="store_true",
                   help="Load problems from results/ instead of regenerating")
    p.add_argument("--skip-validation", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(f"{args.output}/traces", exist_ok=True)

    print("=" * 60)
    print("GRPO TRAINING — FROM SCRATCH VERSION")
    print("=" * 60)
    print(f"Model:  {args.model}")
    print(f"Steps:  {args.steps}")
    print(f"G:      {args.G}  (completions per prompt)")
    print(f"LR:     {args.lr}")
    print(f"beta:   {args.beta}  (KL penalty)")
    print(f"Output: {args.output}")
    print()

    # ── 1. Hardware ───────────────────────────────────────────────────────
    device = get_device()

    # ── 2. Data ───────────────────────────────────────────────────────────
    if args.reuse_data:
        print("Loading existing datasets from results/...")
        train_problems = load_dataset(f"{args.output}/train_problems.json")
        eval_problems  = load_dataset(f"{args.output}/eval_problems.json")
        probe_problems = load_dataset(f"{args.output}/probe_problems.json")
    else:
        print("Generating fresh datasets...")
        train_problems, eval_problems, probe_problems = generate_splits()

    print(f"  Train: {len(train_problems)} | Eval: {len(eval_problems)} | Probe: {len(probe_problems)}")

    # ── 3. Load model ─────────────────────────────────────────────────────
    model, tokenizer = load_model(args.model, device)
    ref_model = make_reference_model(model)

    # ── 4. Validate reward ────────────────────────────────────────────────
    if not args.skip_validation:
        print("\nValidating reward function...")
        hit_rate, _ = validate_reward(model, tokenizer, n_samples=20, device=device)
        if hit_rate < 0.03:
            print("✗ Reward hit rate too low. Fix reward_function() and retry.")
            return
        print(f"✓ Reward OK ({hit_rate:.1%})")

    # ── 5. Config ─────────────────────────────────────────────────────────
    config = GRPOConfig(
        num_steps=args.steps,
        G=args.G,
        learning_rate=args.lr,
        beta=args.beta,
        batch_size=4,
        max_new_tokens=256,
        temperature=0.8,
        log_every=10,
        eval_every=50,
        save_traces_every=50,
        output_dir=args.output,
    )

    # ── 6. Train ──────────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("Starting from-scratch GRPO training...")
    print("─" * 60 + "\n")

    history = train(
        policy_model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        train_problems=train_problems,
        eval_problems=eval_problems,
        probe_problems=probe_problems,
        reward_fn=reward_function,
        config=config,
    )

    # ── 7. Save history ───────────────────────────────────────────────────
    history_path = f"{args.output}/scratch_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\n✓ History saved → {history_path}")

    # ── 8. Print summary ──────────────────────────────────────────────────
    if len(history) >= 2:
        start = history[0]
        end   = history[-1]
        print("\n── RESULTS SUMMARY ──────────────────────")
        print(f"  Accuracy:     {start.get('accuracy', 0):.1%} → {end.get('accuracy', 0):.1%}  "
              f"(+{end.get('accuracy',0) - start.get('accuracy',0):.1%})")
        print(f"  Trace length: {start.get('avg_trace_length', 0):.0f}w → {end.get('avg_trace_length', 0):.0f}w")
        print(f"  Format rate:  {start.get('format_rate', 0):.1%} → {end.get('format_rate', 0):.1%}")
        print("─────────────────────────────────────────")

    # ── 9. Plots ──────────────────────────────────────────────────────────
    try:
        plot_main_results(history, save_path=f"{args.output}/scratch_main_results.png")
        plot_loss_curve(history,   save_path=f"{args.output}/scratch_loss_curve.png")
        plot_format_compliance(history, save_path=f"{args.output}/scratch_format_compliance.png")
        print("✓ Plots saved")
    except Exception as e:
        print(f"  Plotting skipped ({e}) — run python src/visualize.py locally")

    print("\n" + "=" * 60)
    print("Scratch run complete. Next: run experiments/run_ablation.py")
    print("=" * 60)


if __name__ == "__main__":
    main()