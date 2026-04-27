"""
experiments/run_trl.py — Run GRPO training using HuggingFace TRL

incase of failure:
  - Check reward validation output — hit rate must be 5-40%
  - Check VRAM — Qwen2.5-1.5B needs ~3GB in float16
  - Check TRL version — needs >= 0.8.6 for GRPOTrainer

Usage (Colab):
    !python experiments/run_trl.py
    !python experiments/run_trl.py --model Qwen/Qwen2.5-0.5B-Instruct --steps 100
"""

import sys
import os
import json
import argparse

# Make src/ importable regardless of where you run this from
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from datasets import Dataset
from trl import GRPOConfig as TRLGRPOConfig, GRPOTrainer

from src.dataset import (
    generate_splits, format_prompt,
    reward_function, validate_reward, save_dataset
)
from src.model import load_model, get_device, evaluate, save_probe_traces
from src.visualize import plot_main_results, plot_loss_curve


# ---------------------------------------------------------------------------
# ARGUMENT PARSING
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",   default="Qwen/Qwen2.5-1.5B-Instruct",
                   help="HuggingFace model name")
    p.add_argument("--steps",   type=int, default=300,
                   help="Number of training steps")
    p.add_argument("--G",       type=int, default=8,
                   help="Group size (completions per prompt)")
    p.add_argument("--lr",      type=float, default=5e-6,
                   help="Learning rate")
    p.add_argument("--output",  default="results",
                   help="Output directory")
    p.add_argument("--skip-validation", action="store_true",
                   help="Skip reward validation (not recommended)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# TRL REWARD WRAPPER
# ---------------------------------------------------------------------------

def _extract_completion_text(completion) -> str:
    """
    Normalise a TRL completion to plain text.

    TRL's completion format changed across versions:
      Old (<0.9):  completions is list[str]
      New (>=0.9): completions is list[list[dict]] chat message format
                   e.g. [[{"role": "assistant", "content": "..."}]]
    """
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and len(completion) > 0:
        msg = completion[-1]
        if isinstance(msg, dict):
            return msg.get("content", "")
    return str(completion)


def make_trl_reward_fn(eval_problems):
    """
    Build a TRL-compatible reward function.
    Signature: fn(completions, answer, **kwargs) -> list[float]
    The 'answer' kwarg comes from the dataset column named 'answer'.
    """
    def trl_reward(completions, answer, **kwargs):
        return [
            reward_function(_extract_completion_text(c), int(ans))
            for c, ans in zip(completions, answer)
        ]
    return trl_reward


# ---------------------------------------------------------------------------
# DATASET FORMATTING FOR TRL
# ---------------------------------------------------------------------------

def make_hf_dataset(problems: list) -> Dataset:
    """
    TRL's GRPOTrainer expects a HuggingFace Dataset with a 'prompt' column.
    Any extra columns get passed as kwargs to the reward function.

    We add 'answer' so the reward function can check correctness.
    """
    return Dataset.from_dict({
        "prompt": [format_prompt(a, b) for a, b, _ in problems],
        "answer": [str(ans)            for _, _, ans in problems],
    })


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    print("=" * 60)
    print("GRPO TRAINING — TRL VERSION")
    print("=" * 60)
    print(f"Model:  {args.model}")
    print(f"Steps:  {args.steps}")
    print(f"G:      {args.G}")
    print(f"LR:     {args.lr}")
    print(f"Output: {args.output}")
    print()

    # ── 1. Hardware ───────────────────────────────────────────────────────
    device = get_device()

    # ── 2. Data ───────────────────────────────────────────────────────────
    print("Generating datasets...")
    train_problems, eval_problems, probe_problems = generate_splits(
        train_size=200, eval_size=50, probe_size=5
    )
    save_dataset(train_problems, f"{args.output}/train_problems.json")
    save_dataset(eval_problems,  f"{args.output}/eval_problems.json")
    save_dataset(probe_problems, f"{args.output}/probe_problems.json")
    print(f"  Train: {len(train_problems)} | Eval: {len(eval_problems)} | Probe: {len(probe_problems)}")

    # ── 3. Load model ─────────────────────────────────────────────────────
    model, tokenizer = load_model(args.model, device)

    # ── 4. Validate reward BEFORE training ────────────────────────────────
    if not args.skip_validation:
        print("\nValidating reward function on base model...")
        hit_rate, _ = validate_reward(model, tokenizer, n_samples=20, device=device)
        if hit_rate < 0.03:
            print("\n✗ STOPPING: Reward hit rate too low. Fix reward_function() first.")
            return
        if hit_rate > 0.7:
            print("\n✗ STOPPING: Reward hit rate too high. Task may be too easy.")
            return
        print(f"✓ Reward validation passed ({hit_rate:.1%} hit rate)")

    # ── 5. Save baseline traces ───────────────────────────────────────────
    print("\nSaving baseline traces (step 0)...")
    save_probe_traces(model, tokenizer, probe_problems, step=0)

    # ── 6. Baseline eval ──────────────────────────────────────────────────
    print("Running baseline evaluation...")
    baseline = evaluate(model, tokenizer, eval_problems, reward_function)
    print(f"  Accuracy:     {baseline['accuracy']:.1%}")
    print(f"  Trace length: {baseline['avg_trace_length']:.0f} words avg")
    print(f"  Format rate:  {baseline['format_rate']:.1%}")

    history = [{"step": 0, **{k: v for k, v in baseline.items() if k != "results"}}]

    # ── 7. Build TRL dataset + reward fn ─────────────────────────────────
    train_dataset = make_hf_dataset(train_problems)
    trl_reward_fn = make_trl_reward_fn(eval_problems)

    # ── 8. TRL GRPOConfig ─────────────────────────────────────────────────
    # Parameter names changed across TRL versions:
    #   max_new_tokens      → max_completion_length   (breaking change ~0.9+)
    #   num_generations     → num_generations          (stable)
    #   beta                → beta                     (stable)
    # We detect the installed version and use the right names automatically.
    import trl as _trl
    from packaging.version import Version as V

    trl_version = V(_trl.__version__)
    use_new_api = trl_version >= V("0.9.0")

    print(f"  TRL version: {_trl.__version__} → using {'new' if use_new_api else 'old'} API")

    common_kwargs = dict(
        output_dir=f"{args.output}/trl_checkpoints",
        max_steps=args.steps,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,          # effective batch size = 4
        learning_rate=args.lr,
        num_generations=args.G,                 # completions per prompt (was G)
        temperature=0.8,
        beta=0.1,                               # KL penalty weight
        logging_steps=10,
        save_steps=100,
        report_to="none",                       # swap to "wandb" if you want curves
        remove_unused_columns=False,            # keeps 'answer' col for reward fn
    )

    if use_new_api:
        common_kwargs["max_completion_length"] = 256   # TRL >= 0.9
    else:
        common_kwargs["max_new_tokens"] = 256          # TRL < 0.9

    trl_config = TRLGRPOConfig(**common_kwargs)

    # ── 9. Train ──────────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("Starting TRL GRPO training...")
    print("─" * 60)

    trainer = GRPOTrainer(
        model=model,
        args=trl_config,
        train_dataset=train_dataset,
        reward_funcs=trl_reward_fn,
    )

    # Custom callback to run our eval every 50 steps and save traces
    class EvalCallback:
        def __init__(self, eval_interval=50):
            self.eval_interval = eval_interval
            self.step = 0

        def on_step_end(self, args, state, control, **kwargs):
            self.step += 1
            if self.step % self.eval_interval == 0:
                print(f"\n  → Eval at step {self.step}...")
                metrics = evaluate(model, tokenizer, eval_problems, reward_function)
                entry = {"step": self.step, **{k: v for k, v in metrics.items() if k != "results"}}
                history.append(entry)
                print(f"     Accuracy: {metrics['accuracy']:.1%}  "
                      f"Trace: {metrics['avg_trace_length']:.0f}w  "
                      f"Format: {metrics['format_rate']:.1%}")
                save_probe_traces(model, tokenizer, probe_problems, step=self.step)

                # Save history checkpoint
                with open(f"{args.output}/trl_history.json", 'w') as f:
                    json.dump(history, f, indent=2)

    # Note: TRL callback integration varies by version
    # If this doesn't work, eval manually after training
    try:
        from transformers import TrainerCallback

        class GRPOEvalCallback(TrainerCallback):
            def __init__(self):
                self.eval_cb = EvalCallback(eval_interval=50)

            def on_step_end(self, args, state, control, **kwargs):
                self.eval_cb.on_step_end(args, state, control, **kwargs)
                return control

        trainer.add_callback(GRPOEvalCallback())
    except Exception as e:
        print(f"  Note: Callback setup failed ({e}) — will eval after training")

    trainer.train()

    # ── 10. Final evaluation ──────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("Training complete. Running final evaluation...")
    final = evaluate(model, tokenizer, eval_problems, reward_function)
    history.append({"step": args.steps, **{k: v for k, v in final.items() if k != "results"}})

    print(f"\n  Final accuracy:     {final['accuracy']:.1%}  (was {baseline['accuracy']:.1%})")
    print(f"  Final trace length: {final['avg_trace_length']:.0f}w  (was {baseline['avg_trace_length']:.0f}w)")
    print(f"  Final format rate:  {final['format_rate']:.1%}  (was {baseline['format_rate']:.1%})")

    # ── 11. Save final traces ─────────────────────────────────────────────
    save_probe_traces(model, tokenizer, probe_problems, step=args.steps)

    # ── 12. Save everything ───────────────────────────────────────────────
    with open(f"{args.output}/trl_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\n✓ History saved → {args.output}/trl_history.json")

    model.save_pretrained(f"{args.output}/trl_final_model")
    print(f"✓ Model saved   → {args.output}/trl_final_model")

    # ── 13. Quick plots (if matplotlib available) ─────────────────────────
    try:
        plot_main_results(history, save_path=f"{args.output}/trl_main_results.png")
        plot_loss_curve(history,   save_path=f"{args.output}/trl_loss_curve.png")
        print("✓ Plots saved")
    except Exception as e:
        print(f"  Plotting skipped ({e}) — run python src/visualize.py locally")

    print("\n" + "=" * 60)
    print("TRL run complete. Next: run experiments/run_scratch.py")
    print("=" * 60)


if __name__ == "__main__":
    main()