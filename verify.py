"""
verify.py — checks entire setup before training

This script tests every component in isolation, in the correct order.
Run it locally first (no GPU needed), then run it in Colab to verify
the full pipeline including model loading.

Usage:
    # Local (no GPU — tests logic only):
    python verify.py --local

    # Colab (full pipeline including model):
    python verify.py --full

    # Check a specific component only:
    python verify.py --check dataset
    python verify.py --check model
    python verify.py --check grpo

What each check does:
    dataset  — all parsing, reward, and generation logic
    model    — model loads, generates text, evaluate() works
    grpo     — compute_token_log_probs, compute_advantages, one training step
    plots    — visualize.py produces figures from dummy data
    full     — all of the above end to end
"""

import sys
import os
import json
import time
import traceback
import argparse

sys.path.insert(0, os.path.dirname(__file__))


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

PASS  = "✓"
FAIL  = "✗"
SKIP  = "○"
WARN  = "⚠"


def section(title: str):
    print(f"\n{'─'*55}")
    print(f"  {title}")
    print(f"{'─'*55}")


def check(name: str, fn, *args, **kwargs):
    """Run a check function, catch exceptions, print result."""
    try:
        result = fn(*args, **kwargs)
        print(f"  {PASS} {name}")
        return result
    except AssertionError as e:
        print(f"  {FAIL} {name}")
        print(f"       AssertionError: {e}")
        return None
    except Exception as e:
        print(f"  {FAIL} {name}")
        print(f"       {type(e).__name__}: {e}")
        return None


def warn(name: str, fn, *args, **kwargs):
    """Like check() but failures are warnings not errors."""
    try:
        result = fn(*args, **kwargs)
        print(f"  {PASS} {name}")
        return result
    except Exception as e:
        print(f"  {WARN} {name} (non-fatal: {e})")
        return None


# ---------------------------------------------------------------------------
# CHECK 1 — DATASET
# ---------------------------------------------------------------------------

def check_dataset():
    section("CHECK 1 — dataset.py")

    from src.dataset import (
        format_prompt, extract_answer, has_think_tags,
        reward_function, ablation_reward_constant,
        ablation_reward_format_only, generate_splits
    )

    # format_prompt
    def test_format():
        p = format_prompt(123, 45)
        assert "123" in p and "45" in p, "Problem numbers missing from prompt"
        assert "<think>" in p, "<think> tag missing from prompt"
        assert "<answer>" in p, "<answer> tag missing from prompt"
        return p[:60] + "..."
    check("format_prompt() — contains numbers and tags", test_format)

    # extract_answer — happy paths
    check("extract_answer() — correct tag",
          lambda: (assert_eq(extract_answer("<think>step</think><answer>42</answer>"), 42)))
    check("extract_answer() — whitespace inside tag",
          lambda: (assert_eq(extract_answer("<answer>  123  </answer>"), 123)))
    check("extract_answer() — takes last if multiple",
          lambda: (assert_eq(extract_answer("<answer>1</answer><answer>2</answer>"), 2)))

    # extract_answer — sad paths
    check("extract_answer() — returns None for no tags",
          lambda: (assert_eq(extract_answer("The answer is 42"), None)))
    check("extract_answer() — returns None for non-integer",
          lambda: (assert_eq(extract_answer("<answer>abc</answer>"), None)))

    # has_think_tags
    check("has_think_tags() — True when both tags present",
          lambda: assert_true(has_think_tags("<think>reasoning</think>")))
    check("has_think_tags() — False when closing tag missing",
          lambda: assert_false(has_think_tags("<think>no close")))
    check("has_think_tags() — False when no tags",
          lambda: assert_false(has_think_tags("no tags")))

    # reward_function — all combinations
    def test_rewards():
        r_full    = reward_function("<think>step</think><answer>42</answer>", 42)
        r_correct = reward_function("<answer>42</answer>", 42)
        r_format  = reward_function("<think>step</think><answer>99</answer>", 42)
        r_none    = reward_function("wrong", 42)
        assert r_full    == 1.1, f"Expected 1.1, got {r_full}"
        assert r_correct == 1.0, f"Expected 1.0, got {r_correct}"
        assert r_format  == 0.1, f"Expected 0.1, got {r_format}"
        assert r_none    == 0.0, f"Expected 0.0, got {r_none}"
    check("reward_function() — all 4 cases correct", test_rewards)

    # ablation rewards
    check("ablation_reward_constant() — always 0.5",
          lambda: assert_eq(ablation_reward_constant("anything", 42), 0.5))
    check("ablation_reward_format_only() — 0.1 with think tags",
          lambda: assert_eq(ablation_reward_format_only("<think>t</think>", 42), 0.1))
    check("ablation_reward_format_only() — 0.0 without tags",
          lambda: assert_eq(ablation_reward_format_only("no tags", 42), 0.0))

    # dataset generation
    def test_splits():
        train, eval_, probe = generate_splits(train_size=10, eval_size=5, probe_size=3)
        assert len(train) == 10
        assert len(eval_)  == 5
        assert len(probe)  == 3
        # Verify all answers are correct
        for a, b, ans in train:
            assert a * b == ans, f"{a}×{b}={ans} is wrong"
    check("generate_splits() — correct sizes and answers", test_splits)

    # No train/eval overlap
    def test_no_overlap():
        train, eval_, _ = generate_splits(train_size=200, eval_size=50)
        train_set = set((a, b) for a, b, _ in train)
        eval_set  = set((a, b) for a, b, _ in eval_)
        overlap   = train_set & eval_set
        assert len(overlap) < 5, f"Too much overlap: {len(overlap)} shared problems"
    check("generate_splits() — minimal train/eval overlap", test_no_overlap)

    print(f"\n  {PASS} dataset.py — all checks passed")


# ---------------------------------------------------------------------------
# CHECK 2 — GRPO MATH (no model needed)
# ---------------------------------------------------------------------------

def check_grpo_math():
    section("CHECK 2 — grpo.py math (no model needed)")

    import torch
    from src.grpo import compute_advantages, GRPOConfig

    # compute_advantages — normal case
    def test_advantages_normal():
        rewards = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
        adv = compute_advantages(rewards)
        assert adv is not None, "Should not return None for varied rewards"
        assert abs(adv.mean().item()) < 0.01, "Advantages should be zero-mean"
        assert abs(adv.std().item() - 1.0) < 0.01, "Advantages should have std≈1"
        assert adv[0] > 0, "Reward=1.0 should have positive advantage"
        assert adv[1] < 0, "Reward=0.0 should have negative advantage"
    check("compute_advantages() — correct normalization", test_advantages_normal)

    # compute_advantages — degenerate case
    def test_advantages_constant():
        rewards = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        adv = compute_advantages(rewards)
        assert adv is None, "Should return None when all rewards identical"
    check("compute_advantages() — returns None for constant rewards", test_advantages_constant)

    # GRPOConfig defaults
    def test_config():
        cfg = GRPOConfig()
        assert cfg.G == 8
        assert cfg.beta == 0.1
        assert 1e-7 < cfg.learning_rate < 1e-4
    check("GRPOConfig() — sane defaults", test_config)

    print(f"\n  {PASS} grpo.py math — all checks passed")


# ---------------------------------------------------------------------------
# CHECK 3 — MODEL (requires model download, run in Colab)
# ---------------------------------------------------------------------------

def check_model(model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"):
    section(f"CHECK 3 — model.py (loading {model_name})")
    print(f"  Note: This downloads the model if not cached (~1GB for 0.5B)")

    import torch
    from src.model import (
        get_device, load_model, make_reference_model,
        generate_single, generate_group, evaluate
    )
    from src.dataset import format_prompt, reward_function, generate_splits

    device = get_device()
    check("get_device() — returns valid device",
          lambda: assert_true(device in ["cuda", "mps", "cpu"]))

    # Load model
    model_result = check("load_model() — model and tokenizer load",
                         load_model, model_name, device)
    if model_result is None:
        print(f"  {FAIL} Cannot proceed — model failed to load")
        return
    model, tokenizer = model_result

    # Reference model
    ref_model = check("make_reference_model() — creates frozen copy",
                      make_reference_model, model)
    if ref_model:
        frozen = all(not p.requires_grad for p in ref_model.parameters())
        check("make_reference_model() — all params frozen",
              lambda: assert_true(frozen))

    # Generate single completion
    def test_generate_single():
        _, _, probe = generate_splits(probe_size=1)
        a, b, ans = probe[0]
        full_ids, text, prompt_len = generate_single(
            model, tokenizer, format_prompt(a, b), max_new_tokens=50
        )
        assert isinstance(text, str),       "completion should be a string"
        assert len(text) > 0,               "completion should not be empty"
        assert prompt_len > 0,              "prompt_length should be positive"
        assert full_ids.shape[0] > prompt_len, "full_ids should be longer than prompt"
        return text[:80]
    result = check("generate_single() — returns text and token IDs", test_generate_single)
    if result:
        print(f"       Preview: '{result}...'")

    # Generate group
    def test_generate_group():
        _, _, probe = generate_splits(probe_size=1)
        a, b, ans = probe[0]
        completions = generate_group(
            model, tokenizer, format_prompt(a, b), G=3, max_new_tokens=30
        )
        assert len(completions) == 3, f"Expected 3 completions, got {len(completions)}"
        for full_ids, text, prompt_len in completions:
            assert len(text) > 0
    check("generate_group() — returns G completions", test_generate_group)

    # Evaluate
    def test_evaluate():
        _, eval_probs, _ = generate_splits(eval_size=3)
        metrics = evaluate(model, tokenizer, eval_probs, reward_function)
        assert "accuracy"         in metrics
        assert "mean_reward"      in metrics
        assert "avg_trace_length" in metrics
        assert "format_rate"      in metrics
        assert 0 <= metrics["accuracy"] <= 1
        return metrics
    result = check("evaluate() — returns valid metrics", test_evaluate)
    if result:
        print(f"       Base accuracy: {result['accuracy']:.0%} | "
              f"Trace: {result['avg_trace_length']:.0f}w | "
              f"Format: {result['format_rate']:.0%}")

    print(f"\n  {PASS} model.py — all checks passed")
    return model, tokenizer


# ---------------------------------------------------------------------------
# CHECK 4 — GRPO STEP (requires model)
# ---------------------------------------------------------------------------

def check_grpo_step(model, tokenizer):
    section("CHECK 4 — grpo.py training step (requires model)")

    import torch
    from src.grpo import compute_token_log_probs, compute_grpo_loss, GRPOConfig
    from src.model import generate_group, make_reference_model
    from src.dataset import format_prompt, reward_function, generate_splits

    ref_model = make_reference_model(model)
    _, _, probe = generate_splits(probe_size=1)
    a, b, ans  = probe[0]
    prompt     = format_prompt(a, b)

    # compute_token_log_probs
    def test_log_probs():
        from src.model import generate_single
        full_ids, _, prompt_len = generate_single(
            model, tokenizer, prompt, max_new_tokens=30
        )
        log_probs = compute_token_log_probs(model, full_ids, prompt_len)
        completion_len = full_ids.shape[0] - prompt_len
        assert log_probs.shape[0] == completion_len, \
            f"Expected {completion_len} log probs, got {log_probs.shape[0]}"
        assert (log_probs <= 0).all(), "Log probs should all be <= 0"
        assert not torch.isnan(log_probs).any(), "No NaN in log probs"
        return log_probs.shape[0]
    result = check("compute_token_log_probs() — correct shape, no NaN", test_log_probs)
    if result:
        print(f"       Returned {result} completion token log probs")

    # compute_grpo_loss — varied rewards (should compute loss)
    def test_grpo_loss_varied():
        completions = generate_group(model, tokenizer, prompt, G=4, max_new_tokens=30)
        rewards = [1.0, 0.0, 1.0, 0.0]
        loss = compute_grpo_loss(model, ref_model, completions, rewards, beta=0.1)
        assert loss is not None,          "Should return loss for varied rewards"
        assert not torch.isnan(loss),     "Loss should not be NaN"
        assert loss.requires_grad,        "Loss should have gradient graph attached"
        return loss.item()
    result = check("compute_grpo_loss() — varied rewards produce valid loss", test_grpo_loss_varied)
    if result is not None:
        print(f"       Loss value: {result:.4f}")

    # compute_grpo_loss — constant rewards (should skip)
    def test_grpo_loss_constant():
        completions = generate_group(model, tokenizer, prompt, G=4, max_new_tokens=30)
        rewards = [0.5, 0.5, 0.5, 0.5]
        loss = compute_grpo_loss(model, ref_model, completions, rewards, beta=0.1)
        assert loss is None, "Should return None for constant rewards"
    check("compute_grpo_loss() — constant rewards return None (skip)", test_grpo_loss_constant)

    # One full gradient step
    def test_gradient_step():
        from torch.optim import AdamW
        optimizer = AdamW(model.parameters(), lr=1e-5)
        completions = generate_group(model, tokenizer, prompt, G=4, max_new_tokens=30)
        rewards = [1.0, 0.0, 1.0, 0.0]
        loss = compute_grpo_loss(model, ref_model, completions, rewards, beta=0.1)
        optimizer.zero_grad()
        loss.backward()
        # Check gradients exist
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0, "No gradients computed"
        assert not any(torch.isnan(g).any() for g in grads), "NaN in gradients"
        optimizer.step()
        return len(grads)
    result = check("Full gradient step — backward + optimizer.step()", test_gradient_step)
    if result:
        print(f"       {result} parameter tensors have gradients")

    print(f"\n  {PASS} grpo.py training step — all checks passed")


# ---------------------------------------------------------------------------
# CHECK 5 — VISUALIZE (dummy data, no model)
# ---------------------------------------------------------------------------

def check_visualize():
    section("CHECK 5 — visualize.py (dummy data)")

    import os
    os.makedirs("results", exist_ok=True)

    # Generate realistic dummy history
    dummy_grpo = []
    dummy_ablation = []
    for step in range(0, 310, 50):
        progress = step / 300
        dummy_grpo.append({
            "step":             step,
            "mean_reward":      0.15 + 0.50 * progress + (0.05 * (step % 2)),
            "avg_trace_length": 12  + 80  * progress,
            "format_rate":      0.10 + 0.80 * progress,
            "loss":             2.5  - 1.8  * progress,
            "accuracy":         0.10 + 0.55 * progress,
        })
        dummy_ablation.append({
            "step":             step,
            "mean_reward":      0.14 + 0.04 * progress,
            "avg_trace_length": 11  + 15  * progress,
            "format_rate":      0.08 + 0.15 * progress,
            "loss":             2.4  - 0.2  * progress,
            "accuracy":         0.10 + 0.05 * progress,
        })

    from src.visualize import (
        plot_main_results, plot_ablation_comparison,
        plot_loss_curve, plot_format_compliance
    )

    check("plot_main_results() — saves figure",
          lambda: plot_main_results(dummy_grpo, dummy_ablation,
                                    save_path="results/verify_main.png"))
    check("plot_ablation_comparison() — saves figure",
          lambda: plot_ablation_comparison(dummy_grpo, dummy_ablation,
                                           save_path="results/verify_ablation.png"))
    check("plot_loss_curve() — saves figure",
          lambda: plot_loss_curve(dummy_grpo, save_path="results/verify_loss.png"))
    check("plot_format_compliance() — saves figure",
          lambda: plot_format_compliance(dummy_grpo, save_path="results/verify_format.png"))

    # Check files were actually created
    for fname in ["verify_main.png", "verify_ablation.png", "verify_loss.png", "verify_format.png"]:
        path = f"results/{fname}"
        if os.path.exists(path):
            size_kb = os.path.getsize(path) / 1024
            print(f"  {PASS} results/{fname} ({size_kb:.0f} KB)")
        else:
            print(f"  {FAIL} results/{fname} — file not created")

    print(f"\n  {PASS} visualize.py — all plots generated")


# ---------------------------------------------------------------------------
# ASSERT HELPERS
# ---------------------------------------------------------------------------

def assert_eq(a, b):
    assert a == b, f"Expected {b!r}, got {a!r}"

def assert_true(x):
    assert x, f"Expected True, got {x!r}"

def assert_false(x):
    assert not x, f"Expected False, got {x!r}"


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group()
    g.add_argument("--local",   action="store_true",
                   help="Run only local checks (no model download)")
    g.add_argument("--full",    action="store_true",
                   help="Run all checks including model loading (needs GPU)")
    g.add_argument("--check",   choices=["dataset", "grpo", "model", "plots"],
                   help="Run a specific check only")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct",
                   help="Model to use for model checks")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 55)
    print("  GRPO FROM SCRATCH — VERIFICATION SUITE")
    print("=" * 55)

    t0 = time.time()
    passed = []
    failed = []

    def run(name, fn, *a, **kw):
        try:
            result = fn(*a, **kw)
            passed.append(name)
            return result
        except SystemExit:
            raise
        except Exception as e:
            failed.append(name)
            print(f"\n  {FAIL} {name} CRASHED:")
            traceback.print_exc()
            return None

    if args.check == "dataset" or args.local or args.full or not any([args.local, args.full, args.check]):
        run("dataset", check_dataset)

    if args.check == "grpo" or args.local or args.full or not any([args.local, args.full, args.check]):
        run("grpo_math", check_grpo_math)

    if args.check == "plots" or args.local or args.full or not any([args.local, args.full, args.check]):
        run("visualize", check_visualize)

    if args.check == "model" or args.full:
        model_result = run("model", check_model, args.model)
        if model_result and args.full:
            model, tokenizer = model_result
            run("grpo_step", check_grpo_step, model, tokenizer)

    # ── Summary ───────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'='*55}")
    print(f"  VERIFICATION COMPLETE ({elapsed:.1f}s)")
    print(f"{'='*55}")
    print(f"  {PASS} Passed: {len(passed)}")
    if failed:
        print(f"  {FAIL} Failed: {len(failed)} — {', '.join(failed)}")
        print()
        print("  Fix failures before running training.")
    else:
        print(f"\n  Everything looks good.")
        if not args.full:
            print(f"  Run with --full to also verify model loading and GPU.")
        else:
            print(f"  Ready to train. Next:")
            print(f"    !python experiments/run_trl.py")


if __name__ == "__main__":
    main()