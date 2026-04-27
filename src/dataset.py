"""
dataset.py — Problem generation + reward functions
  1. format_prompt()       — defines what the model sees
  2. extract_answer()      — parses model output
  3. has_think_tags()      — checks format compliance
  4. reward_function()     — the main reward (test this exhaustively)
  5. generate_dataset()    — produces train/eval splits
  6. validate_reward()     — sanity check before training
"""

import re
import json
import random
from typing import Optional


# ---------------------------------------------------------------------------
# 1. PROMPT FORMAT
# ---------------------------------------------------------------------------
# We use structured XML-style tags to make parsing unambiguous.
# The <think> tag encourages step-by-step reasoning.
# The <answer> tag makes the final answer easy to extract reliably.
#
# WHY TAGS: If you ask for a free-form answer, extract_answer() becomes
# fragile — the model might put the number anywhere in its response.
# Tags give a deterministic parsing target.

def format_prompt(a: int, b: int) -> str:
    """
    Format a multiplication problem as a prompt.

    The two-tag format is borrowed from GRPO-Zero and DeepSeek-R1:
    - <think>: encourage multi-step reasoning (format reward)
    - <answer>: unambiguous final answer (answer reward)
    """
    return (
        f"Solve: {a} × {b}\n\n"
        "Show your working inside <think></think> tags.\n"
        "Give your final answer inside <answer></answer> tags.\n\n"
        "Response:"
    )


# ---------------------------------------------------------------------------
# 2. OUTPUT PARSING
# ---------------------------------------------------------------------------

def extract_answer(completion: str) -> Optional[int]:
    """
    Extract the integer inside <answer>...</answer> tags.

    Returns None if:
    - No answer tags found
    - Content inside tags is not a valid integer
    - Multiple answer tags (take the last one)

    >>> extract_answer("<think>2+2=4</think><answer>4</answer>")
    4
    >>> extract_answer("The answer is 42")  # no tags
    None
    """
    # Find all <answer> tag matches — take the last one if multiple
    matches = re.findall(r'<answer>\s*(\d+)\s*</answer>', completion)
    if matches:
        return int(matches[-1])
    return None


def has_think_tags(completion: str) -> bool:
    """
    Check if the completion uses <think> format correctly.
    Both opening AND closing tags must be present.

    >>> has_think_tags("<think>step 1</think><answer>4</answer>")
    True
    >>> has_think_tags("<answer>4</answer>")
    False
    """
    return '<think>' in completion and '</think>' in completion


# ---------------------------------------------------------------------------
# 3. REWARD FUNCTIONS
# ---------------------------------------------------------------------------
#
# TWO-COMPONENT REWARD DESIGN (from GRPO-Zero):
#
#   Format reward (0.1): fires when model uses <think> tags
#   Answer reward (1.0): fires when answer is correct
#
# WHY TWO COMPONENTS:
#   Early in training, the model almost never gets the right answer.
#   If format reward = 0 and answer reward = 0, ALL completions score 0.
#   With all-zero rewards → std(rewards) ≈ 0 → advantages ≈ 0 → zero gradient.
#   The format reward (0.1) creates contrast between completions that at least
#   try to reason vs. those that don't, giving the model a signal to start with.

def reward_function(completion: str, correct_answer: int) -> float:
    """
    Main reward function used during training.

    Returns:
        0.0  — no tags, wrong answer
        0.1  — has think tags, wrong answer
        1.0  — no tags, correct answer  (unlikely but possible)
        1.1  — has think tags, correct answer  (ideal)
    """
    reward = 0.0

    # Component 1: format reward — small, but creates early gradient signal
    if has_think_tags(completion):
        reward += 0.1

    # Component 2: answer reward — the main learning signal
    predicted = extract_answer(completion)
    if predicted is not None and predicted == correct_answer:
        reward += 1.0

    return reward


def ablation_reward_constant(completion: str, correct_answer: int) -> float:
    """
    Ablation A: constant reward = no signal.

    Used to prove the reward is causally responsible for improvement.
    If training with this reward also improves accuracy → something is
    wrong with your experimental design.
    """
    return 0.5  # constant, no information


def ablation_reward_format_only(completion: str, correct_answer: int) -> float:
    """
    Ablation B: format reward only, no answer reward.

    Tests whether trace length increases without accuracy improvement.
    Expected result: model learns to use <think> tags but still gets
    answers wrong → trace length ↑, accuracy stays flat.
    """
    return 0.1 if has_think_tags(completion) else 0.0


# ---------------------------------------------------------------------------
# 4. DATASET GENERATION
# ---------------------------------------------------------------------------

def generate_dataset(n: int, seed: int = 42) -> list:
    """
    Generate n multiplication problems.

    Task: 3-digit × 2-digit multiplication
    WHY THIS TASK:
      - Hard enough: base model gets ~10-20% correct (not trivial)
      - Easy enough: verifiable in one line of Python
      - Requires multi-step reasoning: natural <think> block emerges
      - Answer is unambiguous: no parsing ambiguity

    Returns:
        List of (a, b, answer) tuples
    """
    random.seed(seed)
    problems = []
    for _ in range(n):
        a = random.randint(100, 999)   # 3-digit
        b = random.randint(10, 99)     # 2-digit
        problems.append((a, b, a * b))
    return problems


def generate_splits(train_size: int = 200, eval_size: int = 50, probe_size: int = 5):
    """
    Generate train, eval, and probe splits.

    Probe problems are fixed — we log completions on these every N steps
    to watch the "aha moment" emerge qualitatively.

    Returns:
        train_problems, eval_problems, probe_problems
    """
    train = generate_dataset(train_size, seed=42)
    eval_ = generate_dataset(eval_size, seed=1337)   # different seed = no overlap
    probe = generate_dataset(probe_size, seed=9999)  # fixed forever
    return train, eval_, probe


def save_dataset(problems: list, path: str):
    """Save dataset to JSON for reproducibility."""
    data = [{"a": a, "b": b, "answer": ans} for a, b, ans in problems]
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(problems)} problems to {path}")


def load_dataset(path: str) -> list:
    """Load dataset from JSON."""
    with open(path, 'r') as f:
        data = json.load(f)
    return [(d["a"], d["b"], d["answer"]) for d in data]


# ---------------------------------------------------------------------------
# 5. REWARD VALIDATION — RUN THIS BEFORE TRAINING
# ---------------------------------------------------------------------------

def validate_reward(model, tokenizer, n_samples: int = 20, device: str = "cpu"):
    """
    Sanity check: what fraction of BASE MODEL outputs get non-zero reward?

    Target: 5% – 40%
    - Below 5%: reward function too strict OR task too hard
              → model gets zero gradient → nothing to learn
    - Above 50%: task too easy, model already knows it
               → not enough room to improve

    ALWAYS run this before starting training. Fix the reward function
    before writing the training loop.
    """
    import torch

    problems = generate_dataset(n_samples, seed=777)
    rewards = []
    results = []

    model.eval()
    for a, b, answer in problems:
        prompt = format_prompt(a, b)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.8,
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode only the new tokens
        completion = tokenizer.decode(
            output[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        r = reward_function(completion, answer)
        rewards.append(r)
        results.append({
            "problem": f"{a} × {b} = {answer}",
            "completion_preview": completion[:100],
            "reward": r,
            "has_think": has_think_tags(completion),
            "extracted_answer": extract_answer(completion)
        })

    # Summary stats
    hit_rate = sum(1 for r in rewards if r > 0) / len(rewards)
    correct_rate = sum(1 for r in rewards if r >= 1.0) / len(rewards)
    format_rate = sum(1 for res in results if res["has_think"]) / len(results)

    print("\n" + "="*50)
    print("REWARD VALIDATION RESULTS")
    print("="*50)
    print(f"Non-zero reward rate: {hit_rate:.1%}  (target: 5-40%)")
    print(f"Correct answer rate:  {correct_rate:.1%}")
    print(f"Format compliance:    {format_rate:.1%}  (uses <think> tags)")
    print()

    if hit_rate < 0.05:
        print("⚠️  WARNING: Too few non-zero rewards. Fix reward function or simplify task.")
    elif hit_rate > 0.5:
        print("⚠️  WARNING: Too many non-zero rewards. Task may be too easy.")
    else:
        print("✓  Reward function looks healthy. Proceed to training.")

    print("\nExample outputs:")
    for res in results[:3]:
        print(f"  {res['problem']} | reward={res['reward']} | preview: {res['completion_preview'][:60]}...")

    return hit_rate, results


# ---------------------------------------------------------------------------
# Quick self-test (run this file directly to verify parsing logic)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing extract_answer()...")
    assert extract_answer("<think>2×2=4</think><answer>4</answer>") == 4
    assert extract_answer("<answer>  123  </answer>") == 123
    assert extract_answer("The answer is 42") is None
    assert extract_answer("<answer>abc</answer>") is None
    print("  ✓ All extract_answer tests pass")

    print("Testing has_think_tags()...")
    assert has_think_tags("<think>reasoning</think>") is True
    assert has_think_tags("<think>no closing tag") is False
    assert has_think_tags("no tags at all") is False
    print("  ✓ All has_think_tags tests pass")

    print("Testing reward_function()...")
    assert reward_function("<think>step</think><answer>42</answer>", 42) == 1.1
    assert reward_function("<answer>42</answer>", 42) == 1.0
    assert reward_function("<think>step</think><answer>99</answer>", 42) == 0.1
    assert reward_function("wrong answer", 42) == 0.0
    print("  ✓ All reward_function tests pass")

    print("Testing generate_splits()...")
    train, eval_, probe = generate_splits()
    assert len(train) == 200
    assert len(eval_) == 50
    assert len(probe) == 5
    # Verify no overlap between train and eval (different seeds)
    train_set = set((a, b) for a, b, _ in train)
    eval_set = set((a, b) for a, b, _ in eval_)
    overlap = train_set & eval_set
    print(f"  Train/eval overlap: {len(overlap)} problems (expected ~0)")
    print("  ✓ Dataset generation looks correct")

    print("\n✓ All self-tests passed. dataset.py is ready.")