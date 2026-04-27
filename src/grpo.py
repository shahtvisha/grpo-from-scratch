"""
grpo.py — The GRPO training loop, written from scratch

  Algorithm (from DeepSeekMath paper, Section 3):
  ─────────────────────────────────────────────────────────
  For each training step:
    1. Sample a batch of prompts
    2. For each prompt, generate G completions (rollout)
    3. Score each completion with the reward function
    4. Compute group-relative advantages: Â = (r - mean(r)) / std(r)
    5. Compute policy gradient loss: -Â * log π_θ(completion)
    6. Add KL penalty: β * KL(π_θ || π_ref)
    7. Backprop + optimizer step
  ─────────────────────────────────────────────────────────

References:
  - DeepSeekMath (Shao et al., 2024): arXiv:2402.03300  ← original GRPO
  - nano-aha-moment (McGill NLP):     github.com/McGill-NLP/nano-aha-moment
  - GRPO-Zero:                        github.com/policy-gradient/GRPO-Zero
  - aburkov/theLMbook:                github.com/aburkov/theLMbook
"""

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Callable
import json
import os
import time


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

@dataclass
class GRPOConfig:
    """
    All hyperparameters in one place.

    Start with these defaults — they're validated by nano-aha-moment and
    aburkov's notebook on 1.5B models.
    """

    # ── Core algorithm ──────────────────────────────────────────────────────
    G: int = 8
    """
    Group size — completions generated per prompt.
    WHY 8: Standard from DeepSeekMath paper. More = better advantage estimates
    but more VRAM. On T4 Colab with 1.5B model, 8 is the practical limit.
    Minimum useful value: 4. Below that, advantage estimates are too noisy.
    """

    beta: float = 0.1
    """
    KL penalty weight.
    WHY 0.1: From aburkov's validated notebook. Too high → no learning
    (KL dominates). Too low → policy collapse (model outputs gibberish
    that technically maximizes reward but is incoherent).
    If training is unstable: try 0.04 (from DeepSeekMath paper).
    """

    # ── Training ─────────────────────────────────────────────────────────────
    num_steps: int = 300
    """300 steps is enough to see clear improvement on multiplication. 
    Full convergence takes ~500-1000 steps but the trend is visible by 200."""

    batch_size: int = 4
    """Prompts per step. Each prompt generates G completions → 4×8=32 completions/step."""

    learning_rate: float = 5e-6
    """Very low LR. We're fine-tuning a pretrained model, not training from scratch.
    aburkov uses 5e-6. DeepSeekMath uses 1e-6. If loss spikes: halve this."""

    grad_clip: float = 1.0
    """Clip gradient norm to this value. Prevents exploding gradients."""

    # ── Generation ───────────────────────────────────────────────────────────
    max_new_tokens: int = 256
    temperature: float = 0.8
    """Higher temperature = more diverse completions = better advantage spread.
    Too low (0.3): all completions similar → advantages all ~0 → no learning.
    Too high (1.5): incoherent completions → reward always 0 → no learning."""

    # ── Logging ──────────────────────────────────────────────────────────────
    log_every: int = 10
    eval_every: int = 50
    save_traces_every: int = 50
    output_dir: str = "results"


# ---------------------------------------------------------------------------
# CORE MATH: LOG PROBABILITIES
# ---------------------------------------------------------------------------

def compute_token_log_probs(
    model,
    full_ids: torch.Tensor,
    prompt_length: int
) -> torch.Tensor:
    """
    Compute log P(token_t | tokens_0..t-1) for each COMPLETION token.

    This is the most critical function. Get this wrong and nothing works.

    The key insight — we need log probs for only the completion tokens:
    ┌─────────────────────┬──────────────────────────────────────┐
    │   PROMPT TOKENS     │         COMPLETION TOKENS            │
    │  (not our business) │   ← we want log probs for THESE      │
    └─────────────────────┴──────────────────────────────────────┘
    Idx: 0  1  2  3  4  5    6   7   8   9   10  11  12  13
                              ↑
                        prompt_length = 6

    Transformer forward pass gives logits at every position.
    Logit at position i predicts token at position i+1.
    So we shift: log_probs[i] = log P(token[i+1] | token[0..i])

    Args:
        model: policy OR reference model (same function for both)
        full_ids: [seq_len] tensor — prompt + completion concatenated
        prompt_length: number of prompt tokens (to slice out completion)

    Returns:
        [completion_len] tensor of log probabilities, one per completion token
    """
    # Add batch dimension: [seq_len] → [1, seq_len]
    input_ids = full_ids.unsqueeze(0).to(model.device)

    # Forward pass — logits shape: [1, seq_len, vocab_size]
    # NOTE: no torch.no_grad() here — caller decides whether to use gradients
    outputs = model(input_ids=input_ids)
    logits = outputs.logits[0]  # [seq_len, vocab_size]

    # Convert logits to log probabilities
    # log_softmax is numerically more stable than log(softmax(x))
    log_probs = F.log_softmax(logits[:-1], dim=-1)   # [seq_len-1, vocab]
    labels    = input_ids[0, 1:]                      # [seq_len-1]

    # For each position, get log prob of the token that actually appeared
    # gather() selects one value per row: log_probs[i, labels[i]]
    token_log_probs = log_probs.gather(
        dim=-1,
        index=labels.unsqueeze(-1)  # [seq_len-1, 1]
    ).squeeze(-1)                   # [seq_len-1]

    # Return only completion token log probs (skip prompt positions)
    # prompt_length-1 because of the shift (we lost one position from shifting)
    return token_log_probs[prompt_length - 1:]


# ---------------------------------------------------------------------------
# CORE MATH: GRPO LOSS
# ---------------------------------------------------------------------------

def compute_advantages(rewards: List[float]) -> Optional[torch.Tensor]:
    """
    Compute group-relative advantages from a list of rewards.

    Formula: Â_i = (r_i - mean(r)) / (std(r) + ε)

    This is the "group relative" in GRPO.
    Instead of comparing rewards to an absolute baseline (like a value function),
    we compare each reward to the MEAN of the group.

    Result: completions better than average get positive advantage (reinforced)
            completions worse than average get negative advantage (suppressed)

    Returns None if all rewards are the same → no contrast → skip this step.

    >>> compute_advantages([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0])
    tensor([ 0.87, -1.16,  0.87, -1.16, ... ])
    """
    rewards_t = torch.tensor(rewards, dtype=torch.float32)

    std = rewards_t.std()
    if std < 1e-8:
        # All rewards identical → advantages all zero → zero gradient → skip
        # This happens when: all completions correct (std≈0) OR all wrong (std≈0)
        return None

    advantages = (rewards_t - rewards_t.mean()) / (std + 1e-8)
    return advantages


def compute_grpo_loss(
    policy_model,
    ref_model,
    completions: List[Tuple[torch.Tensor, str, int]],
    rewards: List[float],
    beta: float
) -> Optional[torch.Tensor]:
    """
    Compute the GRPO loss for one group of completions.

    Loss = Policy Gradient Loss + KL Penalty

    ┌─── Policy Gradient Loss ────────────────────────────────────────────┐
    │  L_PG = -(1/G) Σ_i Â_i * mean_t[log π_θ(t | context)]             │
    │                                                                      │
    │  Intuition: Increase probability of high-advantage completions,     │
    │             decrease probability of low-advantage ones.             │
    └──────────────────────────────────────────────────────────────────────┘
    ┌─── KL Penalty ──────────────────────────────────────────────────────┐
    │  L_KL = β * KL(π_θ || π_ref)                                        │
    │       ≈ β * mean_t[π_θ(t) * (log π_θ(t) - log π_ref(t))]           │
    │                                                                      │
    │  Intuition: Don't drift too far from the original model.            │
    │  Without this, the model exploits formatting tricks to get reward   │
    │  without actually reasoning better (reward hacking).                │
    └──────────────────────────────────────────────────────────────────────┘

    Args:
        policy_model: the model being trained (requires grad)
        ref_model:    frozen reference model (no grad)
        completions:  list of G (full_ids, text, prompt_len) tuples
        rewards:      list of G reward values
        beta:         KL penalty weight

    Returns:
        scalar loss tensor (with gradient graph attached), or None if skipped
    """
    advantages = compute_advantages(rewards)
    if advantages is None:
        return None  # all same reward, nothing to learn

    policy_model.train()
    total_loss = torch.zeros(1, requires_grad=False).to(policy_model.device)
    # We'll accumulate loss by summing tensors that have gradients

    for i, (full_ids, _, prompt_len) in enumerate(completions):

        # ── Policy log probs (WITH gradients) ────────────────────────────
        policy_log_probs = compute_token_log_probs(
            policy_model, full_ids, prompt_len
        )
        # Shape: [completion_len] with gradient graph attached

        # ── Reference log probs (WITHOUT gradients) ───────────────────────
        with torch.no_grad():
            ref_log_probs = compute_token_log_probs(
                ref_model, full_ids, prompt_len
            )
        # Shape: [completion_len], detached

        # ── Policy gradient term ─────────────────────────────────────────
        # Mean over tokens: each completion contributes one scalar to the loss.
        # Negative because we want to MAXIMIZE reward (gradient ascent),
        # but PyTorch optimizers do gradient DESCENT.
        advantage_i = advantages[i].to(policy_model.device)
        pg_loss = -(advantage_i * policy_log_probs.mean())

        # ── KL divergence term ───────────────────────────────────────────
        # KL(π_θ || π_ref) = Σ π_θ(t) * (log π_θ(t) - log π_ref(t))
        # Approximated per-token using current log probs.
        # policy_log_probs.exp() ≈ π_θ(t) for each token
        kl = (policy_log_probs.exp() * (policy_log_probs - ref_log_probs)).mean()

        # ── Combine ──────────────────────────────────────────────────────
        completion_loss = pg_loss + beta * kl
        total_loss = total_loss + completion_loss

    # Average over the group
    return total_loss / len(completions)


# ---------------------------------------------------------------------------
# TRAINING LOOP
# ---------------------------------------------------------------------------

def train(
    policy_model,
    ref_model,
    tokenizer,
    train_problems: list,
    eval_problems: list,
    probe_problems: list,
    reward_fn: Callable,
    config: GRPOConfig
) -> List[dict]:
    """
    Main GRPO training loop.

    Pseudocode:
    ───────────────────────────────────────────────────────────
    history = []
    for step in range(num_steps):
        batch = sample(train_problems, batch_size)
        batch_loss = 0
        for problem in batch:
            completions = generate_group(model, problem, G)
            rewards     = [reward_fn(c, answer) for c in completions]
            loss        = compute_grpo_loss(model, ref, completions, rewards)
            batch_loss += loss
        backprop(batch_loss)
        optimizer.step()
        if step % eval_every == 0:
            metrics = evaluate(model, eval_problems)
            history.append(metrics)
    return history
    ───────────────────────────────────────────────────────────

    Returns:
        history: list of metric dicts, one per logged step
    """
    import random
    from src.model import generate_group, evaluate, save_probe_traces

    os.makedirs(config.output_dir, exist_ok=True)

    optimizer = AdamW(policy_model.parameters(), lr=config.learning_rate)
    history = []

    # Save baseline traces BEFORE training starts
    print("Saving baseline traces (step 0)...")
    save_probe_traces(policy_model, tokenizer, probe_problems, step=0)

    # Baseline evaluation
    print("Running baseline evaluation...")
    baseline = evaluate(policy_model, tokenizer, eval_problems, reward_fn)
    print(f"  Baseline accuracy: {baseline['accuracy']:.1%}")
    print(f"  Baseline avg trace length: {baseline['avg_trace_length']:.0f} words")

    history.append({"step": 0, **baseline})
    t0 = time.time()

    for step in range(1, config.num_steps + 1):

        # ── Sample batch ──────────────────────────────────────────────────
        batch = random.sample(
            train_problems,
            min(config.batch_size, len(train_problems))
        )

        # ── Accumulate loss over batch ────────────────────────────────────
        batch_loss = None
        batch_rewards = []
        n_skipped = 0

        for (a, b, correct_answer) in batch:
            from src.dataset import format_prompt
            prompt = format_prompt(a, b)

            # Rollout: generate G completions
            completions = generate_group(
                policy_model, tokenizer, prompt,
                G=config.G,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature
            )

            # Score completions
            rewards = [
                reward_fn(text, correct_answer)
                for _, text, _ in completions
            ]
            batch_rewards.extend(rewards)

            # Compute GRPO loss for this problem's group
            loss = compute_grpo_loss(
                policy_model, ref_model,
                completions, rewards,
                config.beta
            )

            if loss is None:
                n_skipped += 1
                continue

            batch_loss = loss if batch_loss is None else batch_loss + loss

        # ── Optimizer step ────────────────────────────────────────────────
        if batch_loss is not None:
            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                policy_model.parameters(),
                config.grad_clip
            )
            optimizer.step()

        # ── Logging ──────────────────────────────────────────────────────
        if step % config.log_every == 0:
            mean_reward = sum(batch_rewards) / len(batch_rewards) if batch_rewards else 0
            loss_val = batch_loss.item() if batch_loss is not None else 0
            elapsed = time.time() - t0
            print(
                f"Step {step:4d}/{config.num_steps} | "
                f"Loss: {loss_val:.4f} | "
                f"Mean reward: {mean_reward:.3f} | "
                f"Skipped: {n_skipped}/{config.batch_size} | "
                f"Elapsed: {elapsed:.0f}s"
            )

        # ── Full eval ────────────────────────────────────────────────────
        if step % config.eval_every == 0:
            print(f"\n  → Running eval at step {step}...")
            metrics = evaluate(policy_model, tokenizer, eval_problems, reward_fn)
            print(f"     Accuracy: {metrics['accuracy']:.1%}  "
                  f"Trace length: {metrics['avg_trace_length']:.0f}w  "
                  f"Format rate: {metrics['format_rate']:.1%}")
            history.append({"step": step, **metrics})

            # Save checkpoint of history so far
            with open(f"{config.output_dir}/training_history.json", 'w') as f:
                json.dump(_serialize_history(history), f, indent=2)

        # ── Save qualitative traces ───────────────────────────────────────
        if step % config.save_traces_every == 0:
            save_probe_traces(
                policy_model, tokenizer, probe_problems, step=step
            )

    print(f"\n✓ Training complete. {config.num_steps} steps in {(time.time()-t0)/60:.1f} min")
    return history


def _serialize_history(history: List[dict]) -> List[dict]:
    """Remove non-serializable items (the per-result dicts) before saving."""
    clean = []
    for entry in history:
        clean.append({
            k: v for k, v in entry.items()
            if k != "results"  # skip per-problem detail, too verbose
        })
    return clean