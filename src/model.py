"""
model.py — Model loading, inference utilities, reference model setup

WRITE THIS SECOND (after dataset.py).

This file has one job: give you a (model, tokenizer) pair that works,
and utilities to run inference cleanly.

Nothing in here should know about GRPO or rewards. Pure model utilities.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Tuple, Optional
import copy


# ---------------------------------------------------------------------------
# HARDWARE DETECTION
# ---------------------------------------------------------------------------

def get_device() -> str:
    """
    Detect the best available device.

    Priority: CUDA (NVIDIA GPU) > MPS (Apple Silicon) > CPU

    On Google Colab free tier (T4 GPU): "cuda"
    On Mac M1/M2/M3: "mps"  — slow but works for 0.5B-1.5B models
    On CPU only: "cpu" — very slow, only use for debugging
    """
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✓ GPU detected: {gpu_name} ({vram_gb:.1f} GB VRAM)")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
        print("✓ Apple Silicon (MPS) detected")
    else:
        device = "cpu"
        print("⚠️  No GPU detected — using CPU (will be slow)")
    return device


def recommend_model(device: str) -> str:
    """
    Recommend the right model size based on available hardware.

    Model sizes and VRAM requirements (float16):
      Qwen2.5-0.5B-Instruct: ~1GB  — works on CPU, fast on any GPU
      Qwen2.5-1.5B-Instruct: ~3GB  — recommended, good balance
      Qwen2.5-3B-Instruct:   ~6GB  — better results, needs T4/better
      Qwen2.5-7B-Instruct:   ~14GB — best results, needs A100
    """
    if device == "cpu" or device == "mps":
        recommended = "Qwen/Qwen2.5-0.5B-Instruct"
        print(f"  Recommended model for {device}: {recommended}")
    elif device == "cuda":
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        if vram < 8:
            recommended = "Qwen/Qwen2.5-0.5B-Instruct"
        elif vram < 16:
            recommended = "Qwen/Qwen2.5-1.5B-Instruct"
        else:
            recommended = "Qwen/Qwen2.5-3B-Instruct"
        print(f"  Recommended model for {vram:.0f}GB VRAM: {recommended}")
    else:
        recommended = "Qwen/Qwen2.5-1.5B-Instruct"

    return recommended


# ---------------------------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------------------------

def load_model(model_name: str, device: str):
    """
    Load model + tokenizer. Returns (model, tokenizer).

    WHY Qwen2.5:
      - Strong base reasoning ability (trained on math)
      - Instruct version responds to <think> tag prompts naturally
      - Small enough to train on free Colab T4
      - Used by TinyZero, nano-aha-moment, GRPO-Zero — well validated

    float16 vs float32:
      float16 halves memory usage with negligible accuracy loss.
      Always use float16 on GPU. Use float32 on CPU (MPS doesn't support float16 well).
    """
    print(f"\nLoading {model_name}...")

    dtype = torch.float16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto"   # handles multi-GPU automatically if available
    )

    # Qwen tokenizer doesn't always set pad_token — fix this
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()  # start in eval mode, grpo.py will switch to train() as needed

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"✓ Loaded {model_name}")
    print(f"  Parameters: {n_params:.0f}M")
    print(f"  dtype: {dtype}")

    return model, tokenizer


def make_reference_model(policy_model):
    """
    Create a frozen copy of the policy model to use as reference.

    The reference model is used to compute the KL penalty:
      KL(π_policy || π_reference)

    It NEVER gets updated. Its job is to remember where training started,
    so the policy doesn't drift too far from sensible outputs.

    WHY deepcopy: we need independent weights, not a reference to the same object.
    """
    ref_model = copy.deepcopy(policy_model)
    ref_model.eval()

    # Freeze all parameters — no gradients computed for reference model
    for param in ref_model.parameters():
        param.requires_grad = False

    print("✓ Reference model created (frozen)")
    return ref_model


# ---------------------------------------------------------------------------
# INFERENCE UTILITIES
# ---------------------------------------------------------------------------

def generate_single(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    greedy: bool = False
) -> Tuple[torch.Tensor, str, int]:
    """
    Generate one completion for a prompt.

    Returns:
        full_ids: tensor of [prompt_tokens + completion_tokens]
        completion_text: decoded completion only
        prompt_length: number of prompt tokens (needed for log prob slicing)

    WHY we return full_ids + prompt_length separately:
        compute_token_log_probs() needs the full sequence to run the forward pass,
        but should only return log probs for COMPLETION tokens (not prompt).
        Returning both lets grpo.py slice correctly.
    """
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_length = inputs['input_ids'].shape[1]

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if not greedy else 1.0,
            do_sample=not greedy,
            pad_token_id=tokenizer.eos_token_id
        )

    full_ids = output[0]  # [prompt_len + completion_len]
    completion_ids = full_ids[prompt_length:]
    completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)

    return full_ids, completion_text, prompt_length


def generate_group(
    model,
    tokenizer,
    prompt: str,
    G: int,
    max_new_tokens: int = 256,
    temperature: float = 0.8
) -> List[Tuple[torch.Tensor, str, int]]:
    """
    Generate G completions for one prompt.

    This is the "rollout" phase of GRPO.
    G independent samples from the same prompt — this is what creates
    the group of completions that advantages are computed over.

    WHY G samples from one prompt (not 1 sample from G prompts):
        Advantages are computed WITHIN a group from the same prompt.
        You need multiple samples of the same problem to normalize rewards.
        Mixing different problems would make advantage computation meaningless.

    Returns:
        List of G (full_ids, completion_text, prompt_length) tuples
    """
    completions = []
    for _ in range(G):
        result = generate_single(
            model, tokenizer, prompt, max_new_tokens, temperature
        )
        completions.append(result)
    return completions


# ---------------------------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------------------------

def evaluate(
    model,
    tokenizer,
    problems: list,
    reward_fn,
    max_new_tokens: int = 256
) -> dict:
    """
    Run full evaluation on a problem set.

    Returns dict with:
        accuracy: fraction of problems answered correctly
        mean_reward: average reward across all problems
        avg_trace_length: average completion length in words
        results: per-problem details
    """
    from src.dataset import has_think_tags

    results = []
    model.eval()

    for a, b, correct_answer in problems:
        from src.dataset import format_prompt
        prompt = format_prompt(a, b)

        # Use greedy decoding for eval (deterministic, comparable across steps)
        _, completion, _ = generate_single(
            model, tokenizer, prompt,
            max_new_tokens=max_new_tokens,
            greedy=True
        )

        r = reward_fn(completion, correct_answer)
        results.append({
            "problem": f"{a} × {b}",
            "correct_answer": correct_answer,
            "completion": completion,
            "reward": r,
            "correct": r >= 1.0,
            "trace_length": len(completion.split()),
            "has_think": has_think_tags(completion)
        })

    accuracy = sum(r["correct"] for r in results) / len(results)
    mean_reward = sum(r["reward"] for r in results) / len(results)
    avg_trace = sum(r["trace_length"] for r in results) / len(results)
    format_rate = sum(r["has_think"] for r in results) / len(results)

    return {
        "accuracy": accuracy,
        "mean_reward": mean_reward,
        "avg_trace_length": avg_trace,
        "format_rate": format_rate,
        "results": results
    }


def save_probe_traces(model, tokenizer, probe_problems: list, step: int,
                      max_new_tokens: int = 256):
    """
    Save completions on fixed probe problems.

    These files are your qualitative evidence for the "aha moment".
    Compare step_0000.txt vs step_0300.txt in your README.
    """
    import os
    from src.dataset import format_prompt

    os.makedirs("results/traces", exist_ok=True)
    path = f"results/traces/step_{step:04d}.txt"

    with open(path, 'w') as f:
        f.write(f"=== Probe Traces — Step {step} ===\n\n")
        for i, (a, b, answer) in enumerate(probe_problems):
            prompt = format_prompt(a, b)
            _, completion, _ = generate_single(
                model, tokenizer, prompt,
                max_new_tokens=max_new_tokens,
                greedy=True  # greedy for reproducibility
            )
            f.write(f"--- Problem {i+1}: {a} × {b} = {answer} ---\n")
            f.write(f"COMPLETION:\n{completion}\n\n")

    print(f"  Saved traces → {path}")