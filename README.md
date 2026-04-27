# GRPO From Scratch: Teaching a 1.5B Model to Reason with RL

## Hypothesis
Can a small LLM learn multi-step arithmetic reasoning through 
pure reinforcement learning — with no labeled reasoning traces, 
no human feedback, no chain-of-thought supervision?

## Method
- Model: Qwen2.5-1.5B-Instruct
- Task: 3-digit × 2-digit multiplication
- Algorithm: GRPO (Group Relative Policy Optimization)
- Reward: Binary verifiable (correct answer = 1.0, wrong = 0.0)
- Ablation: Same setup, reward signal removed

## Results





## Based On
- DeepSeek-R1-Zero (DeepSeek-AI, 2025)
- GRPO: DeepSeekMath (Shao et al., 2024)
- Reflexion (Shinn et al., NeurIPS 2023)