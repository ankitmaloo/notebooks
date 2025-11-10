# nanochat RL Patterns

Patterns and insights from Andrej Karpathy's nanochat project for RL training of LLMs.

## Overview

nanochat is a minimal, end-to-end ChatGPT-style training pipeline that includes:
- Custom Rust BPE tokenizer
- Base pretraining on FineWeb-EDU
- Mid-training on conversations/MCQ/tool-use
- Supervised Fine-Tuning (SFT)
- Optional RL with simplified GRPO

## Philosophy

**Minimal and Hackable**
- ~8,000 lines of code total
- Single cohesive codebase
- No giant config objects
- Direct and readable

**End-to-End**
- Everything from tokenization to serving
- Complete pipeline in one repo
- Self-contained and reproducible

## RL Implementation in nanochat

nanochat uses a **simplified GRPO** approach:

### Key Simplifications

1. **No Trust Region** - No reference model KL constraint
2. **No KL Penalty** - Omits standard KL divergence term
3. **On-Policy Updates** - No PPO-style importance ratios/clipping
4. **Token-Level Normalization** - GAPO-style advantage normalization
5. **Mean-Shift Advantage** - Group-relative advantages

### Implementation Pattern

```python
# Generate responses (on-policy)
responses = model.generate(prompts, do_sample=True, temperature=0.8)

# Compute rewards
rewards = [compute_reward(prompt, resp) for prompt, resp in zip(prompts, responses)]

# Group-relative advantages
advantages = rewards - rewards.mean()

# Policy loss (simplified GRPO)
loss = -(log_probs * advantages).mean()

# Optimize
loss.backward()
optimizer.step()
```

### Reward Function (GSM8K Math)

```python
def compute_reward(response: str, ground_truth: str) -> float:
    """
    Simple binary reward for math problems
    """
    # Extract answer from response
    pred_answer = extract_answer(response)  # Parse #### marker
    true_answer = extract_answer(ground_truth)
    
    # Binary reward
    return 1.0 if pred_answer == true_answer else 0.0
```

## Training Pipeline Structure

### 1. Base Training (Pretraining)
```bash
torchrun --standalone --nproc_per_node=8 -m scripts.base_train \
    --depth=20 \
    --device_batch_size=32
```

- Trains on FineWeb-EDU shards
- ~11.2B tokens for speedrun (d20)
- Uses Muon + AdamW optimizer
- Saves checkpoint for next stage

### 2. Mid-Training
```bash
torchrun --standalone --nproc_per_node=8 -m scripts.mid_train \
    --device_batch_size=16
```

- Adapts to conversation format (SmolTalk)
- Trains on MCQ (MMLU auxiliary)
- Seeds tool use with Python tags
- Prepares for SFT

### 3. Supervised Fine-Tuning
```bash
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft \
    --device_batch_size=8
```

- High-quality conversations
- Enforces test-time formatting
- Non-concatenated training
- Improves instruction following

### 4. RL Training (Optional)
```bash
torchrun --standalone --nproc_per_node=8 -m scripts.chat_rl \
    --dataset=GSM8K
```

- Simplified GRPO on math
- ~1 hour additional training
- Modest improvement on reasoning

## Key Design Decisions

### Optimizer: Muon + AdamW

```python
# Muon for matmul parameters (faster convergence)
muon_params = [p for n, p in model.named_parameters() if 'weight' in n and p.ndim >= 2]

# AdamW for embeddings/biases
adamw_params = [p for n, p in model.named_parameters() if p not in muon_params]

optimizers = [
    Muon(muon_params, lr=0.01, momentum=0.95),
    AdamW(adamw_params, lr=3e-4)
]
```

### Depth as Complexity Slider

```python
# Single --depth parameter controls:
# - Number of layers
# - Hidden size (channels)
# - Learning rates
# - All hyperparameters scale automatically

depth = 20  # speedrun
depth = 26  # ~GPT-2 level
depth = 32  # $1000 tier
```

### Memory Management

```python
# Reduce device_batch_size until fits
device_batch_size = 32  # default
device_batch_size = 16  # for larger models
device_batch_size = 8   # for 40GB GPUs

# Code automatically increases gradient accumulation
# to maintain effective batch size
```

### Bits-Per-Byte Metric

```python
# Report loss in bits-per-byte (tokenizer-invariant)
def compute_bpb(loss: float, tokenizer) -> float:
    """Convert cross-entropy loss to bits per byte"""
    chars_per_token = 4.8  # Measured for this tokenizer
    return loss / np.log(2) / chars_per_token
```

## Practical Patterns

### Single-File Modularity

Each script is self-contained but imports from `nanochat/` package:

```python
from nanochat.gpt import GPT
from nanochat.dataloader import DataLoader
from nanochat.adamw import AdamW
from nanochat.checkpoint_manager import CheckpointManager
```

### Configuration via argparse

```python
# Simple argparse instead of config files
parser = argparse.ArgumentParser()
parser.add_argument('--depth', type=int, default=20)
parser.add_argument('--device_batch_size', type=int, default=32)
parser.add_argument('--learning_rate', type=float, default=None)  # Auto if None
args = parser.parse_args()
```

### Automatic Learning Rate Scaling

```python
def get_learning_rate(depth: int) -> float:
    """Scale learning rate with model size"""
    # Larger models need smaller LR
    return 3e-4 * (20 / depth) ** 0.5
```

### Report Generation

```python
from nanochat.report import Report

report = Report()
report.add_metric("CORE", core_score)
report.add_metric("GSM8K", gsm8k_score)
report.save("report.md")
```

## Scaling Tiers

### Speedrun ($100, 4 hours)
```python
depth = 20
params = 560M
tokens = 11.2B
cost = ~$100
```

### GPT-2 Level ($300, 12 hours)
```python
depth = 26
params = ~1B
tokens = 20B+
cost = ~$300
```

### High Quality ($1000, 42 hours)
```python
depth = 32
params = 1.9B
tokens = 38B
cost = ~$800
```

## Lessons from nanochat

### 1. Simplicity Works
- Removed many RLHF complexities
- Still gets reasonable results
- Much easier to understand and modify

### 2. Scale Matters More Than Algorithm
- Better to train longer with simple method
- Than complex algorithm with less compute

### 3. Good Baselines First
- Strong SFT baseline crucial
- RL adds modest improvements
- Don't skip base/mid/SFT

### 4. Evaluation Matters
- CORE benchmark for base model
- Task-specific evals (GSM8K, etc.)
- Report card for tracking

### 5. Infrastructure Over Features
- Fast data loading
- Efficient checkpointing
- Good logging
- These matter more than fancy algorithms

## Adapting nanochat Patterns

### For Your Project

```python
# Use nanochat's structure but adapt:

# 1. Custom tokenizer or use HF
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# 2. Your data instead of FineWeb
train_dataset = YourDataset(...)

# 3. Your reward function
def compute_reward(prompt, response):
    # Your domain-specific reward
    pass

# 4. Keep the simple training loop
for batch in dataloader:
    responses = model.generate(batch['prompts'])
    rewards = [compute_reward(p, r) for p, r in zip(prompts, responses)]
    advantages = rewards - rewards.mean()
    loss = -(log_probs * advantages).mean()
    loss.backward()
    optimizer.step()
```

## Resources

- nanochat GitHub: https://github.com/karpathy/nanochat
- Speedrun Guide: https://github.com/karpathy/nanochat/discussions/1
- Report Example: See `report.md` after training
- Paper/Course: LLM101n (in development)
