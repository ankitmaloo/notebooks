# verl Patterns and Usage Guide

This document provides common patterns and best practices for using verl (Volcano Engine Reinforcement Learning) for LLM post-training.

## Overview

verl is a flexible, efficient framework for RL training of LLMs. It supports:
- **Algorithms**: PPO, GRPO, DAPO, ReMax, REINFORCE++, RLOO, PRIME
- **Backends**: PyTorch FSDP (recommended for prototyping), Megatron-LM (for production scaling)
- **Inference Engines**: vLLM, SGLang, HuggingFace Transformers

## Installation

### FSDP Backend (Recommended for Research/Prototyping)

```bash
# Install dependencies
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0
pip install vllm==0.5.4 ray==2.10
pip install flash-attn --no-build-isolation

# Install verl
git clone https://github.com/volcengine/verl
cd verl && pip install -e .
```

### Megatron-LM Backend (For Production Scaling)

```bash
# Install Apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation \
    --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" \
    git+https://github.com/NVIDIA/apex

# Install Transformer Engine
pip install git+https://github.com/NVIDIA/TransformerEngine.git@v1.7

# Install Megatron-LM with patches
cd ..
git clone -b core_v0.4.0 https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
cp ../verl/patches/megatron_v4.patch .
git apply megatron_v4.patch
pip install -e .
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

## Basic PPO Training Pattern

```python
from verl import PPOTrainer
from verl.trainer import TrainingConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# Configure training
config = TrainingConfig(
    total_epochs=3,
    batch_size=8,
    learning_rate=1e-6,
    ppo_epochs=4,
    clip_range=0.2,
    kl_coef=0.1,
)

# Create trainer
trainer = PPOTrainer(model, tokenizer, config)

# Train
trainer.train(train_dataloader)
```

## GRPO Training Pattern

GRPO (Group Relative Policy Optimization) is a simplified variant of PPO that:
- Uses group-relative advantages instead of a value function
- Doesn't require a separate critic/value model
- Similar to REINFORCE but with group normalization

```python
from verl import GRPOTrainer

# Similar setup to PPO
trainer = GRPOTrainer(model, tokenizer, config)
trainer.train(train_dataloader)
```

## Reward Function Patterns

### Model-Based Reward

```python
from transformers import pipeline

# Load reward model
reward_model = pipeline("text-classification", model="OpenAssistant/reward-model-deberta-v3-large")

def compute_reward(prompt: str, response: str) -> float:
    """Compute reward using a reward model"""
    text = f"{prompt}\n\n{response}"
    result = reward_model(text)[0]
    return result['score']
```

### Rule-Based Reward

```python
def compute_reward(prompt: str, response: str) -> float:
    """Simple rule-based reward"""
    reward = 0.0
    
    # Length penalty/reward
    words = response.split()
    if 10 <= len(words) <= 100:
        reward += 0.5
    
    # Keyword matching
    if any(keyword in response.lower() for keyword in ['because', 'therefore', 'thus']):
        reward += 0.3
    
    # Formatting
    if response.strip().endswith('.'):
        reward += 0.1
    
    return reward
```

### Verifiable Reward (Math/Code)

```python
import re

def compute_math_reward(prompt: str, response: str, ground_truth: str) -> float:
    """Reward based on correct answer"""
    # Extract answer from response
    match = re.search(r'####\s*(-?\d+(?:\.\d+)?)', response)
    if match:
        predicted = match.group(1)
        return 1.0 if predicted == ground_truth else 0.0
    return 0.0
```

## Multi-Turn RL Pattern

For conversational/agentic tasks:

```python
def rollout_episode(model, tokenizer, initial_prompt, max_turns=10):
    """Multi-turn episode rollout"""
    conversation = [{"role": "user", "content": initial_prompt}]
    rewards = []
    
    for turn in range(max_turns):
        # Generate response
        prompt = format_conversation(conversation)
        response = generate_response(model, tokenizer, prompt)
        conversation.append({"role": "assistant", "content": response})
        
        # Compute reward for this turn
        reward = compute_turn_reward(conversation)
        rewards.append(reward)
        
        # Check if episode should end
        if should_end_episode(response):
            break
        
        # Generate next user message (environment step)
        next_user_msg = get_next_user_message(conversation)
        conversation.append({"role": "user", "content": next_user_msg})
    
    return conversation, rewards
```

## Hybrid Controller Pattern

verl uses a hybrid controller model that combines single-controller and multi-controller paradigms:

```python
from verl.workers import RolloutWorker, TrainerWorker

# Create workers
rollout_worker = RolloutWorker(
    model=model,
    inference_engine="vllm",  # or "sglang", "hf"
    device="cuda:0"
)

trainer_worker = TrainerWorker(
    model=model,
    optimizer=optimizer,
    device="cuda:1"
)

# Execute RL loop
for epoch in range(num_epochs):
    # Rollout phase (generation)
    rollout_data = rollout_worker.generate(prompts)
    
    # Training phase (optimization)
    metrics = trainer_worker.train_step(rollout_data)
```

## Device Mapping Strategies

### Single GPU

```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0"
)
```

### Multi-GPU (FSDP)

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

# Wrap model with FSDP
model = FSDP(
    model,
    auto_wrap_policy=size_based_auto_wrap_policy,
    device_id=torch.cuda.current_device(),
)
```

## Logging and Monitoring

```python
import wandb

# Initialize wandb
wandb.init(project="llm-rl-training", config=config)

# Log metrics during training
wandb.log({
    "loss": loss.item(),
    "reward": reward.mean().item(),
    "kl_div": kl_div.item(),
    "learning_rate": optimizer.param_groups[0]['lr'],
}, step=global_step)
```

## Checkpointing Pattern

```python
from pathlib import Path

def save_checkpoint(model, optimizer, step, checkpoint_dir):
    """Save training checkpoint"""
    save_path = Path(checkpoint_dir) / f"checkpoint_step_{step}"
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model.save_pretrained(save_path)
    
    # Save optimizer state
    torch.save(optimizer.state_dict(), save_path / "optimizer.pt")
    
    # Save training state
    torch.save({
        "step": step,
        "config": config.__dict__,
    }, save_path / "training_state.pt")
```

## Common Issues and Solutions

### OOM (Out of Memory)

1. Reduce batch size
2. Use gradient accumulation
3. Enable gradient checkpointing:
```python
model.gradient_checkpointing_enable()
```
4. Use bf16/fp16 precision
5. Reduce sequence length

### Slow Training

1. Use vLLM for faster generation
2. Enable flash attention
3. Optimize data loading with multiple workers
4. Profile and identify bottlenecks

### Unstable Training

1. Lower learning rate
2. Increase KL penalty coefficient
3. Use gradient clipping
4. Check reward distribution (should not be too large)

## Resources

- verl GitHub: https://github.com/volcengine/verl
- verl Documentation: https://verl.readthedocs.io/
- HybridFlow Paper: https://arxiv.org/abs/2409.19256
