# RL Training Primitives: Utility Functions

This document provides utility functions that make working with the primitives easier.

---

## Data Structure Helpers

```python
from typing import List, Dict, Any, Optional
import torch
import json

# ============================================================================
# Rollout Utilities
# ============================================================================

def rollout_to_dict(rollout: Rollout) -> Dict[str, Any]:
    """Convert rollout to dictionary for serialization"""
    return {
        "prompt": rollout.prompt,
        "generation_text": rollout.generation.text,
        "token_ids": rollout.token_ids,
        "logprobs": rollout.logprobs,
        "ref_logprobs": rollout.ref_logprobs,
        "rewards": rollout.rewards,
        "kl_divergence": rollout.kl_divergence,
        "tool_calls": rollout.tool_calls,
        "tool_results": rollout.tool_results,
        "metadata": rollout.metadata
    }

def dict_to_rollout(data: Dict[str, Any]) -> Rollout:
    """Convert dictionary back to rollout"""
    generation = Generation(
        text=data["generation_text"],
        logprobs=data.get("logprobs"),
        token_ids=data.get("token_ids")
    )
    
    conversation = Conversation(messages=[])  # Would need to serialize/deserialize messages
    
    return Rollout(
        prompt=data["prompt"],
        conversation=conversation,
        generation=generation,
        token_ids=data.get("token_ids", []),
        logprobs=data.get("logprobs", []),
        ref_logprobs=data.get("ref_logprobs"),
        rewards=data.get("rewards", 0.0),
        kl_divergence=data.get("kl_divergence"),
        tool_calls=data.get("tool_calls", []),
        tool_results=data.get("tool_results", []),
        metadata=data.get("metadata", {})
    )

def save_rollouts(rollouts: List[Rollout], filepath: str):
    """Save rollouts to JSON file"""
    data = [rollout_to_dict(r) for r in rollouts]
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

def load_rollouts(filepath: str) -> List[Rollout]:
    """Load rollouts from JSON file"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return [dict_to_rollout(d) for d in data]

# ============================================================================
# Batch Utilities
# ============================================================================

def batch_statistics(batch: Batch) -> Dict[str, float]:
    """Compute statistics over batch"""
    rewards = [r.rewards if isinstance(r.rewards, float) else sum(r.rewards) 
               for r in batch.rollouts]
    kl_divs = [r.kl_divergence for r in batch.rollouts if r.kl_divergence is not None]
    lengths = [len(r.token_ids) for r in batch.rollouts]
    
    import statistics
    
    return {
        "num_rollouts": len(batch),
        "reward_mean": statistics.mean(rewards) if rewards else 0.0,
        "reward_std": statistics.stdev(rewards) if len(rewards) > 1 else 0.0,
        "reward_min": min(rewards) if rewards else 0.0,
        "reward_max": max(rewards) if rewards else 0.0,
        "kl_mean": statistics.mean(kl_divs) if kl_divs else 0.0,
        "kl_std": statistics.stdev(kl_divs) if len(kl_divs) > 1 else 0.0,
        "length_mean": statistics.mean(lengths),
        "length_std": statistics.stdev(lengths) if len(lengths) > 1 else 0.0
    }

def filter_by_reward(batch: Batch, min_reward: float, max_reward: float = float('inf')) -> Batch:
    """Filter rollouts by reward range"""
    def predicate(r: Rollout) -> bool:
        reward = r.rewards if isinstance(r.rewards, float) else sum(r.rewards)
        return min_reward <= reward <= max_reward
    
    return batch.filter(predicate)

def filter_by_kl(batch: Batch, max_kl: float) -> Batch:
    """Filter rollouts by KL divergence threshold"""
    def predicate(r: Rollout) -> bool:
        return r.kl_divergence is None or r.kl_divergence <= max_kl
    
    return batch.filter(predicate)

def sample_batch(batch: Batch, n: int, strategy: str = "random") -> Batch:
    """Sample n rollouts from batch"""
    import random
    
    if strategy == "random":
        sampled = random.sample(batch.rollouts, min(n, len(batch)))
    
    elif strategy == "top":
        sorted_rollouts = sorted(
            batch.rollouts,
            key=lambda r: r.rewards if isinstance(r.rewards, float) else sum(r.rewards),
            reverse=True
        )
        sampled = sorted_rollouts[:n]
    
    elif strategy == "bottom":
        sorted_rollouts = sorted(
            batch.rollouts,
            key=lambda r: r.rewards if isinstance(r.rewards, float) else sum(r.rewards)
        )
        sampled = sorted_rollouts[:n]
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    return Batch(rollouts=sampled, batch_metadata=batch.batch_metadata)

# ============================================================================
# Conversation Utilities
# ============================================================================

def conversation_to_text(conversation: Conversation, format: str = "plain") -> str:
    """Convert conversation to text"""
    if format == "plain":
        lines = []
        for msg in conversation.messages:
            lines.append(f"{msg.role.upper()}: {msg.content}")
        return "\n".join(lines)
    
    elif format == "markdown":
        lines = []
        for msg in conversation.messages:
            lines.append(f"**{msg.role.capitalize()}**: {msg.content}")
            lines.append("")
        return "\n".join(lines)
    
    elif format == "json":
        messages = [
            {"role": m.role, "content": m.content, "metadata": m.metadata}
            for m in conversation.messages
        ]
        return json.dumps(messages, indent=2)
    
    else:
        raise ValueError(f"Unknown format: {format}")

def count_turns(conversation: Conversation, role: Optional[str] = None) -> int:
    """Count number of turns in conversation"""
    if role is None:
        return len(conversation.messages)
    return sum(1 for m in conversation.messages if m.role == role)

def get_last_message(conversation: Conversation, role: Optional[str] = None) -> Optional[Message]:
    """Get last message, optionally filtered by role"""
    if role is None:
        return conversation.messages[-1] if conversation.messages else None
    
    for msg in reversed(conversation.messages):
        if msg.role == role:
            return msg
    return None
```

---

## Reward Function Builders

```python
# ============================================================================
# Common Reward Function Patterns
# ============================================================================

def length_reward(min_length: int = 50, max_length: int = 500, optimal: int = 200) -> Callable[[Rollout], float]:
    """Create length-based reward function"""
    def reward_fn(rollout: Rollout) -> float:
        length = len(rollout.generation.text)
        
        if length < min_length:
            return length / min_length * 0.3
        elif length > max_length:
            return max(0.3, 1.0 - (length - max_length) / max_length)
        else:
            # Gaussian centered at optimal
            distance = abs(length - optimal)
            return max(0.3, 1.0 - (distance / optimal) ** 2)
    
    return reward_fn

def keyword_reward(keywords: List[str], weights: Optional[List[float]] = None) -> Callable[[Rollout], float]:
    """Create keyword-based reward function"""
    if weights is None:
        weights = [1.0] * len(keywords)
    
    def reward_fn(rollout: Rollout) -> float:
        text = rollout.generation.text.lower()
        score = 0.0
        
        for keyword, weight in zip(keywords, weights):
            if keyword.lower() in text:
                score += weight
        
        # Normalize to [0, 1]
        max_score = sum(weights)
        return min(score / max_score, 1.0) if max_score > 0 else 0.0
    
    return reward_fn

def regex_reward(patterns: List[str], mode: str = "any") -> Callable[[Rollout], float]:
    """Create regex-based reward function"""
    import re
    compiled_patterns = [re.compile(p, re.IGNORECASE) for p in patterns]
    
    def reward_fn(rollout: Rollout) -> float:
        text = rollout.generation.text
        matches = [bool(p.search(text)) for p in compiled_patterns]
        
        if mode == "any":
            return 1.0 if any(matches) else 0.0
        elif mode == "all":
            return 1.0 if all(matches) else 0.0
        elif mode == "count":
            return sum(matches) / len(matches)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    return reward_fn

def combine_rewards(*reward_fns: Callable[[Rollout], float], weights: Optional[List[float]] = None) -> Callable[[Rollout], float]:
    """Combine multiple reward functions"""
    if weights is None:
        weights = [1.0] * len(reward_fns)
    
    def combined_fn(rollout: Rollout) -> float:
        scores = [fn(rollout) for fn in reward_fns]
        weighted = sum(s * w for s, w in zip(scores, weights))
        return weighted / sum(weights)
    
    return combined_fn

# Example usage:
# reward_fn = combine_rewards(
#     length_reward(min_length=100, max_length=500),
#     keyword_reward(["explanation", "because", "therefore"]),
#     weights=[0.6, 0.4]
# )
```

---

## Prompt Management

```python
# ============================================================================
# Prompt Loading and Management
# ============================================================================

class PromptDataset:
    """Manage prompts for training"""
    
    def __init__(self, prompts: List[str], shuffle: bool = True):
        self.prompts = prompts
        self.shuffle = shuffle
        self.index = 0
    
    def get_batch(self, batch_size: int, iteration: Optional[int] = None) -> List[str]:
        """Get batch of prompts"""
        if self.shuffle and iteration is not None:
            import random
            random.seed(iteration)
            prompts = random.sample(self.prompts, min(batch_size, len(self.prompts)))
        else:
            # Sequential
            end_idx = min(self.index + batch_size, len(self.prompts))
            prompts = self.prompts[self.index:end_idx]
            
            self.index = end_idx
            if self.index >= len(self.prompts):
                self.index = 0  # Wrap around
        
        return prompts
    
    def __len__(self):
        return len(self.prompts)

def load_prompts_from_file(filepath: str) -> List[str]:
    """Load prompts from file (one per line)"""
    with open(filepath, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def load_prompts_from_jsonl(filepath: str, prompt_key: str = "prompt") -> List[str]:
    """Load prompts from JSONL file"""
    prompts = []
    with open(filepath, 'r') as f:
        for line in f:
            data = json.loads(line)
            prompts.append(data[prompt_key])
    return prompts

def load_prompts_from_dataset(dataset_name: str, split: str = "train", prompt_key: str = "prompt") -> List[str]:
    """Load prompts from HuggingFace dataset"""
    from datasets import load_dataset
    
    dataset = load_dataset(dataset_name, split=split)
    return [item[prompt_key] for item in dataset]

# Example:
# prompts = load_prompts_from_dataset("Anthropic/hh-rlhf", split="train", prompt_key="chosen")
```

---

## Logging and Visualization

```python
# ============================================================================
# Logging Utilities
# ============================================================================

class Logger:
    """Multi-backend logger"""
    
    def __init__(self, backends: List[str] = ["console"], **kwargs):
        self.backends = backends
        self.wandb_run = None
        
        if "wandb" in backends:
            import wandb
            self.wandb_run = wandb.init(**kwargs)
    
    def log(self, data: Dict[str, Any], step: Optional[int] = None):
        """Log data to all backends"""
        if "console" in self.backends:
            self._log_console(data, step)
        
        if "wandb" in self.backends and self.wandb_run:
            import wandb
            wandb.log(data, step=step)
        
        if "tensorboard" in self.backends:
            self._log_tensorboard(data, step)
    
    def _log_console(self, data: Dict[str, Any], step: Optional[int]):
        """Log to console"""
        step_str = f"[Step {step}] " if step is not None else ""
        items = [f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" 
                 for k, v in data.items()]
        print(f"{step_str}{' | '.join(items)}")
    
    def _log_tensorboard(self, data: Dict[str, Any], step: Optional[int]):
        """Log to tensorboard"""
        if not hasattr(self, 'tb_writer'):
            from torch.utils.tensorboard import SummaryWriter
            self.tb_writer = SummaryWriter()
        
        for key, value in data.items():
            if isinstance(value, (int, float)):
                self.tb_writer.add_scalar(key, value, step)
    
    def close(self):
        """Close all backends"""
        if self.wandb_run:
            self.wandb_run.finish()
        
        if hasattr(self, 'tb_writer'):
            self.tb_writer.close()

# ============================================================================
# Visualization Utilities
# ============================================================================

def plot_training_curves(stats_history: List[TrainingStats], save_path: Optional[str] = None):
    """Plot training curves"""
    import matplotlib.pyplot as plt
    
    iterations = list(range(len(stats_history)))
    losses = [s.loss for s in stats_history]
    rewards = [s.rewards_mean for s in stats_history]
    kls = [s.kl_divergence for s in stats_history]
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    
    axes[0].plot(iterations, losses)
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].grid(True)
    
    axes[1].plot(iterations, rewards)
    axes[1].set_ylabel("Mean Reward")
    axes[1].set_title("Mean Reward")
    axes[1].grid(True)
    
    axes[2].plot(iterations, kls)
    axes[2].set_ylabel("KL Divergence")
    axes[2].set_xlabel("Iteration")
    axes[2].set_title("KL Divergence")
    axes[2].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_reward_distribution(batch: Batch, save_path: Optional[str] = None):
    """Plot reward distribution for batch"""
    import matplotlib.pyplot as plt
    
    rewards = [r.rewards if isinstance(r.rewards, float) else sum(r.rewards) 
               for r in batch.rollouts]
    
    plt.figure(figsize=(10, 6))
    plt.hist(rewards, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel("Reward")
    plt.ylabel("Count")
    plt.title(f"Reward Distribution (n={len(rewards)})")
    plt.axvline(sum(rewards) / len(rewards), color='red', linestyle='--', label='Mean')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
```

---

## Common Extension Helpers

```python
# ============================================================================
# Reference Model Update Helpers
# ============================================================================

def create_ema_updater(decay: float = 0.999):
    """Create EMA update function"""
    def update_fn(ref_model, policy_model):
        with torch.no_grad():
            for ref_param, policy_param in zip(ref_model.parameters(), policy_model.parameters()):
                ref_param.data.mul_(decay).add_(policy_param.data, alpha=1-decay)
        return ref_model
    return update_fn

def create_periodic_updater(update_every: int = 10):
    """Create periodic update function"""
    counter = {"count": 0}
    
    def update_fn(ref_model, policy_model):
        counter["count"] += 1
        if counter["count"] % update_every == 0:
            import copy
            ref_model = copy.deepcopy(policy_model)
            ref_model.eval()
            for param in ref_model.parameters():
                param.requires_grad = False
        return ref_model
    
    return update_fn

# ============================================================================
# KL Computation Helpers
# ============================================================================

def create_truncated_kl(threshold: float = 1.0):
    """Create truncated KL function"""
    def kl_fn(logprobs: List[float], ref_logprobs: List[float]) -> float:
        import math
        kl = 0.0
        for lp, ref_lp in zip(logprobs, ref_logprobs):
            prob = math.exp(lp)
            token_kl = prob * (lp - ref_lp)
            kl += min(token_kl, threshold)
        return kl
    return kl_fn

def create_adaptive_kl(confidence_threshold: float = 0.1):
    """Create adaptive KL function"""
    def kl_fn(logprobs: List[float], ref_logprobs: List[float]) -> float:
        import math
        kl = 0.0
        for lp, ref_lp in zip(logprobs, ref_logprobs):
            prob = math.exp(lp)
            if prob > confidence_threshold:
                token_kl = prob * (lp - ref_lp)
                kl += token_kl
        return kl
    return kl_fn

# ============================================================================
# Advantage Computation Helpers
# ============================================================================

def create_normalized_advantages():
    """Create normalized advantage function"""
    def advantage_fn(rewards: List[float]) -> List[float]:
        import statistics
        mean_r = statistics.mean(rewards)
        std_r = statistics.stdev(rewards) if len(rewards) > 1 else 1.0
        return [(r - mean_r) / (std_r + 1e-8) for r in rewards]
    return advantage_fn

def create_rank_based_advantages():
    """Create rank-based advantage function"""
    def advantage_fn(rewards: List[float]) -> List[float]:
        # Assign advantages based on rank
        indexed = [(r, i) for i, r in enumerate(rewards)]
        sorted_indexed = sorted(indexed, key=lambda x: x[0], reverse=True)
        
        advantages = [0.0] * len(rewards)
        for rank, (_, idx) in enumerate(sorted_indexed):
            # Map rank to [-1, 1]
            advantages[idx] = 1.0 - (2 * rank / (len(rewards) - 1))
        
        return advantages
    return advantage_fn

# ============================================================================
# Tool Execution Helpers
# ============================================================================

def create_mock_tool_executor():
    """Create mock tool executor for testing"""
    def executor(tool_call: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": tool_call.get("id", "call_0"),
            "output": f"Mock result for {tool_call.get('name', 'unknown')}",
            "success": True,
            "metadata": {"mock": True}
        }
    return executor

def create_python_executor(timeout: int = 5):
    """Create Python code executor"""
    def executor(tool_call: Dict[str, Any]) -> Dict[str, Any]:
        if tool_call.get("name") != "python":
            return {"id": tool_call.get("id"), "output": "Not a Python call", "success": False}
        
        code = tool_call.get("arguments", {}).get("code", "")
        
        try:
            import subprocess
            result = subprocess.run(
                ["python", "-c", code],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            return {
                "id": tool_call.get("id"),
                "output": result.stdout or result.stderr,
                "success": result.returncode == 0,
                "metadata": {"returncode": result.returncode}
            }
        except subprocess.TimeoutExpired:
            return {
                "id": tool_call.get("id"),
                "output": "Execution timed out",
                "success": False
            }
        except Exception as e:
            return {
                "id": tool_call.get("id"),
                "output": str(e),
                "success": False
            }
    
    return executor
```

---

## Testing Utilities

```python
# ============================================================================
# Testing Helpers
# ============================================================================

def create_mock_rollout(
    prompt: str = "Test prompt",
    generation_text: str = "Test generation",
    reward: float = 0.5,
    num_tokens: int = 10
) -> Rollout:
    """Create mock rollout for testing"""
    generation = Generation(
        text=generation_text,
        logprobs=[-1.0] * num_tokens,
        token_ids=list(range(num_tokens))
    )
    
    return Rollout(
        prompt=prompt,
        conversation=Conversation(messages=[]),
        generation=generation,
        token_ids=list(range(num_tokens)),
        logprobs=[-1.0] * num_tokens,
        ref_logprobs=[-0.9] * num_tokens,
        rewards=reward,
        kl_divergence=0.1
    )

def create_mock_batch(size: int = 8) -> Batch:
    """Create mock batch for testing"""
    rollouts = [
        create_mock_rollout(
            prompt=f"Prompt {i}",
            generation_text=f"Generation {i}",
            reward=i / size
        )
        for i in range(size)
    ]
    return Batch(rollouts=rollouts)

def validate_rollout(rollout: Rollout) -> List[str]:
    """Validate rollout and return list of issues"""
    issues = []
    
    if not rollout.prompt:
        issues.append("Empty prompt")
    
    if not rollout.generation.text:
        issues.append("Empty generation")
    
    if rollout.logprobs and len(rollout.logprobs) != len(rollout.token_ids):
        issues.append(f"Logprobs length ({len(rollout.logprobs)}) != token_ids length ({len(rollout.token_ids)})")
    
    if rollout.ref_logprobs and len(rollout.ref_logprobs) != len(rollout.logprobs):
        issues.append("Ref logprobs length != logprobs length")
    
    if isinstance(rollout.rewards, float):
        if rollout.rewards < -10 or rollout.rewards > 10:
            issues.append(f"Suspicious reward value: {rollout.rewards}")
    
    return issues

def validate_batch(batch: Batch) -> Dict[str, Any]:
    """Validate batch and return validation report"""
    all_issues = []
    
    for i, rollout in enumerate(batch.rollouts):
        issues = validate_rollout(rollout)
        if issues:
            all_issues.append({
                "rollout_index": i,
                "issues": issues
            })
    
    return {
        "num_rollouts": len(batch),
        "num_invalid": len(all_issues),
        "invalid_rollouts": all_issues,
        "is_valid": len(all_issues) == 0
    }

# ============================================================================
# Debugging Utilities
# ============================================================================

def inspect_rollout(rollout: Rollout, max_length: int = 200):
    """Print detailed rollout inspection"""
    print("=" * 80)
    print("ROLLOUT INSPECTION")
    print("=" * 80)
    
    print(f"\nPrompt ({len(rollout.prompt)} chars):")
    print(rollout.prompt[:max_length] + "..." if len(rollout.prompt) > max_length else rollout.prompt)
    
    print(f"\nGeneration ({len(rollout.generation.text)} chars):")
    print(rollout.generation.text[:max_length] + "..." if len(rollout.generation.text) > max_length else rollout.generation.text)
    
    print(f"\nReward: {rollout.rewards}")
    
    if rollout.kl_divergence is not None:
        print(f"KL Divergence: {rollout.kl_divergence:.6f}")
    
    print(f"\nTokens: {len(rollout.token_ids)}")
    print(f"Has logprobs: {rollout.logprobs is not None}")
    print(f"Has ref_logprobs: {rollout.ref_logprobs is not None}")
    
    if rollout.tool_calls:
        print(f"\nTool Calls ({len(rollout.tool_calls)}):")
        for call in rollout.tool_calls:
            print(f"  - {call.get('name', 'unknown')}")
    
    if rollout.metadata:
        print(f"\nMetadata:")
        for key, value in rollout.metadata.items():
            print(f"  {key}: {value}")
    
    print("=" * 80)

def compare_rollouts(rollout1: Rollout, rollout2: Rollout):
    """Compare two rollouts"""
    print("=" * 80)
    print("ROLLOUT COMPARISON")
    print("=" * 80)
    
    def compare_field(name, val1, val2):
        if val1 == val2:
            print(f"{name}: EQUAL")
        else:
            print(f"{name}: DIFFERENT")
            print(f"  Rollout 1: {val1}")
            print(f"  Rollout 2: {val2}")
    
    compare_field("Prompt", rollout1.prompt, rollout2.prompt)
    compare_field("Generation", rollout1.generation.text, rollout2.generation.text)
    compare_field("Reward", rollout1.rewards, rollout2.rewards)
    compare_field("KL Divergence", rollout1.kl_divergence, rollout2.kl_divergence)
    compare_field("Num Tokens", len(rollout1.token_ids), len(rollout2.token_ids))
    
    print("=" * 80)
```

---

## Configuration Management

```python
# ============================================================================
# Configuration Helpers
# ============================================================================

def create_training_config(
    # Model
    model_name: str = "meta-llama/Llama-3-8B",
    use_flash_attention: bool = True,
    
    # Generation
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    
    # Training
    learning_rate: float = 1e-5,
    batch_size: int = 8,
    num_iterations: int = 100,
    
    # Algorithm
    algorithm: str = "grpo",
    kl_coef: float = 0.1,
    
    # Reference model
    ref_update_strategy: str = "ema",
    ema_decay: float = 0.999,
    
    # Logging
    log_every: int = 1,
    eval_every: int = 10,
    checkpoint_every: int = 20,
    
    **kwargs
) -> Dict[str, Any]:
    """Create complete training configuration"""
    
    config = {
        "model": {
            "name": model_name,
            "use_flash_attention": use_flash_attention
        },
        "generation": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p
        },
        "training": {
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "num_iterations": num_iterations
        },
        "algorithm": {
            "name": algorithm,
            "kl_coef": kl_coef
        },
        "reference_model": {
            "update_strategy": ref_update_strategy,
            "ema_decay": ema_decay
        },
        "logging": {
            "log_every": log_every,
            "eval_every": eval_every,
            "checkpoint_every": checkpoint_every
        }
    }
    
    # Add any additional kwargs
    config["custom"] = kwargs
    
    return config

def save_config(config: Dict[str, Any], filepath: str):
    """Save configuration to JSON"""
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)

def load_config(filepath: str) -> Dict[str, Any]:
    """Load configuration from JSON"""
    with open(filepath, 'r') as f:
        return json.load(f)
```

---

## Quick Start Template

```python
# ============================================================================
# Quick Start Template
# ============================================================================

def quick_start_training(
    model_name: str,
    prompts: List[str],
    reward_fn: Callable[[Rollout], float],
    num_iterations: int = 100,
    **kwargs
):
    """
    Quick start training with sensible defaults.
    
    Usage:
        prompts = ["Explain AI", "What is ML?", ...]
        
        def my_reward(rollout):
            return len(rollout.generation.text) / 100
        
        quick_start_training(
            model_name="meta-llama/Llama-3-8B",
            prompts=prompts,
            reward_fn=my_reward,
            num_iterations=50
        )
    """
    
    # Setup
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    import copy
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print("Creating reference model...")
    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=kwargs.get("learning_rate", 1e-5))
    
    # Configs
    gen_config = GenerationConfig(
        max_new_tokens=kwargs.get("max_new_tokens", 256),
        temperature=kwargs.get("temperature", 0.7),
        top_p=kwargs.get("top_p", 0.9)
    )
    
    algorithm_config = create_algorithm_config(
        kwargs.get("algorithm", "grpo"),
        {"kl_coef": kwargs.get("kl_coef", 0.1)}
    )
    
    # Prompt loader
    prompt_dataset = PromptDataset(prompts, shuffle=True)
    
    def load_prompts(iteration: int) -> List[str]:
        return prompt_dataset.get_batch(kwargs.get("batch_size", 8), iteration)
    
    # State
    initial_state = TrainingState(
        iteration=0,
        global_step=0,
        model=model,
        optimizer=optimizer,
        ref_model=ref_model
    )
    
    # Logger
    logger = Logger(backends=["console"])
    
    def log_fn(state, stats):
        logger.log({
            "iteration": state.iteration,
            "loss": stats.loss,
            "reward": stats.rewards_mean,
            "kl": stats.kl_divergence
        }, step=state.iteration)
    
    # Train
    print(f"Starting training for {num_iterations} iterations...")
    final_state = train(
        initial_state=initial_state,
        prompt_loader=load_prompts,
        num_iterations=num_iterations,
        gen_config=gen_config,
        algorithm_config=algorithm_config,
        reward_fn=reward_fn,
        logger=log_fn
    )
    
    print("Training complete!")
    
    # Plot
    if kwargs.get("plot", True):
        plot_training_curves(final_state.stats_history)
    
    return final_state
```

---

These utilities make it easy to:
- Work with rollouts and batches
- Build custom reward functions
- Manage prompts and datasets
- Log and visualize training
- Test and debug components
- Configure training runs
- Get started quickly

All while maintaining the functional, composable architecture!
