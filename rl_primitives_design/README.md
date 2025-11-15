# RL Training Primitives: Functional Architecture

A comprehensive, function-based architecture for building hackable, modular RL training pipelines for LLM post-training.

## Overview

This architecture is built on **pure functions** instead of classes, making it extremely **hackable** and **composable**. Every component can be swapped, extended, or customized by passing different functions as arguments.

### Core Design Principles

1. **Functions, not classes** - All primitives are pure functions
2. **Explicit data flow** - No hidden state, everything flows through parameters
3. **Maximum hackability** - Swap any component by passing a different function
4. **Clear extension points** - Every primitive accepts custom functions for behavior
5. **Composability** - Functions naturally compose to build complex pipelines

## Documents

### 📘 [ARCHITECTURE.md](./ARCHITECTURE.md)
Complete architecture specification with:
- **Data Structures**: All NamedTuples and core types
- **Inference Primitives**: Model loading, generation, multi-turn conversations
- **Reward Primitives**: Scalar/vector rewards, aggregation, reward models
- **KL Divergence Primitives**: Different KL methods, reference model handling
- **Rollout Primitives**: Data collection, tool tracking, multi-turn rollouts
- **Algorithm Primitives**: GRPO, DAPO, PPO with extension points
- **Training Loop Primitives**: Iteration management, logging, checkpointing
- **Composition Patterns**: How primitives fit together
- **Extension Scenarios**: Examples of customization

### 📗 [EXAMPLES.md](./EXAMPLES.md)
Complete working examples including:
- **Basic GRPO Training** - Simple end-to-end example
- **Vector Rewards** - Multi-objective rewards with custom aggregation
- **Algorithm Switching** - Dynamic algorithm changes mid-training
- **Tool Call Tracking** - Multi-turn rollouts with tool use
- **Custom KL Divergence** - Custom KL computation and reference updates
- **Complete Pipeline** - Full custom training pipeline with all features

### 📙 [UTILITIES.md](./UTILITIES.md)
Utility functions and helpers:
- **Data Structure Helpers** - Serialization, batch operations
- **Reward Function Builders** - Common reward patterns
- **Prompt Management** - Loading and managing training prompts
- **Logging & Visualization** - Multi-backend logging, plotting
- **Extension Helpers** - EMA updaters, KL functions, advantage computation
- **Testing Utilities** - Mock objects, validation, debugging
- **Quick Start Template** - Get started in 5 lines of code

## Quick Start

### Installation

```bash
pip install torch transformers datasets wandb
```

### 5-Minute Example

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Load model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8B")

# 2. Define reward
def my_reward(rollout):
    return len(rollout.generation.text) / 100  # Simple length reward

# 3. Train
from rl_primitives import quick_start_training

prompts = ["Explain quantum computing", "What is ML?"] * 50

final_state = quick_start_training(
    model_name="meta-llama/Llama-3-8B",
    prompts=prompts,
    reward_fn=my_reward,
    num_iterations=100
)
```

## Architecture Highlights

### 1. Inference/Generation Primitives

```python
# Load model with custom loader
model = load_model(config, model_loader=my_vllm_loader)

# Generate with custom backend
generation = generate(model, tokenizer, prompt, config, generator=vllm_generator)

# Multi-turn with tool calls
rollout = create_multiturn_rollout(
    model, tokenizer, conversation, gen_config,
    tool_parser=parse_xml_tools,
    tool_executor=execute_tools
)
```

### 2. Reward Primitives

```python
# Scalar reward
reward = compute_scalar_reward(rollout, reward_fn=correctness_checker)

# Vector rewards with custom aggregation
vector_rewards = compute_vector_reward(rollout, [correctness, safety, helpfulness])
final_reward = aggregate_vector_rewards(vector_rewards, weighted_sum)

# Reward shaping
shaped_reward = apply_reward_shaping(rollout, base_reward, kl_penalty_fn)
```

### 3. KL Divergence Primitives

```python
# Custom KL computation
kl = compute_forward_kl(logprobs, ref_logprobs, kl_fn=truncated_kl)

# Custom reference model updates
ref_model = update_reference_model(ref_model, policy, update_fn=ema_update)

# Different KL strategies
kl_fn = create_truncated_kl(threshold=1.0)
kl_fn = create_adaptive_kl(confidence_threshold=0.1)
```

### 4. Algorithm Primitives

```python
# GRPO
advantages = compute_grpo_advantages(batch, advantage_fn=normalized_advantages)
loss = compute_grpo_loss(batch, advantages, kl_coef=0.1)

# DAPO
pairs = compute_dapo_pairs(batch, pair_selector=top_vs_bottom)
loss = compute_dapo_loss(pairs, beta=0.1)

# PPO
advantages, returns = compute_ppo_advantages(batch, value_fn, gamma=0.99)
loss = compute_ppo_loss(batch, old_logprobs, advantages, clip_epsilon=0.2)

# Switch algorithms mid-training
algorithm_config = get_algorithm_for_iteration(iteration)
```

### 5. Training Loop Primitives

```python
# Full training loop with all extension points
final_state = train(
    initial_state=state,
    prompt_loader=load_prompts_fn,  # Custom prompt loading
    num_iterations=100,
    gen_config=gen_config,
    algorithm_config=algorithm_config,
    reward_fn=custom_reward,  # Custom reward
    ref_model_update_fn=ema_update,  # Custom ref model updates
    checkpoint_fn=save_checkpoint,  # Custom checkpointing
    eval_fn=evaluate,  # Custom evaluation
    logger=log_to_wandb  # Custom logging
)
```

## Extension Examples

### Example 1: Switching from GRPO to DAPO

```python
def adaptive_algorithm(iteration):
    if iteration < 50:
        return create_algorithm_config("grpo", {"kl_coef": 0.1})
    else:
        return create_algorithm_config("dapo", {"beta": 0.1})

# Use in training loop
for iteration in range(100):
    config = adaptive_algorithm(iteration)
    # ... training iteration
```

### Example 2: Multi-Objective Rewards

```python
# Define objectives
reward_fns = [
    correctness_reward,
    helpfulness_reward,
    safety_reward
]

# Combine with custom aggregation
def my_reward(rollout):
    vector = compute_vector_reward(rollout, reward_fns)
    # Safety is critical - zero out if unsafe
    if vector[2] < 0.8:
        return 0.0
    return weighted_sum(vector, [0.5, 0.3, 0.2])
```

### Example 3: Custom KL Penalty

```python
def custom_kl_fn(logprobs, ref_logprobs):
    """Only penalize high-confidence divergences"""
    kl = 0.0
    for lp, ref_lp in zip(logprobs, ref_logprobs):
        prob = math.exp(lp)
        if prob > 0.2:  # Only high-confidence tokens
            kl += prob * (lp - ref_lp)
    return kl

# Use in rollout creation
rollout = create_rollout(model, tokenizer, prompt, gen_config, ref_model)
custom_kl = custom_kl_fn(rollout.logprobs, rollout.ref_logprobs)
rollout = rollout._replace(kl_divergence=custom_kl)
```

### Example 4: Tool Use with Multi-Turn

```python
def tool_aware_reward(rollout):
    """Reward successful tool use"""
    base_reward = correctness_reward(rollout)
    
    if rollout.tool_calls:
        success_rate = sum(r['success'] for r in rollout.tool_results) / len(rollout.tool_results)
        tool_bonus = success_rate * 0.5
        return base_reward + tool_bonus
    
    return base_reward

rollout = create_multiturn_rollout(
    model, tokenizer, conversation, gen_config,
    tool_parser=parse_xml_tools,
    tool_executor=execute_function_calls,
    reward_fn=tool_aware_reward
)
```

### Example 5: EMA Reference Model

```python
def ema_update(ref_model, policy_model, decay=0.999):
    with torch.no_grad():
        for ref_param, policy_param in zip(ref_model.parameters(), policy_model.parameters()):
            ref_param.data.mul_(decay).add_(policy_param.data, alpha=1-decay)
    return ref_model

# Use in training
final_state = train(
    # ...
    ref_model_update_fn=lambda ref, policy: ema_update(ref, policy, decay=0.995)
)
```

## Key Features

### ✅ Hackability

- **Swap reward functions**: Pass any `Callable[[Rollout], float]`
- **Change KL computation**: Pass custom `kl_fn`
- **Switch algorithms**: Change `algorithm_config` mid-training
- **Update reference model**: Pass different `ref_model_update_fn`
- **Track tool calls**: Pass `tool_parser` and `tool_executor`
- **Custom generation**: Pass custom `generator` for vLLM, SGLang, etc.

### ✅ Modularity

Every primitive is:
- **Independent**: No hidden dependencies
- **Composable**: Functions compose naturally
- **Testable**: Pure functions easy to test
- **Reusable**: Use same primitives across projects

### ✅ Extension Points

Clear extension points in every primitive:
- `model_loader`, `tokenizer_loader` - Custom model loading
- `generator`, `logprob_extractor` - Custom generation
- `reward_fn`, `aggregator`, `shaping_fn` - Custom rewards
- `kl_fn`, `ref_model_fn`, `update_fn` - Custom KL
- `advantage_fn`, `loss_fn`, `pair_selector` - Custom algorithms
- `checkpoint_fn`, `eval_fn`, `logger` - Custom training loop

### ✅ Type Safety

Using NamedTuples provides:
- **Clear contracts**: Know exactly what data flows where
- **IDE support**: Auto-completion and type hints
- **Immutability**: NamedTuples are immutable by default
- **Self-documenting**: Field names make code readable

## Comparison: Functions vs Classes

### Traditional Class-Based Approach

```python
class RewardModel:
    def __init__(self, config):
        self.config = config
        self.model = self.load_model()
    
    def compute_reward(self, rollout):
        # Tightly coupled to class state
        return self.model(rollout)

# Hard to swap behavior without subclassing
```

### Functional Approach

```python
def compute_reward(rollout, reward_fn):
    """Just pass a different function!"""
    return reward_fn(rollout)

# Easy to swap
reward = compute_reward(rollout, correctness_fn)
reward = compute_reward(rollout, safety_fn)
reward = compute_reward(rollout, lambda r: custom_logic(r))
```

## Data Flow Example

```
Prompts
  ↓
create_batch_rollouts
  ├─→ generate_with_logprobs (model, tokenizer, prompt)
  │     └─→ Generation (text, logprobs, token_ids)
  │
  ├─→ get_reference_logprobs (ref_model, token_ids)
  │     └─→ ref_logprobs
  │
  ├─→ compute_forward_kl (logprobs, ref_logprobs)
  │     └─→ kl_divergence
  │
  └─→ reward_fn (rollout)
        └─→ reward

Rollouts (List[Rollout])
  ↓
Batch
  ↓
compute_grpo_advantages
  └─→ advantages

compute_grpo_loss (batch, advantages)
  └─→ loss

optimizer.step()
  └─→ Updated model

update_reference_model (ref_model, model)
  └─→ Updated ref_model
```

All data flows explicitly through function parameters!

## Best Practices

### 1. Keep Functions Pure

```python
# Good: Pure function
def compute_reward(rollout: Rollout) -> float:
    return len(rollout.generation.text) / 100

# Avoid: Side effects
def compute_reward_bad(rollout: Rollout) -> float:
    self.total_rewards += reward  # Mutating state
    return reward
```

### 2. Use Closures for Configuration

```python
# Good: Closure captures configuration
def create_reward_fn(min_length=100):
    def reward_fn(rollout):
        length = len(rollout.generation.text)
        return length / min_length if length >= min_length else 0.0
    return reward_fn

reward_fn = create_reward_fn(min_length=200)
```

### 3. Compose Functions

```python
# Compose multiple reward functions
reward_fn = combine_rewards(
    length_reward(min_length=100),
    keyword_reward(["because", "therefore"]),
    safety_check_reward(),
    weights=[0.5, 0.3, 0.2]
)
```

### 4. Use Type Hints

```python
from typing import Callable, List

def create_rollout(
    model: Any,
    tokenizer: Any,
    prompt: str,
    reward_fn: Optional[Callable[[Rollout], float]] = None
) -> Rollout:
    ...
```

## Performance Considerations

### Batching

```python
# Generate in batches for efficiency
batch_results = batch_generate(
    model, tokenizer, prompts,
    gen_config,
    batch_size=16  # Process 16 at a time
)
```

### Reference Model Caching

```python
# Cache reference logprobs if not updating ref model often
ref_logprobs_cache = {}

def cached_ref_logprobs(ref_model, prompt, token_ids):
    key = (prompt, tuple(token_ids))
    if key not in ref_logprobs_cache:
        ref_logprobs_cache[key] = get_reference_logprobs(ref_model, prompt, token_ids)
    return ref_logprobs_cache[key]
```

### Gradient Accumulation

```python
# Accumulate gradients for larger effective batch size
for i, mini_batch in enumerate(batches):
    loss = compute_loss(mini_batch, algorithm_config)
    loss = loss / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

## Debugging Tips

### 1. Inspect Rollouts

```python
from utilities import inspect_rollout

rollout = create_rollout(model, tokenizer, prompt, gen_config)
inspect_rollout(rollout)  # Pretty-print all fields
```

### 2. Validate Batches

```python
from utilities import validate_batch

batch = create_batch_rollouts(...)
validation_report = validate_batch(batch)

if not validation_report["is_valid"]:
    print("Invalid rollouts:", validation_report["invalid_rollouts"])
```

### 3. Plot Training Curves

```python
from utilities import plot_training_curves

plot_training_curves(final_state.stats_history, save_path="training_curves.png")
```

## Contributing

This is a functional architecture - extending it is as simple as writing new functions!

### Adding a New Algorithm

```python
def compute_my_algorithm_loss(batch: Batch, params: Dict[str, Any]) -> torch.Tensor:
    """Your custom algorithm"""
    # Implement loss computation
    return loss

# Register in dispatcher
def compute_loss_by_algorithm(batch, algorithm_config, **kwargs):
    if algorithm_config["algorithm"] == "my_algorithm":
        return compute_my_algorithm_loss(batch, algorithm_config["params"])
    # ... existing algorithms
```

### Adding a New Reward Type

```python
def my_custom_reward(rollout: Rollout) -> float:
    """Your custom reward logic"""
    # Implement reward computation
    return reward

# Use it anywhere
rollout = create_rollout(model, tokenizer, prompt, gen_config, reward_fn=my_custom_reward)
```

## License

MIT

## Citation

```bibtex
@software{rl_primitives_functional,
  title={RL Training Primitives: Functional Architecture},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/rl-primitives}
}
```

---

**Built with functions, not classes. Hack away! 🚀**
