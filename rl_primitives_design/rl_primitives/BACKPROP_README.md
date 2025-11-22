# BackpropModule Documentation

## Overview

The `BackpropModule` is a production-ready component for handling gradient computation and model updates in RL training of Large Language Models. It provides a unified interface for multiple RL algorithms (PPO, GRPO, REINFORCE) with comprehensive support for:

- **Generalized Advantage Estimation (GAE)** - Variance reduction for advantage computation
- **PPO Clipped Loss** - Proximal Policy Optimization with importance sampling
- **KL Divergence** - Policy regularization against reference model
- **Reference Model Management** - Hard copy or EMA updates
- **Gradient Clipping** - Stabilize training with gradient norm clipping
- **Multiple Algorithms** - Flexible support for PPO, GRPO, and REINFORCE

## Quick Start

```python
from rl_primitives import BackpropModule, BackpropConfig
import torch

# Create configuration
config = BackpropConfig(
    gamma=0.99,
    gae_lambda=0.95,
    ppo_clip_range=0.2,
    kl_coef=0.1,
    max_grad_norm=1.0,
)

# Initialize module
backprop = BackpropModule(
    model=your_model,
    optimizer=your_optimizer,
    config=config,
)

# Compute advantages
advantages = backprop.compute_advantages(rewards, values)

# Compute PPO loss
loss_dict = backprop.compute_ppo_loss(
    current_log_probs, old_log_probs, advantages
)

# Update model
backprop.update(loss_dict['total_loss'])

# Update reference model periodically
backprop.update_ref_model(method="ema")
```

## Architecture Design

Based on the modular architecture from `v2.md`, the BackpropModule separates gradient computation from:
- Environment interactions (handled by Environment module)
- Rollout collection (handled by RolloutManager)
- Reward computation (handled by RewardComputer)

This separation enables:
1. **Algorithm flexibility** - Switch between PPO/GRPO/REINFORCE easily
2. **Reusability** - Same backprop module across different experiments
3. **Clarity** - Each component has a single responsibility

## Key Methods

### 1. Advantage Computation

#### `compute_advantages(rewards, values=None, dones=None, gamma=None, lam=None)`

Computes advantages using either:
- **GAE (with values)**: Generalized Advantage Estimation for low variance
- **Simple discounting (without values)**: For GRPO/REINFORCE

```python
# PPO-style with GAE
advantages = backprop.compute_advantages(
    rewards=rewards,      # shape: (batch, timesteps)
    values=values,        # shape: (batch, timesteps+1)
    dones=dones,          # shape: (batch, timesteps)
    gamma=0.99,
    lam=0.95,
)

# GRPO-style without value function
advantages = backprop.compute_advantages(
    rewards=rewards,      # shape: (batch, timesteps)
    gamma=0.99,
)
```

**GAE Formula:**
```
A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
where δ_t = r_t + γV(s_{t+1}) - V(s_t)
```

**References:**
- [GAE Paper](https://arxiv.org/abs/1506.02438)
- [PyTorch Implementation](https://nn.labml.ai/rl/ppo/gae.html)
- [TorchRL Documentation](https://docs.pytorch.org/rl/0.7/reference/generated/torchrl.objectives.value.GAE.html)

#### `compute_relative_advantages(rewards, group_size=None)`

GRPO-style group-relative advantages:

```python
# For GRPO: multiple samples per prompt
advantages = backprop.compute_relative_advantages(
    rewards=rewards,      # shape: (num_groups, group_size)
    group_size=8,
)
# Returns: rewards - group_mean
```

### 2. KL Divergence

#### `compute_kl(prompts, responses, attention_mask=None, ref_override=None)`

Computes KL divergence between current policy and reference policy:

```python
kl_div = backprop.compute_kl(
    prompts=prompt_tokens,
    responses=response_tokens,
    attention_mask=attention_mask,
)
```

**KL Formula:**
```
KL(π || π_ref) = E[log π(a|s) - log π_ref(a|s)]
```

### 3. Loss Computation

#### `compute_loss(log_probs, advantages, kl_div=None, entropy=None, kl_weight=None)`

Basic policy gradient loss (REINFORCE/GRPO):

```python
loss_dict = backprop.compute_loss(
    log_probs=log_probs,
    advantages=advantages,
    kl_div=kl_div,         # Optional KL penalty
    entropy=entropy,        # Optional entropy bonus
)
```

**Loss Formula:**
```
L = -E[log π(a|s) * A] + β_kl * KL - β_ent * H
```

#### `compute_ppo_loss(current_log_probs, old_log_probs, advantages, ...)`

PPO clipped loss with optional value function:

```python
loss_dict = backprop.compute_ppo_loss(
    current_log_probs=current_log_probs,
    old_log_probs=old_log_probs,
    advantages=advantages,
    values=values,              # Optional
    old_values=old_values,      # Optional
    returns=returns,            # Optional
    entropy=entropy,            # Optional
    kl_div=kl_div,             # Optional
)
```

**PPO Formula:**
```
L^CLIP = E[min(r_t(θ)A_t, clip(r_t(θ), 1-ε, 1+ε)A_t)]
where r_t(θ) = π_θ(a|s) / π_θ_old(a|s)
```

**Returns dictionary with:**
- `total_loss`: Combined loss for backprop
- `policy_loss`: Clipped policy gradient loss
- `value_loss`: Value function MSE (if values provided)
- `entropy_loss`: Entropy bonus
- `kl_loss`: KL divergence penalty
- `ratio_mean`, `ratio_std`: Importance sampling statistics

### 4. Model Updates

#### `update(loss, retain_graph=False)`

Perform gradient descent step with optional gradient clipping:

```python
metrics = backprop.update(loss)
# Returns: {'grad_norm': ..., 'loss': ...}
```

**Steps:**
1. Zero gradients
2. Backward pass
3. Compute gradient norm
4. Clip gradients (if configured)
5. Optimizer step

#### `update_ref_model(method=None, alpha=None)`

Update reference model for KL divergence:

```python
# Hard copy (complete synchronization)
backprop.update_ref_model(method="copy")

# EMA update (smooth blending)
backprop.update_ref_model(method="ema", alpha=0.99)
# ref = 0.99 * ref + 0.01 * current
```

## Configuration

### BackpropConfig

```python
@dataclass
class BackpropConfig:
    # Advantage computation
    gamma: float = 0.99              # Discount factor
    gae_lambda: float = 0.95         # GAE lambda

    # PPO settings
    ppo_clip_range: float = 0.2      # Clipping epsilon
    ppo_epochs: int = 4              # Update epochs per batch

    # Loss coefficients
    value_loss_coef: float = 0.1     # Value function weight
    entropy_coef: float = 0.01       # Entropy bonus
    kl_coef: float = 0.1             # KL penalty

    # Optimization
    max_grad_norm: float = 1.0       # Gradient clipping (None=disable)

    # Reference model
    ref_update_method: str = "copy"  # "copy" or "ema"
    ema_alpha: float = 0.99          # EMA coefficient

    # Algorithm-specific
    normalize_advantages: bool = True
    clip_value_loss: bool = True
```

## Algorithm Support

### PPO (Proximal Policy Optimization)

**Requirements:**
- Value function model
- Multiple update epochs per batch
- Clipped importance sampling

**Example:**
```python
config = BackpropConfig(
    ppo_clip_range=0.2,
    ppo_epochs=4,
    value_loss_coef=0.1,
)

# Collect trajectories
advantages = backprop.compute_advantages(rewards, values, dones)

# Multiple epochs on same data
for epoch in range(config.ppo_epochs):
    loss_dict = backprop.compute_ppo_loss(
        current_log_probs, old_log_probs, advantages,
        values, old_values, returns
    )
    backprop.update(loss_dict['total_loss'])
```

### GRPO (Group Relative Policy Optimization)

**Requirements:**
- Multiple samples per prompt
- Group-relative advantages
- No value function needed

**Example:**
```python
config = BackpropConfig(
    kl_coef=0.1,
    normalize_advantages=True,
)

# Generate multiple samples per prompt
# rewards shape: (num_prompts, samples_per_prompt)

advantages = backprop.compute_relative_advantages(
    rewards, group_size=8
)

loss_dict = backprop.compute_loss(log_probs, advantages)
backprop.update(loss_dict['total_loss'])
```

### REINFORCE

**Requirements:**
- Simple policy gradient
- Optional baseline subtraction

**Example:**
```python
config = BackpropConfig(
    entropy_coef=0.01,
    normalize_advantages=True,
)

# Simple discounted returns
advantages = backprop.compute_advantages(rewards)

loss_dict = backprop.compute_loss(log_probs, advantages)
backprop.update(loss_dict['total_loss'])
```

## Helper Functions

### `compute_returns(rewards, values, dones, gamma)`

Compute target returns for value function training:

```python
from rl_primitives.backprop import compute_returns

returns = compute_returns(rewards, values, dones, gamma=0.99)
# Returns: r_t + γ*V(s_{t+1})*(1-done)
```

### `compute_entropy(logits, mask=None)`

Compute policy entropy for exploration:

```python
from rl_primitives.backprop import compute_entropy

entropy = compute_entropy(logits, mask=attention_mask)
# Returns: -Σ p(a) log p(a)
```

## Usage Patterns

### Pattern 1: Standard PPO Training

```python
# Setup
backprop = BackpropModule(model, optimizer, value_model, config)

# Training loop
for batch in dataloader:
    # Collect trajectories
    trajectories, rewards, values = collect_rollouts(batch)

    # Compute advantages
    advantages = backprop.compute_advantages(rewards, values)

    # Store old log probs
    old_log_probs = get_log_probs(trajectories)

    # Multiple PPO epochs
    for _ in range(config.ppo_epochs):
        current_log_probs = get_log_probs(trajectories)

        loss_dict = backprop.compute_ppo_loss(
            current_log_probs, old_log_probs, advantages,
            values, old_values, returns
        )

        backprop.update(loss_dict['total_loss'])

    # Update reference model periodically
    if step % 100 == 0:
        backprop.update_ref_model(method="ema")
```

### Pattern 2: GRPO with KL Penalty

```python
# Setup
backprop = BackpropModule(model, optimizer, config)

# Training loop
for prompts in dataloader:
    # Generate multiple responses per prompt
    responses = model.generate(prompts, num_samples=8)
    rewards = compute_rewards(prompts, responses)

    # Group-relative advantages
    advantages = backprop.compute_relative_advantages(
        rewards.view(-1, 8), group_size=8
    )

    # Get log probs
    log_probs = get_log_probs(prompts, responses)

    # Compute KL
    kl_div = backprop.compute_kl(prompts, responses)

    # Loss with KL penalty
    loss_dict = backprop.compute_loss(
        log_probs, advantages.flatten(),
        kl_div=kl_div, kl_weight=0.1
    )

    backprop.update(loss_dict['total_loss'])
```

### Pattern 3: Simple REINFORCE

```python
# Setup
backprop = BackpropModule(model, optimizer, config)

# Training loop
for batch in dataloader:
    # Generate and collect
    responses = model.generate(batch['prompts'])
    rewards = compute_rewards(batch['prompts'], responses)

    # Simple advantages (discounted returns)
    advantages = backprop.compute_advantages(rewards)

    # Get log probs
    log_probs = get_log_probs(batch['prompts'], responses)

    # Simple policy gradient
    loss_dict = backprop.compute_loss(log_probs, advantages)

    backprop.update(loss_dict['total_loss'])
```

## Advanced Features

### Custom Advantage Normalization

```python
# Disable automatic normalization
config = BackpropConfig(normalize_advantages=False)

# Manual normalization with custom scheme
advantages = backprop.compute_advantages(rewards, values)
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
advantages = torch.clamp(advantages, -10, 10)  # Clip outliers
```

### Dynamic KL Coefficient Scheduling

```python
# Start with low KL penalty, increase over time
initial_kl = 0.01
final_kl = 0.5

for step in range(total_steps):
    # Linear schedule
    kl_weight = initial_kl + (final_kl - initial_kl) * (step / total_steps)

    loss_dict = backprop.compute_loss(
        log_probs, advantages, kl_div, kl_weight=kl_weight
    )
```

### Gradient Accumulation

```python
accumulation_steps = 4

for i, batch in enumerate(dataloader):
    loss_dict = backprop.compute_ppo_loss(...)

    # Scale loss by accumulation steps
    loss = loss_dict['total_loss'] / accumulation_steps

    # Update with retain_graph for all but last step
    if (i + 1) % accumulation_steps == 0:
        backprop.update(loss, retain_graph=False)
    else:
        backprop.update(loss, retain_graph=True)
```

## Best Practices

### 1. Advantage Normalization

✅ **Do:** Normalize advantages for stable training
```python
config = BackpropConfig(normalize_advantages=True)
```

❌ **Don't:** Use raw advantages with high variance
```python
advantages = rewards  # No normalization
```

### 2. Gradient Clipping

✅ **Do:** Clip gradients to prevent exploding gradients
```python
config = BackpropConfig(max_grad_norm=1.0)
```

❌ **Don't:** Disable clipping without careful monitoring
```python
config = BackpropConfig(max_grad_norm=None)
```

### 3. Reference Model Updates

✅ **Do:** Update reference model periodically
```python
if step % 100 == 0:
    backprop.update_ref_model(method="ema", alpha=0.99)
```

❌ **Don't:** Never update reference model (stale KL)
```python
# Never calling update_ref_model
```

### 4. PPO Epochs

✅ **Do:** Use 3-5 PPO epochs per batch
```python
config = BackpropConfig(ppo_epochs=4)
```

❌ **Don't:** Use too many epochs (overfitting)
```python
config = BackpropConfig(ppo_epochs=20)
```

### 5. Value Function Coefficient

✅ **Do:** Balance policy and value losses
```python
config = BackpropConfig(value_loss_coef=0.1)
```

❌ **Don't:** Overwhelm policy loss with value loss
```python
config = BackpropConfig(value_loss_coef=10.0)
```

## Troubleshooting

### Issue: High gradient norms

**Solution:** Lower learning rate or increase gradient clipping
```python
config = BackpropConfig(max_grad_norm=0.5)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-7)
```

### Issue: Policy diverging from reference

**Solution:** Increase KL coefficient or update reference more frequently
```python
config = BackpropConfig(kl_coef=0.5)
# Update ref model every 50 steps instead of 100
```

### Issue: High variance in advantages

**Solution:** Use GAE with higher lambda for more smoothing
```python
config = BackpropConfig(gae_lambda=0.98)  # More bias, less variance
```

### Issue: Value function not learning

**Solution:** Increase value loss coefficient or learning rate
```python
config = BackpropConfig(value_loss_coef=0.5)
```

## References

### Papers
- [PPO: Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)
- [GAE: Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)
- [GRPO: Group Relative Policy Optimization](https://arxiv.org/abs/2402.03300)
- [InstructGPT: Training language models to follow instructions](https://arxiv.org/abs/2203.02155)

### Implementations
- [TorchRL GAE](https://docs.pytorch.org/rl/0.7/reference/generated/torchrl.objectives.value.GAE.html)
- [nanochat RL Training](https://github.com/karpathy/nanochat)
- [labml.ai PPO](https://nn.labml.ai/rl/ppo/gae.html)

### Tutorials
- [GAE: Maths and Code (Towards Data Science)](https://towardsdatascience.com/generalized-advantage-estimate-maths-and-code-b5d5bd3ce737/)
- [PyTorch RL Tutorials (bentrevett)](https://github.com/bentrevett/pytorch-rl)

## License

Part of RL Primitives package - see main LICENSE file.
