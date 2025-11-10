# RL Algorithms Quick Reference

Quick reference for reinforcement learning algorithms used in LLM post-training.

## PPO (Proximal Policy Optimization)

### Overview
- Most popular algorithm for RLHF
- Uses clipped objective to prevent large policy updates
- Requires value function (critic) alongside policy (actor)
- On-policy algorithm

### Key Hyperparameters
```python
ppo_epochs = 4                  # Number of optimization epochs per batch
clip_range = 0.2                # Clipping parameter
value_loss_coef = 0.1           # Value function loss coefficient
entropy_coef = 0.01             # Entropy bonus coefficient
kl_coef = 0.1                   # KL divergence penalty
learning_rate = 1e-6            # Learning rate
```

### When to Use
- Standard RLHF training
- When you have compute for separate value function
- Need stable, reliable training

### Pros/Cons
✅ Stable training  
✅ Well-studied and documented  
✅ Good performance  
❌ Requires value function  
❌ More complex implementation  
❌ Slower than some alternatives  

## GRPO (Group Relative Policy Optimization)

### Overview
- Simplified PPO variant used in nanochat
- No separate value function needed
- Uses group-relative advantages instead
- Behaves similarly to REINFORCE with baseline

### Key Hyperparameters
```python
group_size = 8                  # Number of samples per prompt
temperature = 0.8               # Sampling temperature
kl_coef = 0.1                   # KL divergence penalty
learning_rate = 1e-6            # Learning rate
```

### When to Use
- Simpler alternative to PPO
- Limited compute (no need for critic)
- Quick prototyping

### Pros/Cons
✅ Simpler than PPO  
✅ No value function needed  
✅ Fast to implement  
❌ Potentially less sample efficient  
❌ Can be unstable with bad rewards  

## REINFORCE

### Overview
- Classic policy gradient algorithm
- Simple and foundational
- High variance without baselines
- On-policy algorithm

### Key Hyperparameters
```python
learning_rate = 1e-5            # Learning rate (often higher than PPO)
entropy_coef = 0.01             # Entropy bonus
baseline = 'mean_reward'        # Variance reduction
```

### When to Use
- Simple tasks
- Teaching/learning purposes
- Baseline for comparison

### Pros/Cons
✅ Simplest algorithm  
✅ Easy to understand  
✅ Fast to implement  
❌ High variance  
❌ Sample inefficient  
❌ Unstable training  

## On-Policy Distillation

### Overview
- Not strictly RL, but related
- Train student to mimic teacher on teacher-generated data
- Uses KL divergence loss
- Useful for model compression

### Key Hyperparameters
```python
temperature = 2.0               # Distillation temperature
alpha = 0.5                     # Balance between KL and CE loss
num_generations = 4             # Samples per prompt
learning_rate = 2e-5            # Learning rate
```

### When to Use
- Compressing larger models
- Transfer learning
- Improving smaller models with larger ones

### Pros/Cons
✅ Effective for compression  
✅ Simpler than RL  
✅ Stable training  
❌ Requires teacher model  
❌ Not true RL optimization  

## DPO (Direct Preference Optimization)

### Overview
- Optimizes directly on preference data
- No separate reward model needed
- Simpler than RLHF pipeline
- Offline algorithm

### Key Hyperparameters
```python
beta = 0.1                      # KL penalty coefficient
learning_rate = 5e-7            # Learning rate
reference_free = False          # Whether to use reference model
```

### When to Use
- Have preference pairs (chosen/rejected)
- Want simpler than full RLHF
- Offline optimization

### Pros/Cons
✅ No reward model needed  
✅ Simpler than PPO  
✅ Direct optimization  
❌ Requires preference data  
❌ Offline only  
❌ May be less flexible  

## Comparing Algorithms

### Sample Efficiency
1. DPO (offline, uses fixed dataset)
2. PPO
3. GRPO
4. REINFORCE

### Implementation Complexity
1. REINFORCE (simplest)
2. GRPO
3. DPO
4. PPO (most complex)

### Stability
1. DPO
2. PPO
3. GRPO
4. REINFORCE

### Compute Requirements
- Lowest: REINFORCE, GRPO, DPO
- Highest: PPO (needs value function)

## Algorithm Selection Guide

Choose based on your constraints:

**Limited Compute → GRPO or REINFORCE**
- Single GPU
- Small models
- Quick experiments

**Stability Priority → PPO or DPO**
- Production systems
- Important applications
- Large investments

**Have Preference Data → DPO**
- Human/AI feedback pairs
- Simpler pipeline desired

**Need Flexibility → PPO**
- Complex reward functions
- Multi-task training
- Advanced features

**Quick Prototyping → GRPO**
- Fast iteration
- Testing ideas
- Research

## Implementation Tips

### Reward Scaling
```python
# Normalize rewards to reasonable range
rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
rewards = torch.clamp(rewards, -10, 10)
```

### KL Penalty
```python
# Prevent model from deviating too far from reference
kl_penalty = kl_coef * (current_logprobs - ref_logprobs)
total_reward = reward - kl_penalty
```

### Gradient Clipping
```python
# Prevent exploding gradients
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### Learning Rate Scheduling
```python
from torch.optim.lr_scheduler import CosineAnnealingLR

scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
```

## Common Patterns

### Episode Rollout
```python
# Generate responses
responses = model.generate(prompts, max_new_tokens=100)

# Compute rewards
rewards = [compute_reward(p, r) for p, r in zip(prompts, responses)]

# Compute advantages
advantages = compute_advantages(rewards, values=None)  # GRPO/REINFORCE
# or
advantages = compute_advantages(rewards, values)  # PPO
```

### Training Step
```python
# Compute loss
policy_loss = compute_policy_loss(logprobs, old_logprobs, advantages)
value_loss = compute_value_loss(values, returns)  # PPO only
entropy_loss = compute_entropy(logits)

# Total loss
loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_loss

# Optimize
loss.backward()
optimizer.step()
optimizer.zero_grad()
```

## References

- PPO Paper: https://arxiv.org/abs/1707.06347
- RLHF Paper: https://arxiv.org/abs/2203.02155
- DPO Paper: https://arxiv.org/abs/2305.18290
- InstructGPT Paper: https://arxiv.org/abs/2203.02155
- GRPO (used in DeepSeek): https://arxiv.org/abs/2402.03300
