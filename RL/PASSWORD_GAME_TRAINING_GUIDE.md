# Password Game PPO Training Guide

## Quick Start

### 1. Check Prerequisites

```bash
# Check GPU
nvidia-smi

# Verify Python
python --version  # Should be 3.10+
```

### 2. Install Dependencies

```bash
# Install PyTorch with CUDA
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121

# Install Flash Attention
pip install flash-attn --no-build-isolation

# Install other packages
pip install transformers>=4.45.0 accelerate datasets tokenizers wandb tensorboard
```

### 3. Set Up API Keys

Create a `keys.py` file in the project root:

```python
# keys.py
WANDB_API_KEY = "your_wandb_key"
HF_TOKEN = "your_huggingface_token"
OPENAI_API_KEY = "your_openai_key"  # For password game dynamic elements
```

### 4. Run Training

```bash
# Option A: Use launch script
./launch_password_ppo.sh default

# Option B: Direct Python
python verl_password_game_ppo.py

# Option C: Jupyter notebook
jupyter notebook verl_password_game_training.ipynb
```

---

## Understanding the Password Game

### Rule Accumulation Example

```
Turn 1:
  Rules: [1. At least 5 characters]
  Password: "Hello"
  ✓ Passes

Turn 2:
  Rules: [1. At least 5 characters,
          2. Include a number]
  Password: "Hello1"
  ✓ Passes both

Turn 3:
  Rules: [1. At least 5 characters,
          2. Include a number,
          3. Include uppercase]
  Password: "Hello1"
  ✓ Passes all three (already has uppercase!)

Turn 4:
  Rules: [1-3 above,
          4. Include special character]
  Password: "Hello1!"
  ✓ Passes all four
```

### Challenge: Later Rules Are Harder

```
Turn 8:
  Rules: [1-7 above,
          8. Roman numerals multiply to 35]
  Need: V (5) × VII (7) = 35
  Example: "Hello1!StarbucksVVII"

Turn 13:
  Rules: [1-12 above,
          13. Include country name: Albania]
  Example: "Hello1!StarbucksVVIIAlbania"
```

The password must grow to satisfy all rules while staying concise!

---

## PPO Algorithm Flow

### Episode Collection Phase

```
┌─────────────────────────────────────────────────────┐
│ 1. RESET ENVIRONMENT                                 │
│    - New game, rule 1                                │
│    - Initial prompt                                  │
└─────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────┐
│ 2. GENERATE RESPONSE                                 │
│    Policy(prompt) → password                         │
│    - Use current policy (Qwen model)                 │
│    - Sample with temperature                         │
│    - Parse thinking if enabled                       │
└─────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────┐
│ 3. GET REWARD                                        │
│    Environment(password) → reward                    │
│    - Check all rules                                 │
│    - Calculate shaped reward                         │
│    - Advance to next rule                            │
└─────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────┐
│ 4. STORE TRANSITION                                  │
│    Save: (state, action, reward, next_state)        │
└─────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────┐
│ 5. CHECK DONE                                        │
│    - If not done: go to step 2 with next rule       │
│    - If done: episode complete                       │
└─────────────────────────────────────────────────────┘
```

### PPO Update Phase

```
┌─────────────────────────────────────────────────────┐
│ 1. COMPUTE OLD PROBABILITIES                         │
│    - Log probs from current policy (before update)  │
│    - Values from value head                          │
│    - Log probs from reference model (frozen)         │
└─────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────┐
│ 2. COMPUTE ADVANTAGES (GAE)                          │
│    A_t = Σ (γλ)^k δ_{t+k}                           │
│    where δ_t = r_t + γV(s_{t+1}) - V(s_t)           │
│    - Smooth credit assignment                        │
│    - Balance bias vs. variance                       │
└─────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────┐
│ 3. PPO POLICY UPDATE (4 epochs)                      │
│    For each epoch:                                   │
│      - Compute new log probs                         │
│      - Compute ratio = exp(new - old)                │
│      - Clip ratio to [1-ε, 1+ε]                      │
│      - Loss = -min(ratio×A, clip×A)                  │
│      - Backprop and update                           │
└─────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────┐
│ 4. VALUE HEAD UPDATE                                 │
│    Loss = MSE(V(s), returns)                         │
└─────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────┐
│ 5. KL PENALTY                                        │
│    Loss += β × KL(policy || reference)              │
│    - Prevents policy from drifting too far           │
│    - Adaptive β (increases if KL too high)           │
└─────────────────────────────────────────────────────┘
```

---

## Reward Shaping Strategy

### Components

| Component | Weight | Purpose |
|-----------|--------|---------|
| **Rules Passing** | +1.0 per rule | Core objective |
| **Progress Bonus** | +2.0 | Encourage advancing |
| **Length Penalty** | -0.1 per char | Prefer concise passwords |
| **Success Bonus** | +10.0 | Major milestone reward |

### Example Calculation

```python
# Turn 5: Try to satisfy 5 rules
password = "Hello1!Pepsi"  # 12 characters
feedback = {
    'total_passing': 4,  # Only passes 4 rules (failed one)
    'length': 12
}

# Calculation
reward = 0.0
reward += 4 * 1.0        # Rules passing: +4.0
reward += 0.0            # Progress bonus: 0 (failed to advance)
reward -= 12 * 0.1       # Length penalty: -1.2
reward += 0.0            # Success bonus: 0 (< 10 rules)

total_reward = 2.8
```

### Why Shaped Rewards?

**Without shaping:**
```
Turn 1: password="Hello"     → reward = 0 (game not done)
Turn 2: password="Hello1"    → reward = 0 (game not done)
Turn 3: password="Hello1!"   → reward = 0 (game not done)
...
Turn 10: game ends           → reward = 5.0 (if 5 rules passed)

Problem: All reward at the end! Hard to learn which actions helped.
```

**With shaping:**
```
Turn 1: password="Hello"     → reward = 0.5 (1 rule - 0.5 length)
Turn 2: password="Hello1"    → reward = 2.4 (2 rules + 2 progress - 0.6)
Turn 3: password="Hello1!"   → reward = 2.3 (3 rules + 2 progress - 0.7)
...

Problem solved: Immediate feedback for each action!
```

---

## Hyperparameter Tuning Guide

### Learning Rate

```python
# Too low (1e-8)
Problem: Training too slow, no improvement
Symptom: Eval reward stays near baseline
Fix: Increase to 1e-7 or 5e-7

# Good (5e-7)
Result: Steady improvement
Symptom: Eval reward increases smoothly

# Too high (1e-5)
Problem: Policy unstable, erratic performance
Symptom: Reward spikes up and down
Fix: Decrease to 5e-7 or 1e-7
```

### Clip Range

```python
# Too low (0.1)
Problem: Too conservative, slow learning
Use case: When policy is already good, fine-tuning

# Good (0.2)
Result: Balanced exploration and stability
Default: Recommended starting point

# Too high (0.5)
Problem: Policy can change too much, unstable
Symptom: High KL divergence, erratic rewards
```

### KL Coefficient

```python
# Too low (0.01)
Problem: Policy drifts from reference, forgets
Symptom: KL divergence > 1.0, performance drops

# Good (0.1 with adaptive)
Result: Policy improves but stays grounded
Note: Adaptive adjustment is enabled by default

# Too high (0.5)
Problem: Policy too constrained, can't explore
Symptom: Low KL divergence, no improvement
```

### Batch Size & Episodes

```python
# Small batch (batch_size=2, episodes=50)
Pro: Fast iterations, less memory
Con: High variance, noisy gradients
Use: Quick experiments, debugging

# Medium batch (batch_size=4, episodes=100)
Pro: Balanced variance and speed
Con: None (recommended)
Use: Standard training

# Large batch (batch_size=8, episodes=200)
Pro: Low variance, stable gradients
Con: Slow iterations, more memory
Use: Final production runs
```

---

## Monitoring Training

### Key Metrics to Watch

#### 1. train/reward
```
Good:  Steadily increasing (0.5 → 1.0 → 2.0 → ...)
Bad:   Flat or decreasing
Action: Increase LR, increase progress_bonus
```

#### 2. train/rules_satisfied
```
Good:  Increasing over time (1 → 2 → 3 → 4 ...)
Bad:   Stuck at 1-2 rules
Action: Increase progress_bonus, check reward function
```

#### 3. train/kl_div
```
Good:  0.01 - 0.1 (controlled drift)
Bad:   > 0.5 (policy drifting too far)
Action: Let adaptive KL handle it, or increase kl_coef
```

#### 4. train/policy_loss
```
Good:  Decreasing over time
Bad:   Increasing or erratic
Action: Check clip_range, reduce LR
```

### WandB Dashboard Example

```
Epoch 1:
  train/reward: 1.2 ± 0.8
  train/rules_satisfied: 2.1
  eval/reward: 1.0
  eval/success_rate: 0.10 (10%)

Epoch 3:
  train/reward: 3.5 ± 1.2
  train/rules_satisfied: 4.8
  eval/reward: 3.2
  eval/success_rate: 0.35 (35%)

Epoch 5:
  train/reward: 6.8 ± 1.5
  train/rules_satisfied: 7.2
  eval/reward: 6.5
  eval/success_rate: 0.65 (65%)
```

---

## Common Issues & Solutions

### Issue 1: Model Not Learning

**Symptoms:**
- Reward stays near baseline
- Rules satisfied = 1-2 constantly

**Debug:**
```python
# Check if rewards are being computed
print(f"Reward: {reward}")  # Should be non-zero

# Check if gradients flowing
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.norm()}")

# Check advantages
print(f"Advantages: {advantages.mean()}, {advantages.std()}")
```

**Solutions:**
- Increase learning rate to 1e-6
- Increase progress_bonus to 5.0
- Check reward function is correct
- Verify environment is working

### Issue 2: Policy Collapse

**Symptoms:**
- Model generates same password every time
- Entropy near zero
- KL divergence increases rapidly

**Debug:**
```python
# Check generation diversity
passwords = [generate() for _ in range(10)]
print(f"Unique passwords: {len(set(passwords))}")

# Check entropy
print(f"Entropy: {entropy.item()}")
```

**Solutions:**
- Increase entropy_coef to 0.02
- Increase temperature to 0.9
- Reduce kl_coef to 0.05
- Check reference model is frozen

### Issue 3: High Variance

**Symptoms:**
- Reward jumps wildly
- Training unstable
- Can't tell if improving

**Debug:**
```python
# Check reward distribution
print(f"Reward: {rewards.mean()} ± {rewards.std()}")
print(f"Min: {rewards.min()}, Max: {rewards.max()}")
```

**Solutions:**
- Increase batch_size to 8
- Increase samples_per_state to 4
- Normalize advantages (already enabled)
- Use longer evaluation (50+ episodes)

### Issue 4: Out of Memory

**Symptoms:**
- CUDA OOM error
- Training crashes

**Solutions:**
```python
# Reduce batch size
config.batch_size = 2
config.samples_per_state = 1

# Reduce sequence lengths
config.max_prompt_length = 768
config.max_new_tokens = 128

# Reduce PPO epochs
config.ppo_epochs = 2

# Enable gradient checkpointing (add to model)
model.gradient_checkpointing_enable()
```

---

## Advanced: Multi-Turn Credit Assignment

### The Challenge

```
Episode:
  Turn 1: "Hello"      → +0.5
  Turn 2: "Hello1"     → +2.4
  Turn 3: "Hello1!"    → +2.3
  Turn 4: "Hello1!May" → +2.2

Question: How much credit does Turn 1 deserve for Turn 4's reward?
```

### Solution: GAE (Generalized Advantage Estimation)

**Formula:**
```
A_t = Σ_{k=0}^∞ (γλ)^k δ_{t+k}
where δ_t = r_t + γV(s_{t+1}) - V(s_t)
```

**Parameters:**
- `γ` (gamma): Discount factor (0.99)
  - Higher = more future credit
  - Lower = more immediate credit

- `λ` (lambda): GAE parameter (0.95)
  - λ=0: Only immediate TD error (high bias, low variance)
  - λ=1: Full Monte Carlo (low bias, high variance)
  - λ=0.95: Balanced (recommended)

**Effect:**
```
With γ=0.99, λ=0.95:

Turn 1 advantage = 0.5×δ₁ + 0.94×δ₂ + 0.89×δ₃ + 0.84×δ₄ + ...
                   ↑        ↑         ↑         ↑
                   now      soon      later     much later

Turn 1 gets credit for all future rewards, but discounted!
```

---

## Comparison: Password Game vs. Other Tasks

### vs. Atari Games

| Aspect | Password Game | Atari |
|--------|---------------|-------|
| **State** | Text (rules) | Image (screen) |
| **Action** | Text generation | Discrete buttons |
| **Episodes** | 1-26 turns | Hundreds of frames |
| **Reward** | Sparse, shaped | Dense, game score |
| **Challenge** | Reasoning, constraints | Reflexes, planning |

### vs. Dialogue RL

| Aspect | Password Game | Dialogue |
|--------|---------------|----------|
| **Turns** | 1-26 (fixed rules) | Variable (user driven) |
| **Goal** | Satisfy constraints | User satisfaction |
| **Reward** | Automatic (rule checker) | Human feedback |
| **Difficulty** | Increases predictably | Varies by user |

### vs. Math Problem Solving

| Aspect | Password Game | Math Problems |
|--------|---------------|---------------|
| **Constraints** | Accumulate | Independent |
| **Verification** | Automatic | Need solution checker |
| **Reasoning** | Creative (password construction) | Logical (step-by-step) |
| **Success** | Partial (satisfy N rules) | Binary (correct/wrong) |

---

## Next Steps

### Experiment Ideas

1. **Different Models**
   ```python
   # Try larger model
   config.model_name = "Qwen/Qwen2.5-1.5B"

   # Try different architecture
   config.model_name = "meta-llama/Llama-3.2-1B"
   ```

2. **Reward Variations**
   ```python
   # More length penalty (shorter passwords)
   config.length_penalty_scale = 0.2

   # Higher success bonus (aim for completion)
   config.final_success_bonus = 20.0

   # Bonus for specific hard rules
   if rule_index == 8:  # Roman numerals
       reward += 5.0
   ```

3. **Curriculum Learning**
   ```python
   # Start with easier rules
   if epoch < 2:
       config.max_turns_per_episode = 5
   elif epoch < 4:
       config.max_turns_per_episode = 10
   else:
       config.max_turns_per_episode = 15
   ```

4. **Ensemble Training**
   ```python
   # Train multiple policies, use best
   policies = [train_policy(seed=i) for i in range(5)]
   best = max(policies, key=lambda p: evaluate(p))
   ```

### Research Questions

1. Does thinking mode actually help?
   - Run with/without thinking, compare

2. What's the optimal reward balance?
   - Sweep length_penalty_scale

3. Can we learn to generalize?
   - Train on some rules, test on others

4. What's the sample efficiency?
   - Plot rules vs. episodes

---

## Resources

### Papers
- [PPO](https://arxiv.org/abs/1707.06347)
- [GAE](https://arxiv.org/abs/1506.02438)
- [RLHF](https://arxiv.org/abs/2203.02155)

### Code References
- [OpenAI Spinning Up](https://spinningup.openai.com/)
- [CleanRL](https://github.com/vwxyzjn/cleanrl)
- [Transformer RL](https://github.com/lvwerra/trl)

### Community
- WandB Reports
- GitHub Discussions
- Discord/Slack

---

## FAQ

**Q: How long does training take?**
A: On H100, ~2-4 hours for 5 epochs. On A100, ~4-6 hours.

**Q: Can I use multiple GPUs?**
A: Current implementation is single-GPU. DDP would require modifications.

**Q: What if I don't have API keys?**
A: Password game uses OpenAI for dynamic elements. Without it, some rules will fail.

**Q: Can I train on CPU?**
A: Not recommended. GPU is required for reasonable speed.

**Q: How do I resume training?**
A: Load checkpoint and continue. Need to implement checkpoint loading (TODO).

**Q: What's the minimum GPU memory?**
A: ~24GB for Qwen-0.6B with current settings. Reduce batch size for less.

**Q: Can I use LoRA instead of full fine-tuning?**
A: Yes, but would need to add LoRA adapter. Full fine-tuning works better for RL.

**Q: Why PPO and not DPO?**
A: DPO requires preference data. This task has automatic rewards, perfect for PPO.

---

## Changelog

**v1.0** (2025-01-10)
- Initial release
- PPO implementation
- Multi-turn episode handling
- Shaped rewards
- Thinking mode support
- WandB logging

---

## Contributing

Contributions welcome! Areas for improvement:

1. **Checkpoint Loading**: Resume from saved checkpoints
2. **DDP Support**: Multi-GPU training
3. **LoRA Option**: Memory-efficient training
4. **Better Parsing**: Smarter password extraction
5. **Rule Analysis**: Per-rule success rates
6. **Visualization**: Interactive training dashboard

---

## Acknowledgments

- OpenAI for PPO algorithm
- Qwen team for excellent models
- Password game creators for the challenge
- RL community for techniques and insights

---

**Happy Training! 🚀**
