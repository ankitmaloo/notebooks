# Verl-Based PPO Training for Password Game

A complete PPO (Proximal Policy Optimization) implementation for training Qwen3-0.6B on the multi-turn password game task.

## Overview

The password game is a unique RL challenge where:
- **Cumulative Rules**: Each turn adds a new constraint
- **Multi-turn Episodes**: Must satisfy ALL previous + current rules
- **Complex Reasoning**: Rules involve wordplay, math, elements, dates, etc.
- **Dynamic Elements**: Captcha, countries, Wordle answers, moon phases

## Key Features

### 1. Multi-Turn Episode Handling
- Full game runs as episodes (up to 26 rules)
- Each step requires satisfying all accumulated rules
- Proper GAE (Generalized Advantage Estimation) for multi-turn credit assignment

### 2. Shaped Reward System
```python
Reward = (Rules Passing × 1.0) + Progress Bonus - (Length × 0.1) + Success Bonus
```
- **Rules Passing**: +1 per satisfied rule
- **Progress Bonus**: +2 for advancing past previous best
- **Length Penalty**: -0.1 per character (encourages conciseness)
- **Success Bonus**: +10 for completing 10+ rules

### 3. Thinking Mode Support
- Leverages Qwen's thinking capabilities
- Parses `<think>` tags for reasoning
- Separate temperature for thinking vs. response

### 4. PPO Implementation
- **Clipped Policy**: Prevents destructive updates
- **Value Function**: Separate critic head
- **KL Divergence**: Adaptive penalty vs. reference model
- **GAE**: Multi-step advantage estimation
- **Entropy Bonus**: Encourages exploration

### 5. Single GPU Training
- Optimized for H100 (or A100)
- No DDP (Distributed Data Parallel)
- Flash Attention 2 for efficiency
- BF16 precision

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Password Game Environment                 │
│  - 26 cumulative rules                                       │
│  - Dynamic: captcha, country, wordle, moon phase            │
│  - Reward: rules + progress - length + success              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   Policy Network (Actor)                     │
│  - Qwen3-0.6B (full params, no LoRA)                       │
│  - Thinking mode enabled                                     │
│  - Temperature: 0.7                                          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   Value Network (Critic)                     │
│  - Single linear layer on hidden states                     │
│  - Predicts expected return                                  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                      PPO Update Loop                         │
│  1. Collect episodes (full game runs)                       │
│  2. Compute advantages (GAE)                                 │
│  3. Update policy (clipped objective)                        │
│  4. Update value (MSE loss)                                  │
│  5. KL penalty (vs. reference model)                         │
└─────────────────────────────────────────────────────────────┘
```

## Files

- **`verl_password_game_ppo.py`**: Main training script (standalone)
- **`verl_password_game_training.ipynb`**: Interactive Jupyter notebook
- **`PASSWORD_GAME_PPO_README.md`**: This file
- **`../tasks/password-game/game.py`**: Password game implementation

## Installation

```bash
# Install PyTorch with CUDA
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121

# Install Flash Attention
pip install flash-attn --no-build-isolation

# Install other dependencies
pip install transformers>=4.45.0 accelerate datasets tokenizers
pip install wandb tensorboard pandas matplotlib tqdm requests
```

## Usage

### Option 1: Command Line

```bash
# Basic training
python verl_password_game_ppo.py

# Custom configuration
python verl_password_game_ppo.py \
    --model Qwen/Qwen2.5-0.6B \
    --epochs 5 \
    --episodes-per-epoch 100 \
    --batch-size 4 \
    --lr 5e-7 \
    --output-dir ./my_run \
    --wandb-project my-project

# Disable thinking mode
python verl_password_game_ppo.py --no-thinking
```

### Option 2: Jupyter Notebook

```bash
jupyter notebook verl_password_game_training.ipynb
```

### Option 3: Python Import

```python
from verl_password_game_ppo import PasswordGamePPOConfig, train_ppo_password_game

config = PasswordGamePPOConfig(
    model_name="Qwen/Qwen2.5-0.6B",
    num_epochs=5,
    num_episodes_per_epoch=100,
    learning_rate=5e-7,
    enable_thinking=True,
)

train_ppo_password_game(config)
```

## Configuration

Key hyperparameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | `Qwen/Qwen2.5-0.6B` | HuggingFace model |
| `num_epochs` | 5 | Training epochs |
| `num_episodes_per_epoch` | 100 | Episodes per epoch |
| `batch_size` | 4 | Episode batch size |
| `samples_per_state` | 2 | Parallel samples per state |
| `learning_rate` | 5e-7 | Learning rate |
| `ppo_epochs` | 4 | PPO update epochs |
| `clip_range` | 0.2 | PPO clip epsilon |
| `gamma` | 0.99 | Discount factor |
| `gae_lambda` | 0.95 | GAE lambda |
| `kl_coef` | 0.1 | KL penalty coefficient (adaptive) |
| `max_turns_per_episode` | 10 | Max rules to attempt |
| `enable_thinking` | True | Use Qwen thinking mode |
| `progress_bonus` | 2.0 | Bonus per new rule |
| `length_penalty_scale` | 0.1 | Penalty per character |
| `final_success_bonus` | 10.0 | Bonus for 10+ rules |

## Training Process

### Phase 1: Baseline (Step 0)
- Evaluate untrained model
- Typically satisfies 1-2 rules
- Establishes performance floor

### Phase 2: Early Training (Steps 1-100)
- Model learns basic rules (length, uppercase, numbers)
- Progress bonus encourages exploration
- Rules satisfied: 2-4

### Phase 3: Mid Training (Steps 100-300)
- Model learns complex rules (roman numerals, elements)
- Begins optimizing password length
- Rules satisfied: 4-7

### Phase 4: Late Training (Steps 300+)
- Fine-tuning password construction
- Balancing all constraints
- Rules satisfied: 7-10+

### Expected Timeline (H100)
- **Baseline**: 2-5 minutes
- **Epoch**: ~30-45 minutes
- **Full training (5 epochs)**: 2-4 hours

## Reward Shaping Details

The reward function is crucial for this task:

```python
def calculate_shaped_reward(password, feedback, previous_turn):
    reward = 0.0

    # Base: Rules passing
    rules_passing = feedback['total_passing']
    reward += rules_passing * 1.0

    # Progress bonus: Advancing beyond previous best
    expected = previous_turn + 1
    if rules_passing >= expected:
        reward += 2.0  # progress_bonus

    # Length penalty: Shorter is better
    reward -= len(password) * 0.1  # length_penalty_scale

    # Success bonus: Complete 10+ rules
    if rules_passing >= 10:
        reward += 10.0  # final_success_bonus

    return reward
```

### Why This Works

1. **Sparse Rewards**: Game naturally has sparse rewards (pass/fail per rule)
2. **Progress Signal**: Bonus for advancing encourages exploration
3. **Length Optimization**: Penalty prevents overly long passwords
4. **Success Incentive**: Large bonus for major milestones

## Multi-Turn Handling

The password game is inherently multi-turn:

### Episode Structure
```
Turn 1: Rule 1 → Password A → Reward R1
Turn 2: Rule 1,2 → Password B → Reward R2
Turn 3: Rule 1,2,3 → Password C → Reward R3
...
```

### Key Challenges
1. **Credit Assignment**: Which action led to success?
   - Solution: GAE with γ=0.99, λ=0.95

2. **Compounding Difficulty**: Later rules are harder
   - Solution: Progress bonus scales with turn

3. **State Representation**: Must include all previous rules
   - Solution: Prompt includes full rule history

### GAE (Generalized Advantage Estimation)
```python
A_t = Σ (γλ)^k δ_{t+k}
where δ_t = r_t + γV(s_{t+1}) - V(s_t)
```

This provides smooth credit assignment across turns.

## Monitoring

### WandB Metrics

**Training**:
- `train/reward`: Episode reward
- `train/rules_satisfied`: Rules passed
- `train/loss`: Total loss
- `train/policy_loss`: PPO policy loss
- `train/value_loss`: Critic loss
- `train/kl_div`: KL vs. reference
- `train/kl_coef`: Adaptive KL coefficient

**Evaluation**:
- `eval/reward`: Mean reward
- `eval/rules_satisfied`: Mean rules
- `eval/success_rate`: % with ≥5 rules

### Checkpoints

Models saved to `output_dir/`:
- `best_model/`: Best validation performance
- `checkpoint-{step}/`: Periodic checkpoints
- `final_model/`: End of training
- `config.json`: Configuration
- `summary.json`: Training summary

## Troubleshooting

### GPU OOM (Out of Memory)
```python
# Reduce batch size
config.batch_size = 2
config.samples_per_state = 1

# Reduce sequence length
config.max_prompt_length = 768
config.max_new_tokens = 128

# Use gradient accumulation
config.gradient_accumulation_steps = 8
```

### Poor Performance
```python
# Increase learning rate
config.learning_rate = 1e-6

# Increase progress bonus
config.progress_bonus = 5.0

# More PPO epochs
config.ppo_epochs = 6

# Lower KL penalty
config.kl_coef = 0.05
```

### Model Not Exploring
```python
# Increase temperature
config.temperature = 0.9

# Increase entropy coefficient
config.entropy_coef = 0.02

# Reduce KL penalty
config.kl_coef = 0.05
```

### Too Much KL Drift
```python
# Increase KL coefficient
config.kl_coef = 0.2

# Use adaptive KL (enabled by default)
# Will automatically adjust based on kl_target
```

## Differences from Standard RL

### vs. Single-Turn RL
- **Episodes**: Multiple turns per episode
- **State**: Accumulates all previous rules
- **Reward**: Shaped for progress, not just terminal
- **Credit**: GAE for multi-step assignment

### vs. Supervised Learning
- **No Labels**: Learns from reward signal
- **Exploration**: Samples from policy distribution
- **Gradual**: Improves via small policy updates
- **Robust**: Less prone to overfitting

### vs. DPO/RLHF
- **Environment**: Interactive game, not preferences
- **Multi-turn**: Sequential decision making
- **Shaped Reward**: Explicit reward function
- **On-policy**: Samples from current policy

## Advanced: Custom Reward Functions

You can customize the reward function in `verl_password_game_ppo.py`:

```python
def _calculate_shaped_reward(self, password, feedback, previous_turn):
    # Your custom reward logic here

    # Example: Heavily penalize failures
    if feedback['total_passing'] < previous_turn:
        return -10.0

    # Example: Reward specific rules more
    for rule_info in feedback['rules_checked']:
        if rule_info['rule_index'] == 8:  # Roman numerals
            if rule_info['passes']:
                reward += 5.0  # Extra bonus

    return reward
```

## Citation

If you use this code, please cite:

```bibtex
@software{verl_password_game_ppo,
  title = {Verl-Based PPO for Password Game},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/yourrepo}
}
```

## References

- **PPO Paper**: [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- **GAE Paper**: [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)
- **Qwen Model**: [Qwen Technical Report](https://arxiv.org/abs/2309.16609)
- **Password Game**: Original game by Neal.fun

## License

MIT License - See LICENSE file for details

## Contact

For questions or issues, please open a GitHub issue or contact the maintainers.
