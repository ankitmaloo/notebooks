# Password Game - Qwen3-0.6B RL Training with VERL

Complete implementation of reinforcement learning training for the Password Game task using Qwen3-0.6B and VERL framework.

## Overview

This notebook trains Qwen3-0.6B to play the **Password Game** - a challenging cumulative constraint satisfaction task with 26 progressive rules. The model learns through PPO (Proximal Policy Optimization) to generate passwords that satisfy increasingly complex rule combinations.

## What's Included

### Main Notebook
- **`password_game_verl_training.ipynb`** - Complete end-to-end training pipeline

### Supporting Infrastructure
All infrastructure from subagent work:
- Password Game environment wrapper (created by subagents)
- Baseline evaluation tools (created by subagents)
- VERL PPO training implementation (created by subagents)
- Comprehensive documentation (created by subagents)

## Quick Start

### Prerequisites
- Single H100 GPU (or equivalent)
- Python 3.10+
- CUDA 12.1+

### 1. Start the Notebook

```bash
cd /home/user/notebooks
jupyter notebook password_game_verl_training.ipynb
```

### 2. Run All Cells

The notebook will:
1. ✅ Install VERL and dependencies (~10 minutes)
2. ✅ Start Password Game API server
3. ✅ Load Qwen3-0.6B model
4. ✅ Run baseline evaluation (5 episodes)
5. ✅ Train with PPO (3 epochs, ~2-4 hours)
6. ✅ Evaluate trained model (10 episodes)
7. ✅ Compare results

### 3. Monitor Training

Training metrics are logged to Weights & Biases:
- Average reward per epoch
- Rules satisfied
- Success rate
- Individual episode performance

## Task Details

### The Password Game

**26 Progressive Rules** (cumulative):
- Rule 0: At least 5 characters
- Rule 1: Include a number
- Rule 2: Include uppercase letter
- Rule 3: Include special character
- Rule 4: Digits sum to 25
- Rule 5: Include a month
- ... (continues through Rule 25)

**Reward Structure**:
```python
reward = (rules_satisfied × 1.0) - (password_length × 0.01)
```

**Success**: Satisfy all 26 rules

## Architecture

```
Password Game Environment (FastAPI)
          ↓
   Password Wrapper (PasswordGameEnv)
          ↓
   Qwen3-0.6B Policy Network
          ↓
   PPO Training (with Value Head)
          ↓
   Improved Password Generation
```

## Training Configuration

### Model
- **Base Model**: Qwen/Qwen3-0.6B
- **Precision**: bfloat16 (optimal for H100)
- **Flash Attention**: Enabled
- **Parameters**: ~600M

### PPO Hyperparameters
```python
learning_rate: 1e-6
num_epochs: 3
episodes_per_epoch: 20
max_steps_per_episode: 30
clip_range: 0.2
gamma: 0.99
gae_lambda: 0.95
```

### Generation
```python
temperature: 0.7
top_p: 0.9
top_k: 50
max_new_tokens: 256
```

## Expected Results

### Baseline (Untrained)
- **Average Reward**: 0-2
- **Rules Satisfied**: 1-3 / 26
- **Success Rate**: 0-10%

### After Training (3 epochs)
- **Average Reward**: 5-10
- **Rules Satisfied**: 5-10 / 26
- **Success Rate**: 30-60%

### Training Time
- **Single epoch**: ~40-60 minutes
- **Full training (3 epochs)**: ~2-4 hours on H100

## Key Features

### 1. Proper Multi-Turn Handling
The Password Game requires cumulative rule satisfaction:
- Turn 1: Satisfy rule 0
- Turn 2: Satisfy rules 0 + 1
- Turn 3: Satisfy rules 0 + 1 + 2
- ... continues

The implementation properly handles this with:
- State tracking across turns
- Credit assignment via GAE
- Shaped rewards for progress

### 2. Qwen3-Specific Optimizations
- Official chat template
- Proper tokenizer configuration
- bfloat16 precision for H100
- Flash Attention 2

### 3. Shaped Reward System
```python
reward_components = {
    'progress': +1.0 per rule satisfied,
    'length_penalty': -0.01 per character,
    'episode_bonus': +10 for completing 5+ rules
}
```

### 4. Comprehensive Logging
- Real-time WandB tracking
- Episode-level statistics
- Best model checkpointing
- Comparative analysis

## File Structure

```
/home/user/notebooks/
├── password_game_verl_training.ipynb     # Main notebook ⭐
├── PASSWORD_GAME_README.md               # This file
├── checkpoints/
│   └── password_game_best/               # Best model saved here
└── tasks/password-game/
    ├── game.py                           # Password game logic
    ├── main.py                           # FastAPI server
    └── utils.py                          # Game utilities
```

## Cell-by-Cell Guide

### Section 1: Setup (Cells 1-4)
- Check GPU availability
- Install VERL and dependencies
- Import libraries

### Section 2: Environment (Cells 5-7)
- Start Password Game API server
- Test connectivity
- Define environment wrapper

### Section 3: Model Loading (Cells 8-10)
- Configure model parameters
- Load Qwen3-0.6B tokenizer
- Load model with proper dtype

### Section 4: Baseline (Cells 11-14)
- Define prompt formatting
- Implement generation function
- Run baseline episodes
- Compute baseline statistics

### Section 5: PPO Setup (Cells 15-20)
- Configure PPO hyperparameters
- Initialize WandB
- Create value head
- Define GAE computation
- Implement trajectory collection

### Section 6: Training (Cell 21)
- Main training loop
- Trajectory collection
- Model updates
- Checkpointing

### Section 7: Evaluation (Cells 22-23)
- Load best model
- Run evaluation episodes
- Compare with baseline

### Section 8: Cleanup (Cell 24)
- Stop API server
- Close WandB
- Finish

## Customization

### Adjust Training Length
```python
ppo_config = PPOConfig(
    num_epochs=5,              # Increase for more training
    num_episodes_per_epoch=30  # More episodes per epoch
)
```

### Change Model
```python
config = ModelConfig(
    model_name="Qwen/Qwen3-1.5B",  # Use larger model
    precision="bfloat16"
)
```

### Tune Generation
```python
config.temperature = 1.0  # More creative
config.top_p = 0.95       # More diverse
```

### Adjust Rewards
```python
ppo_config.reward_per_rule = 2.0     # Increase rule reward
ppo_config.length_penalty = 0.02     # Stronger length penalty
```

## Troubleshooting

### API Server Not Starting
```bash
# Manually start server
cd /home/user/notebooks/tasks/password-game
uvicorn main:app --port 8000 --reload
```

### Out of Memory
```python
# Reduce batch size or use gradient checkpointing
model.gradient_checkpointing_enable()
```

### Slow Training
```python
# Reduce episodes per epoch
ppo_config.num_episodes_per_epoch = 10
```

### Poor Performance
```python
# Increase training
ppo_config.num_epochs = 10
ppo_config.learning_rate = 5e-7  # Lower LR for stability
```

## Advanced Usage

### Manual Episode Run
```python
env = PasswordGameEnv()
state = env.reset()
password = generate_password(model, tokenizer, state.all_rules, config)
feedback = env.get_feedback(password)
obs, reward, done, info = env.submit(password)
```

### Custom Reward Function
```python
def custom_reward(rules_satisfied, password_length, advanced):
    base = rules_satisfied * 1.0
    penalty = password_length * 0.02
    bonus = 5.0 if advanced else 0.0
    return base - penalty + bonus
```

### Save/Load Models
```python
# Save
model.save_pretrained("./my_checkpoint")
tokenizer.save_pretrained("./my_checkpoint")

# Load
model = AutoModelForCausalLM.from_pretrained("./my_checkpoint")
```

## Implementation Notes

### Why VERL?
- Production-ready RL framework
- Optimized for LLM post-training
- Supports PPO, GRPO, and more
- Good integration with HuggingFace

### Why Qwen3-0.6B?
- Small enough for fast iteration
- Large enough for complex reasoning
- Excellent instruction following
- Official chat template support

### Why Single GPU?
- Simpler setup (no DDP complexity)
- Faster debugging
- Sufficient for 0.6B model
- Easy to scale up later

### Why PPO?
- Stable training
- Well-understood algorithm
- Good for sparse rewards
- Proven for RLHF

## References

### Documentation Created by Subagents
The subagents created extensive documentation in their work:
- **Environment Design**: Complete Password Game wrapper with error handling
- **Baseline Evaluation**: Statistical evaluation framework with visualizations
- **VERL Integration**: Installation guides and configuration patterns
- **PPO Training**: Full implementation with GAE and value heads

### External Resources
- [VERL GitHub](https://github.com/volcengine/verl)
- [Qwen3 Model Card](https://huggingface.co/Qwen/Qwen3-0.6B)
- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [Password Game](https://neal.fun/password-game/)

## Next Steps

After running this notebook:

1. **Analyze Results**
   - Review WandB dashboard
   - Identify which rules are hardest
   - Examine failure cases

2. **Iterate**
   - Adjust reward shaping
   - Tune hyperparameters
   - Increase training time

3. **Scale Up**
   - Try Qwen3-1.5B or 3B
   - Implement curriculum learning
   - Add more sophisticated prompting

4. **Deploy**
   - Export best model
   - Create inference API
   - Build demo application

## Support

For issues or questions:
- Check troubleshooting section above
- Review WandB logs for training issues
- Examine API server logs: `/home/user/notebooks/tasks/password-game/`

## License

This implementation uses:
- Qwen3 (Apache 2.0 / Tongyi Qianwen License)
- VERL (Apache 2.0)
- PyTorch (BSD)

## Acknowledgments

- **Qwen Team** for the excellent 0.6B model
- **VERL Team** for the RL framework
- **Password Game** for the challenging task
- **Subagents** for comprehensive infrastructure development

---

**Created**: 2025-11-10
**Model**: Qwen3-0.6B
**Framework**: VERL + PyTorch
**Task**: Password Game (26 rules)
**Hardware**: Single H100 GPU
