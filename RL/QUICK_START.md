# Password Game PPO - Quick Start

## 🚀 30-Second Start

```bash
cd /home/user/notebooks/RL
./launch_password_ppo.sh default
```

That's it! Training will begin.

---

## 📁 What You Got

### Core Files
1. **`verl_password_game_ppo.py`** - Main training script (1,050 lines)
2. **`verl_password_game_training.ipynb`** - Interactive notebook
3. **`launch_password_ppo.sh`** - Easy launcher

### Documentation
4. **`PASSWORD_GAME_PPO_README.md`** - Complete reference
5. **`PASSWORD_GAME_TRAINING_GUIDE.md`** - Training guide
6. **`PASSWORD_GAME_PPO_INDEX.md`** - System overview
7. **`QUICK_START.md`** - This file

---

## ⚡ Quick Commands

### Training
```bash
# Standard training (5 epochs, ~3 hours)
./launch_password_ppo.sh default

# Quick test (3 epochs, ~1.5 hours)
./launch_password_ppo.sh quick-test

# Long run (10 epochs, ~6 hours)
./launch_password_ppo.sh long-run
```

### Python
```python
# Direct execution
python verl_password_game_ppo.py --epochs 5 --lr 5e-7

# Custom config
from verl_password_game_ppo import PasswordGamePPOConfig, train_ppo_password_game
config = PasswordGamePPOConfig(learning_rate=1e-6, num_epochs=3)
train_ppo_password_game(config)
```

### Jupyter
```bash
jupyter notebook verl_password_game_training.ipynb
```

---

## 🎯 What It Does

### The Task
Password game with **26 cumulative rules**:
- Turn 1: "At least 5 characters"
- Turn 2: Previous + "Include a number"
- Turn 3: Previous + "Include uppercase"
- ...
- Turn 26: All previous rules!

### The Model
- **Qwen3-0.6B** with thinking mode
- **PPO** for policy optimization
- **GAE** for multi-turn credit
- **Shaped rewards** for progress

### The Goal
- Satisfy as many rules as possible
- Keep password concise
- Learn to reason about constraints

---

## 📊 Expected Results

| Epoch | Reward | Rules | Success Rate |
|-------|--------|-------|--------------|
| 0 (baseline) | 1-2 | 1-2 | 0-10% |
| 1-2 | 3-4 | 3-4 | 20-30% |
| 3-4 | 5-7 | 5-7 | 40-60% |
| 5+ | 7-10 | 7-10 | 60-80% |

**Success** = Satisfying ≥5 rules

---

## 🛠️ Key Hyperparameters

```python
learning_rate = 5e-7         # Policy learning rate
clip_range = 0.2             # PPO clip epsilon
gamma = 0.99                 # Discount factor
gae_lambda = 0.95            # GAE lambda
progress_bonus = 2.0         # Reward for advancing
length_penalty = 0.1         # Penalty per character
final_success_bonus = 10.0   # Bonus for 10+ rules
```

---

## 🔍 Monitor Training

### WandB Metrics
- `train/reward` - Should increase (1 → 2 → 4 → 8)
- `train/rules_satisfied` - Should increase (1 → 3 → 5 → 7)
- `train/kl_div` - Should be 0.01-0.1 (controlled)
- `eval/success_rate` - Should increase (0.1 → 0.5 → 0.7)

### Checkpoints
- `best_model/` - Best validation performance
- `checkpoint-N/` - Periodic saves
- `final_model/` - End of training

---

## 🐛 Troubleshooting

### Training too slow?
```python
config.batch_size = 2
config.gradient_accumulation_steps = 8
```

### Out of memory?
```python
config.max_prompt_length = 768
config.max_new_tokens = 128
```

### Not learning?
```python
config.learning_rate = 1e-6
config.progress_bonus = 5.0
```

---

## 📚 Read More

1. **`PASSWORD_GAME_PPO_INDEX.md`** - System overview
2. **`PASSWORD_GAME_PPO_README.md`** - Full reference
3. **`PASSWORD_GAME_TRAINING_GUIDE.md`** - Detailed guide

---

## 🎓 Key Concepts

### PPO (Proximal Policy Optimization)
Prevents destructive policy updates via clipping:
```python
ratio = new_prob / old_prob
clipped = clip(ratio, 1-ε, 1+ε)
loss = -min(ratio × advantage, clipped × advantage)
```

### GAE (Generalized Advantage Estimation)
Multi-turn credit assignment:
```python
A_t = Σ (γλ)^k δ_{t+k}
# Turn 1 gets credit for all future rewards (discounted)
```

### Shaped Rewards
Dense feedback instead of sparse:
```python
reward = rules_passing - length_penalty + progress_bonus
# Immediate signal for each action
```

---

## ✨ Features

✓ Multi-turn episode handling
✓ Cumulative rule management
✓ Shaped reward system
✓ GAE for credit assignment
✓ Thinking mode support
✓ Adaptive KL coefficient
✓ WandB logging
✓ Checkpointing
✓ Comprehensive docs

---

## 🚦 Status Check

Before training, verify:

```bash
# GPU available?
nvidia-smi

# Dependencies installed?
python -c "import torch, transformers, wandb; print('OK')"

# Password game works?
python -c "from game import PasswordGame; g=PasswordGame(); print(g.get_current_rule())"

# API keys set?
python -c "from keys import WANDB_API_KEY; print('OK' if WANDB_API_KEY else 'Missing')"
```

All "OK"? You're ready to train!

---

## 🎯 Your First Run

```bash
# 1. Navigate
cd /home/user/notebooks/RL

# 2. Quick test (1-2 hours)
./launch_password_ppo.sh quick-test

# 3. Monitor
# Check WandB dashboard URL in output

# 4. Evaluate
python -c "
from verl_password_game_ppo import evaluate_policy
# ... load model and run evaluation
"
```

---

## 💡 Tips

1. **Start with quick-test** to verify setup
2. **Monitor WandB** for training progress
3. **Check rules_satisfied** not just reward
4. **Save best model** for production use
5. **Experiment** with hyperparameters

---

## 📞 Need Help?

1. Check **`PASSWORD_GAME_TRAINING_GUIDE.md`** for common issues
2. Review **`PASSWORD_GAME_PPO_README.md`** for detailed docs
3. Run **`verl_password_game_training.ipynb`** for interactive debugging
4. Read error messages carefully (usually point to solution)

---

## 🏆 Success Criteria

After training, your model should:
- ✓ Satisfy 7-10 rules (vs. 1-2 baseline)
- ✓ Have 60-80% success rate (≥5 rules)
- ✓ Generate reasonable passwords
- ✓ Show understanding of constraints

---

## 🔬 Experiment Ideas

1. **Longer training**: 10-20 epochs
2. **Higher LR**: 1e-6 instead of 5e-7
3. **More progress bonus**: 5.0 instead of 2.0
4. **No thinking mode**: Compare performance
5. **Larger model**: Qwen-1.5B instead of 0.6B

---

**Ready? Let's train!**

```bash
./launch_password_ppo.sh default
```

Good luck! 🚀
