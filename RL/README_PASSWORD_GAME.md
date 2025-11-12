# Password Game RL Training Integration

Complete integration of the Password Game environment for reinforcement learning training.

## Quick Start

### 1. Install Dependencies
```bash
pip install openai
```

### 2. Add Notebook Cells
Copy cells from `password_game_cells_ready.txt` into your RL notebook (starting at cell 19):
- Cell 19a: Install OpenAI
- Cell 19b: Section header (markdown)
- Cell 20: Import password game
- Cell 21: PasswordGameEnvironment class
- Cell 22: PasswordGameDataset class
- Cell 23: Configuration
- Cell 24: Initialize environment and datasets
- Cell 25: Reward function

### 3. Modify Training Loop
Update your training loop (cell ~34) to handle password game batches:
- Extract `games` and `target_rules` from batch
- Expand for `samples_per_prompt`
- Pass to reward calculation
- Add dataset refresh between epochs

See `PASSWORD_GAME_INTEGRATION.md` for detailed modifications.

### 4. Run Training
Execute the notebook and monitor:
- Reward improvements (baseline vs trained)
- Rule progression (5 → 15 rules)
- Success rate (% positive rewards)

## Files Overview

| File | Purpose |
|------|---------|
| **README_PASSWORD_GAME.md** | This file - quick start guide |
| **PASSWORD_GAME_INTEGRATION.md** | Complete documentation with all code |
| **PASSWORD_GAME_SUMMARY.md** | Overview, architecture, and reference |
| **password_game_cells_ready.txt** | Copy-paste ready cells for notebook |
| **INSTALLATION_NOTES.md** | Dependency setup and troubleshooting |
| **test_password_game_integration.py** | Test script to verify integration |

## What You Get

### PasswordGameEnvironment
Manages game instances with:
- Rule progression (start with 5 rules, scale to 15+)
- Prompt formatting with instructions and rules
- Reward calculation (+1 per rule, -0.1 per char)
- Statistics tracking (success rate, games completed)

### PasswordGameDataset
PyTorch Dataset providing:
- Training samples with game instances
- Target rule indices for progressive difficulty
- Sample refresh between epochs
- Batch-compatible format

### Reward System
- **+1 point** per passing rule
- **-0.1 point** per character (encourages brevity)
- Example: 4 rules passed, 6 chars = 4 - 0.6 = **3.4 reward**

## The Password Game

26 progressive rules including:
1. Minimum length (5+ characters)
2. Include number
3. Include uppercase
4. Include special character
5. Digits sum to 25
6. Include month name
7. Include roman numeral
8. Include sponsor (Pepsi/Starbucks/Shell)
9. Roman numerals multiply to 35
10. Include CAPTCHA (dynamic)
11. Include Wordle answer (dynamic)
12. Include periodic element
13. Include moon phase emoji (dynamic)
14. Include country name (dynamic)
15. Include leap year
... and 11 more!

## Configuration

```python
@dataclass
class PasswordGameConfig:
    use_rule_progression: bool = True    # Progressive difficulty
    min_rules: int = 5                   # Start with 5 rules
    max_rules: int = 15                  # Scale to 15 rules
    progression_rate: float = 0.1        # Progression speed
```

## Expected Performance

### Baseline (Untrained)
- Reward: -2 to 0
- Passing rules: 0-2 out of 5
- Success rate: 0-20%

### After Training (2-3 epochs)
- Reward: 2 to 5+
- Passing rules: 3-7 out of 5-10
- Success rate: 40-60%

### Well-Trained
- Reward: 5-10+
- Passing rules: 8-12 out of 10-15
- Success rate: 60-80%

## Integration Example

```python
# Create environment
password_env = PasswordGameEnvironment(
    use_rule_progression=True,
    min_rules=5,
    max_rules=15,
    progression_rate=0.1
)

# Create dataset
train_dataset = PasswordGameDataset(
    num_samples=1000,
    environment=password_env,
    seed=42
)

# Training loop
for batch in dataloader:
    prompts = batch['prompt']
    games = batch['game']
    target_rules = batch['target_rule']

    # Generate responses
    responses = model.generate(prompts)

    # Calculate rewards
    rewards = [
        calculate_reward(p, r, game, target_rule)
        for p, r, game, target_rule in zip(prompts, responses, games, target_rules)
    ]

    # PPO update
    loss = ppo_loss(rewards, ...)
    loss.backward()
```

## Troubleshooting

### ImportError: No module named 'openai'
```bash
pip install openai
```

### ImportError: No module named 'keys'
Create `tasks/password-game/keys.py`:
```python
OPENAI_API_KEY = ""  # Optional
```

### Rewards stay negative
- Decrease `max_rules` (easier task)
- Increase training epochs
- Check model is learning (loss decreasing)

### Rule progression too fast/slow
Adjust `progression_rate`:
- Lower (0.05): Slower, more stable
- Higher (0.2): Faster, more aggressive

## Next Steps

1. **Read** `INSTALLATION_NOTES.md` for setup details
2. **Review** `PASSWORD_GAME_INTEGRATION.md` for complete code
3. **Copy** cells from `password_game_cells_ready.txt`
4. **Modify** training loop as described
5. **Run** and monitor training
6. **Analyze** results and tune configuration

## Architecture

```
PasswordGameEnvironment
  ├─> create_game_instance() → PasswordGame
  ├─> format_prompt() → Prompt with rules
  ├─> calculate_reward() → Score
  └─> update_progression() → Difficulty scaling

PasswordGameDataset
  ├─> __getitem__() → Sample with game
  ├─> refresh_samples() → New games
  └─> Compatible with DataLoader

Training Loop
  ├─> Batch: prompts, games, target_rules
  ├─> Generate: model responses
  ├─> Reward: calculate_reward()
  ├─> Update: PPO loss & optimize
  └─> Progress: rule difficulty scaling
```

## Key Methods

### Environment
```python
env.create_game_instance()           # New PasswordGame
env.format_prompt(game, rule_idx)    # Prompt string
env.calculate_reward(game, pwd, idx) # Score
env.get_stats()                      # Statistics
```

### Dataset
```python
dataset[i]                           # Get sample
dataset.refresh_samples()            # New games
len(dataset)                         # Sample count
```

### Game
```python
game.get_current_rule()              # Rule text
game.calculate_reward(password)      # Score
game.get_rule_feedback(password)     # Detailed feedback
```

## Files Location

```
/home/user/notebooks/RL/
├── README_PASSWORD_GAME.md              (this file)
├── PASSWORD_GAME_INTEGRATION.md         (full docs)
├── PASSWORD_GAME_SUMMARY.md             (overview)
├── password_game_cells_ready.txt        (cells)
├── INSTALLATION_NOTES.md                (setup)
└── test_password_game_integration.py    (test)

/home/user/notebooks/tasks/password-game/
├── game.py                              (game logic)
├── utils.py                             (helpers)
└── main.py                              (API)
```

## Support

1. **Installation issues**: See `INSTALLATION_NOTES.md`
2. **Integration questions**: See `PASSWORD_GAME_INTEGRATION.md`
3. **Architecture details**: See `PASSWORD_GAME_SUMMARY.md`
4. **Testing**: Run `test_password_game_integration.py`

## Quick Commands

```bash
# Install dependencies
pip install openai

# Test integration
cd /home/user/notebooks/RL
python test_password_game_integration.py

# View cells
cat password_game_cells_ready.txt

# Read full docs
cat PASSWORD_GAME_INTEGRATION.md
```

## Training Tips

1. **Start small**: Use `min_rules=5, max_rules=10` initially
2. **Monitor closely**: Watch reward trends and rule progression
3. **Adjust learning rate**: May need lower LR for complex constraints
4. **Use thinking mode**: If available in your model (helps with reasoning)
5. **Refresh often**: Call `dataset.refresh_samples()` each epoch
6. **Log everything**: Track rule-specific success rates

## Success Criteria

Training is working if:
- Rewards increase over time (baseline → positive)
- Rule progression advances smoothly (5 → 10 → 15)
- Success rate improves (20% → 40% → 60%)
- Model generates valid passwords (not gibberish)
- Loss decreases steadily

## Common Patterns

### Good passwords (5 rules):
```
"12345A!"       → Rules 0-3 (length, number, upper, special)
"98ABC!may"     → Rules 0-5 (+ digit sum, month)
```

### Good passwords (10 rules):
```
"9876BA!mayIVpepsict5rk"  → Multiple constraints satisfied
```

The model learns to combine constraints efficiently!

---

**Ready to train!** Start with `INSTALLATION_NOTES.md` for setup, then follow the quick start above.

For questions or issues, check the detailed documentation files.
