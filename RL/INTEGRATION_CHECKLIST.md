# Password Game Integration - Implementation Checklist

Use this checklist to ensure correct integration into your RL training notebook.

## Pre-Integration Setup

- [ ] OpenAI package installed: `pip install openai`
- [ ] Password game files exist: `/home/user/notebooks/tasks/password-game/game.py`
- [ ] RL training notebook is ready: `verl_qwen_rule_task.ipynb` or similar
- [ ] Have read: `README_PASSWORD_GAME.md` for overview

## Step 1: Add Notebook Cells (Cells 19-25)

### Cell 19a: Install Dependencies
- [ ] Add cell with: `!pip install -q openai`
- [ ] Run cell and verify: "✓ OpenAI installed"

### Cell 19b: Section Header (Markdown)
- [ ] Add markdown cell with section title
- [ ] Copy from: `password_game_cells_ready.txt` (Cell 19b)

### Cell 20: Import Password Game
- [ ] Add import cell
- [ ] Copy from: `password_game_cells_ready.txt` (Cell 20)
- [ ] Run cell and verify: "✓ Password game imported (26 rules)"

### Cell 21: PasswordGameEnvironment Class
- [ ] Add class definition
- [ ] Copy from: `password_game_cells_ready.txt` (Cell 21)
- [ ] Run cell and verify: "✓ PasswordGameEnvironment class defined"

### Cell 22: PasswordGameDataset Class
- [ ] Add class definition
- [ ] Copy from: `password_game_cells_ready.txt` (Cell 22)
- [ ] Run cell and verify: "✓ PasswordGameDataset class defined"

### Cell 23: Configuration
- [ ] Add PasswordGameConfig dataclass
- [ ] Copy from: `password_game_cells_ready.txt` (Cell 23)
- [ ] Run cell and verify configuration printed

### Cell 24: Initialize Environment and Datasets
- [ ] Add initialization code
- [ ] Copy from: `password_game_cells_ready.txt` (Cell 24)
- [ ] Run cell and verify:
  - [ ] "✓ Password game environment initialized"
  - [ ] "✓ Datasets created"
  - [ ] Example sample displayed

### Cell 25: Reward Function
- [ ] Add calculate_reward() and extract_password_from_response()
- [ ] Copy from: `password_game_cells_ready.txt` (Cell 25)
- [ ] Run cell and verify:
  - [ ] "✓ Password game reward function defined"
  - [ ] Test rewards displayed for sample passwords

## Step 2: Modify Training Loop (Cell ~34)

### 2.1: Update Batch Extraction
Location: Inside training loop, after `batch = next(epoch_iter)`

- [ ] Find line: `prompts = batch['prompt']`
- [ ] Find line: `rules = batch['rule']`
- [ ] Replace with:
  ```python
  prompts = batch['prompt']
  games = batch['game']
  target_rules = batch['target_rule']
  ```

### 2.2: Add Sample Expansion
Location: After batch extraction, before ROLLOUT section

- [ ] Add after batch extraction:
  ```python
  # Expand for samples_per_prompt
  expanded_games = games * config.samples_per_prompt
  expanded_target_rules = target_rules * config.samples_per_prompt
  ```

### 2.3: Update Reward Calculation
Location: In REWARDS section, after generating responses

- [ ] Find line with: `calculate_reward(p, r, rule)`
- [ ] Replace with:
  ```python
  rewards = torch.tensor([
      calculate_reward(p, r, game, target_rule)
      for p, r, game, target_rule in zip(
          expanded_prompts, all_responses,
          expanded_games, expanded_target_rules
      )
  ], device=DEVICE, dtype=dtype)
  ```

### 2.4: Add Dataset Refresh
Location: At the start of epoch loop (after `for epoch in range(...)`)

- [ ] Add at start of epoch loop:
  ```python
  # Refresh dataset samples for new epoch
  if epoch > 0:
      train_dataset.refresh_samples()
      print(f"✓ Refreshed training samples for epoch {epoch+1}")
  ```

### 2.5: Add Rule Progression Logging
Location: In LOGGING section (after existing wandb.log)

- [ ] Add after reward logging:
  ```python
  if global_step % config.log_interval == 0:
      stats = password_env.get_stats()
      wandb.log({
          "rule_progression/current_max_rules": stats['current_max_rules'],
          "rule_progression/success_rate": stats['success_rate'],
      }, step=global_step)
  ```

## Step 3: Modify Evaluation Function (Cell ~26)

### 3.1: Update Batch Processing
Location: Inside evaluate_model function, in the batch loop

- [ ] Find line: `rules = [item['rule'] for item in batch]`
- [ ] Replace with:
  ```python
  games = [item['game'] for item in batch]
  target_rules = [item['target_rule'] for item in batch]
  ```

### 3.2: Update Reward Calculation
Location: Inside evaluation loop, where rewards are calculated

- [ ] Find line: `for gen_ids, prompt, rule in zip(...)`
- [ ] Replace with:
  ```python
  for gen_ids, prompt, game, target_rule in zip(generated_ids, prompts, games, target_rules):
      if config.enable_thinking and config.parse_thinking:
          _, response = parse_thinking_response(gen_ids, tokenizer)
      else:
          response = tokenizer.decode(gen_ids, skip_special_tokens=True)
      reward = calculate_reward(prompt, response, game, target_rule)
      all_rewards.append(reward)
      total_reward += reward
  ```

## Step 4: Verify Integration

### 4.1: Run Test Cells
- [ ] Run all new cells (19-25) in order
- [ ] Verify no import errors
- [ ] Check reward function test output

### 4.2: Test Dataset
- [ ] Create a test sample: `sample = train_dataset[0]`
- [ ] Verify keys: `'prompt'`, `'game'`, `'target_rule'`, `'sample_id'`
- [ ] Check prompt length: Should be 500-2000 chars
- [ ] Verify game instance: `isinstance(sample['game'], PasswordGame)`

### 4.3: Test Reward Function
- [ ] Test with simple password: `calculate_reward("", "12345A!", train_dataset[0]['game'], 3)`
- [ ] Should return positive reward (at least 2-3)
- [ ] Test with invalid password: `calculate_reward("", "abc", train_dataset[0]['game'], 3)`
- [ ] Should return negative reward

### 4.4: Check Configuration
- [ ] Verify `password_game_config` exists
- [ ] Check `password_env` is initialized
- [ ] Confirm `train_dataset` and `val_dataset` created
- [ ] Test: `password_env.get_stats()` returns dict with keys

## Step 5: Run Training

### 5.1: Baseline Evaluation
- [ ] Run baseline evaluation cell
- [ ] Record baseline reward (should be around -2 to 0)
- [ ] Save baseline metrics to file

### 5.2: Start Training
- [ ] Run training loop
- [ ] Monitor first few steps for errors
- [ ] Check WandB logging works
- [ ] Verify rewards are being calculated

### 5.3: Monitor Progress
- [ ] Watch loss decrease
- [ ] Watch mean reward increase
- [ ] Check rule progression advances (5 → 6 → 7...)
- [ ] Monitor success rate improves

### 5.4: Validation Checks
- [ ] Validation runs without errors
- [ ] Validation reward shows improvement
- [ ] Best model saved when validation improves

## Step 6: Post-Training Analysis

### 6.1: Final Evaluation
- [ ] Run final evaluation
- [ ] Compare to baseline
- [ ] Check improvement: `final_reward - baseline_reward > 0`

### 6.2: Statistics
- [ ] Review environment stats: `password_env.get_stats()`
- [ ] Check total games played
- [ ] Check success rate achieved
- [ ] Verify rule progression reached target

### 6.3: Sample Analysis
- [ ] Generate sample passwords with trained model
- [ ] Check rule feedback: `game.get_rule_feedback(password)`
- [ ] Identify which rules are learned best
- [ ] Identify which rules need more training

## Troubleshooting Checklist

If you encounter errors, check:

### Import Errors
- [ ] OpenAI installed: `pip list | grep openai`
- [ ] Path to password game is correct
- [ ] All imports successful (no red errors)

### Training Errors
- [ ] Batch keys match: `'game'`, `'target_rule'` not `'rule'`
- [ ] Reward function has 4 arguments: `prompt, response, game, target_rule`
- [ ] Dataset is using PasswordGameDataset, not RuleDataset

### Reward Errors
- [ ] Rewards are not all zero
- [ ] Rewards include both positive and negative values
- [ ] Reward calculation doesn't crash
- [ ] Password extraction works (check extract_password_from_response)

### Memory Errors
- [ ] Reduce batch_size if OOM
- [ ] Reduce samples_per_prompt
- [ ] Enable gradient checkpointing
- [ ] Clear cache: `torch.cuda.empty_cache()`

## Configuration Tuning Checklist

After initial training, tune these:

### Rule Progression
- [ ] Too easy? Increase `min_rules` or `max_rules`
- [ ] Too hard? Decrease `max_rules` or slow `progression_rate`
- [ ] Good progression curve in logs

### Learning
- [ ] Rewards improving but slowly? Increase learning rate
- [ ] Unstable training? Decrease learning rate
- [ ] Good loss curve in WandB

### Dataset
- [ ] Not enough variety? Increase `num_train_samples`
- [ ] Too slow per epoch? Decrease `num_train_samples`
- [ ] Refreshing between epochs

## Success Metrics Checklist

Training is successful if:

- [ ] Baseline reward: -2 to 0
- [ ] Final reward: 2 to 5+ (improvement of 4-7 points)
- [ ] Success rate: Started at 0-20%, ended at 40-60%+
- [ ] Rule progression: Advanced from 5 to 10-15 rules
- [ ] Loss decreased steadily
- [ ] Model generates valid passwords (not random gibberish)
- [ ] At least 3-5 rules consistently passing

## Documentation References

For each step, refer to:

- **Cell code**: `password_game_cells_ready.txt`
- **Detailed instructions**: `PASSWORD_GAME_INTEGRATION.md`
- **Overview**: `PASSWORD_GAME_SUMMARY.md`
- **Installation help**: `INSTALLATION_NOTES.md`
- **Quick reference**: `README_PASSWORD_GAME.md`

## Final Verification

Before considering integration complete:

- [ ] All cells run without errors
- [ ] Training loop executes successfully
- [ ] Rewards are being calculated correctly
- [ ] Rule progression is working
- [ ] Dataset refreshes between epochs
- [ ] Evaluation function works
- [ ] WandB logging captures all metrics
- [ ] Model checkpoints are being saved
- [ ] Can see improvement over baseline

## Next Steps After Integration

- [ ] Run full training (2-3 epochs)
- [ ] Analyze which rules are hardest
- [ ] Experiment with different configurations
- [ ] Try increasing max_rules to 20-26
- [ ] Compare different model sizes
- [ ] Try different reward formulations

---

## Quick Reference Commands

```bash
# View cell code
cat password_game_cells_ready.txt | grep -A 50 "CELL 20"

# Test integration
python test_password_game_integration.py

# Check imports
python -c "from game import PasswordGame; print('OK')"

# View documentation
cat README_PASSWORD_GAME.md
```

---

**Complete this checklist as you integrate. Mark items with [x] when done.**

Good luck with your RL training! 🚀
