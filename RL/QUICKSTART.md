# Password Game Baseline Evaluation - Quick Start

## What You Have

A complete baseline evaluation system for testing Qwen3-0.6B on the Password Game task.

## Files Created

```
/home/user/notebooks/RL/
├── password_game_baseline_eval.ipynb    # Main evaluation notebook
├── compare_results.py                   # Compare baseline vs trained
├── test_baseline_eval.py                # Test setup before running
├── requirements_baseline_eval.txt       # Python dependencies
├── PASSWORD_GAME_EVALUATION.md          # Full documentation
└── QUICKSTART.md                        # This file
```

## Quick Start (3 Steps)

### Step 1: Install Dependencies

```bash
cd /home/user/notebooks/RL
pip install -r requirements_baseline_eval.txt
```

### Step 2: Test Setup

```bash
python test_baseline_eval.py
```

This verifies:
- All packages installed
- Password game accessible
- GPU available
- Model can be loaded

### Step 3: Run Evaluation

```bash
jupyter notebook password_game_baseline_eval.ipynb
```

Or run all cells at once:

```bash
jupyter nbconvert --to notebook --execute password_game_baseline_eval.ipynb --output baseline_results.ipynb
```

## What Gets Evaluated

The notebook runs **20 episodes** where the model plays the Password Game:

1. **Game Rules**: 26 progressive rules (must satisfy all previous rules at each step)
2. **Model Task**: Generate passwords that satisfy rules
3. **Metrics**: Success rate, rules satisfied, reward, password length, steps taken
4. **Output**: Comprehensive analysis with visualizations

## Expected Runtime

- **Setup (Step 1-2)**: ~5 minutes
- **Evaluation (Step 3)**: ~15-30 minutes (depends on GPU)
  - Model loading: ~2 minutes
  - Per episode: ~30-60 seconds
  - 20 episodes: ~10-20 minutes
  - Visualization: ~1 minute

## Results

After running, you'll get:

```
baseline_eval_<timestamp>/
├── config.json                   # Configuration
├── metrics.json                  # Aggregated metrics
├── baseline_export.json          # For comparison
├── summary_plots.png             # Visual overview
├── rule_performance.png          # Per-rule success rates
├── correlations.png              # Correlation analysis
├── results_table.csv             # Episode details
├── summary_statistics.csv        # Statistics table
└── episodes/                     # Individual episodes
    ├── episode_000.json
    ├── episode_001.json
    └── ... (20 total)
```

## Key Metrics to Watch

| Metric | What It Means | Typical Baseline |
|--------|---------------|------------------|
| **Success Rate** | % of games completed | 0-5% |
| **Avg Rules Satisfied** | Rules passed (out of 26) | 2-8 rules |
| **Avg Final Reward** | Score (+1/rule, -0.1/char) | -5 to +2 |
| **Avg Password Length** | Characters in password | 20-100 chars |
| **Avg Steps Taken** | Attempts per episode | 15-30 steps |

## After Baseline: Training

Once you have baseline results:

1. **Train** your model with RL (PPO, GRPO, etc.)
2. **Re-run** evaluation with trained model
3. **Compare** using `compare_results.py`:

```bash
python compare_results.py \
    --baseline baseline_eval_123/baseline_export.json \
    --trained trained_eval_456/trained_export.json \
    --output-dir comparison_results
```

This generates:
- Improvement metrics
- Side-by-side visualizations
- Rule-level comparison
- Statistical analysis

## Customization

Edit the notebook configuration cell:

```python
@dataclass
class EvalConfig:
    num_episodes: int = 20          # Change to 10 for quick test
    max_steps_per_episode: int = 30 # Max attempts per game
    temperature: float = 0.7        # Sampling temperature
    # ... more options
```

## Troubleshooting

### Issue: Out of Memory

```python
# In notebook config
precision: str = "float16"  # Instead of bfloat16
max_new_tokens: int = 128   # Instead of 256
```

### Issue: Model outputs explanations, not passwords

Already handled! The notebook includes response cleaning:
- Strips newlines
- Removes "Password:" prefixes
- Extracts first line only

### Issue: Slow execution

**Normal**: 20 episodes takes ~15-30 minutes
**Speed up**:
- Reduce `num_episodes` to 10
- Reduce `max_steps_per_episode` to 20
- Use GPU (CPU is 10x slower)

### Issue: Import errors

```bash
# Reinstall packages
pip install --upgrade -r requirements_baseline_eval.txt

# Or install individually
pip install torch transformers accelerate
pip install pandas matplotlib seaborn tqdm
```

## Understanding Results

### Good Baseline (for training)
- Low success rate (more room to improve)
- Some rules satisfied (model understands task)
- Consistent behavior (reproducible)

### Example Output
```
================================
BASELINE EVALUATION RESULTS
================================
Episodes: 20

Success Rate: 5.0%

Rules Satisfied: 4.2 ± 1.8
  - Min: 2
  - Max: 9
  - Total Rules: 26

Final Reward: -1.5 ± 3.2

Password Length: 35.4 ± 12.3

Steps Taken: 22.1 ± 5.4
================================
```

**Interpretation**:
- ✓ Model satisfies 4-5 rules on average (baseline established)
- ✓ Only 5% complete games (room for improvement)
- ✓ Training target: 50%+ success rate, 15+ rules

## Next Steps

1. ✅ Run baseline evaluation
2. 📊 Analyze results (which rules fail?)
3. 🧠 Design RL training strategy
4. 🚀 Train model (PPO/GRPO)
5. 📈 Re-evaluate and compare
6. 🔁 Iterate based on improvements

## Full Documentation

See `PASSWORD_GAME_EVALUATION.md` for:
- Detailed configuration options
- Prompt engineering strategies
- Advanced analysis techniques
- Training integration guide
- Troubleshooting tips

## Support

**Files to check**:
1. `QUICKSTART.md` (this file) - Quick start
2. `PASSWORD_GAME_EVALUATION.md` - Full guide
3. `test_baseline_eval.py` - Setup verification
4. Notebook comments - In-line documentation

**Testing before full run**:
```bash
# Verify setup
python test_baseline_eval.py

# Quick evaluation (just 3 episodes)
# Edit notebook: num_episodes = 3
jupyter notebook password_game_baseline_eval.ipynb
```

## Example Commands

```bash
# Full workflow
cd /home/user/notebooks/RL

# 1. Install
pip install -r requirements_baseline_eval.txt

# 2. Test
python test_baseline_eval.py

# 3. Run baseline
jupyter nbconvert --to notebook --execute password_game_baseline_eval.ipynb

# 4. Check results
ls -lh baseline_eval_*/
cat baseline_eval_*/summary_statistics.csv

# 5. (Later) Compare with trained model
python compare_results.py \
    --baseline baseline_eval_*/baseline_export.json \
    --trained trained_eval_*/trained_export.json
```

## Tips

- 🎯 **Start small**: Test with 3 episodes first
- 📝 **Document**: Keep notes on observations
- 🔍 **Analyze failures**: Which rules are hardest?
- 🎛️ **Tune prompts**: Try different system prompts
- 📊 **Track trends**: Compare across runs
- 💾 **Save everything**: You'll need it for comparison

## Ready to Go!

You now have everything needed for baseline evaluation:

✅ Complete evaluation notebook
✅ Proper Qwen3 prompting
✅ Comprehensive metrics
✅ Visualization tools
✅ Comparison utilities
✅ Full documentation

**Run the test, then the evaluation, and you'll have your baseline!**

---

*For questions or issues, refer to PASSWORD_GAME_EVALUATION.md*
