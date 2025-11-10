# Password Game Baseline Evaluation System

Complete baseline evaluation for Qwen3-0.6B on the Password Game task.

## Start Here

Choose your path:

### 🚀 I want to start immediately
→ Read **[QUICKSTART.md](QUICKSTART.md)** (3 steps to running)

### 📚 I want complete documentation
→ Read **[PASSWORD_GAME_EVALUATION.md](PASSWORD_GAME_EVALUATION.md)** (full guide)

### 📊 I want to understand what was built
→ Read **[BASELINE_EVAL_SUMMARY.md](BASELINE_EVAL_SUMMARY.md)** (system overview)

## Files Overview

```
Baseline Evaluation System/
│
├── 📓 password_game_baseline_eval.ipynb  ← Main evaluation notebook
│
├── 🐍 compare_results.py                 ← Compare baseline vs trained
├── 🐍 test_baseline_eval.py              ← Verify setup
│
├── 📦 requirements_baseline_eval.txt     ← Dependencies
│
├── 📖 README_BASELINE.md                 ← This file (index)
├── 📖 QUICKSTART.md                      ← Quick start (3 steps)
├── 📖 PASSWORD_GAME_EVALUATION.md        ← Full documentation
└── 📖 BASELINE_EVAL_SUMMARY.md           ← System overview
```

## Quick Start (30 seconds)

```bash
cd /home/user/notebooks/RL
pip install -r requirements_baseline_eval.txt
python test_baseline_eval.py
jupyter notebook password_game_baseline_eval.ipynb
```

## What This Does

Evaluates **untrained Qwen3-0.6B** on Password Game to establish baseline performance:

- **Runs**: 20 independent game episodes
- **Collects**: Success rate, rules satisfied, reward, password length, steps
- **Analyzes**: Per-rule performance, correlations, distributions
- **Visualizes**: 9 comprehensive plots
- **Exports**: Data for post-training comparison

## Expected Output

```
baseline_eval_<timestamp>/
├── 6 visualization PNG files
├── 4 data files (JSON + CSV)
└── 20 individual episode JSON files
```

## Results (Typical Baseline)

| Metric | Baseline | Post-Training Goal |
|--------|----------|-------------------|
| Success Rate | 0-5% | 50%+ |
| Rules Satisfied | 2-8 / 26 | 20+ / 26 |
| Final Reward | -5 to +2 | 10+ |

## Next Steps After Baseline

1. **Train** model with RL (PPO/GRPO)
2. **Re-evaluate** with trained model
3. **Compare** using `compare_results.py`
4. **Iterate** based on improvements

## Documentation Map

| Want to... | Read... |
|------------|---------|
| Start quickly | [QUICKSTART.md](QUICKSTART.md) |
| Understand system | [BASELINE_EVAL_SUMMARY.md](BASELINE_EVAL_SUMMARY.md) |
| Learn details | [PASSWORD_GAME_EVALUATION.md](PASSWORD_GAME_EVALUATION.md) |
| Verify setup | Run `test_baseline_eval.py` |
| See examples | All docs have examples |

## Key Features

- ✅ Proper Qwen chat template
- ✅ Comprehensive metrics (5 + 26 rules)
- ✅ Statistical analysis
- ✅ Beautiful visualizations
- ✅ Reproducible (fixed seed)
- ✅ Easy comparison tool
- ✅ Full documentation
- ✅ Test suite

## Help

**Setup issues?**
```bash
python test_baseline_eval.py
```

**Usage questions?**
Read [QUICKSTART.md](QUICKSTART.md)

**Deep dive?**
Read [PASSWORD_GAME_EVALUATION.md](PASSWORD_GAME_EVALUATION.md)

**System overview?**
Read [BASELINE_EVAL_SUMMARY.md](BASELINE_EVAL_SUMMARY.md)

## Author

Created for baseline evaluation of RL-trained models on the Password Game task.

## License

[Your License]

---

**Ready to establish your baseline? Start with [QUICKSTART.md](QUICKSTART.md)!**
