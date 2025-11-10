# Password Game Baseline Evaluation Guide

This guide explains how to use the baseline evaluation system for the Password Game task with Qwen3-0.6B.

## Overview

The baseline evaluation system tests an **untrained** Qwen3-0.6B model on the Password Game task to establish performance metrics before RL training. This provides a reference point to measure training improvements.

## Files

- **`password_game_baseline_eval.ipynb`**: Main evaluation notebook
- **`compare_results.py`**: Script to compare baseline vs post-training results
- **`PASSWORD_GAME_EVALUATION.md`**: This guide

## Quick Start

### 1. Run Baseline Evaluation

Open and run the notebook:

```bash
jupyter notebook password_game_baseline_eval.ipynb
```

Or run all cells:

```bash
jupyter nbconvert --to notebook --execute password_game_baseline_eval.ipynb
```

### 2. Review Results

The notebook generates comprehensive results in `baseline_eval_<timestamp>/`:

```
baseline_eval_1234567890/
├── config.json                    # Evaluation configuration
├── metrics.json                   # Aggregated metrics
├── baseline_export.json           # Full data for comparison
├── summary_plots.png              # Visual summary
├── rule_performance.png           # Per-rule success rates
├── correlations.png               # Correlation analysis
├── results_table.csv              # Detailed episode results
├── summary_statistics.csv         # Summary statistics
└── episodes/                      # Individual episode data
    ├── episode_000.json
    ├── episode_001.json
    └── ...
```

### 3. Compare with Trained Model

After training your model, run another evaluation and compare:

```bash
python compare_results.py \
    --baseline baseline_eval_1234567890/baseline_export.json \
    --trained trained_eval_9876543210/trained_export.json \
    --output-dir comparison_results
```

## Metrics Collected

### Overall Metrics

| Metric | Description |
|--------|-------------|
| **Success Rate** | % of episodes where all rules satisfied |
| **Avg Rules Satisfied** | Mean number of rules satisfied (out of 26) |
| **Avg Final Reward** | Mean reward score (+1 per rule, -0.1 per char) |
| **Avg Password Length** | Mean length of final passwords |
| **Avg Steps Taken** | Mean number of password attempts per episode |

### Rule-Level Metrics

- **Per-Rule Success Rate**: Success rate for each individual rule (0-25)
- **Rule Progression**: How rules are satisfied over steps
- **Failed Rules Analysis**: Which rules are hardest for the model

### Episode-Level Data

Each episode records:
- All attempted passwords
- Rule progression at each step
- Detailed rule feedback (which rules pass/fail)
- Final password and game state
- Game-specific values (captcha, country, wordle answer, moon phase)

## Configuration

Customize evaluation in the notebook:

```python
@dataclass
class EvalConfig:
    # Model
    model_name: str = "Qwen/Qwen2.5-0.6B"

    # Evaluation
    num_episodes: int = 20              # More episodes = better statistics
    max_steps_per_episode: int = 30     # Max attempts per game

    # Generation
    temperature: float = 0.7            # Higher = more exploration
    top_p: float = 0.9
    top_k: int = 50
    max_new_tokens: int = 256

    # Prompting
    use_chat_template: bool = True      # Use Qwen chat template
    system_prompt: str = "..."          # System instruction

    # Output
    output_dir: str = "./baseline_eval_{timestamp}"
    save_episodes: bool = True          # Save individual episodes

    # Reproducibility
    seed: int = 42
```

## Prompt Engineering

The evaluation uses **Qwen's chat template** for optimal performance:

```python
messages = [
    {
        "role": "system",
        "content": "You are an AI assistant playing the Password Game..."
    },
    {
        "role": "user",
        "content": "Step 1: Create a password satisfying these rules:\n1. ..."
    }
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
```

### System Prompt Strategy

The system prompt instructs the model to:
1. Read rules carefully
2. Review all previous rules
3. Generate password satisfying ALL rules
4. Output ONLY the password (no explanations)
5. Keep password as short as possible

### User Prompt Format

Each step shows:
- Step number
- All current rules (cumulative)
- Clear request for password

## Understanding the Password Game

The Password Game is a rule-based challenge:

1. **Start**: Rule 0 (password must be ≥5 chars)
2. **Each Step**: Model generates password
3. **Validation**: Check if password satisfies all current rules
4. **Advance**: If all rules pass, move to next rule
5. **Continue**: Until all 26 rules satisfied or max steps reached

### Sample Rules

- Rule 0: At least 5 characters
- Rule 1: Include a number
- Rule 2: Include uppercase letter
- Rule 3: Include special character
- Rule 4: Digits sum to 25
- Rule 5: Include month name
- ...
- Rule 25: Include 3 consecutive alphabetical letters

**Total**: 26 rules (indices 0-25)

### Reward Function

```python
reward = (rules_satisfied * 1.0) - (password_length * 0.1)
```

- **+1** for each satisfied rule
- **-0.1** for each character in password
- **Goal**: Maximize rules while minimizing length

## Expected Baseline Performance

Typical untrained Qwen3-0.6B performance:

| Metric | Expected Range |
|--------|---------------|
| Success Rate | 0-5% |
| Avg Rules Satisfied | 2-8 rules (out of 26) |
| Avg Final Reward | -5 to +2 |
| Avg Password Length | 20-100 chars |
| Avg Steps Taken | 15-30 steps |

**Note**: Actual performance varies based on temperature and prompting.

## Using Baseline for Training

### 1. Identify Weak Rules

Look at `rule_performance.png` to see which rules fail most:

```python
# Load baseline
with open('baseline_export.json', 'r') as f:
    baseline = json.load(f)

# Find worst rules
rule_perf = baseline['rule_performance']
worst_rules = sorted(rule_perf.items(), key=lambda x: x[1])[:5]
print("Worst performing rules:", worst_rules)
```

### 2. Design Reward Shaping

Use baseline insights to shape rewards:

```python
# Example: Bonus for hard rules
hard_rules = {4, 8, 16, 22, 23}  # From baseline analysis

def shaped_reward(password, rules_satisfied):
    base_reward = rules_satisfied - len(password) * 0.1

    # Bonus for hard rules
    bonus = sum(0.5 for r in hard_rules if rule_satisfied(r))

    return base_reward + bonus
```

### 3. Track Training Progress

Compare checkpoints against baseline:

```python
# After each training checkpoint
python compare_results.py \
    --baseline baseline_export.json \
    --trained checkpoint_1000_export.json \
    --output-dir checkpoint_1000_comparison
```

### 4. Set Training Goals

Use baseline to set realistic targets:

```python
# Example targets
baseline_reward = 0.5      # From baseline
target_reward = 10.0       # 20x improvement
target_success = 0.50      # 50% success rate
target_rules = 20          # Average 20/26 rules
```

## Troubleshooting

### Issue: Model outputs explanations instead of passwords

**Solution**: Strengthen system prompt or add post-processing:

```python
# Clean response
password = generated_text.strip()
if '\n' in password:
    password = password.split('\n')[0]
for prefix in ['Password:', 'password:']:
    if password.startswith(prefix):
        password = password[len(prefix):].strip()
```

### Issue: Model gives up too early

**Solution**:
- Increase `max_steps_per_episode`
- Adjust temperature (higher = more exploration)
- Improve prompting to encourage persistence

### Issue: Very long passwords

**Solution**:
- Emphasize "shortest possible" in system prompt
- Increase length penalty in reward function
- Show examples of concise passwords

### Issue: Low GPU memory

**Solution**:
- Use `precision: "float16"` instead of `bfloat16`
- Reduce `max_new_tokens`
- Process episodes sequentially (already default)

## Advanced Usage

### Custom Evaluation Episodes

Run specific episodes with detailed logging:

```python
# In notebook
result = run_episode(episode_id=0, verbose=True)
print(result.final_password)
print(result.rule_feedback)
```

### Analyze Best Episode

Find and examine the best performing episode:

```python
best = max(results, key=lambda r: r.rules_satisfied)
print(f"Best episode: {best.episode_id}")
print(f"Password: {best.final_password}")
print(f"Rules: {best.rules_satisfied}/{best.total_rules}")

# Load detailed data
with open(f'episodes/episode_{best.episode_id:03d}.json', 'r') as f:
    details = json.load(f)
    print(details['passwords'])  # All attempts
```

### Export for Training

Use baseline episodes as training data:

```python
# Load all episodes
import glob
episodes = []
for path in glob.glob('episodes/*.json'):
    with open(path, 'r') as f:
        episodes.append(json.load(f))

# Extract successful passwords
successful = [
    ep for ep in episodes
    if ep['rules_satisfied'] > threshold
]

# Use for training initialization or demonstration
```

## Comparison Script Usage

### Basic Comparison

```bash
python compare_results.py \
    --baseline baseline_export.json \
    --trained trained_export.json
```

### Custom Output Directory

```bash
python compare_results.py \
    --baseline baseline_export.json \
    --trained trained_export.json \
    --output-dir my_comparison
```

### Programmatic Usage

```python
from compare_results import compare_evaluations

results = compare_evaluations(
    baseline_path='baseline_export.json',
    trained_path='trained_export.json',
    output_dir='comparison'
)

# Access improvements
print(f"Success rate improved by: {results['improvements']['success_rate']['absolute_improvement']:.2%}")
print(f"Reward improved by: {results['improvements']['avg_final_reward']['absolute_improvement']:.2f}")

# Access rule-level comparison
for rule_idx, data in results['rule_comparison'].items():
    if data['improvement'] > 0.2:  # 20% improvement
        print(f"Rule {rule_idx} greatly improved!")
```

## Integration with Training

### Before Training

```python
# 1. Run baseline evaluation
# 2. Save baseline_export.json
# 3. Analyze weak points
# 4. Design reward function
```

### During Training

```python
# After each checkpoint:
# 1. Run evaluation with trained model
# 2. Compare with baseline
# 3. Track improvement trends
# 4. Adjust hyperparameters if needed
```

### After Training

```python
# 1. Run final evaluation
# 2. Generate comprehensive comparison
# 3. Analyze which rules improved most
# 4. Document results
```

## Example Workflow

```bash
# 1. Run baseline evaluation
jupyter nbconvert --to notebook --execute password_game_baseline_eval.ipynb

# 2. Train model (your training code)
python train_ppo.py --config config.json

# 3. Evaluate trained model (modify notebook to load trained model)
jupyter nbconvert --to notebook --execute password_game_trained_eval.ipynb

# 4. Compare results
python compare_results.py \
    --baseline baseline_eval_123/baseline_export.json \
    --trained trained_eval_456/trained_export.json \
    --output-dir final_comparison

# 5. Review comparison
open final_comparison/comparison_overall.png
open final_comparison/comparison_rules.png
open final_comparison/comparison_improvements.png
```

## Tips for Best Results

1. **Run enough episodes**: 20+ for statistical validity
2. **Use consistent seed**: For reproducible baselines
3. **Save all data**: Enable `save_episodes=True`
4. **Document configuration**: Keep config.json with results
5. **Version control**: Track baseline alongside code
6. **Multiple baselines**: Test different temperatures/prompts
7. **Analyze failures**: Study failed rules to improve training

## Citation

If you use this evaluation system in research:

```bibtex
@software{password_game_baseline_eval,
  title={Password Game Baseline Evaluation for Qwen3-0.6B},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/yourrepo}
}
```

## Support

For issues or questions:
1. Check this documentation
2. Review notebook comments
3. Examine example outputs
4. Debug with `verbose=True`
5. Check model compatibility

## License

[Your License Here]
