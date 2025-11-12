# Password Game Environment Integration for RL Training

## Overview

This document provides notebook cells to integrate the Password Game environment into your RL training notebook.

**What this provides:**
- `PasswordGameEnvironment`: Wrapper class for managing game instances
- `PasswordGameDataset`: Dataset class for generating training prompts
- Reward calculation using the game's built-in method
- Rule progression support (start easy, gradually increase difficulty)
- Full integration with the existing PPO training loop

## Game Details

- **26 progressive rules** with increasing complexity
- **Reward scheme**: +1 per passing rule, -0.1 per character
- **Dynamic elements**: CAPTCHA, Wordle answer, country name, moon phase
- **Special rules**: Roman numerals, periodic elements, leap years, etc.

## Installation Instructions

Insert these cells **after the existing environment section** (around cell 19 in the verl_qwen_rule_task.ipynb notebook).

---

## CELL 19: Section Header (Markdown)

```markdown
## 🎮 Password Game Environment

Integration with the Password Game from tasks/password-game/

**Game Overview**:
- 26 progressive rules with increasing difficulty
- Reward: +1 per rule passed, -0.1 per character
- Rules include: length requirements, character types, special constraints (CAPTCHA, Wordle, moon phase, etc.)

**Features**:
- Rule progression (start with fewer rules, gradually increase)
- Dynamic prompt formatting with instructions and current rules
- Integration with PasswordGame's built-in reward calculation
```

---

## CELL 20: Import Password Game (Code)

```python
import sys
import os
from pathlib import Path

# Add password game directory to path
password_game_dir = Path("/home/user/notebooks/tasks/password-game")
if str(password_game_dir) not in sys.path:
    sys.path.insert(0, str(password_game_dir))

# Import password game components
from game import PasswordGame, rules, instructions

print(f"✓ Password game imported ({len(rules)} rules)")
```

---

## CELL 21: PasswordGameEnvironment Class (Code)

```python
class PasswordGameEnvironment:
    """
    Wrapper for PasswordGame that manages game instances and provides
    RL training interface compatible with the batch training loop.
    """
    def __init__(
        self,
        use_rule_progression: bool = True,
        min_rules: int = 5,
        max_rules: int = 26,
        progression_rate: float = 0.1,
        seed: int = 42
    ):
        """
        Args:
            use_rule_progression: Start with fewer rules, gradually increase
            min_rules: Minimum number of rules to start with
            max_rules: Maximum number of rules (up to 26)
            progression_rate: Rate at which to increase rules (0-1)
            seed: Random seed
        """
        self.use_rule_progression = use_rule_progression
        self.min_rules = min_rules
        self.max_rules = max_rules
        self.progression_rate = progression_rate
        self.seed = seed
        self.current_max_rules = min_rules if use_rule_progression else max_rules
        self.games_completed = 0

        # Track statistics
        self.total_games = 0
        self.successful_games = 0

    def create_game_instance(self) -> PasswordGame:
        """Create a new PasswordGame instance."""
        return PasswordGame()

    def get_current_max_rules(self) -> int:
        """Get current maximum rules based on progression."""
        if not self.use_rule_progression:
            return self.max_rules

        # Gradually increase max rules based on games completed
        progress = min(1.0, self.games_completed * self.progression_rate / 100)
        current_max = self.min_rules + int((self.max_rules - self.min_rules) * progress)
        return min(current_max, self.max_rules)

    def format_prompt(self, game: PasswordGame, target_rule_index: int = None) -> str:
        """
        Format prompt for the LLM including instructions and current rules.

        Args:
            game: PasswordGame instance
            target_rule_index: Optional specific rule index to target (for training)
                             If None, uses game.current_rule

        Returns:
            Formatted prompt string
        """
        if target_rule_index is not None:
            # Override current rule for training purposes
            rule_index = target_rule_index
        else:
            rule_index = game.current_rule

        # Get all rules up to current
        all_rules = []
        for i in range(rule_index + 1):
            if i < len(rules):
                rule_text = rules[i]
                # Format dynamic rules
                if "{captcha}" in rule_text:
                    rule_text = rule_text.format(captcha=game.captcha)
                elif "{country}" in rule_text:
                    rule_text = rule_text.format(country=game.country)
                all_rules.append(f"Rule {i+1}: {rule_text}")

        # Get current rule (highlighted)
        current_rule = all_rules[-1] if all_rules else "No rules yet"

        # Format as chat messages
        prompt_text = f"""You are playing the Password Game. Create a password that satisfies ALL the rules below.

INSTRUCTIONS:
{instructions.strip()}

ALL ACTIVE RULES:
{chr(10).join(all_rules)}

CURRENT RULE (just added):
{current_rule}

Your task: Generate a password that satisfies ALL {len(all_rules)} rules above. Output ONLY the password string, nothing else."""

        messages = [
            {"role": "system", "content": "You are an expert at creating passwords that satisfy multiple complex constraints."},
            {"role": "user", "content": prompt_text}
        ]

        # Format using tokenizer chat template if available
        if hasattr(tokenizer, 'apply_chat_template'):
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

        # Fallback formatting
        return f"System: {messages[0]['content']}\\n\\nUser: {messages[1]['content']}\\n\\nAssistant:"

    def calculate_reward(self, game: PasswordGame, password: str, target_rule_index: int = None) -> float:
        """
        Calculate reward using the game's built-in reward calculation.

        Args:
            game: PasswordGame instance
            password: Generated password to evaluate
            target_rule_index: Optional specific rule index to evaluate up to

        Returns:
            Reward value (+1 per passing rule, -0.1 per character)
        """
        if target_rule_index is not None:
            # Temporarily set current rule for reward calculation
            original_rule = game.current_rule
            game.current_rule = target_rule_index + 1
            reward = game.calculate_reward(password)
            game.current_rule = original_rule
            return reward

        return game.calculate_reward(password)

    def get_rule_feedback(self, game: PasswordGame, password: str) -> Dict:
        """Get detailed feedback on which rules pass/fail."""
        return game.get_rule_feedback(password)

    def update_progression(self, reward: float, rules_used: int):
        """Update rule progression based on performance."""
        self.total_games += 1

        # Consider game successful if reward > 0 (more rules passed than length penalty)
        if reward > 0:
            self.successful_games += 1

        # Update current max rules
        if self.use_rule_progression:
            old_max = self.current_max_rules
            self.current_max_rules = self.get_current_max_rules()
            if self.current_max_rules > old_max:
                print(f"✓ Rule progression: {old_max} → {self.current_max_rules} rules")

        self.games_completed += 1

    def get_stats(self) -> Dict:
        """Get environment statistics."""
        success_rate = self.successful_games / max(1, self.total_games)
        return {
            "total_games": self.total_games,
            "successful_games": self.successful_games,
            "success_rate": success_rate,
            "current_max_rules": self.current_max_rules,
            "progression_enabled": self.use_rule_progression
        }

print("✓ PasswordGameEnvironment class defined")
```

---

## CELL 22: PasswordGameDataset Class (Code)

```python
class PasswordGameDataset(Dataset):
    """
    Dataset for password game training that generates prompts on-the-fly.
    Each sample is a password game instance at a specific rule index.
    """
    def __init__(
        self,
        num_samples: int,
        environment: PasswordGameEnvironment,
        seed: int = 42
    ):
        """
        Args:
            num_samples: Number of training samples
            environment: PasswordGameEnvironment instance
            seed: Random seed for reproducibility
        """
        self.num_samples = num_samples
        self.environment = environment
        self.seed = seed
        self.rng = random.Random(seed)

        # Pre-generate game instances and target rules
        self.samples = []
        current_max_rules = self.environment.get_current_max_rules()

        for i in range(num_samples):
            # Create a new game instance for each sample
            game = self.environment.create_game_instance()

            # Select a random rule index to target (between 0 and current_max_rules-1)
            # This allows the model to practice different difficulty levels
            target_rule = self.rng.randint(0, min(current_max_rules - 1, len(rules) - 1))

            self.samples.append({
                'game': game,
                'target_rule': target_rule,
                'sample_id': i
            })

        print(f"✓ Dataset created: {num_samples} samples, max {current_max_rules} rules")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Get a sample with prompt and metadata.

        Returns:
            Dict with:
                - prompt: Formatted prompt string
                - game: PasswordGame instance
                - target_rule: Target rule index
                - sample_id: Unique sample identifier
        """
        sample = self.samples[idx]

        # Format prompt for this game and target rule
        prompt = self.environment.format_prompt(
            sample['game'],
            target_rule_index=sample['target_rule']
        )

        return {
            'prompt': prompt,
            'game': sample['game'],
            'target_rule': sample['target_rule'],
            'sample_id': sample['sample_id']
        }

    def refresh_samples(self):
        """Refresh samples with new game instances (call between epochs)."""
        current_max_rules = self.environment.get_current_max_rules()

        for i in range(self.num_samples):
            game = self.environment.create_game_instance()
            target_rule = self.rng.randint(0, min(current_max_rules - 1, len(rules) - 1))

            self.samples[i] = {
                'game': game,
                'target_rule': target_rule,
                'sample_id': i
            }

print("✓ PasswordGameDataset class defined")
```

---

## CELL 23: Configuration (Code)

```python
# Add to existing VerlPPOConfig or create new config section
@dataclass
class PasswordGameConfig:
    """Configuration for password game environment."""
    # Rule progression
    use_rule_progression: bool = True
    min_rules: int = 5
    max_rules: int = 15  # Start with 15 rules (subset of 26)
    progression_rate: float = 0.1

    # Training focus
    focus_on_failed_rules: bool = True  # Prioritize rules that fail more often
    min_passing_threshold: float = 0.5  # Minimum pass rate before progression

# Add to main config
password_game_config = PasswordGameConfig()

print("Password Game Config:")
print(f"  Rule progression: {password_game_config.use_rule_progression}")
print(f"  Rules: {password_game_config.min_rules} → {password_game_config.max_rules}")
print(f"  Progression rate: {password_game_config.progression_rate}")
```

---

## CELL 24: Initialize Environment and Dataset (Code)

```python
# Initialize password game environment
password_env = PasswordGameEnvironment(
    use_rule_progression=password_game_config.use_rule_progression,
    min_rules=password_game_config.min_rules,
    max_rules=password_game_config.max_rules,
    progression_rate=password_game_config.progression_rate,
    seed=config.seed
)

print("✓ Password game environment initialized")
print(f"  Current max rules: {password_env.get_current_max_rules()}")

# Create datasets
train_dataset = PasswordGameDataset(
    num_samples=config.num_train_samples,
    environment=password_env,
    seed=config.seed
)

val_dataset = PasswordGameDataset(
    num_samples=config.num_val_samples,
    environment=password_env,
    seed=config.seed + 1000
)

print(f"✓ Datasets created: Train={len(train_dataset)}, Val={len(val_dataset)}")

# Test dataset sampling
test_sample = train_dataset[0]
print(f"\n--- Example Sample ---")
print(f"Target rule: Rule {test_sample['target_rule'] + 1}")
print(f"Prompt preview (first 500 chars):\n{test_sample['prompt'][:500]}...")
```

---

## CELL 25: Reward Function (Code)

```python
def calculate_reward(prompt: str, response: str, game: PasswordGame, target_rule: int) -> float:
    """
    Calculate reward for a password game response.

    Args:
        prompt: The prompt (unused, kept for compatibility)
        response: Generated password string
        game: PasswordGame instance with game state
        target_rule: Target rule index being trained on

    Returns:
        Reward value: +1 per passing rule, -0.1 per character
    """
    # Extract just the password from response (model might add extra text)
    password = extract_password_from_response(response)

    # Calculate reward using game's method
    reward = password_env.calculate_reward(game, password, target_rule_index=target_rule)

    return reward

def extract_password_from_response(response: str) -> str:
    """
    Extract the actual password from model response.
    Model might generate thinking or explanation text.

    Args:
        response: Full model response

    Returns:
        Extracted password string
    """
    # Remove leading/trailing whitespace
    response = response.strip()

    # If response has multiple lines, take the last non-empty line
    # (often the password comes after thinking)
    lines = [line.strip() for line in response.split('\n') if line.strip()]
    if not lines:
        return ""

    # Take the last line as the password
    password = lines[-1]

    # If still too long (>500 chars), take first word
    if len(password) > 500:
        words = password.split()
        password = words[0] if words else password[:100]

    return password

print("✓ Password game reward function defined")

# Test the reward function
test_game = PasswordGame()
test_passwords = [
    "12345",  # Passes rule 0 (length) and 1 (number)
    "12345A!",  # Passes rules 0, 1, 2 (uppercase), 3 (special)
    "invalid",  # Fails most rules
]

print("\n--- Reward Function Test ---")
for pwd in test_passwords:
    reward = test_game.calculate_reward(pwd)
    feedback = test_game.get_rule_feedback(pwd)
    print(f"Password: '{pwd}'")
    print(f"  Reward: {reward:.2f}")
    print(f"  Passing rules: {feedback['total_passing']}/{len(feedback['rules_checked'])}")
```

---

## CELL 26: Training Loop Modifications (Markdown)

```markdown
## 🔄 Training Loop Integration

**Required modifications to the training loop:**

### 1. Update batch extraction:

Replace:
```python
prompts = batch['prompt']
rules = batch['rule']
```

With:
```python
prompts = batch['prompt']
games = batch['game']
target_rules = batch['target_rule']
```

### 2. Update reward calculation:

Replace:
```python
rewards = torch.tensor([calculate_reward(p, r, rule) for p, r, rule in zip(expanded_prompts, all_responses, expanded_rules)], ...)
```

With:
```python
rewards = torch.tensor([
    calculate_reward(p, r, game, target_rule)
    for p, r, game, target_rule in zip(expanded_prompts, all_responses, expanded_games, expanded_target_rules)
], device=DEVICE, dtype=dtype)
```

Where `expanded_games` and `expanded_target_rules` are created by:
```python
expanded_games = games * config.samples_per_prompt
expanded_target_rules = target_rules * config.samples_per_prompt
```

### 3. Add dataset refresh between epochs:

At the start of each epoch loop:
```python
if step == 0 and epoch > 0:  # Start of new epoch
    train_dataset.refresh_samples()
    print(f"✓ Refreshed training samples for epoch {epoch+1}")
```

### 4. Track rule progression:

After each epoch:
```python
stats = password_env.get_stats()
wandb.log({
    "rule_progression/current_max_rules": stats['current_max_rules'],
    "rule_progression/success_rate": stats['success_rate'],
    "rule_progression/total_games": stats['total_games']
}, step=global_step)
```

### 5. Update evaluation function:

Modify `evaluate_model` to handle password game batch format:
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

The password game environment is now ready for training!
```

---

## Next Steps

1. **Insert cells 19-26** into your RL training notebook
2. **Modify the training loop** as described in Cell 26
3. **Run the notebook** and monitor:
   - Rule progression (should gradually increase from 5 to 15 rules)
   - Reward improvements (baseline vs trained)
   - Success rate (percentage of games with positive reward)

## Configuration Tips

- Start with `min_rules=5, max_rules=10` for faster initial training
- Increase `max_rules` to 15-20 once the model learns basic patterns
- Adjust `progression_rate` based on training speed (higher = faster progression)
- Set `use_rule_progression=False` to train on all rules from the start (harder)

## Expected Performance

- **Baseline**: Reward around -2 to 0 (random passwords fail most rules)
- **After training**: Reward should improve to 2-5+ (passing 3-7 rules consistently)
- **Rule progression**: Should advance from 5 to 15 rules over 2-3 epochs

## Troubleshooting

**Issue**: Model generates long responses instead of just passwords
- **Solution**: Improve the `extract_password_from_response` function or adjust prompt

**Issue**: Rewards stay negative
- **Solution**: Decrease `max_rules` or increase `min_rules` range

**Issue**: Rule progression too fast/slow
- **Solution**: Adjust `progression_rate` parameter

## Files Location

- Game implementation: `/home/user/notebooks/tasks/password-game/game.py`
- Integration cells: `/home/user/notebooks/RL/password_game_integration_cells.py`
- This documentation: `/home/user/notebooks/RL/PASSWORD_GAME_INTEGRATION.md`
