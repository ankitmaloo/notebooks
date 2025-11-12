# Password Game Environment Integration Cells
# These cells should be inserted into the RL training notebook around cell 19

# ============================================================================
# CELL 19 - Markdown Header
# ============================================================================
CELL_19_MARKDOWN = """## 🎮 Password Game Environment

Integration with the Password Game from tasks/password-game/

**Game Overview**:
- 26 progressive rules with increasing difficulty
- Reward: +1 per rule passed, -0.1 per character
- Rules include: length requirements, character types, special constraints (CAPTCHA, Wordle, moon phase, etc.)

**Features**:
- Rule progression (start with fewer rules, gradually increase)
- Dynamic prompt formatting with instructions and current rules
- Integration with PasswordGame's built-in reward calculation
"""

# ============================================================================
# CELL 20 - Import Password Game Files
# ============================================================================
CELL_20_CODE = """import sys
import os
from pathlib import Path

# Add password game directory to path
password_game_dir = Path("/home/user/notebooks/tasks/password-game")
if str(password_game_dir) not in sys.path:
    sys.path.insert(0, str(password_game_dir))

# Import password game components
from game import PasswordGame, rules, instructions
print(f"✓ Password game imported ({len(rules)} rules)")
"""

# ============================================================================
# CELL 21 - PasswordGameEnvironment Wrapper Class
# ============================================================================
CELL_21_CODE = '''class PasswordGameEnvironment:
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
'''

# ============================================================================
# CELL 22 - PasswordGameDataset Class
# ============================================================================
CELL_22_CODE = '''class PasswordGameDataset(Dataset):
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
'''

# ============================================================================
# CELL 23 - Configuration Update
# ============================================================================
CELL_23_MARKDOWN = """## ⚙️ Password Game Configuration

Update the config to include password game specific settings:
"""

CELL_23_CODE = '''# Add to existing VerlPPOConfig or create new config section
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
'''

# ============================================================================
# CELL 24 - Initialize Environment and Dataset
# ============================================================================
CELL_24_CODE = '''# Initialize password game environment
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
print(f"\\n--- Example Sample ---")
print(f"Target rule: Rule {test_sample['target_rule'] + 1}")
print(f"Prompt preview (first 300 chars):\\n{test_sample['prompt'][:300]}...")
'''

# ============================================================================
# CELL 25 - Reward Function Implementation
# ============================================================================
CELL_25_MARKDOWN = """## 🎁 Password Game Reward Function

Implements the reward calculation using the PasswordGame's built-in method:
- +1 point for each rule that passes
- -0.1 point for each character (encourages shorter passwords)
"""

CELL_25_CODE = '''def calculate_reward(prompt: str, response: str, game: PasswordGame, target_rule: int) -> float:
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
    # Take the first line or first word as the password attempt
    password = response.strip().split('\\n')[0].strip()

    # If response is too long, truncate (likely includes explanation)
    if len(password) > 500:
        # Try to find a reasonable password substring
        words = password.split()
        if words:
            password = words[0]  # Take first word

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
    lines = [line.strip() for line in response.split('\\n') if line.strip()]
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

print("\\n--- Reward Function Test ---")
for pwd in test_passwords:
    reward = test_game.calculate_reward(pwd)
    feedback = test_game.get_rule_feedback(pwd)
    print(f"Password: '{pwd}'")
    print(f"  Reward: {reward}")
    print(f"  Passing rules: {feedback['total_passing']}/{len(feedback['rules_checked'])}")
'''

# ============================================================================
# CELL 26 - Training Loop Integration Notes
# ============================================================================
CELL_26_MARKDOWN = """## 🔄 Training Loop Integration

**Required modifications to the training loop:**

1. **Batch processing**: Extract game instances and target rules from batch
   ```python
   games = [item['game'] for item in batch]
   target_rules = [item['target_rule'] for item in batch]
   ```

2. **Reward calculation**: Pass game and target_rule to calculate_reward
   ```python
   rewards = [
       calculate_reward(prompt, response, game, target_rule)
       for prompt, response, game, target_rule in zip(prompts, responses, games, target_rules)
   ]
   ```

3. **Epoch refresh**: Refresh dataset samples between epochs for variety
   ```python
   train_dataset.refresh_samples()
   ```

4. **Rule progression**: Update environment progression after each epoch
   ```python
   stats = password_env.get_stats()
   wandb.log({"rule_progression/current_max_rules": stats['current_max_rules']})
   ```

The password game environment is now ready for training!
"""

print("Password Game Integration Cells - Ready to use")
print("Insert cells 19-26 into your RL training notebook")
