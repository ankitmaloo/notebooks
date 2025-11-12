# Password Game Integration - Installation Notes

## Dependencies

The password game integration requires the OpenAI package for fetching dynamic game elements (Wordle answer, moon phase). While the game has fallbacks, it's recommended to install it.

### Required Installation

Add this cell **before Cell 20** (the password game import cell):

```python
# Install OpenAI for password game dynamic elements
!pip install -q openai

print("✓ OpenAI installed for password game")
```

### Alternative: Modify utils.py to handle missing OpenAI

If you don't want to install OpenAI, you can modify `/home/user/notebooks/tasks/password-game/utils.py` to handle the missing import gracefully:

**Option 1: Comment out OpenAI import**

Edit `utils.py` and wrap the OpenAI import:

```python
# utils.py
import os
import json

try:
    from openai import OpenAI
    from keys import *
    client = OpenAI(api_key=OPENAI_API_KEY)
    OPENAI_AVAILABLE = True
except (ImportError, NameError):
    OPENAI_AVAILABLE = False
    client = None

# ... rest of file

def utils_get_wordle() -> dict:
    """Gets today's Wordle answer using web search."""
    if not OPENAI_AVAILABLE:
        return {"answer": "crane", "date": "2024-01-01"}
    # ... existing code

def utils_get_emoji() -> dict:
    """Gets today's actual moon phase emoji using web search."""
    if not OPENAI_AVAILABLE:
        return {"answer": "🌕", "date": "2024-01-01"}
    # ... existing code
```

**Option 2: Use fallback values only**

Edit `game.py` to skip the dynamic fetching:

```python
# In game.py, change these methods:

def get_wordle_answer():
    """Fetch today's Wordle answer from web."""
    return "crane"  # Always use fallback

def get_current_moon_phase():
    """Get current moon phase emoji."""
    return "🌕"  # Always use full moon
```

## Recommended Approach

**For full functionality**: Install OpenAI
```bash
pip install openai
```

**For quick testing**: Use fallback values (moon phase and Wordle will be static)

The game will work either way, but with OpenAI installed, each game instance gets real dynamic values for:
- Wordle answer (Rule 11)
- Moon phase emoji (Rule 13)

## Complete Installation Sequence for Notebook

Insert these cells in order:

### Cell 19a: Install OpenAI (NEW)
```python
!pip install -q openai
print("✓ OpenAI installed")
```

### Cell 19b: Section Header (Markdown)
```markdown
## 🎮 Password Game Environment
...
```

### Cell 20: Import Password Game
```python
import sys
from pathlib import Path

password_game_dir = Path("/home/user/notebooks/tasks/password-game")
sys.path.insert(0, str(password_game_dir))

from game import PasswordGame, rules, instructions
print(f"✓ Password game imported ({len(rules)} rules)")
```

## Verification

Run this to verify the import works:

```python
# Test import
try:
    from game import PasswordGame
    game = PasswordGame()
    print(f"✓ Password game working")
    print(f"  CAPTCHA: {game.captcha}")
    print(f"  Country: {game.country}")
    print(f"  Wordle: {game.wordle_answer}")
    print(f"  Moon: {game.moon_phase}")
except Exception as e:
    print(f"✗ Error: {e}")
    print("  Install openai: !pip install openai")
```

## Full Dependency List

The RL training notebook should have these installed:

```python
# From the training notebook setup cell:
!pip install -q torch==2.4.0 torchvision==0.19.0
!pip install -q flash-attn --no-build-isolation
!pip install -q transformers>=4.45.0 accelerate datasets tokenizers
!pip install -q wandb tensorboard pandas matplotlib tqdm

# ADD for password game:
!pip install -q openai
```

## Environment Variables

The password game's utils.py expects these keys (for OpenAI API):

```python
# keys.py should contain:
OPENAI_API_KEY = "your-api-key-here"
WANDB_API_KEY = "your-wandb-key-here"
HF_TOKEN = "your-hf-token-here"
```

If you don't have OPENAI_API_KEY, the game will:
- Use fallback Wordle answer: "crane"
- Raise exception for moon phase (which is caught and uses fallback)

## Troubleshooting

### Issue: `No module named 'openai'`
**Solution**: Run `!pip install openai` in a notebook cell before importing

### Issue: `No module named 'keys'`
**Solution**: Create `/home/user/notebooks/tasks/password-game/keys.py` with:
```python
OPENAI_API_KEY = ""  # Optional, will use fallbacks
```

### Issue: Import works but moon phase fails
**Solution**: This is expected if OPENAI_API_KEY is not set. The game catches this and continues.

### Issue: Game imports but training fails
**Solution**: Check that:
1. All cells from `password_game_cells_ready.txt` are added
2. Training loop is modified as described
3. `calculate_reward` function signature matches

## Summary

**Quick Start (with OpenAI)**:
```bash
pip install openai
```

**Quick Start (without OpenAI)**:
- Modify `utils.py` to handle missing import
- Or modify `game.py` to skip dynamic fetching
- Game will use static fallback values

Both approaches work for training, but dynamic values add variety to the training data.
