# Password Game Environment - Quick Start Guide

## Installation & Setup

### 1. Install Dependencies

```bash
cd /home/user/notebooks/tasks/password-game
pip install -r requirements.txt
```

### 2. Start the API Server

```bash
# Terminal 1: Start the API server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Server will be available at `http://localhost:8000`

### 3. Verify Installation

```bash
# Terminal 2: Run tests
python test_password_game_env.py
```

## Common Use Cases

### Use Case 1: Basic Game Playing

```python
from password_game_env import PasswordGameEnv

# Create environment
env = PasswordGameEnv()

# Start game
state = env.start_game()
print(f"Rule: {state.current_rule}")

# Test password
feedback = env.get_feedback("Hello123!")
print(f"Passing: {feedback.total_passing} rules")
print(f"Reward: {feedback.reward}")

# Submit if good
if feedback.total_passing > 3:
    result = env.submit_password("Hello123!")
    print(f"Success: {result.success}")

env.close()
```

### Use Case 2: Multi-Step Game

```python
from password_game_env import PasswordGameEnv

with PasswordGameEnv() as env:
    state = env.start_game()

    # Play through multiple rules
    passwords = ["Hello", "Hello1", "Hello1A", "Hello1A!"]

    for pwd in passwords:
        if not env.is_active():
            break

        result = env.submit_password(pwd)
        if result.success:
            print(f"Rule {state.current_rule_index}: '{pwd}' → advanced")
            state = result.new_state or state
        else:
            print(f"Failed: {result.error}")
            break
```

### Use Case 3: Batch Processing

```python
from password_game_env import BatchPasswordGameEnv

# Create 4 parallel environments
batch_env = BatchPasswordGameEnv(num_envs=4)

# Start all games
states = batch_env.reset()

# Test passwords in parallel
passwords = ["Test1!", "Test2@", "Test3#", "Test4$"]
feedbacks = batch_env.get_feedback_batch(passwords)

# Print results
for i, fb in enumerate(feedbacks):
    if fb:
        print(f"Env {i}: {fb.total_passing} passing, reward={fb.reward}")

# Submit all
results = batch_env.submit_batch(passwords)

batch_env.close()
```

### Use Case 4: Simple RL Training

```python
from password_game_env import PasswordGameEnv, create_observation_dict

env = PasswordGameEnv()

# Training loop
for episode in range(100):
    state = env.reset()
    episode_reward = 0

    while env.is_active():
        # Create observation
        obs = create_observation_dict(state)

        # Your agent selects action (password)
        password = your_agent.select_action(obs)

        # Execute
        result = env.submit_password(password)

        if result.success:
            reward = result.reward if result.game_ended else 0.0
            episode_reward += reward

            # Update your agent
            your_agent.update(obs, password, reward)

            if result.game_ended:
                break

            state = result.new_state

    print(f"Episode {episode}: Reward = {episode_reward}")

env.close()
```

### Use Case 5: Debugging with Detailed Feedback

```python
from password_game_env import PasswordGameEnv

env = PasswordGameEnv()
state = env.start_game()

password = "Test123!"
feedback = env.get_feedback(password)

# Analyze each rule
print(f"\nPassword: '{password}'")
print(f"Length: {feedback.length}")
print(f"Total Passing: {feedback.total_passing}/{len(feedback.rules_checked)}")
print(f"Reward: {feedback.reward}\n")

print("Rule-by-rule breakdown:")
for rule in feedback.rules_checked:
    status = "✓" if rule["passes"] else "✗"
    print(f"{status} Rule {rule['rule_index']}: {rule['rule_text']}")

# Find first failure
first_fail = feedback.first_failing_rule()
if first_fail:
    print(f"\nFirst failing: {first_fail['rule_text']}")

env.close()
```

### Use Case 6: Error Handling

```python
from password_game_env import PasswordGameEnv

env = PasswordGameEnv(
    base_url="http://localhost:8000",
    timeout=10.0,
    max_retries=2
)

# Check if server is up
state = env.start_game()
if state is None:
    print("Error: Could not connect to server")
    print("Make sure the API server is running:")
    print("  uvicorn main:app --reload")
    exit(1)

# Safe submission with error handling
result = env.submit_password("Test123")
if not result.success:
    print(f"Submission failed: {result.error}")
    # Handle error (retry, reset, etc.)
    state = env.reset()

env.close()
```

## API Quick Reference

### Environment Methods

| Method | Description | Returns |
|--------|-------------|---------|
| `start_game()` | Start new game | `GameState` |
| `get_state(refresh=False)` | Get current state | `GameState` |
| `get_feedback(password)` | Test password | `FeedbackResult` |
| `submit_password(password, give_up=False)` | Submit and advance | `SubmitResult` |
| `end_game()` | End current game | `bool` |
| `reset()` | Reset to new game | `GameState` |
| `is_active()` | Check if active | `bool` |
| `close()` | Clean up resources | `None` |

### Batch Environment Methods

| Method | Description | Returns |
|--------|-------------|---------|
| `reset()` | Reset all envs | `List[GameState]` |
| `get_states(refresh=False)` | Get all states | `List[GameState]` |
| `get_feedback_batch(passwords)` | Test passwords | `List[FeedbackResult]` |
| `submit_batch(passwords, give_up=None)` | Submit all | `List[SubmitResult]` |
| `close()` | Clean up all | `None` |

### Data Classes

**GameState**
- `token: str` - Session ID
- `current_rule_index: int` - Current rule (0-25)
- `current_rule: str` - Rule text
- `all_rules: List[str]` - All rules so far
- `game_active: bool` - Is game active?

**FeedbackResult**
- `password: str` - Tested password
- `length: int` - Password length
- `total_passing: int` - Number passing
- `reward: float` - Calculated reward
- `rules_checked: List[Dict]` - Detailed results
- Methods: `passing_rules()`, `failing_rules()`, `first_failing_rule()`

**SubmitResult**
- `success: bool` - Did submission work?
- `game_ended: bool` - Did game end?
- `gave_up: bool` - Did player give up?
- `reward: float` - Final reward (if ended)
- `new_state: GameState` - New state (if continues)
- `error: str` - Error message (if failed)

## Common Patterns

### Pattern: Try Different Passwords

```python
env = PasswordGameEnv()
state = env.start_game()

candidates = ["Test1", "Test1!", "Test1!A", "Test1!A9"]
best_password = None
best_score = -float('inf')

for pwd in candidates:
    feedback = env.get_feedback(pwd)
    if feedback and feedback.reward > best_score:
        best_score = feedback.reward
        best_password = pwd

print(f"Best: '{best_password}' with score {best_score}")
result = env.submit_password(best_password)

env.close()
```

### Pattern: Continue Until Stuck

```python
env = PasswordGameEnv()
state = env.start_game()

max_attempts = 3
attempts = 0

while env.is_active() and attempts < max_attempts:
    # Your logic to generate password
    password = generate_password_for_rule(state.current_rule)

    feedback = env.get_feedback(password)

    if feedback and feedback.total_passing == len(feedback.rules_checked):
        # All rules pass, submit
        result = env.submit_password(password)
        state = result.new_state
        attempts = 0  # Reset attempts
    else:
        # Didn't pass all rules
        attempts += 1

if attempts >= max_attempts:
    print("Stuck, giving up")
    env.submit_password(password, give_up=True)

env.close()
```

### Pattern: Batch Training with Different Strategies

```python
batch_env = BatchPasswordGameEnv(num_envs=3)

# Each env uses different strategy
strategies = [strategy_A, strategy_B, strategy_C]

states = batch_env.reset()

while any(env.is_active() for env in batch_env.envs):
    # Generate passwords with different strategies
    passwords = [
        strategy(state) if state else ""
        for strategy, state in zip(strategies, states)
    ]

    # Execute in parallel
    results = batch_env.submit_batch(passwords)

    # Update states
    states = [
        result.new_state if result.success and not result.game_ended else state
        for result, state in zip(results, states)
    ]

batch_env.close()
```

## Troubleshooting

### Problem: Connection refused

```
Error: Failed to start game: Connection error: ...
```

**Solution:** Start the API server
```bash
cd /home/user/notebooks/tasks/password-game
uvicorn main:app --reload
```

### Problem: Invalid token (HTTP 404)

```
Error: HTTP error: 404 - Game session not found
```

**Solution:** Reset environment to get new token
```python
state = env.reset()
```

### Problem: Batch size mismatch

```
ValueError: Expected 4 passwords, got 2
```

**Solution:** Provide correct number of passwords
```python
batch_env = BatchPasswordGameEnv(num_envs=4)
passwords = ["p1", "p2", "p3", "p4"]  # Must be 4
feedbacks = batch_env.get_feedback_batch(passwords)
```

### Problem: Timeout errors

```
Error: Request timeout: ...
```

**Solution:** Increase timeout
```python
env = PasswordGameEnv(timeout=60.0)  # 60 seconds
```

### Problem: No response from feedback

```python
feedback = env.get_feedback("test")
# feedback is None
```

**Solution:** Check if game is active
```python
if not env.is_active():
    print("Game not active, call start_game() first")
    state = env.start_game()
```

## Performance Tips

1. **Use cached state when possible**
   ```python
   state = env.get_state()  # Fast (cached)
   state = env.get_state(refresh=True)  # Slower (API call)
   ```

2. **Batch operations for parallel work**
   ```python
   # Slower: Sequential
   for i in range(4):
       env = PasswordGameEnv()
       env.start_game()

   # Faster: Parallel
   batch_env = BatchPasswordGameEnv(num_envs=4)
   batch_env.reset()
   ```

3. **Reuse environment instances**
   ```python
   # Slower: Create new env each time
   for episode in range(100):
       env = PasswordGameEnv()
       env.start_game()
       # ...
       env.close()

   # Faster: Reuse env
   env = PasswordGameEnv()
   for episode in range(100):
       env.reset()
       # ...
   env.close()
   ```

4. **Use context managers for cleanup**
   ```python
   # Automatic cleanup
   with PasswordGameEnv() as env:
       # Use env
   # Automatically cleaned up
   ```

## Next Steps

1. **Read full documentation:** See `README_ENV.md`
2. **Understand architecture:** See `ARCHITECTURE.md`
3. **Run examples:** `python example_usage.py`
4. **Run tests:** `python test_password_game_env.py`
5. **Start training:** Integrate with your RL framework

## Resources

- **Main wrapper:** `/home/user/notebooks/tasks/password-game/password_game_env.py`
- **Examples:** `/home/user/notebooks/tasks/password-game/example_usage.py`
- **Tests:** `/home/user/notebooks/tasks/password-game/test_password_game_env.py`
- **Documentation:** `/home/user/notebooks/tasks/password-game/README_ENV.md`
- **Architecture:** `/home/user/notebooks/tasks/password-game/ARCHITECTURE.md`

## Support

For issues or questions:
1. Check the documentation
2. Run the test suite
3. Review example usage
4. Check server logs (if API issue)

---

**Quick Start Guide v1.0.0**
**Last Updated: 2025-11-10**
