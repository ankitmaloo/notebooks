# RewardComputer - Quick Start Guide

## What is RewardComputer?

RewardComputer separates reward computation from environment logic, allowing batch-wise comparison of trajectories. This is essential for:
- **GRPO**: Comparing trajectories to rank them
- **Pareto Optimization**: Finding optimal trade-offs between multiple objectives
- **SVRL**: Balancing task quality, verification cost, and efficiency

## Three Reward Methods

### 1. Absolute Rewards
Each trajectory scored independently.

```python
rewards = reward_computer.compute_rewards(trajectories, method="absolute")
# Returns: [0.8, 0.6, 0.9, 0.7]  # Independent scores
```

**Use when**: Standard RL, each trajectory evaluated on its own merits

### 2. Relative Rewards (GRPO)
Trajectories ranked against each other.

```python
rewards = reward_computer.compute_rewards(trajectories, method="relative")
# Returns: [0.5, -1.2, 1.4, -0.7]  # Normalized ranks (mean=0, std=1)
```

**Use when**: GRPO training, only updating on top-k trajectories

### 3. Pareto Rewards
Multi-objective optimization using Pareto frontier.

```python
rewards = reward_computer.compute_rewards(trajectories, method="pareto")
# Returns: [1.0, 1.0, 0.5, 1.0]  # 1.0 = on frontier, <1.0 = dominated
```

**Use when**: Multiple competing objectives (accuracy vs efficiency vs safety)

## Quick Examples

### Example 1: SVRL (Self-Verification)

```python
from rl_primitives.reward_computer import SVRLRewardComputer, TrajectoryState

# Define how to evaluate task completion
def evaluate_task(traj: TrajectoryState) -> float:
    # Return 0-1 score for task quality
    return check_correctness(traj.response)

# Create SVRL reward computer
rc = SVRLRewardComputer(
    task_evaluator=evaluate_task,
    verification_weight=0.1,    # Penalty for using verification budget
    efficiency_weight=0.05      # Bonus for completing quickly
)

# Create trajectories with metadata
trajectories = [
    TrajectoryState(
        step_count=20,
        response="My solution",
        metadata={
            'verification_spent': 5,   # How much budget used
            'initial_budget': 10,       # Total budget available
            'max_steps': 50             # Maximum steps allowed
        }
    ),
    # ... more trajectories
]

# Compute rewards (use "relative" for GRPO)
rewards = rc.compute_rewards(trajectories, method="relative")
```

### Example 2: Pareto Multi-Objective

```python
from rl_primitives.reward_computer import ParetoRewardComputer

# Define your objectives (higher = better for all)
def accuracy(traj):
    return evaluate_accuracy(traj.response)

def efficiency(traj):
    return 1.0 - (traj.step_count / 50)  # Fewer steps = better

def safety(traj):
    return evaluate_safety(traj.response)

# Create Pareto reward computer
rc = ParetoRewardComputer(
    objective_functions=[accuracy, efficiency, safety],
    objective_names=['accuracy', 'efficiency', 'safety']
)

# Compute Pareto rewards
rewards = rc.compute_rewards(trajectories, method="pareto")

# Get which trajectories are on Pareto frontier
frontier_indices = rc.get_pareto_frontier_indices(trajectories)
print(f"Frontier: {frontier_indices}")  # e.g., [0, 2, 5, 7]
```

### Example 3: Custom Reward Computer

```python
from rl_primitives.reward_computer import RewardComputer
import numpy as np

class MyCustomRewardComputer(RewardComputer):
    def score_trajectory(self, traj: TrajectoryState) -> float:
        # Your custom scoring logic
        quality = evaluate_quality(traj.response)
        cost = traj.step_count * 0.01
        return quality - cost

# Use it
rc = MyCustomRewardComputer(normalize=True)
rewards = rc.compute_rewards(trajectories, method="absolute")
```

## Integration with Training Loop

```python
# 1. Collect trajectories (environment handles interactions)
trajectories = rollout_manager.collect_rollouts(batch_size=32)

# 2. Compute rewards (RewardComputer compares trajectories)
rewards = reward_computer.compute_rewards(trajectories, method="relative")

# 3. For GRPO: only update on top-50% trajectories
threshold = np.percentile(rewards, 50)
good_trajectories = [t for t, r in zip(trajectories, rewards) if r > threshold]

# 4. Compute advantages and update model
advantages = backprop.compute_advantages(rewards)
loss = backprop.compute_loss(trajectories, advantages)
backprop.update(loss)
```

## Key Classes

### TrajectoryState
```python
@dataclass
class TrajectoryState:
    step_count: int          # Number of steps in trajectory
    response: Any            # Final response/action
    done: bool = True        # Whether complete
    metadata: Dict = None    # Custom data (put anything here)
```

### SVRLRewardComputer
```python
SVRLRewardComputer(
    task_evaluator,          # Function: TrajectoryState -> float (0-1)
    verification_weight,     # Penalty for verification cost
    efficiency_weight,       # Bonus for step efficiency
    normalize                # Normalize rewards (default: True)
)
```

### ParetoRewardComputer
```python
ParetoRewardComputer(
    objective_functions,     # List of functions: TrajectoryState -> float
    objective_names,         # Optional names for logging
    weights,                 # Optional weights for single-objective fallback
    normalize                # Normalize rewards (default: True)
)
```

## Common Patterns

### Pattern 1: GRPO Training
```python
# Use relative rewards + top-k selection
rewards = rc.compute_rewards(trajectories, method="relative")
top_k = int(0.5 * len(trajectories))
sorted_indices = np.argsort(rewards)[-top_k:]
good_trajectories = [trajectories[i] for i in sorted_indices]
```

### Pattern 2: Pareto Diversity
```python
# Get Pareto frontier for diverse solutions
frontier_idx = rc.get_pareto_frontier_indices(trajectories)
frontier_trajectories = [trajectories[i] for i in frontier_idx]

# Monitor objective statistics
stats = rc.compute_objective_statistics(trajectories)
print(f"Accuracy: {stats['accuracy']['mean']:.3f}")
print(f"Efficiency: {stats['efficiency']['mean']:.3f}")
```

### Pattern 3: Mixed Objectives
```python
# Use weighted combination for single reward
rc = ParetoRewardComputer(
    objective_functions=[accuracy, efficiency, safety],
    weights=[0.5, 0.3, 0.2]  # Prioritize accuracy
)

# Get single reward value
single_reward = rc.score_trajectory(traj)  # Uses weights

# Or use full Pareto method
pareto_rewards = rc.compute_rewards(trajectories, method="pareto")
```

## Tips

1. **Normalization**: Keep `normalize=True` for stable training
2. **Relative for GRPO**: Always use `method="relative"` with GRPO
3. **Pareto for Trade-offs**: Use `method="pareto"` when objectives conflict
4. **Metadata**: Put custom data in `trajectory.metadata` dict
5. **Objective Scale**: Make sure all objectives are "higher is better"

## Running Examples

```bash
# Built-in test
python -m rl_primitives.reward_computer

# Comprehensive examples
python rl_primitives/examples/reward_computer_usage.py
```

## Files

- **Implementation**: `/home/user/notebooks/rl_primitives_design/rl_primitives/reward_computer.py` (822 lines)
- **Examples**: `/home/user/notebooks/rl_primitives_design/rl_primitives/examples/reward_computer_usage.py` (306 lines)
- **Full Docs**: `/home/user/notebooks/rl_primitives_design/rl_primitives/examples/README_reward_computer.md`

## Next Steps

1. ✅ RewardComputer implemented
2. Use with RolloutManager to collect trajectories
3. Use with BackpropModule for training
4. See `v2.md` for full architecture

## Questions?

See full documentation in `README_reward_computer.md` or run examples for more details.
