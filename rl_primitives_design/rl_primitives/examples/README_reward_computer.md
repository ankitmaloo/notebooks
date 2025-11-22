# RewardComputer Component

## Overview

The `RewardComputer` is a core component of the RL primitives architecture that handles reward computation after trajectory collection. It separates reward logic from environment logic, enabling batch-wise comparison of trajectories (essential for GRPO and Pareto-based methods).

## Architecture

Based on the design in `/home/user/notebooks/rl_primitives_design/v2.md`, the RewardComputer provides:

1. **Base RewardComputer class** - Abstract interface for all reward computations
2. **SVRLRewardComputer** - For self-verification tasks with budget constraints
3. **ParetoRewardComputer** - For multi-objective optimization using Pareto ranking

## Key Features

### 1. Multiple Reward Computation Methods

- **Absolute Rewards**: Direct scoring of each trajectory independently
- **Relative Rewards**: GRPO-style ranking (compares trajectories against each other)
- **Pareto Rewards**: Multi-objective Pareto frontier ranking

### 2. Efficient Pareto Frontier Computation

Implements fast Pareto dominance algorithm from [Stack Overflow](https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python):
- O(n²) worst case, but much faster in practice
- Handles any number of objectives
- Computes Pareto ranks (0=frontier, 1=dominated by frontier, etc.)

### 3. Production-Ready Implementation

- Full type annotations
- Comprehensive docstrings
- NumPy/SciPy for efficient computation
- Modular and extensible design
- Built-in normalization options
- Extensive examples and tests

## Usage

### Basic Usage

```python
from rl_primitives.reward_computer import SVRLRewardComputer, TrajectoryState

# Define task evaluator
def evaluate_task(trajectory: TrajectoryState) -> float:
    return trajectory.metadata.get('accuracy', 0.0)

# Create reward computer
reward_computer = SVRLRewardComputer(
    task_evaluator=evaluate_task,
    verification_weight=0.1,
    efficiency_weight=0.05
)

# Compute rewards
trajectories = rollout_manager.collect_rollouts(32)
rewards = reward_computer.compute_rewards(trajectories, method="relative")
```

### GRPO-Style Relative Ranking

```python
# For GRPO, use relative rewards to rank trajectories
rewards = reward_computer.compute_rewards(trajectories, method="relative")

# Rewards are normalized ranks (mean=0, std=1)
# Top trajectories get positive rewards
# Bottom trajectories get negative rewards
```

### Multi-Objective Pareto Optimization

```python
from rl_primitives.reward_computer import ParetoRewardComputer

# Define objective functions
def accuracy(traj): return traj.metadata['accuracy']
def efficiency(traj): return 1.0 - (traj.step_count / 50)
def safety(traj): return traj.metadata['safety']

# Create Pareto reward computer
pareto_rc = ParetoRewardComputer(
    objective_functions=[accuracy, efficiency, safety],
    objective_names=['accuracy', 'efficiency', 'safety']
)

# Compute Pareto-based rewards
rewards = pareto_rc.compute_rewards(trajectories, method="pareto")

# Get Pareto frontier indices
frontier = pareto_rc.get_pareto_frontier_indices(trajectories)
```

## Components

### 1. Base RewardComputer

Abstract base class providing:
- `compute_rewards(trajectories, method)` - Main entry point
- `absolute_rewards(trajectories)` - Independent trajectory scoring
- `relative_rewards(trajectories)` - GRPO-style ranking
- `pareto_rewards(trajectories)` - Multi-objective Pareto ranking
- `score_trajectory(trajectory)` - Single trajectory scoring (abstract)
- `score_trajectory_multiobjective(trajectory)` - Multi-objective scoring

### 2. SVRLRewardComputer

Specialized for self-verification tasks:
- Balances task quality, verification cost, and efficiency
- Configurable weights for each objective
- Built-in support for verification budget tracking

**Constructor Parameters:**
- `task_evaluator`: Function to evaluate task completion (returns 0-1)
- `verification_weight`: Penalty weight for verification cost (default: 0.1)
- `efficiency_weight`: Bonus weight for step efficiency (default: 0.05)
- `normalize`: Whether to normalize rewards (default: True)

**Expected Trajectory Metadata:**
- `verification_spent`: Amount of verification budget used
- `initial_budget`: Total verification budget available
- `max_steps`: Maximum allowed steps

### 3. ParetoRewardComputer

Specialized for multi-objective optimization:
- Computes Pareto frontier using efficient algorithm
- Ranks trajectories based on Pareto dominance
- Supports any number of objectives
- Provides objective statistics for monitoring

**Constructor Parameters:**
- `objective_functions`: List of objective evaluation functions
- `objective_names`: Optional names for objectives (for logging)
- `weights`: Optional weights for single-objective fallback
- `normalize`: Whether to normalize rewards (default: True)

**Key Methods:**
- `get_pareto_frontier_indices(trajectories)` - Get frontier trajectory indices
- `compute_objective_statistics(trajectories)` - Get per-objective statistics

## TrajectoryState

Flexible container for trajectory data:

```python
@dataclass
class TrajectoryState:
    step_count: int          # Number of steps taken
    response: Any            # Final response/action
    done: bool = True        # Whether trajectory is complete
    metadata: Dict = None    # Additional trajectory-specific data
```

Environments can extend this or use metadata dict for custom fields.

## Examples

See `reward_computer_usage.py` for comprehensive examples:

1. **Example 1**: Basic absolute rewards
2. **Example 2**: GRPO relative ranking
3. **Example 3**: SVRL with verification budget
4. **Example 4**: Pareto multi-objective optimization
5. **Example 5**: Factory pattern usage

Run examples:
```bash
cd /home/user/notebooks/rl_primitives_design
python rl_primitives/examples/reward_computer_usage.py
```

## Implementation Details

### Pareto Frontier Algorithm

The Pareto frontier computation uses an efficient iterative algorithm:

1. For each point, check if any other point dominates it
2. Point A dominates B if A ≥ B in all objectives AND A > B in at least one
3. Keep only non-dominated points (the Pareto frontier)
4. For Pareto ranking, iteratively find frontiers and assign ranks

**Time Complexity**: O(n² × m) where n=points, m=objectives
**Space Complexity**: O(n × m)

### Normalization

Rewards can be optionally normalized to mean=0, std=1:
- Helps with training stability
- Makes rewards comparable across different scales
- Can be disabled if needed

### Extension Points

To create custom reward computers:

```python
class MyCustomRewardComputer(RewardComputer):
    def score_trajectory(self, trajectory: TrajectoryState) -> float:
        # Your custom scoring logic
        return my_score_function(trajectory)

    def score_trajectory_multiobjective(self, trajectory: TrajectoryState) -> np.ndarray:
        # Optional: for Pareto method support
        return np.array([obj1(trajectory), obj2(trajectory)])
```

## Integration with RL Training

The RewardComputer integrates with other RL primitives:

```python
# 1. Environment handles interactions (no rewards)
trajectories = rollout_manager.collect_rollouts(batch_size=32)

# 2. RewardComputer computes rewards AFTER collection
rewards = reward_computer.compute_rewards(trajectories, method="relative")

# 3. BackpropModule uses rewards for training
advantages = backprop.compute_advantages(rewards)
loss = backprop.compute_loss(trajectories, advantages)
backprop.update(loss)
```

## References

- **Pareto Optimization**:
  - [Fast Pareto front calculation in Python](https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python)
  - [Multi-objective optimization - Wikipedia](https://en.wikipedia.org/wiki/Multi-objective_optimization)
  - [pymoo library](https://pymoo.org/)

- **GRPO Algorithm**: Group Relative Policy Optimization uses relative ranking

- **Architecture**: Based on v2.md separation of concerns

## Testing

Run the built-in tests:
```bash
# Basic functionality test
python -m rl_primitives.reward_computer

# Comprehensive examples
python rl_primitives/examples/reward_computer_usage.py
```

## Performance Considerations

1. **Batch Processing**: All methods operate on batches for efficiency
2. **NumPy Operations**: Uses vectorized operations throughout
3. **Pareto Efficiency**: Iterative algorithm is much faster than naive O(n²)
4. **Memory**: Efficient for typical batch sizes (32-256 trajectories)

## Future Enhancements

Potential extensions:
- [ ] Support for hierarchical objectives
- [ ] Online Pareto frontier updates
- [ ] Crowding distance computation for diversity
- [ ] Support for constrained optimization
- [ ] Automatic objective normalization
- [ ] Multi-task reward computers

## File Location

**Main Implementation**: `/home/user/notebooks/rl_primitives_design/rl_primitives/reward_computer.py`

**Examples**: `/home/user/notebooks/rl_primitives_design/rl_primitives/examples/reward_computer_usage.py`
