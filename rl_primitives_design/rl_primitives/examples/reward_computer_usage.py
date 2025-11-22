"""
Example usage of RewardComputer for RL training.

This demonstrates how to use the different reward computation methods
for various RL training scenarios.
"""

import numpy as np
import sys
import importlib.util
from pathlib import Path

# Load reward_computer module directly to avoid torch dependency
reward_computer_path = Path(__file__).parent.parent / "reward_computer.py"
spec = importlib.util.spec_from_file_location("reward_computer", reward_computer_path)
reward_computer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reward_computer)

# Import classes from the loaded module
RewardComputer = reward_computer.RewardComputer
SVRLRewardComputer = reward_computer.SVRLRewardComputer
ParetoRewardComputer = reward_computer.ParetoRewardComputer
TrajectoryState = reward_computer.TrajectoryState
create_reward_computer = reward_computer.create_reward_computer


def example_1_basic_absolute_rewards():
    """Example 1: Basic absolute rewards for simple task scoring."""
    print("\n" + "="*60)
    print("Example 1: Basic Absolute Rewards")
    print("="*60)

    # Create a custom reward computer
    class SimpleTaskRewardComputer(RewardComputer):
        def score_trajectory(self, trajectory: TrajectoryState) -> float:
            # Score based on response quality and efficiency
            quality = trajectory.metadata.get('quality', 0.5)
            efficiency = 1.0 - (trajectory.step_count / 50)
            return quality + 0.2 * efficiency

    # Create mock trajectories
    trajectories = [
        TrajectoryState(step_count=10, response="Fast solution",
                       metadata={'quality': 0.7}),
        TrajectoryState(step_count=30, response="Slow solution",
                       metadata={'quality': 0.9}),
        TrajectoryState(step_count=20, response="Balanced solution",
                       metadata={'quality': 0.8}),
    ]

    # Compute rewards
    reward_computer = SimpleTaskRewardComputer(normalize=True)
    rewards = reward_computer.compute_rewards(trajectories, method="absolute")

    print("\nTrajectories and their absolute rewards:")
    for i, (traj, reward) in enumerate(zip(trajectories, rewards)):
        print(f"  {i}. {traj.response:20s} (steps={traj.step_count:2d}): "
              f"reward={reward:+.3f}")


def example_2_grpo_relative_rewards():
    """Example 2: GRPO-style relative ranking."""
    print("\n" + "="*60)
    print("Example 2: GRPO Relative Rewards")
    print("="*60)

    # Create custom reward computer
    class MathTaskRewardComputer(RewardComputer):
        def score_trajectory(self, trajectory: TrajectoryState) -> float:
            # Score based on correctness and reasoning steps
            correct = trajectory.metadata.get('correct', False)
            reasoning_quality = trajectory.metadata.get('reasoning_quality', 0.5)

            base_score = 1.0 if correct else 0.0
            return base_score + 0.3 * reasoning_quality

    # Create trajectories with varying quality
    trajectories = [
        TrajectoryState(step_count=15, response="Wrong answer",
                       metadata={'correct': False, 'reasoning_quality': 0.3}),
        TrajectoryState(step_count=20, response="Correct with good reasoning",
                       metadata={'correct': True, 'reasoning_quality': 0.8}),
        TrajectoryState(step_count=18, response="Correct with poor reasoning",
                       metadata={'correct': True, 'reasoning_quality': 0.4}),
        TrajectoryState(step_count=25, response="Wrong but good reasoning",
                       metadata={'correct': False, 'reasoning_quality': 0.7}),
    ]

    reward_computer = MathTaskRewardComputer(normalize=False)

    # Compare absolute vs relative rewards
    abs_rewards = reward_computer.compute_rewards(trajectories, method="absolute")
    rel_rewards = reward_computer.compute_rewards(trajectories, method="relative")

    print("\nComparison of absolute vs relative rewards:")
    print(f"{'Response':<35s} {'Absolute':>10s} {'Relative':>10s}")
    print("-" * 58)
    for traj, abs_r, rel_r in zip(trajectories, abs_rewards, rel_rewards):
        print(f"{traj.response:<35s} {abs_r:>10.3f} {rel_r:>10.3f}")

    print("\nInterpretation:")
    print("- Relative rewards rank trajectories against each other")
    print("- Best trajectory gets highest relative reward (positive)")
    print("- Worst trajectory gets lowest relative reward (negative)")
    print("- This is useful for GRPO where we only update on top-k trajectories")


def example_3_svrl_verification():
    """Example 3: SVRL with verification budget."""
    print("\n" + "="*60)
    print("Example 3: SVRL with Verification Budget")
    print("="*60)

    def task_evaluator(traj: TrajectoryState) -> float:
        """Evaluate if the task was completed correctly."""
        return traj.metadata.get('task_score', 0.0)

    # Create SVRL reward computer
    reward_computer = SVRLRewardComputer(
        task_evaluator=task_evaluator,
        verification_weight=0.1,
        efficiency_weight=0.05,
        normalize=False
    )

    # Create trajectories with different verification strategies
    trajectories = [
        # Over-verification: Used too much budget
        TrajectoryState(
            step_count=25,
            response="Solution with excessive verification",
            metadata={
                'task_score': 0.9,
                'verification_spent': 9,
                'initial_budget': 10,
                'max_steps': 50
            }
        ),
        # Efficient verification: Good balance
        TrajectoryState(
            step_count=15,
            response="Solution with balanced verification",
            metadata={
                'task_score': 0.85,
                'verification_spent': 3,
                'initial_budget': 10,
                'max_steps': 50
            }
        ),
        # No verification: Fast but lower quality
        TrajectoryState(
            step_count=8,
            response="Solution with no verification",
            metadata={
                'task_score': 0.7,
                'verification_spent': 0,
                'initial_budget': 10,
                'max_steps': 50
            }
        ),
    ]

    rewards = reward_computer.compute_rewards(trajectories, method="absolute")

    print("\nSVRL Rewards (balancing task quality, verification, and efficiency):")
    print(f"{'Response':<45s} {'Task':>6s} {'VerBudget':>10s} {'Steps':>6s} {'Reward':>8s}")
    print("-" * 80)
    for traj, reward in zip(trajectories, rewards):
        ver_used = traj.metadata['verification_spent']
        ver_budget = traj.metadata['initial_budget']
        task = traj.metadata['task_score']
        print(f"{traj.response:<45s} {task:>6.2f} {ver_used:>4d}/{ver_budget:<4d} "
              f"{traj.step_count:>6d} {reward:>8.3f}")


def example_4_pareto_multiobjective():
    """Example 4: Pareto-based multi-objective optimization."""
    print("\n" + "="*60)
    print("Example 4: Pareto Multi-Objective Optimization")
    print("="*60)

    # Define multiple objectives
    def accuracy_fn(traj: TrajectoryState) -> float:
        return traj.metadata.get('accuracy', 0.0)

    def safety_fn(traj: TrajectoryState) -> float:
        return traj.metadata.get('safety', 0.0)

    def efficiency_fn(traj: TrajectoryState) -> float:
        return 1.0 - (traj.step_count / 50)

    # Create Pareto reward computer
    reward_computer = ParetoRewardComputer(
        objective_functions=[accuracy_fn, safety_fn, efficiency_fn],
        objective_names=['accuracy', 'safety', 'efficiency'],
        normalize=False
    )

    # Create trajectories representing different trade-offs
    trajectories = [
        # High accuracy, moderate safety, low efficiency
        TrajectoryState(
            step_count=40,
            response="Very accurate but slow",
            metadata={'accuracy': 0.95, 'safety': 0.7}
        ),
        # Balanced across all objectives
        TrajectoryState(
            step_count=25,
            response="Balanced solution",
            metadata={'accuracy': 0.75, 'safety': 0.75}
        ),
        # High safety, moderate accuracy, high efficiency
        TrajectoryState(
            step_count=12,
            response="Safe and fast",
            metadata={'accuracy': 0.65, 'safety': 0.95}
        ),
        # Dominated solution (worse in all objectives than balanced)
        TrajectoryState(
            step_count=35,
            response="Poor solution",
            metadata={'accuracy': 0.6, 'safety': 0.5}
        ),
    ]

    # Get Pareto frontier
    frontier_indices = reward_computer.get_pareto_frontier_indices(trajectories)

    # Compute rewards
    rewards = reward_computer.compute_rewards(trajectories, method="pareto")

    print("\nMulti-Objective Scores and Pareto Frontier:")
    print(f"{'ID':>3s} {'Response':<25s} {'Accuracy':>10s} {'Safety':>8s} "
          f"{'Efficiency':>11s} {'Reward':>8s} {'Frontier':>9s}")
    print("-" * 85)

    for i, (traj, reward) in enumerate(zip(trajectories, rewards)):
        scores = reward_computer.score_trajectory_multiobjective(traj)
        on_frontier = "✓" if i in frontier_indices else ""
        print(f"{i:>3d} {traj.response:<25s} {scores[0]:>10.2f} {scores[1]:>8.2f} "
              f"{scores[2]:>11.2f} {reward:>8.3f} {on_frontier:>9s}")

    print("\nPareto Frontier Interpretation:")
    print("- Trajectories on the frontier represent optimal trade-offs")
    print("- No frontier trajectory is strictly better than another")
    print("- Dominated trajectories are worse in all objectives than at least one frontier point")

    # Show objective statistics
    stats = reward_computer.compute_objective_statistics(trajectories)
    print("\nObjective Statistics:")
    for obj_name, obj_stats in stats.items():
        print(f"  {obj_name:12s}: mean={obj_stats['mean']:.3f}, "
              f"std={obj_stats['std']:.3f}, "
              f"range=[{obj_stats['min']:.3f}, {obj_stats['max']:.3f}]")


def example_5_factory_pattern():
    """Example 5: Using the factory function."""
    print("\n" + "="*60)
    print("Example 5: Factory Pattern for Creating Reward Computers")
    print("="*60)

    # Example task evaluator
    def my_task_evaluator(traj: TrajectoryState) -> float:
        return traj.metadata.get('score', 0.5)

    # Create SVRL reward computer using factory
    svrl_rc = create_reward_computer(
        task_type="svrl",
        task_evaluator=my_task_evaluator,
        verification_weight=0.15,
        efficiency_weight=0.05
    )

    # Example objectives
    def obj1(traj): return traj.metadata.get('obj1', 0.5)
    def obj2(traj): return traj.metadata.get('obj2', 0.5)

    # Create Pareto reward computer using factory
    pareto_rc = create_reward_computer(
        task_type="pareto",
        objective_functions=[obj1, obj2],
        objective_names=['objective_1', 'objective_2']
    )

    print("\nCreated reward computers using factory pattern:")
    print(f"  SVRL: {type(svrl_rc).__name__}")
    print(f"  Pareto: {type(pareto_rc).__name__}")
    print("\nFactory pattern allows for easy configuration and instantiation")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("RewardComputer Usage Examples")
    print("="*60)

    example_1_basic_absolute_rewards()
    example_2_grpo_relative_rewards()
    example_3_svrl_verification()
    example_4_pareto_multiobjective()
    example_5_factory_pattern()

    print("\n" + "="*60)
    print("All examples completed successfully!")
    print("="*60)
