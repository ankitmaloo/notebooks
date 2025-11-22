"""
RewardComputer Module for RL Training

This module provides reward computation for RL training with support for:
- Absolute rewards: Direct trajectory scoring
- Relative rewards: GRPO-style ranking
- Pareto rewards: Multi-objective Pareto ranking

Based on the architecture defined in v2.md.

References:
- Fast Pareto frontier computation: https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
- pymoo library: https://pymoo.org/
- Multi-objective optimization: https://en.wikipedia.org/wiki/Multi-objective_optimization
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass
import numpy as np
from scipy import stats
from scipy.spatial import distance


@dataclass
class TrajectoryState:
    """
    Represents a completed trajectory state.

    This is a flexible container that environments can extend with
    their own fields.

    Attributes:
        step_count: Number of steps in the trajectory
        response: Final response or action
        done: Whether trajectory is complete
        metadata: Additional trajectory-specific data
    """
    step_count: int
    response: Any
    done: bool = True
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class RewardComputer(ABC):
    """
    Base class for computing rewards after collecting trajectories.

    This separates reward computation from environment logic, allowing
    for batch-wise comparison of trajectories (essential for GRPO and
    Pareto-based methods).

    Usage:
        reward_computer = SVRLRewardComputer(task_evaluator=my_evaluator)
        trajectories = rollout_manager.collect_rollouts(32)
        rewards = reward_computer.compute_rewards(trajectories, method="relative")
    """

    def __init__(self, normalize: bool = True):
        """
        Initialize RewardComputer.

        Args:
            normalize: Whether to normalize rewards (z-score normalization)
        """
        self.normalize = normalize

    def compute_rewards(
        self,
        trajectories: List[TrajectoryState],
        method: str = "absolute"
    ) -> np.ndarray:
        """
        Compute rewards for a batch of completed trajectories.

        This is the main entry point for reward computation. It supports
        multiple methods for computing rewards based on the training algorithm.

        Args:
            trajectories: List of completed trajectory states
            method: Reward computation method. One of:
                - "absolute": Direct scoring of each trajectory
                - "relative": GRPO-style ranking (compares trajectories)
                - "pareto": Multi-objective Pareto ranking

        Returns:
            Array of rewards with shape (num_trajectories,)

        Raises:
            ValueError: If method is not recognized
        """
        if not trajectories:
            return np.array([])

        if method == "absolute":
            rewards = self.absolute_rewards(trajectories)
        elif method == "relative":
            rewards = self.relative_rewards(trajectories)
        elif method == "pareto":
            rewards = self.pareto_rewards(trajectories)
        else:
            raise ValueError(
                f"Unknown reward method: {method}. "
                f"Must be one of: absolute, relative, pareto"
            )

        # Optional normalization
        if self.normalize and len(rewards) > 1:
            rewards = self._normalize_rewards(rewards)

        return rewards

    def absolute_rewards(self, trajectories: List[TrajectoryState]) -> np.ndarray:
        """
        Compute absolute rewards by scoring each trajectory independently.

        This is the standard approach where each trajectory is scored
        based on its own merits, without comparison to others.

        Args:
            trajectories: List of completed trajectory states

        Returns:
            Array of rewards with shape (num_trajectories,)
        """
        rewards = np.array([
            self.score_trajectory(trajectory)
            for trajectory in trajectories
        ])
        return rewards

    def relative_rewards(self, trajectories: List[TrajectoryState]) -> np.ndarray:
        """
        Compute relative rewards using GRPO-style ranking.

        This method ranks trajectories against each other and converts
        ranks to normalized rewards. This is the approach used in GRPO
        and other rank-based RL algorithms.

        The ranking process:
        1. Score all trajectories
        2. Rank them (best = highest rank)
        3. Normalize ranks to have mean=0, std=1

        Args:
            trajectories: List of completed trajectory states

        Returns:
            Array of normalized rank-based rewards with shape (num_trajectories,)
        """
        # Score each trajectory
        scores = np.array([
            self.score_trajectory(trajectory)
            for trajectory in trajectories
        ])

        # Convert scores to ranks (1 = worst, n = best)
        # ties are averaged
        ranks = stats.rankdata(scores, method='average')

        # Normalize ranks to mean=0, std=1
        if len(ranks) > 1:
            normalized_ranks = (ranks - ranks.mean()) / (ranks.std() + 1e-8)
        else:
            normalized_ranks = np.zeros_like(ranks)

        return normalized_ranks

    def pareto_rewards(self, trajectories: List[TrajectoryState]) -> np.ndarray:
        """
        Compute rewards based on Pareto dominance in multi-objective space.

        This method:
        1. Evaluates each trajectory on multiple objectives
        2. Computes Pareto frontier (non-dominated points)
        3. Assigns rewards based on Pareto rank

        Trajectories on the Pareto frontier get highest rewards.
        Dominated trajectories get rewards based on their distance
        to the frontier.

        Args:
            trajectories: List of completed trajectory states

        Returns:
            Array of Pareto-based rewards with shape (num_trajectories,)
        """
        # Get multi-dimensional objectives for each trajectory
        # Shape: (num_trajectories, num_objectives)
        objectives = np.array([
            self.score_trajectory_multiobjective(trajectory)
            for trajectory in trajectories
        ])

        # Compute Pareto ranks (0 = on frontier, 1 = dominated by frontier, etc.)
        pareto_ranks = self._compute_pareto_ranks(objectives)

        # Convert ranks to rewards (lower rank = higher reward)
        # Rank 0 (frontier) gets reward 1.0
        # Each subsequent rank gets progressively lower reward
        max_rank = pareto_ranks.max()
        if max_rank > 0:
            rewards = 1.0 - (pareto_ranks / (max_rank + 1))
        else:
            rewards = np.ones_like(pareto_ranks, dtype=float)

        return rewards

    @abstractmethod
    def score_trajectory(self, trajectory: TrajectoryState) -> float:
        """
        Score a single trajectory (for absolute/relative rewards).

        This method should be implemented by subclasses to define
        how individual trajectories are scored.

        Args:
            trajectory: A completed trajectory state

        Returns:
            Scalar score (higher is better)

        Example:
            def score_trajectory(self, trajectory):
                task_reward = self.evaluate_task_completion(trajectory)
                efficiency_penalty = -0.1 * trajectory.step_count
                return task_reward + efficiency_penalty
        """
        raise NotImplementedError("Subclasses must implement score_trajectory()")

    def score_trajectory_multiobjective(
        self,
        trajectory: TrajectoryState
    ) -> np.ndarray:
        """
        Score a trajectory on multiple objectives (for Pareto rewards).

        Default implementation returns single-objective score.
        Override this for true multi-objective optimization.

        Args:
            trajectory: A completed trajectory state

        Returns:
            Array of objective values with shape (num_objectives,)
            Higher values are better for all objectives.

        Example:
            def score_trajectory_multiobjective(self, trajectory):
                return np.array([
                    self.accuracy_score(trajectory),
                    self.efficiency_score(trajectory),
                    self.safety_score(trajectory)
                ])
        """
        # Default: convert single score to multi-objective
        return np.array([self.score_trajectory(trajectory)])

    def _normalize_rewards(self, rewards: np.ndarray) -> np.ndarray:
        """
        Normalize rewards to have mean=0, std=1.

        Args:
            rewards: Array of rewards

        Returns:
            Normalized rewards
        """
        mean = rewards.mean()
        std = rewards.std()

        if std > 1e-8:
            return (rewards - mean) / std
        else:
            return rewards - mean

    def _compute_pareto_ranks(self, objectives: np.ndarray) -> np.ndarray:
        """
        Compute Pareto ranks for multi-objective optimization.

        This uses an efficient iterative algorithm to compute ranks:
        - Rank 0: Points on the Pareto frontier (non-dominated)
        - Rank 1: Points dominated only by rank 0
        - Rank 2: Points dominated only by rank 0 or 1
        - etc.

        Args:
            objectives: Array of shape (n_points, n_objectives)
                       Higher values are better for all objectives.

        Returns:
            Array of Pareto ranks with shape (n_points,)
        """
        n_points = len(objectives)
        ranks = np.zeros(n_points, dtype=int)
        remaining_indices = np.arange(n_points)
        remaining_objectives = objectives.copy()

        current_rank = 0

        while len(remaining_indices) > 0:
            # Find Pareto frontier in remaining points
            frontier_mask = self._find_pareto_frontier(remaining_objectives)
            frontier_indices = remaining_indices[frontier_mask]

            # Assign current rank to frontier points
            ranks[frontier_indices] = current_rank

            # Remove frontier points and continue
            remaining_indices = remaining_indices[~frontier_mask]
            remaining_objectives = remaining_objectives[~frontier_mask]

            current_rank += 1

            # Safety check to prevent infinite loops
            if current_rank > n_points:
                break

        return ranks

    def _find_pareto_frontier(
        self,
        objectives: np.ndarray,
        return_mask: bool = True
    ) -> np.ndarray:
        """
        Find Pareto-efficient (non-dominated) points.

        A point is Pareto-efficient if no other point dominates it.
        Point A dominates point B if A is >= B in all objectives and
        strictly > B in at least one objective.

        This uses an efficient iterative algorithm that's much faster
        than naive O(n^2) comparison.

        Algorithm from: https://stackoverflow.com/questions/32791911

        Args:
            objectives: Array of shape (n_points, n_objectives)
                       Higher values are better for all objectives.
            return_mask: If True, return boolean mask. Otherwise return indices.

        Returns:
            Boolean mask or indices of Pareto-efficient points
        """
        if len(objectives) == 0:
            return np.array([], dtype=bool) if return_mask else np.array([], dtype=int)

        # For maximization, we want to find points that are NOT dominated
        # Point i is dominated if there exists point j where:
        #   objectives[j] >= objectives[i] in all dimensions AND
        #   objectives[j] > objectives[i] in at least one dimension

        n_points = objectives.shape[0]
        is_efficient = np.ones(n_points, dtype=bool)

        for i in range(n_points):
            if is_efficient[i]:
                # Keep points that are not dominated by point i
                # A point j is NOT dominated by i if:
                #   objectives[j] > objectives[i] in at least one dimension
                is_efficient[is_efficient] = np.any(
                    objectives[is_efficient] > objectives[i],
                    axis=1
                )
                # Point i itself is efficient
                is_efficient[i] = True

        if return_mask:
            return is_efficient
        else:
            return np.where(is_efficient)[0]

    def _distance_to_pareto_frontier(
        self,
        point: np.ndarray,
        frontier_points: np.ndarray
    ) -> float:
        """
        Compute minimum distance from a point to the Pareto frontier.

        Args:
            point: Single point with shape (n_objectives,)
            frontier_points: Frontier points with shape (n_frontier, n_objectives)

        Returns:
            Minimum Euclidean distance to frontier
        """
        if len(frontier_points) == 0:
            return 0.0

        distances = distance.cdist(
            point.reshape(1, -1),
            frontier_points,
            metric='euclidean'
        )
        return distances.min()


class SVRLRewardComputer(RewardComputer):
    """
    Reward computer for Self-Verification RL tasks.

    This computer is designed for tasks where the agent must balance:
    - Task completion quality
    - Verification cost (budget spent on verification)
    - Efficiency (steps taken)

    Example usage:
        reward_computer = SVRLRewardComputer(
            task_evaluator=my_task_eval_function,
            verification_weight=0.1,
            efficiency_weight=0.05
        )

        trajectories = rollout_manager.collect_rollouts(32)
        rewards = reward_computer.compute_rewards(trajectories, method="relative")
    """

    def __init__(
        self,
        task_evaluator: Callable[[TrajectoryState], float],
        verification_weight: float = 0.1,
        efficiency_weight: float = 0.05,
        normalize: bool = True
    ):
        """
        Initialize SVRL reward computer.

        Args:
            task_evaluator: Function that evaluates task completion quality.
                           Should return a score in [0, 1] where 1 is perfect.
            verification_weight: Weight for verification cost penalty
            efficiency_weight: Weight for efficiency bonus (fewer steps = better)
            normalize: Whether to normalize final rewards
        """
        super().__init__(normalize=normalize)
        self.task_evaluator = task_evaluator
        self.verification_weight = verification_weight
        self.efficiency_weight = efficiency_weight

    def score_trajectory(self, trajectory: TrajectoryState) -> float:
        """
        Score a single SVRL trajectory.

        The score combines:
        1. Task completion quality (primary objective)
        2. Verification efficiency (penalty for excessive verification)
        3. Step efficiency (bonus for completing quickly)

        Args:
            trajectory: Completed trajectory state

        Returns:
            Scalar score combining all objectives
        """
        # Primary objective: task quality
        task_score = self.task_evaluator(trajectory)

        # Verification efficiency
        # Assume metadata contains verification info
        verification_spent = trajectory.metadata.get('verification_spent', 0)
        initial_budget = trajectory.metadata.get('initial_budget', 1)

        if initial_budget > 0:
            verification_penalty = -self.verification_weight * (
                verification_spent / initial_budget
            )
        else:
            verification_penalty = 0.0

        # Step efficiency (bonus for early completion)
        max_steps = trajectory.metadata.get('max_steps', 50)
        step_efficiency_bonus = self.efficiency_weight * (
            max(0, (max_steps - trajectory.step_count) / max_steps)
        )

        total_score = task_score + verification_penalty + step_efficiency_bonus

        return total_score

    def score_trajectory_multiobjective(
        self,
        trajectory: TrajectoryState
    ) -> np.ndarray:
        """
        Score SVRL trajectory on multiple objectives.

        Returns:
            Array with [task_quality, verification_efficiency, step_efficiency]
        """
        task_score = self.task_evaluator(trajectory)

        verification_spent = trajectory.metadata.get('verification_spent', 0)
        initial_budget = trajectory.metadata.get('initial_budget', 1)
        verification_efficiency = 1.0 - (verification_spent / initial_budget)

        max_steps = trajectory.metadata.get('max_steps', 50)
        step_efficiency = (max_steps - trajectory.step_count) / max_steps

        return np.array([
            task_score,
            verification_efficiency,
            step_efficiency
        ])


class ParetoRewardComputer(RewardComputer):
    """
    Reward computer for multi-objective optimization using Pareto ranking.

    This computer is designed for tasks with multiple competing objectives
    where there's no single "best" solution, only trade-offs.

    Example usage:
        def accuracy_fn(traj):
            return evaluate_accuracy(traj.response)

        def efficiency_fn(traj):
            return 1.0 - (traj.step_count / 50)

        def safety_fn(traj):
            return evaluate_safety(traj.response)

        reward_computer = ParetoRewardComputer(
            objective_functions=[accuracy_fn, efficiency_fn, safety_fn],
            objective_names=["accuracy", "efficiency", "safety"]
        )

        trajectories = rollout_manager.collect_rollouts(32)
        rewards = reward_computer.compute_rewards(trajectories, method="pareto")
    """

    def __init__(
        self,
        objective_functions: List[Callable[[TrajectoryState], float]],
        objective_names: Optional[List[str]] = None,
        weights: Optional[np.ndarray] = None,
        normalize: bool = True
    ):
        """
        Initialize Pareto reward computer.

        Args:
            objective_functions: List of functions that each evaluate one objective.
                                Each function should return a scalar score (higher is better).
            objective_names: Optional names for objectives (for logging/debugging)
            weights: Optional weights for combining objectives in single-objective mode.
                    If None, uses uniform weights.
            normalize: Whether to normalize final rewards
        """
        super().__init__(normalize=normalize)
        self.objective_functions = objective_functions
        self.num_objectives = len(objective_functions)

        if objective_names is None:
            self.objective_names = [f"objective_{i}" for i in range(self.num_objectives)]
        else:
            assert len(objective_names) == self.num_objectives
            self.objective_names = objective_names

        if weights is None:
            self.weights = np.ones(self.num_objectives) / self.num_objectives
        else:
            assert len(weights) == self.num_objectives
            self.weights = np.array(weights) / np.sum(weights)  # Normalize weights

    def score_trajectory(self, trajectory: TrajectoryState) -> float:
        """
        Score trajectory using weighted combination of objectives.

        This is used for absolute/relative reward methods.
        For true Pareto optimization, use method="pareto" which calls
        score_trajectory_multiobjective instead.

        Args:
            trajectory: Completed trajectory state

        Returns:
            Weighted combination of all objectives
        """
        objectives = self.score_trajectory_multiobjective(trajectory)
        return np.dot(objectives, self.weights)

    def score_trajectory_multiobjective(
        self,
        trajectory: TrajectoryState
    ) -> np.ndarray:
        """
        Evaluate trajectory on all objectives.

        Args:
            trajectory: Completed trajectory state

        Returns:
            Array of objective values with shape (num_objectives,)
        """
        objectives = np.array([
            obj_fn(trajectory)
            for obj_fn in self.objective_functions
        ])
        return objectives

    def get_pareto_frontier_indices(
        self,
        trajectories: List[TrajectoryState]
    ) -> np.ndarray:
        """
        Get indices of trajectories on the Pareto frontier.

        Useful for analysis and visualization.

        Args:
            trajectories: List of completed trajectory states

        Returns:
            Array of indices corresponding to frontier trajectories
        """
        objectives = np.array([
            self.score_trajectory_multiobjective(trajectory)
            for trajectory in trajectories
        ])

        return self._find_pareto_frontier(objectives, return_mask=False)

    def compute_objective_statistics(
        self,
        trajectories: List[TrajectoryState]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute statistics for each objective across all trajectories.

        Useful for logging and monitoring training progress.

        Args:
            trajectories: List of completed trajectory states

        Returns:
            Dictionary mapping objective names to their statistics
            (mean, std, min, max)
        """
        objectives = np.array([
            self.score_trajectory_multiobjective(trajectory)
            for trajectory in trajectories
        ])

        stats = {}
        for i, name in enumerate(self.objective_names):
            obj_values = objectives[:, i]
            stats[name] = {
                'mean': float(obj_values.mean()),
                'std': float(obj_values.std()),
                'min': float(obj_values.min()),
                'max': float(obj_values.max())
            }

        return stats


# ============================================================================
# Example usage and utilities
# ============================================================================

def create_reward_computer(
    task_type: str,
    **kwargs
) -> RewardComputer:
    """
    Factory function to create appropriate reward computer.

    Args:
        task_type: Type of task ("svrl", "pareto", or "custom")
        **kwargs: Additional arguments passed to the reward computer

    Returns:
        Configured reward computer instance

    Example:
        # For SVRL
        rc = create_reward_computer(
            "svrl",
            task_evaluator=my_eval_fn,
            verification_weight=0.1
        )

        # For Pareto
        rc = create_reward_computer(
            "pareto",
            objective_functions=[acc_fn, eff_fn, safety_fn]
        )
    """
    if task_type == "svrl":
        return SVRLRewardComputer(**kwargs)
    elif task_type == "pareto":
        return ParetoRewardComputer(**kwargs)
    elif task_type == "custom":
        # User must provide a custom RewardComputer subclass
        custom_class = kwargs.pop('reward_computer_class')
        return custom_class(**kwargs)
    else:
        raise ValueError(
            f"Unknown task_type: {task_type}. "
            f"Must be one of: svrl, pareto, custom"
        )


if __name__ == "__main__":
    """
    Example usage and tests.
    """
    print("RewardComputer Module - Example Usage\n")

    # Create mock trajectories
    print("Creating mock trajectories...")
    trajectories = [
        TrajectoryState(
            step_count=10,
            response="Solution A",
            metadata={
                'verification_spent': 5,
                'initial_budget': 10,
                'max_steps': 50,
                'accuracy': 0.9,
                'efficiency': 0.8
            }
        ),
        TrajectoryState(
            step_count=20,
            response="Solution B",
            metadata={
                'verification_spent': 8,
                'initial_budget': 10,
                'max_steps': 50,
                'accuracy': 0.95,
                'efficiency': 0.6
            }
        ),
        TrajectoryState(
            step_count=15,
            response="Solution C",
            metadata={
                'verification_spent': 3,
                'initial_budget': 10,
                'max_steps': 50,
                'accuracy': 0.85,
                'efficiency': 0.7
            }
        ),
    ]

    # Example 1: SVRL Reward Computer
    print("\n" + "="*60)
    print("Example 1: SVRL Reward Computer")
    print("="*60)

    def mock_task_evaluator(traj: TrajectoryState) -> float:
        return traj.metadata.get('accuracy', 0.0)

    svrl_rc = SVRLRewardComputer(
        task_evaluator=mock_task_evaluator,
        verification_weight=0.1,
        efficiency_weight=0.05
    )

    for method in ['absolute', 'relative']:
        rewards = svrl_rc.compute_rewards(trajectories, method=method)
        print(f"\n{method.capitalize()} rewards:")
        for i, (traj, reward) in enumerate(zip(trajectories, rewards)):
            print(f"  Trajectory {i}: {reward:.4f} (steps={traj.step_count})")

    # Example 2: Pareto Reward Computer
    print("\n" + "="*60)
    print("Example 2: Pareto Reward Computer")
    print("="*60)

    def accuracy_fn(traj: TrajectoryState) -> float:
        return traj.metadata.get('accuracy', 0.0)

    def efficiency_fn(traj: TrajectoryState) -> float:
        return traj.metadata.get('efficiency', 0.0)

    def verification_efficiency_fn(traj: TrajectoryState) -> float:
        spent = traj.metadata.get('verification_spent', 0)
        budget = traj.metadata.get('initial_budget', 1)
        return 1.0 - (spent / budget)

    pareto_rc = ParetoRewardComputer(
        objective_functions=[accuracy_fn, efficiency_fn, verification_efficiency_fn],
        objective_names=['accuracy', 'efficiency', 'verification_efficiency']
    )

    # Compute multiobjective scores
    print("\nObjective scores:")
    for i, traj in enumerate(trajectories):
        scores = pareto_rc.score_trajectory_multiobjective(traj)
        print(f"  Trajectory {i}: {scores}")

    # Compute Pareto rewards
    pareto_rewards = pareto_rc.compute_rewards(trajectories, method="pareto")
    print("\nPareto rewards:")
    for i, reward in enumerate(pareto_rewards):
        print(f"  Trajectory {i}: {reward:.4f}")

    # Get Pareto frontier
    frontier_indices = pareto_rc.get_pareto_frontier_indices(trajectories)
    print(f"\nPareto frontier indices: {frontier_indices}")

    # Get objective statistics
    stats = pareto_rc.compute_objective_statistics(trajectories)
    print("\nObjective statistics:")
    for obj_name, obj_stats in stats.items():
        print(f"  {obj_name}:")
        for stat_name, stat_value in obj_stats.items():
            print(f"    {stat_name}: {stat_value:.4f}")

    print("\n" + "="*60)
    print("Examples complete!")
    print("="*60)
