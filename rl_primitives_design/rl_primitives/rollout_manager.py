"""
RolloutManager: Production-ready rollout collection for RL training.

This module provides efficient parallel rollout management with support for:
- Variable-length trajectories
- Batched inference for efficiency
- Progress tracking and logging
- Memory-efficient storage
- Distributed rollouts (future-proof)

Author: Generated using rl-training-code skill
Date: 2025-11-22
"""

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch


# Type aliases for clarity
StateType = Any  # Generic state type - will be defined by environment
RolloutID = int


@dataclass
class RolloutStats:
    """Statistics for a single rollout."""

    rollout_id: RolloutID
    start_time: float
    end_time: Optional[float] = None
    num_steps: int = 0
    total_tokens: int = 0

    @property
    def duration(self) -> Optional[float]:
        """Compute rollout duration if completed."""
        if self.end_time is not None:
            return self.end_time - self.start_time
        return None

    @property
    def steps_per_second(self) -> Optional[float]:
        """Compute steps per second if completed."""
        duration = self.duration
        if duration and duration > 0:
            return self.num_steps / duration
        return None


@dataclass
class BatchStats:
    """Statistics for a batch of rollouts."""

    batch_id: int
    start_time: float
    end_time: Optional[float] = None
    num_rollouts: int = 0
    completed_rollouts: int = 0
    total_steps: int = 0
    total_tokens: int = 0

    # Track trajectory lengths
    trajectory_lengths: List[int] = field(default_factory=list)

    @property
    def duration(self) -> Optional[float]:
        """Compute batch duration if completed."""
        if self.end_time is not None:
            return self.end_time - self.start_time
        return None

    @property
    def avg_trajectory_length(self) -> float:
        """Average trajectory length."""
        if self.trajectory_lengths:
            return np.mean(self.trajectory_lengths)
        return 0.0

    @property
    def throughput(self) -> Optional[float]:
        """Trajectories per second."""
        duration = self.duration
        if duration and duration > 0:
            return self.completed_rollouts / duration
        return None

    def summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            "batch_id": self.batch_id,
            "num_rollouts": self.num_rollouts,
            "completed": self.completed_rollouts,
            "total_steps": self.total_steps,
            "total_tokens": self.total_tokens,
            "duration": self.duration,
            "throughput": self.throughput,
            "avg_length": self.avg_trajectory_length,
            "min_length": min(self.trajectory_lengths) if self.trajectory_lengths else 0,
            "max_length": max(self.trajectory_lengths) if self.trajectory_lengths else 0,
        }


@dataclass
class RolloutConfig:
    """Configuration for RolloutManager."""

    # Parallel execution
    num_parallel: int = 32
    maintain_parallel_count: bool = False  # Replace completed rollouts immediately

    # Batching configuration
    max_batch_size: int = 32  # Maximum batch size for inference
    adaptive_batching: bool = True  # Dynamically adjust batch size

    # Memory management
    max_trajectory_buffer: int = 1000  # Max trajectories to keep in memory
    offload_to_disk: bool = False  # Offload to disk if buffer full

    # Logging and monitoring
    log_interval: int = 10  # Log every N steps
    verbose: bool = True
    track_stats: bool = True

    # Distributed settings (future-proof)
    distributed: bool = False
    rank: int = 0
    world_size: int = 1

    # Progress tracking
    enable_progress_bar: bool = True

    def __post_init__(self):
        """Validate configuration."""
        assert self.num_parallel > 0, "num_parallel must be positive"
        assert self.max_batch_size > 0, "max_batch_size must be positive"
        assert self.max_trajectory_buffer > 0, "max_trajectory_buffer must be positive"


class RolloutManager:
    """
    Manages parallel rollout collection with variable-length trajectories.

    This class handles the complexity of running multiple trajectories in parallel
    with different completion times, while maintaining efficient batched inference.

    Key Features:
    - Parallel rollout execution with async completion
    - Batched inference for efficiency
    - Variable-length trajectory support
    - Progress tracking and statistics
    - Memory-efficient storage
    - Future-proof for distributed execution

    Example:
        >>> env = MyEnvironment(inference_module)
        >>> config = RolloutConfig(num_parallel=32, verbose=True)
        >>> manager = RolloutManager(env, config)
        >>>
        >>> # Collect 100 trajectories
        >>> trajectories = manager.collect_rollouts(min_trajectories=100)
        >>>
        >>> # Access statistics
        >>> stats = manager.get_statistics()
        >>> print(f"Average trajectory length: {stats['avg_trajectory_length']}")

    Args:
        env: Environment instance (must implement reset(), step(), is_terminal(), etc.)
        config: RolloutConfig instance for configuration
        logger: Optional logger instance
    """

    def __init__(
        self,
        env: Any,  # BaseEnvironment type
        config: Optional[RolloutConfig] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """Initialize RolloutManager."""
        self.env = env
        self.config = config or RolloutConfig()
        self.logger = logger or self._setup_logger()

        # Active rollouts: {rollout_id: state}
        self.active_rollouts: Dict[RolloutID, StateType] = {}

        # Completed trajectories buffer
        self.completed_buffer: deque = deque(maxlen=self.config.max_trajectory_buffer)

        # Statistics tracking
        self.rollout_stats: Dict[RolloutID, RolloutStats] = {}
        self.batch_stats: List[BatchStats] = []
        self.current_batch_stats: Optional[BatchStats] = None

        # Counters
        self.next_rollout_id: RolloutID = 0
        self.batch_counter: int = 0
        self.total_steps: int = 0
        self.total_trajectories: int = 0

        # Callbacks (for monitoring and custom logic)
        self.callbacks: Dict[str, List[Callable]] = defaultdict(list)

        self.logger.info(f"RolloutManager initialized with config: {self.config}")

    def _setup_logger(self) -> logging.Logger:
        """Setup default logger."""
        logger = logging.getLogger("RolloutManager")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO if self.config.verbose else logging.WARNING)
        return logger

    def register_callback(self, event: str, callback: Callable) -> None:
        """
        Register a callback for specific events.

        Supported events:
        - 'on_rollout_start': Called when a new rollout starts
        - 'on_rollout_complete': Called when a rollout completes
        - 'on_batch_start': Called when a new batch starts
        - 'on_batch_complete': Called when a batch completes
        - 'on_step': Called after each step

        Args:
            event: Event name
            callback: Callable to invoke
        """
        self.callbacks[event].append(callback)
        self.logger.debug(f"Registered callback for event: {event}")

    def _trigger_callbacks(self, event: str, *args, **kwargs) -> None:
        """Trigger all callbacks for an event."""
        for callback in self.callbacks.get(event, []):
            try:
                callback(*args, **kwargs)
            except Exception as e:
                self.logger.error(f"Callback error for {event}: {e}")

    def initialize_batch(self, num_rollouts: Optional[int] = None) -> Dict[RolloutID, StateType]:
        """
        Start a new batch of rollouts.

        Args:
            num_rollouts: Number of rollouts to start (default: config.num_parallel)

        Returns:
            Dictionary mapping rollout IDs to initial states
        """
        num_rollouts = num_rollouts or self.config.num_parallel

        self.logger.info(f"Initializing batch {self.batch_counter} with {num_rollouts} rollouts")

        # Clear active rollouts
        self.active_rollouts = {}

        # Initialize batch statistics
        self.current_batch_stats = BatchStats(
            batch_id=self.batch_counter,
            start_time=time.time(),
            num_rollouts=num_rollouts,
        )

        # Start new rollouts
        for i in range(num_rollouts):
            rollout_id = self.next_rollout_id
            self.next_rollout_id += 1

            # Reset environment to get initial state
            initial_state = self.env.reset()
            self.active_rollouts[rollout_id] = initial_state

            # Initialize rollout statistics
            if self.config.track_stats:
                self.rollout_stats[rollout_id] = RolloutStats(
                    rollout_id=rollout_id,
                    start_time=time.time(),
                )

            # Trigger callback
            self._trigger_callbacks('on_rollout_start', rollout_id, initial_state)

        # Trigger batch start callback
        self._trigger_callbacks('on_batch_start', self.batch_counter, num_rollouts)

        self.batch_counter += 1

        return self.active_rollouts

    def batch_step(self, states: List[StateType]) -> List[StateType]:
        """
        Step multiple states efficiently using batched inference.

        This method handles batched environment stepping, which is crucial for
        efficient inference with LLMs. The environment should implement batched
        operations internally.

        Args:
            states: List of states to step

        Returns:
            List of new states after stepping
        """
        if not states:
            return []

        # Check if environment has batch_step method
        if hasattr(self.env, 'batch_step'):
            # Use environment's native batch stepping
            new_states = self.env.batch_step(states)
        else:
            # Fall back to sequential stepping (less efficient)
            self.logger.warning(
                "Environment doesn't implement batch_step. "
                "Using sequential stepping (inefficient)."
            )
            new_states = [self.env.step(state) for state in states]

        return new_states

    def step_active(self) -> Dict[str, Any]:
        """
        Step all active rollouts and handle completions.

        This method:
        1. Batches all active states for efficient inference
        2. Steps them through the environment
        3. Checks for terminal states
        4. Moves completed rollouts to buffer
        5. Optionally starts new rollouts to maintain parallel count

        Returns:
            Dictionary with:
                - 'active': List of active rollout IDs
                - 'just_completed': List of (rollout_id, final_state) tuples
                - 'num_active': Number of active rollouts
                - 'num_completed': Number of just-completed rollouts
        """
        if not self.active_rollouts:
            return {
                "active": [],
                "just_completed": [],
                "num_active": 0,
                "num_completed": 0,
            }

        # Get active rollout IDs and states
        active_ids = list(self.active_rollouts.keys())
        active_states = [self.active_rollouts[rid] for rid in active_ids]

        # Batch step all active states
        new_states = self.batch_step(active_states)

        # Process results
        just_completed = []
        still_active = {}

        for rollout_id, new_state in zip(active_ids, new_states):
            # Update statistics
            if self.config.track_stats and rollout_id in self.rollout_stats:
                self.rollout_stats[rollout_id].num_steps += 1
                # Track tokens if state has this info
                if hasattr(new_state, 'num_tokens'):
                    self.rollout_stats[rollout_id].total_tokens += new_state.num_tokens

            # Check if terminal
            if self.env.is_terminal(new_state):
                # Rollout completed
                just_completed.append((rollout_id, new_state))
                self.completed_buffer.append(new_state)

                # Finalize statistics
                if self.config.track_stats and rollout_id in self.rollout_stats:
                    stats = self.rollout_stats[rollout_id]
                    stats.end_time = time.time()

                    # Update batch stats
                    if self.current_batch_stats:
                        self.current_batch_stats.completed_rollouts += 1
                        self.current_batch_stats.total_steps += stats.num_steps
                        self.current_batch_stats.total_tokens += stats.total_tokens
                        self.current_batch_stats.trajectory_lengths.append(stats.num_steps)

                self.total_trajectories += 1

                # Trigger callback
                self._trigger_callbacks('on_rollout_complete', rollout_id, new_state)

                # Log completion
                if self.config.verbose and rollout_id in self.rollout_stats:
                    stats = self.rollout_stats[rollout_id]
                    self.logger.info(
                        f"Rollout {rollout_id} completed: "
                        f"{stats.num_steps} steps in {stats.duration:.2f}s "
                        f"({stats.steps_per_second:.2f} steps/s)"
                    )
            else:
                # Still active
                still_active[rollout_id] = new_state

        # Update active rollouts
        self.active_rollouts = still_active

        # Optionally start new rollouts to maintain parallel count
        if self.config.maintain_parallel_count and just_completed:
            num_to_start = len(just_completed)
            self.logger.debug(f"Starting {num_to_start} new rollouts to maintain parallel count")

            for _ in range(num_to_start):
                rollout_id = self.next_rollout_id
                self.next_rollout_id += 1

                initial_state = self.env.reset()
                self.active_rollouts[rollout_id] = initial_state

                if self.config.track_stats:
                    self.rollout_stats[rollout_id] = RolloutStats(
                        rollout_id=rollout_id,
                        start_time=time.time(),
                    )

                self._trigger_callbacks('on_rollout_start', rollout_id, initial_state)

        # Update global step counter
        self.total_steps += len(active_ids)

        # Trigger step callback
        self._trigger_callbacks('on_step', self.total_steps, len(still_active))

        # Periodic logging
        if self.config.verbose and self.total_steps % self.config.log_interval == 0:
            self.logger.info(
                f"Step {self.total_steps}: "
                f"{len(still_active)} active, "
                f"{len(just_completed)} just completed, "
                f"{self.total_trajectories} total trajectories"
            )

        return {
            "active": list(still_active.keys()),
            "just_completed": just_completed,
            "num_active": len(still_active),
            "num_completed": len(just_completed),
        }

    def collect_rollouts(
        self,
        min_trajectories: int,
        max_steps_per_rollout: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> List[StateType]:
        """
        Collect rollouts until minimum number of trajectories is reached.

        This is the main method for collecting training data. It handles:
        - Starting batches of parallel rollouts
        - Stepping them until completion
        - Handling variable-length trajectories
        - Progress tracking and logging

        Args:
            min_trajectories: Minimum number of trajectories to collect
            max_steps_per_rollout: Maximum steps per rollout (optional timeout)
            timeout: Overall timeout in seconds (optional)

        Returns:
            List of completed trajectory states
        """
        self.logger.info(
            f"Starting rollout collection: target={min_trajectories} trajectories"
        )

        all_trajectories = []
        start_time = time.time()

        # Progress tracking
        if self.config.enable_progress_bar:
            try:
                from tqdm import tqdm
                pbar = tqdm(total=min_trajectories, desc="Collecting rollouts")
            except ImportError:
                self.logger.warning("tqdm not available, disabling progress bar")
                pbar = None
        else:
            pbar = None

        while len(all_trajectories) < min_trajectories:
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                self.logger.warning(
                    f"Collection timeout reached. "
                    f"Collected {len(all_trajectories)}/{min_trajectories} trajectories"
                )
                break

            # Initialize new batch
            self.initialize_batch()

            batch_steps = 0

            # Run until all rollouts in this batch complete
            while self.active_rollouts:
                # Check max steps per rollout
                if max_steps_per_rollout and batch_steps >= max_steps_per_rollout:
                    self.logger.warning(
                        f"Max steps per rollout reached ({max_steps_per_rollout}). "
                        f"Terminating {len(self.active_rollouts)} active rollouts."
                    )
                    # Force terminate remaining rollouts
                    for rollout_id, state in self.active_rollouts.items():
                        all_trajectories.append(state)
                        if pbar:
                            pbar.update(1)
                    self.active_rollouts = {}
                    break

                # Step all active rollouts
                step_info = self.step_active()

                # Add completed trajectories
                for _, final_state in step_info["just_completed"]:
                    all_trajectories.append(final_state)
                    if pbar:
                        pbar.update(1)

                batch_steps += 1

                # Early exit if we have enough
                if len(all_trajectories) >= min_trajectories:
                    # Terminate remaining active rollouts
                    for rollout_id, state in self.active_rollouts.items():
                        self.logger.debug(
                            f"Early terminating rollout {rollout_id} "
                            f"(have enough trajectories)"
                        )
                    self.active_rollouts = {}
                    break

            # Finalize batch statistics
            if self.current_batch_stats:
                self.current_batch_stats.end_time = time.time()
                self.batch_stats.append(self.current_batch_stats)

                # Log batch summary
                if self.config.verbose:
                    summary = self.current_batch_stats.summary()
                    self.logger.info(f"Batch {summary['batch_id']} completed: {summary}")

                # Trigger callback
                self._trigger_callbacks(
                    'on_batch_complete',
                    self.current_batch_stats.batch_id,
                    self.current_batch_stats
                )

        if pbar:
            pbar.close()

        total_time = time.time() - start_time

        self.logger.info(
            f"Collection complete: {len(all_trajectories)} trajectories "
            f"in {total_time:.2f}s ({len(all_trajectories)/total_time:.2f} traj/s)"
        )

        return all_trajectories

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about rollout collection.

        Returns:
            Dictionary with various statistics including:
                - Total trajectories collected
                - Total steps taken
                - Average trajectory length
                - Throughput metrics
                - Batch statistics
        """
        if not self.batch_stats:
            return {
                "total_trajectories": self.total_trajectories,
                "total_steps": self.total_steps,
                "num_batches": 0,
            }

        all_lengths = []
        for batch in self.batch_stats:
            all_lengths.extend(batch.trajectory_lengths)

        return {
            "total_trajectories": self.total_trajectories,
            "total_steps": self.total_steps,
            "num_batches": len(self.batch_stats),
            "avg_trajectory_length": np.mean(all_lengths) if all_lengths else 0.0,
            "std_trajectory_length": np.std(all_lengths) if all_lengths else 0.0,
            "min_trajectory_length": min(all_lengths) if all_lengths else 0,
            "max_trajectory_length": max(all_lengths) if all_lengths else 0,
            "total_duration": sum(b.duration for b in self.batch_stats if b.duration),
            "avg_batch_throughput": np.mean([
                b.throughput for b in self.batch_stats if b.throughput
            ]) if self.batch_stats else 0.0,
            "trajectory_length_distribution": self._get_length_distribution(all_lengths),
        }

    def _get_length_distribution(self, lengths: List[int]) -> Dict[str, int]:
        """Get distribution of trajectory lengths."""
        if not lengths:
            return {}

        distribution = {}
        for length in lengths:
            distribution[length] = distribution.get(length, 0) + 1

        return distribution

    def reset(self) -> None:
        """Reset the manager to initial state."""
        self.logger.info("Resetting RolloutManager")

        self.active_rollouts = {}
        self.completed_buffer.clear()
        self.rollout_stats = {}
        self.batch_stats = []
        self.current_batch_stats = None

        self.next_rollout_id = 0
        self.batch_counter = 0
        self.total_steps = 0
        self.total_trajectories = 0

    def save_statistics(self, filepath: str) -> None:
        """
        Save statistics to a file.

        Args:
            filepath: Path to save statistics (JSON format)
        """
        import json

        stats = self.get_statistics()

        # Add batch details
        stats["batches"] = [batch.summary() for batch in self.batch_stats]

        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2)

        self.logger.info(f"Statistics saved to {filepath}")

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"RolloutManager("
            f"num_parallel={self.config.num_parallel}, "
            f"active={len(self.active_rollouts)}, "
            f"completed={self.total_trajectories}, "
            f"steps={self.total_steps})"
        )

    # Future-proof methods for distributed execution

    def sync_distributed(self) -> None:
        """
        Synchronize rollouts across distributed workers.

        This is a placeholder for future distributed support.
        Will implement proper synchronization when distributed
        execution is needed.
        """
        if not self.config.distributed:
            return

        # TODO: Implement distributed synchronization
        # - Gather completed trajectories from all workers
        # - Redistribute work if needed
        # - Synchronize statistics

        raise NotImplementedError(
            "Distributed rollout collection not yet implemented. "
            "Set config.distributed=False for now."
        )

    def get_worker_stats(self) -> Dict[str, Any]:
        """
        Get statistics for this worker in distributed setting.

        Returns worker-specific statistics for aggregation.
        """
        if not self.config.distributed:
            return self.get_statistics()

        stats = self.get_statistics()
        stats["rank"] = self.config.rank
        stats["world_size"] = self.config.world_size

        return stats


# Utility functions

def create_rollout_manager(
    env: Any,
    num_parallel: int = 32,
    maintain_parallel: bool = False,
    verbose: bool = True,
    **kwargs
) -> RolloutManager:
    """
    Convenience function to create a RolloutManager with common settings.

    Args:
        env: Environment instance
        num_parallel: Number of parallel rollouts
        maintain_parallel: Whether to maintain parallel count
        verbose: Enable verbose logging
        **kwargs: Additional config parameters

    Returns:
        Configured RolloutManager instance
    """
    config = RolloutConfig(
        num_parallel=num_parallel,
        maintain_parallel_count=maintain_parallel,
        verbose=verbose,
        **kwargs
    )

    return RolloutManager(env, config)


if __name__ == "__main__":
    # Example usage and testing
    print("RolloutManager module loaded successfully!")
    print("\nExample usage:")
    print("""
    from rl_primitives.rollout_manager import RolloutManager, RolloutConfig

    # Create configuration
    config = RolloutConfig(
        num_parallel=32,
        maintain_parallel_count=False,
        verbose=True,
        track_stats=True,
    )

    # Initialize manager
    manager = RolloutManager(env, config)

    # Collect trajectories
    trajectories = manager.collect_rollouts(min_trajectories=100)

    # Get statistics
    stats = manager.get_statistics()
    print(f"Collected {len(trajectories)} trajectories")
    print(f"Average length: {stats['avg_trajectory_length']:.2f}")

    # Save statistics
    manager.save_statistics("rollout_stats.json")
    """)
