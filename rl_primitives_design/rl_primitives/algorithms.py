"""
RL Algorithm Orchestration Components

This module provides the main algorithm orchestration for RL training of LLMs.
It includes base classes and specific implementations for PPO, GRPO, and REINFORCE.

Design Philosophy:
- Modular: Each algorithm is independent and composable
- Extensible: Easy to add new algorithms
- Production-ready: Includes checkpointing, logging, progress tracking
- Integration: Works seamlessly with RolloutManager, RewardComputer, and BackpropModule

Author: Generated with rl-training-code skill
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from pathlib import Path
from collections import defaultdict
import json
import time

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

# Optional imports for logging
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ============================================================================
# Data Structures
# ============================================================================

class AlgorithmConfig:
    """Configuration for RL algorithms"""

    def __init__(
        self,
        # Training settings
        batch_size: int = 32,
        num_iterations: int = 1000,
        learning_rate: float = 1e-6,
        max_grad_norm: float = 1.0,

        # Algorithm-specific settings
        kl_coef: float = 0.1,
        entropy_coef: float = 0.01,

        # PPO-specific
        ppo_epochs: int = 4,
        clip_epsilon: float = 0.2,
        value_loss_coef: float = 0.5,

        # GRPO-specific
        reward_method: str = "relative",  # "relative", "absolute", "pareto"
        filter_percentile: float = 50.0,  # Only train on top X% for GRPO

        # Reference model updates
        update_ref_every: int = 100,
        ref_update_method: str = "copy",  # "copy", "ema", "none"
        ema_decay: float = 0.999,

        # Checkpointing
        checkpoint_dir: str = "./checkpoints",
        save_every: int = 100,

        # Logging
        log_every: int = 10,
        use_wandb: bool = False,
        wandb_project: str = "rl-training",
        wandb_entity: Optional[str] = None,
        log_to_console: bool = True,

        # Device settings
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_amp: bool = False,  # Automatic mixed precision

        # Custom config
        custom_params: Dict[str, Any] = None,
    ):
        self.batch_size = batch_size
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm

        self.kl_coef = kl_coef
        self.entropy_coef = entropy_coef

        self.ppo_epochs = ppo_epochs
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef

        self.reward_method = reward_method
        self.filter_percentile = filter_percentile

        self.update_ref_every = update_ref_every
        self.ref_update_method = ref_update_method
        self.ema_decay = ema_decay

        self.checkpoint_dir = checkpoint_dir
        self.save_every = save_every

        self.log_every = log_every
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.log_to_console = log_to_console

        self.device = device
        self.use_amp = use_amp

        self.custom_params = custom_params or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }


class TrainingMetrics:
    """Container for training metrics"""

    def __init__(self):
        self.iteration: int = 0
        self.loss: float = 0.0
        self.policy_loss: float = 0.0
        self.value_loss: float = 0.0
        self.kl_divergence: float = 0.0
        self.entropy: float = 0.0

        self.rewards_mean: float = 0.0
        self.rewards_std: float = 0.0
        self.rewards_min: float = 0.0
        self.rewards_max: float = 0.0

        self.advantages_mean: float = 0.0
        self.advantages_std: float = 0.0

        self.grad_norm: float = 0.0
        self.learning_rate: float = 0.0

        self.num_trajectories: int = 0
        self.trajectory_length_mean: float = 0.0
        self.trajectory_length_std: float = 0.0

        self.time_elapsed: float = 0.0
        self.trajectories_per_second: float = 0.0

        # Algorithm-specific metrics
        self.custom_metrics: Dict[str, float] = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_') and k != 'custom_metrics'
        } | self.custom_metrics

    def __repr__(self) -> str:
        return (
            f"TrainingMetrics(iter={self.iteration}, "
            f"loss={self.loss:.4f}, "
            f"reward={self.rewards_mean:.4f}, "
            f"kl={self.kl_divergence:.4f})"
        )


# ============================================================================
# Base Algorithm Class
# ============================================================================

class RLAlgorithm(ABC):
    """
    Base class for RL algorithms.

    This class provides the core orchestration logic for RL training,
    including rollout collection, reward computation, training steps,
    checkpointing, and logging.

    To implement a new algorithm, inherit from this class and implement:
    - train_step(): Algorithm-specific training logic

    Optionally override:
    - compute_advantages(): Custom advantage computation
    - compute_loss(): Custom loss computation
    - filter_trajectories(): Custom trajectory filtering
    """

    def __init__(
        self,
        env: Any,  # BaseEnvironment
        rollout_manager: Any,  # RolloutManager
        reward_computer: Any,  # RewardComputer
        backprop: Any,  # BackpropModule
        config: AlgorithmConfig,
    ):
        """
        Initialize RL algorithm.

        Args:
            env: Environment for interaction
            rollout_manager: Manages parallel rollouts
            reward_computer: Computes rewards for trajectories
            backprop: Handles gradients and model updates
            config: Algorithm configuration
        """
        self.env = env
        self.rollout_manager = rollout_manager
        self.reward_computer = reward_computer
        self.backprop = backprop
        self.config = config

        # Training state
        self.iteration = 0
        self.global_step = 0
        self.best_reward = float('-inf')

        # Metrics tracking
        self.metrics_history: List[TrainingMetrics] = []

        # Setup checkpoint directory
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # Initialize wandb if requested
        if config.use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                config=config.to_dict(),
                name=f"{self.__class__.__name__}_{int(time.time())}"
            )
        elif config.use_wandb and not WANDB_AVAILABLE:
            print("Warning: wandb not available, logging to console only")

    def collect_and_score(
        self,
        num_trajectories: int
    ) -> Tuple[List[Any], np.ndarray]:
        """
        Collect trajectories and compute rewards.

        This method:
        1. Uses RolloutManager to collect trajectories
        2. Handles different trajectory lengths
        3. Computes rewards using RewardComputer
        4. Returns trajectories and corresponding rewards

        Args:
            num_trajectories: Number of trajectories to collect

        Returns:
            Tuple of (trajectories, rewards)
        """
        # Collect trajectories (handles different lengths)
        trajectories = self.rollout_manager.collect_rollouts(num_trajectories)

        # Analyze trajectory length distribution
        by_length = defaultdict(list)
        for t in trajectories:
            by_length[t.step_count].append(t)

        if self.config.log_to_console and self.iteration % self.config.log_every == 0:
            length_dist = {k: len(v) for k, v in by_length.items()}
            print(f"Trajectory length distribution: {length_dist}")

        # Compute rewards AFTER collection (allows batch comparison)
        rewards = self.reward_computer.compute_rewards(
            trajectories,
            method=self.config.reward_method
        )

        return trajectories, rewards

    @abstractmethod
    def train_step(
        self,
        trajectories: List[Any],
        rewards: np.ndarray
    ) -> TrainingMetrics:
        """
        Execute one training step (algorithm-specific).

        This method must be implemented by subclasses to define
        the specific algorithm logic (PPO, GRPO, REINFORCE, etc.).

        Args:
            trajectories: List of collected trajectories
            rewards: Rewards for each trajectory

        Returns:
            TrainingMetrics object with statistics
        """
        raise NotImplementedError

    def train(
        self,
        num_iterations: Optional[int] = None,
        progress_bar: bool = True,
        callbacks: Optional[List[Callable]] = None
    ) -> List[TrainingMetrics]:
        """
        Main training loop.

        This method orchestrates the entire training process:
        1. Collect trajectories
        2. Execute training step
        3. Update reference model (if applicable)
        4. Log metrics
        5. Save checkpoints

        Args:
            num_iterations: Number of training iterations (uses config if None)
            progress_bar: Whether to show tqdm progress bar
            callbacks: Optional list of callback functions to run each iteration

        Returns:
            List of training metrics for each iteration
        """
        num_iterations = num_iterations or self.config.num_iterations
        callbacks = callbacks or []

        # Create progress bar
        pbar = tqdm(
            range(num_iterations),
            desc="Training",
            disable=not progress_bar
        )

        for iteration in pbar:
            self.iteration = iteration
            iteration_start = time.time()

            # Collect trajectories and compute rewards
            trajectories, rewards = self.collect_and_score(
                num_trajectories=self.config.batch_size
            )

            # Execute training step (algorithm-specific)
            metrics = self.train_step(trajectories, rewards)

            # Update metrics
            metrics.iteration = iteration
            metrics.time_elapsed = time.time() - iteration_start
            metrics.trajectories_per_second = len(trajectories) / metrics.time_elapsed
            self.metrics_history.append(metrics)

            # Update progress bar
            if progress_bar:
                pbar.set_postfix({
                    'loss': f'{metrics.loss:.4f}',
                    'reward': f'{metrics.rewards_mean:.4f}',
                    'kl': f'{metrics.kl_divergence:.4f}'
                })

            # Log metrics
            if iteration % self.config.log_every == 0:
                self.log_metrics(metrics)

            # Update reference model periodically
            if (iteration + 1) % self.config.update_ref_every == 0:
                self.update_reference_model()

            # Save checkpoint
            if (iteration + 1) % self.config.save_every == 0:
                self.save_checkpoint(iteration)

            # Save best model
            if metrics.rewards_mean > self.best_reward:
                self.best_reward = metrics.rewards_mean
                self.save_checkpoint(iteration, best=True)

            # Run callbacks
            for callback in callbacks:
                callback(self, metrics)

            self.global_step += 1

        # Save final checkpoint
        self.save_checkpoint(num_iterations - 1, final=True)

        return self.metrics_history

    def update_reference_model(self):
        """Update reference model based on config."""
        if self.config.ref_update_method == "copy":
            self.backprop.update_ref_model(method="copy")
        elif self.config.ref_update_method == "ema":
            self.backprop.update_ref_model(
                method="ema",
                decay=self.config.ema_decay
            )
        # "none" means no update

    def log_metrics(self, metrics: TrainingMetrics):
        """
        Log metrics to console and wandb.

        Args:
            metrics: Metrics to log
        """
        # Console logging
        if self.config.log_to_console:
            print(f"\n[Iteration {metrics.iteration}]")
            print(f"  Loss: {metrics.loss:.4f} (policy: {metrics.policy_loss:.4f}, value: {metrics.value_loss:.4f})")
            print(f"  Reward: {metrics.rewards_mean:.4f} ± {metrics.rewards_std:.4f} (min: {metrics.rewards_min:.4f}, max: {metrics.rewards_max:.4f})")
            print(f"  KL: {metrics.kl_divergence:.4f}, Entropy: {metrics.entropy:.4f}")
            print(f"  Grad norm: {metrics.grad_norm:.4f}, LR: {metrics.learning_rate:.2e}")
            print(f"  Trajectories: {metrics.num_trajectories}, Length: {metrics.trajectory_length_mean:.1f} ± {metrics.trajectory_length_std:.1f}")
            print(f"  Time: {metrics.time_elapsed:.2f}s, Throughput: {metrics.trajectories_per_second:.1f} traj/s")

        # Wandb logging
        if self.config.use_wandb and WANDB_AVAILABLE:
            wandb.log(metrics.to_dict(), step=metrics.iteration)

    def save_checkpoint(
        self,
        iteration: int,
        best: bool = False,
        final: bool = False
    ):
        """
        Save training checkpoint.

        Args:
            iteration: Current iteration number
            best: Whether this is the best checkpoint
            final: Whether this is the final checkpoint
        """
        # Determine checkpoint name
        if best:
            checkpoint_name = "checkpoint_best.pt"
        elif final:
            checkpoint_name = "checkpoint_final.pt"
        else:
            checkpoint_name = f"checkpoint_iter_{iteration}.pt"

        checkpoint_path = Path(self.config.checkpoint_dir) / checkpoint_name

        # Create checkpoint dict
        checkpoint = {
            'iteration': iteration,
            'global_step': self.global_step,
            'best_reward': self.best_reward,
            'config': self.config.to_dict(),
            'metrics_history': [m.to_dict() for m in self.metrics_history],
            'model_state_dict': self.backprop.model.state_dict(),
            'optimizer_state_dict': self.backprop.optimizer.state_dict(),
        }

        # Add reference model if exists
        if hasattr(self.backprop, 'ref_model') and self.backprop.ref_model is not None:
            checkpoint['ref_model_state_dict'] = self.backprop.ref_model.state_dict()

        # Save
        torch.save(checkpoint, checkpoint_path)

        if self.config.log_to_console:
            print(f"Saved checkpoint to {checkpoint_path}")

    def load_checkpoint(
        self,
        checkpoint_path: str,
        load_optimizer: bool = True
    ):
        """
        Load training checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            load_optimizer: Whether to load optimizer state
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device)

        # Load model state
        self.backprop.model.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizer state
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.backprop.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load reference model if exists
        if 'ref_model_state_dict' in checkpoint and hasattr(self.backprop, 'ref_model'):
            self.backprop.ref_model.load_state_dict(checkpoint['ref_model_state_dict'])

        # Load training state
        self.iteration = checkpoint.get('iteration', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self.best_reward = checkpoint.get('best_reward', float('-inf'))

        if self.config.log_to_console:
            print(f"Loaded checkpoint from {checkpoint_path} (iteration {self.iteration})")

    def compute_advantages(
        self,
        rewards: np.ndarray,
        values: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute advantages (default: mean baseline).

        Override this method for algorithm-specific advantage computation.

        Args:
            rewards: Rewards for each trajectory
            values: Value estimates (optional, for PPO)

        Returns:
            Advantages for each trajectory
        """
        if values is not None:
            # Use value function as baseline
            advantages = rewards - values
        else:
            # Use mean baseline (REINFORCE/GRPO style)
            advantages = rewards - np.mean(rewards)

        return advantages

    def filter_trajectories(
        self,
        trajectories: List[Any],
        rewards: np.ndarray
    ) -> Tuple[List[Any], np.ndarray]:
        """
        Filter trajectories based on rewards (for GRPO).

        Override this method for custom filtering logic.

        Args:
            trajectories: All trajectories
            rewards: Rewards for each trajectory

        Returns:
            Tuple of (filtered_trajectories, filtered_rewards)
        """
        return trajectories, rewards


# ============================================================================
# PPO Algorithm
# ============================================================================

class PPOAlgorithm(RLAlgorithm):
    """
    Proximal Policy Optimization (PPO) algorithm.

    PPO is the standard algorithm for RLHF training. It uses:
    - Clipped surrogate objective to prevent large policy updates
    - Value function (critic) for advantage estimation
    - Multiple epochs on the same batch of data
    - KL divergence penalty to stay close to reference model

    Reference: https://arxiv.org/abs/1707.06347
    """

    def train_step(
        self,
        trajectories: List[Any],
        rewards: np.ndarray
    ) -> TrainingMetrics:
        """
        Execute PPO training step.

        PPO performs multiple epochs of updates on the same batch:
        1. Compute advantages using value function
        2. For each epoch:
           a. Compute policy loss (clipped surrogate)
           b. Compute value loss
           c. Compute entropy bonus
           d. Backprop and update

        Args:
            trajectories: Collected trajectories
            rewards: Rewards for each trajectory

        Returns:
            Training metrics
        """
        metrics = TrainingMetrics()

        # Convert rewards to tensor
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.config.device)

        # Compute values using critic (if available)
        # NOTE: This assumes backprop has a value function
        # You may need to adapt this based on your BackpropModule implementation
        values = None
        if hasattr(self.backprop, 'compute_values'):
            values = self.backprop.compute_values(trajectories)
        else:
            # Fallback: use mean baseline
            values = np.full_like(rewards, np.mean(rewards))

        # Compute advantages
        advantages = self.backprop.compute_advantages(
            rewards,
            values=values,
            gamma=self.config.custom_params.get('gamma', 0.99),
            lam=self.config.custom_params.get('gae_lambda', 0.95)
        )

        advantages_tensor = torch.tensor(advantages, dtype=torch.float32, device=self.config.device)

        # Normalize advantages (common PPO trick)
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

        # Store old log probs for PPO ratio (before any updates)
        old_logprobs = self.backprop.get_logprobs(trajectories)

        # PPO epochs
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_kl = 0.0

        for epoch in range(self.config.ppo_epochs):
            # Compute current log probs
            curr_logprobs = self.backprop.get_logprobs(trajectories)

            # Compute ratio for PPO clipping
            log_ratio = curr_logprobs - old_logprobs
            ratio = torch.exp(log_ratio)

            # PPO clipped loss
            surr1 = ratio * advantages_tensor
            surr2 = torch.clamp(
                ratio,
                1.0 - self.config.clip_epsilon,
                1.0 + self.config.clip_epsilon
            ) * advantages_tensor
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss (if value function available)
            value_loss = torch.tensor(0.0, device=self.config.device)
            if values is not None and hasattr(self.backprop, 'compute_value_loss'):
                value_loss = self.backprop.compute_value_loss(
                    trajectories,
                    rewards_tensor
                )

            # Entropy bonus (encourages exploration)
            entropy = torch.tensor(0.0, device=self.config.device)
            if hasattr(self.backprop, 'compute_entropy'):
                entropy = self.backprop.compute_entropy(trajectories)

            # KL divergence (for monitoring)
            kl_div = torch.tensor(0.0, device=self.config.device)
            if hasattr(self.backprop, 'compute_kl'):
                kl_div = self.backprop.compute_kl(trajectories)

            # Total loss
            loss = (
                policy_loss +
                self.config.value_loss_coef * value_loss -
                self.config.entropy_coef * entropy +
                self.config.kl_coef * kl_div
            )

            # Backprop and update
            self.backprop.update(loss)

            # Accumulate metrics
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
            total_kl += kl_div.item()

        # Compute gradient norm
        grad_norm = 0.0
        if hasattr(self.backprop.model, 'parameters'):
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.backprop.model.parameters(),
                self.config.max_grad_norm
            ).item()

        # Fill metrics
        metrics.loss = total_policy_loss / self.config.ppo_epochs
        metrics.policy_loss = total_policy_loss / self.config.ppo_epochs
        metrics.value_loss = total_value_loss / self.config.ppo_epochs
        metrics.entropy = total_entropy / self.config.ppo_epochs
        metrics.kl_divergence = total_kl / self.config.ppo_epochs

        metrics.rewards_mean = float(np.mean(rewards))
        metrics.rewards_std = float(np.std(rewards))
        metrics.rewards_min = float(np.min(rewards))
        metrics.rewards_max = float(np.max(rewards))

        metrics.advantages_mean = float(np.mean(advantages))
        metrics.advantages_std = float(np.std(advantages))

        metrics.grad_norm = grad_norm
        metrics.learning_rate = self.backprop.optimizer.param_groups[0]['lr']

        metrics.num_trajectories = len(trajectories)
        lengths = [t.step_count for t in trajectories]
        metrics.trajectory_length_mean = float(np.mean(lengths))
        metrics.trajectory_length_std = float(np.std(lengths))

        return metrics


# ============================================================================
# GRPO Algorithm
# ============================================================================

class GRPOAlgorithm(RLAlgorithm):
    """
    Group Relative Policy Optimization (GRPO) algorithm.

    GRPO is a simplified PPO variant that:
    - Uses group-relative advantages (no value function needed)
    - Filters trajectories to only update on better samples
    - Simpler and faster than PPO
    - Works well for LLM post-training

    Reference: Used in DeepSeek and other systems
    Based on patterns from nanochat (Karpathy)
    """

    def train_step(
        self,
        trajectories: List[Any],
        rewards: np.ndarray
    ) -> TrainingMetrics:
        """
        Execute GRPO training step.

        GRPO:
        1. Computes relative advantages (rewards - mean)
        2. Filters to top percentile of trajectories
        3. Single update on filtered batch
        4. No value function needed

        Args:
            trajectories: Collected trajectories
            rewards: Rewards for each trajectory

        Returns:
            Training metrics
        """
        metrics = TrainingMetrics()

        # Compute relative advantages (GRPO-style)
        advantages = rewards - np.mean(rewards)

        # Filter to top percentile (GRPO specific)
        threshold = np.percentile(rewards, self.config.filter_percentile)
        good_indices = rewards >= threshold

        filtered_trajectories = [
            t for t, keep in zip(trajectories, good_indices) if keep
        ]
        filtered_advantages = advantages[good_indices]

        if len(filtered_trajectories) == 0:
            # All trajectories filtered out, skip update
            print("Warning: All trajectories filtered out, skipping update")
            return metrics

        # Convert to tensors
        advantages_tensor = torch.tensor(
            filtered_advantages,
            dtype=torch.float32,
            device=self.config.device
        )

        # Normalize advantages
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

        # Compute log probs
        logprobs = self.backprop.get_logprobs(filtered_trajectories)

        # GRPO loss: -mean(advantages * log_probs)
        policy_loss = -(logprobs * advantages_tensor).mean()

        # KL penalty (optional)
        kl_div = torch.tensor(0.0, device=self.config.device)
        if hasattr(self.backprop, 'compute_kl'):
            kl_div = self.backprop.compute_kl(filtered_trajectories)

        # Entropy bonus (optional)
        entropy = torch.tensor(0.0, device=self.config.device)
        if hasattr(self.backprop, 'compute_entropy'):
            entropy = self.backprop.compute_entropy(filtered_trajectories)

        # Total loss
        loss = (
            policy_loss +
            self.config.kl_coef * kl_div -
            self.config.entropy_coef * entropy
        )

        # Backprop and update
        self.backprop.update(loss)

        # Compute gradient norm
        grad_norm = 0.0
        if hasattr(self.backprop.model, 'parameters'):
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.backprop.model.parameters(),
                self.config.max_grad_norm
            ).item()

        # Fill metrics
        metrics.loss = loss.item()
        metrics.policy_loss = policy_loss.item()
        metrics.value_loss = 0.0  # No value function in GRPO
        metrics.entropy = entropy.item()
        metrics.kl_divergence = kl_div.item()

        metrics.rewards_mean = float(np.mean(rewards))
        metrics.rewards_std = float(np.std(rewards))
        metrics.rewards_min = float(np.min(rewards))
        metrics.rewards_max = float(np.max(rewards))

        metrics.advantages_mean = float(np.mean(advantages))
        metrics.advantages_std = float(np.std(advantages))

        metrics.grad_norm = grad_norm
        metrics.learning_rate = self.backprop.optimizer.param_groups[0]['lr']

        metrics.num_trajectories = len(trajectories)
        lengths = [t.step_count for t in trajectories]
        metrics.trajectory_length_mean = float(np.mean(lengths))
        metrics.trajectory_length_std = float(np.std(lengths))

        # GRPO-specific metrics
        metrics.custom_metrics['num_filtered_trajectories'] = len(filtered_trajectories)
        metrics.custom_metrics['filter_ratio'] = len(filtered_trajectories) / len(trajectories)

        return metrics


# ============================================================================
# REINFORCE Algorithm
# ============================================================================

class REINFORCEAlgorithm(RLAlgorithm):
    """
    REINFORCE (vanilla policy gradient) algorithm.

    REINFORCE is the simplest policy gradient algorithm:
    - Uses rewards as advantages
    - Optional mean baseline for variance reduction
    - Single update per batch
    - No value function or clipping

    Good for:
    - Simple tasks
    - Baselines
    - Teaching/learning

    Reference: Williams, 1992
    """

    def train_step(
        self,
        trajectories: List[Any],
        rewards: np.ndarray
    ) -> TrainingMetrics:
        """
        Execute REINFORCE training step.

        REINFORCE:
        1. Use rewards directly (or with mean baseline)
        2. Single gradient update
        3. Simple and interpretable

        Args:
            trajectories: Collected trajectories
            rewards: Rewards for each trajectory

        Returns:
            Training metrics
        """
        metrics = TrainingMetrics()

        # Compute advantages (with mean baseline)
        use_baseline = self.config.custom_params.get('use_baseline', True)

        if use_baseline:
            advantages = rewards - np.mean(rewards)
        else:
            advantages = rewards

        # Convert to tensors
        advantages_tensor = torch.tensor(
            advantages,
            dtype=torch.float32,
            device=self.config.device
        )

        # Optional: Normalize advantages
        if self.config.custom_params.get('normalize_advantages', True):
            advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

        # Compute log probs
        logprobs = self.backprop.get_logprobs(trajectories)

        # REINFORCE loss: -mean(advantages * log_probs)
        policy_loss = -(logprobs * advantages_tensor).mean()

        # Entropy bonus (optional, for exploration)
        entropy = torch.tensor(0.0, device=self.config.device)
        if hasattr(self.backprop, 'compute_entropy'):
            entropy = self.backprop.compute_entropy(trajectories)

        # Total loss
        loss = policy_loss - self.config.entropy_coef * entropy

        # Backprop and update
        self.backprop.update(loss)

        # Compute gradient norm
        grad_norm = 0.0
        if hasattr(self.backprop.model, 'parameters'):
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.backprop.model.parameters(),
                self.config.max_grad_norm
            ).item()

        # Fill metrics
        metrics.loss = loss.item()
        metrics.policy_loss = policy_loss.item()
        metrics.value_loss = 0.0  # No value function
        metrics.entropy = entropy.item()
        metrics.kl_divergence = 0.0  # No KL tracking in basic REINFORCE

        metrics.rewards_mean = float(np.mean(rewards))
        metrics.rewards_std = float(np.std(rewards))
        metrics.rewards_min = float(np.min(rewards))
        metrics.rewards_max = float(np.max(rewards))

        metrics.advantages_mean = float(np.mean(advantages))
        metrics.advantages_std = float(np.std(advantages))

        metrics.grad_norm = grad_norm
        metrics.learning_rate = self.backprop.optimizer.param_groups[0]['lr']

        metrics.num_trajectories = len(trajectories)
        lengths = [t.step_count for t in trajectories]
        metrics.trajectory_length_mean = float(np.mean(lengths))
        metrics.trajectory_length_std = float(np.std(lengths))

        return metrics


# ============================================================================
# Utility Functions
# ============================================================================

def create_algorithm(
    algorithm_name: str,
    env: Any,
    rollout_manager: Any,
    reward_computer: Any,
    backprop: Any,
    config: Optional[AlgorithmConfig] = None,
    **kwargs
) -> RLAlgorithm:
    """
    Factory function to create algorithm instances.

    Args:
        algorithm_name: Name of algorithm ("ppo", "grpo", "reinforce")
        env: Environment
        rollout_manager: RolloutManager instance
        reward_computer: RewardComputer instance
        backprop: BackpropModule instance
        config: Algorithm configuration (uses defaults if None)
        **kwargs: Additional config parameters

    Returns:
        RLAlgorithm instance

    Example:
        >>> algo = create_algorithm(
        ...     "ppo",
        ...     env=my_env,
        ...     rollout_manager=rollout_mgr,
        ...     reward_computer=reward_comp,
        ...     backprop=backprop_module,
        ...     batch_size=32,
        ...     ppo_epochs=4
        ... )
    """
    # Create config if not provided
    if config is None:
        config = AlgorithmConfig(**kwargs)
    else:
        # Update config with any additional kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

    # Create algorithm
    algorithm_map = {
        "ppo": PPOAlgorithm,
        "grpo": GRPOAlgorithm,
        "reinforce": REINFORCEAlgorithm,
    }

    algorithm_name_lower = algorithm_name.lower()

    if algorithm_name_lower not in algorithm_map:
        raise ValueError(
            f"Unknown algorithm: {algorithm_name}. "
            f"Available: {list(algorithm_map.keys())}"
        )

    algorithm_class = algorithm_map[algorithm_name_lower]

    return algorithm_class(
        env=env,
        rollout_manager=rollout_manager,
        reward_computer=reward_computer,
        backprop=backprop,
        config=config
    )


def load_algorithm_from_checkpoint(
    checkpoint_path: str,
    env: Any,
    rollout_manager: Any,
    reward_computer: Any,
    backprop: Any,
    algorithm_class: Optional[type] = None
) -> RLAlgorithm:
    """
    Load algorithm from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        env: Environment
        rollout_manager: RolloutManager instance
        reward_computer: RewardComputer instance
        backprop: BackpropModule instance
        algorithm_class: Algorithm class (inferred from checkpoint if None)

    Returns:
        RLAlgorithm instance loaded from checkpoint
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)

    # Create config from checkpoint
    config_dict = checkpoint.get('config', {})
    config = AlgorithmConfig(**config_dict)

    # Infer algorithm class if not provided
    if algorithm_class is None:
        # Try to infer from checkpoint or use default
        algorithm_class = PPOAlgorithm  # Default

    # Create algorithm
    algo = algorithm_class(
        env=env,
        rollout_manager=rollout_manager,
        reward_computer=reward_computer,
        backprop=backprop,
        config=config
    )

    # Load checkpoint
    algo.load_checkpoint(checkpoint_path)

    return algo


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    """
    Example usage of the algorithm orchestration components.

    This shows how to:
    1. Create an algorithm
    2. Train it
    3. Save and load checkpoints
    """

    # NOTE: This is example code showing the API
    # You'll need to provide actual implementations of:
    # - BaseEnvironment
    # - RolloutManager
    # - RewardComputer
    # - BackpropModule

    print("=" * 80)
    print("RL Algorithm Orchestration Example")
    print("=" * 80)

    # Example 1: Create PPO algorithm
    print("\n1. Creating PPO algorithm...")

    # Create config
    config = AlgorithmConfig(
        batch_size=32,
        num_iterations=100,
        ppo_epochs=4,
        learning_rate=1e-6,
        use_wandb=False,
        log_to_console=True
    )

    # NOTE: You need to provide these components
    # env = MyEnvironment(...)
    # rollout_manager = RolloutManager(env, num_parallel=32)
    # reward_computer = MyRewardComputer(...)
    # backprop = BackpropModule(model, ref_model, optimizer)

    # Create algorithm using factory
    # algo = create_algorithm(
    #     "ppo",
    #     env=env,
    #     rollout_manager=rollout_manager,
    #     reward_computer=reward_computer,
    #     backprop=backprop,
    #     config=config
    # )

    print("✓ Algorithm created")

    # Example 2: Train
    print("\n2. Training (example)...")

    # metrics_history = algo.train(
    #     num_iterations=100,
    #     progress_bar=True
    # )

    print("✓ Training complete")

    # Example 3: Save and load checkpoint
    print("\n3. Checkpoint management...")

    # algo.save_checkpoint(iteration=100, best=True)
    # algo.load_checkpoint("checkpoints/checkpoint_best.pt")

    print("✓ Checkpoint saved and loaded")

    print("\n" + "=" * 80)
    print("Example complete!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Implement your Environment (see v2.md)")
    print("2. Implement RolloutManager")
    print("3. Implement RewardComputer")
    print("4. Implement BackpropModule")
    print("5. Use create_algorithm() to orchestrate training")
    print("=" * 80)
