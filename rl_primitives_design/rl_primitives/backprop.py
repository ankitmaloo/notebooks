"""
BackpropModule - Handles gradient computation and model updates for RL training

This module provides a unified interface for different RL algorithms (PPO, GRPO, REINFORCE)
with support for:
- Generalized Advantage Estimation (GAE)
- PPO clipped loss
- KL divergence computation
- Reference model management
- Gradient clipping
- Multiple update strategies

Based on:
- PPO: https://arxiv.org/abs/1707.06347
- GAE: https://arxiv.org/abs/1506.02438
- GRPO: https://arxiv.org/abs/2402.03300
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, List, Dict, Any, Tuple, Literal
from dataclasses import dataclass
import numpy as np


@dataclass
class BackpropConfig:
    """Configuration for BackpropModule"""
    # Advantage computation
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # GAE lambda parameter

    # PPO settings
    ppo_clip_range: float = 0.2  # PPO clipping parameter
    ppo_epochs: int = 4  # Number of PPO update epochs

    # Loss coefficients
    value_loss_coef: float = 0.1  # Value function loss coefficient
    entropy_coef: float = 0.01  # Entropy bonus coefficient
    kl_coef: float = 0.1  # KL divergence penalty coefficient

    # Optimization
    max_grad_norm: Optional[float] = 1.0  # Gradient clipping (None to disable)

    # Reference model updates
    ref_update_method: Literal["copy", "ema"] = "copy"
    ema_alpha: float = 0.99  # For EMA updates

    # Algorithm-specific
    normalize_advantages: bool = True  # Normalize advantages
    clip_value_loss: bool = True  # Clip value loss (PPO style)


class BackpropModule:
    """
    Handles all gradient computation and model updates for RL training.

    Supports multiple RL algorithms:
    - PPO (Proximal Policy Optimization) with value function
    - GRPO (Group Relative Policy Optimization) without value function
    - REINFORCE with optional baseline

    Example:
        >>> backprop = BackpropModule(
        ...     model=policy_model,
        ...     ref_model=reference_model,
        ...     optimizer=optimizer,
        ...     config=BackpropConfig()
        ... )
        >>>
        >>> # Compute advantages
        >>> advantages = backprop.compute_advantages(rewards, values)
        >>>
        >>> # Compute and apply PPO loss
        >>> loss_dict = backprop.compute_ppo_loss(trajectories, advantages)
        >>> backprop.update(loss_dict['total_loss'])
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        ref_model: Optional[nn.Module] = None,
        value_model: Optional[nn.Module] = None,
        config: Optional[BackpropConfig] = None,
    ):
        """
        Initialize BackpropModule.

        Args:
            model: The policy model to train
            optimizer: Optimizer for the policy model
            ref_model: Reference model for KL divergence (optional, defaults to copy of model)
            value_model: Value function model for actor-critic methods (optional)
            config: Configuration object (uses defaults if None)
        """
        self.model = model
        self.optimizer = optimizer
        self.value_model = value_model
        self.config = config or BackpropConfig()

        # Initialize reference model
        if ref_model is None:
            # Create a copy of the model as reference
            self.ref_model = self._create_ref_model(model)
        else:
            self.ref_model = ref_model

        # Set reference model to eval mode
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

    def _create_ref_model(self, model: nn.Module) -> nn.Module:
        """Create a reference model by copying the policy model"""
        import copy
        ref_model = copy.deepcopy(model)
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False
        return ref_model

    # ============================================================================
    # Advantage Computation
    # ============================================================================

    def compute_advantages(
        self,
        rewards: Union[torch.Tensor, np.ndarray, List[float]],
        values: Optional[Union[torch.Tensor, np.ndarray, List[float]]] = None,
        dones: Optional[Union[torch.Tensor, np.ndarray, List[bool]]] = None,
        gamma: Optional[float] = None,
        lam: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Compute advantages using GAE or simple cumulative rewards.

        If values are provided, uses Generalized Advantage Estimation (GAE).
        Otherwise, uses simple discounted cumulative rewards (REINFORCE-style).

        Args:
            rewards: Rewards for each timestep, shape (batch_size, num_steps) or (num_steps,)
            values: Value estimates for each timestep (optional), same shape as rewards
            dones: Done flags for each timestep (optional), same shape as rewards
            gamma: Discount factor (uses config.gamma if None)
            lam: GAE lambda parameter (uses config.gae_lambda if None)

        Returns:
            advantages: Computed advantages, same shape as rewards

        References:
            - GAE paper: https://arxiv.org/abs/1506.02438
            - Implementation: https://nn.labml.ai/rl/ppo/gae.html
        """
        gamma = gamma if gamma is not None else self.config.gamma
        lam = lam if lam is not None else self.config.gae_lambda

        # Convert to tensors
        rewards = self._to_tensor(rewards)

        if values is None:
            # Simple discounted cumulative rewards (REINFORCE / GRPO)
            advantages = self._discount_cumsum(rewards, gamma)
        else:
            # Generalized Advantage Estimation (PPO)
            values = self._to_tensor(values)
            dones = self._to_tensor(dones) if dones is not None else torch.zeros_like(rewards)
            advantages = self._compute_gae(rewards, values, dones, gamma, lam)

        # Normalize advantages if configured
        if self.config.normalize_advantages:
            advantages = self._normalize(advantages)

        return advantages

    def _compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        gamma: float,
        lam: float,
    ) -> torch.Tensor:
        """
        Compute Generalized Advantage Estimation.

        GAE formula:
            A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
        where:
            δ_t = r_t + γV(s_{t+1}) - V(s_t)

        Args:
            rewards: Rewards tensor, shape (batch_size, num_steps)
            values: Value predictions, shape (batch_size, num_steps + 1)
            dones: Done flags, shape (batch_size, num_steps)
            gamma: Discount factor
            lam: GAE lambda parameter

        Returns:
            advantages: GAE advantages, shape (batch_size, num_steps)
        """
        batch_size, num_steps = rewards.shape
        advantages = torch.zeros_like(rewards)

        # Compute TD residuals: δ_t = r_t + γV(s_{t+1})(1-done) - V(s_t)
        next_values = values[:, 1:]  # V(s_{t+1})
        current_values = values[:, :-1]  # V(s_t)

        deltas = rewards + gamma * next_values * (1 - dones) - current_values

        # Compute GAE using reverse iteration
        gae = 0
        for t in reversed(range(num_steps)):
            gae = deltas[:, t] + gamma * lam * (1 - dones[:, t]) * gae
            advantages[:, t] = gae

        return advantages

    def _discount_cumsum(
        self,
        rewards: torch.Tensor,
        gamma: float,
    ) -> torch.Tensor:
        """
        Compute discounted cumulative sum of rewards.

        Returns[t] = rewards[t] + gamma * rewards[t+1] + gamma^2 * rewards[t+2] + ...

        Args:
            rewards: Rewards tensor, shape (batch_size, num_steps) or (num_steps,)
            gamma: Discount factor

        Returns:
            discounted_returns: Discounted cumulative rewards, same shape as rewards
        """
        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        batch_size, num_steps = rewards.shape
        returns = torch.zeros_like(rewards)

        running_return = 0
        for t in reversed(range(num_steps)):
            running_return = rewards[:, t] + gamma * running_return
            returns[:, t] = running_return

        if squeeze:
            returns = returns.squeeze(0)

        return returns

    def compute_relative_advantages(
        self,
        rewards: Union[torch.Tensor, List[float]],
        group_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Compute group-relative advantages (GRPO-style).

        For each group of samples, advantages are:
            A_i = r_i - mean(r_group)

        This is used in GRPO where multiple samples are generated per prompt,
        and we want to reinforce samples that are better than the group average.

        Args:
            rewards: Rewards, shape (batch_size,) or (num_groups, group_size)
            group_size: Size of each group (if rewards are flat)

        Returns:
            advantages: Relative advantages, same shape as rewards

        References:
            - GRPO paper: https://arxiv.org/abs/2402.03300
            - nanochat implementation
        """
        rewards = self._to_tensor(rewards)

        if group_size is not None and rewards.dim() == 1:
            # Reshape flat rewards into groups
            rewards = rewards.view(-1, group_size)

        # Compute mean per group
        if rewards.dim() == 2:
            group_means = rewards.mean(dim=1, keepdim=True)
            advantages = rewards - group_means
        else:
            # Single group
            advantages = rewards - rewards.mean()

        return advantages

    # ============================================================================
    # KL Divergence
    # ============================================================================

    def compute_kl(
        self,
        prompts: torch.Tensor,
        responses: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        ref_override: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """
        Compute KL divergence between current policy and reference policy.

        KL(π || π_ref) = E[log π(a|s) - log π_ref(a|s)]

        Args:
            prompts: Input prompt tokens, shape (batch_size, prompt_len)
            responses: Generated response tokens, shape (batch_size, response_len)
            attention_mask: Attention mask (optional)
            ref_override: Use this reference model instead of self.ref_model (optional)

        Returns:
            kl_div: KL divergence per sample, shape (batch_size,)
        """
        ref_model = ref_override if ref_override is not None else self.ref_model

        # Get log probabilities from current model
        with torch.enable_grad():
            current_logprobs = self._get_log_probs(
                self.model, prompts, responses, attention_mask
            )

        # Get log probabilities from reference model
        with torch.no_grad():
            ref_logprobs = self._get_log_probs(
                ref_model, prompts, responses, attention_mask
            )

        # KL divergence: E[log π - log π_ref]
        kl_div = current_logprobs - ref_logprobs

        return kl_div

    def _get_log_probs(
        self,
        model: nn.Module,
        prompts: torch.Tensor,
        responses: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get log probabilities of responses given prompts.

        Args:
            model: Language model
            prompts: Input tokens
            responses: Generated tokens
            attention_mask: Attention mask

        Returns:
            log_probs: Sum of log probabilities per sample
        """
        # Concatenate prompts and responses
        input_ids = torch.cat([prompts, responses], dim=1)

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs

        # Get log probabilities
        log_probs = F.log_softmax(logits, dim=-1)

        # Gather log probs of actual tokens (shift by 1 for language modeling)
        prompt_len = prompts.shape[1]
        response_log_probs = []

        for i in range(responses.shape[1]):
            token_idx = prompt_len + i
            if token_idx < logits.shape[1]:
                token_log_probs = log_probs[:, token_idx - 1, :]
                actual_tokens = responses[:, i]
                selected_log_probs = torch.gather(
                    token_log_probs, 1, actual_tokens.unsqueeze(1)
                ).squeeze(1)
                response_log_probs.append(selected_log_probs)

        # Sum log probs across sequence
        response_log_probs = torch.stack(response_log_probs, dim=1)

        # Mask if provided
        if attention_mask is not None:
            response_mask = attention_mask[:, prompt_len:]
            response_log_probs = response_log_probs * response_mask

        return response_log_probs.sum(dim=1)

    # ============================================================================
    # Loss Computation
    # ============================================================================

    def compute_loss(
        self,
        log_probs: torch.Tensor,
        advantages: torch.Tensor,
        kl_div: Optional[torch.Tensor] = None,
        entropy: Optional[torch.Tensor] = None,
        kl_weight: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute basic policy gradient loss (REINFORCE / GRPO style).

        Loss = -E[log π(a|s) * A] + kl_coef * KL - entropy_coef * H

        Args:
            log_probs: Log probabilities of actions, shape (batch_size,)
            advantages: Advantage estimates, shape (batch_size,)
            kl_div: KL divergence (optional), shape (batch_size,)
            entropy: Policy entropy (optional), shape (batch_size,)
            kl_weight: KL penalty weight (uses config.kl_coef if None)

        Returns:
            loss_dict: Dictionary with 'total_loss' and individual components
        """
        kl_weight = kl_weight if kl_weight is not None else self.config.kl_coef

        # Policy gradient loss
        pg_loss = -(log_probs * advantages).mean()

        loss_dict = {
            'pg_loss': pg_loss,
            'total_loss': pg_loss,
        }

        # Add KL penalty
        if kl_div is not None and kl_weight > 0:
            kl_loss = kl_div.mean()
            loss_dict['kl_loss'] = kl_loss
            loss_dict['total_loss'] = loss_dict['total_loss'] + kl_weight * kl_loss

        # Add entropy bonus (negative because we want to maximize entropy)
        if entropy is not None and self.config.entropy_coef > 0:
            entropy_loss = -entropy.mean()
            loss_dict['entropy_loss'] = entropy_loss
            loss_dict['total_loss'] = loss_dict['total_loss'] + self.config.entropy_coef * entropy_loss

        return loss_dict

    def compute_ppo_loss(
        self,
        current_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        values: Optional[torch.Tensor] = None,
        old_values: Optional[torch.Tensor] = None,
        returns: Optional[torch.Tensor] = None,
        entropy: Optional[torch.Tensor] = None,
        kl_div: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute PPO loss with clipping.

        PPO objective:
            L^CLIP(θ) = E[min(r_t(θ)A_t, clip(r_t(θ), 1-ε, 1+ε)A_t)]
        where:
            r_t(θ) = π_θ(a|s) / π_θ_old(a|s)

        Args:
            current_log_probs: Log probs from current policy, shape (batch_size,)
            old_log_probs: Log probs from old policy, shape (batch_size,)
            advantages: Advantage estimates, shape (batch_size,)
            values: Current value predictions (optional), shape (batch_size,)
            old_values: Old value predictions (optional), shape (batch_size,)
            returns: Target returns for value function (optional), shape (batch_size,)
            entropy: Policy entropy (optional), shape (batch_size,)
            kl_div: KL divergence (optional), shape (batch_size,)

        Returns:
            loss_dict: Dictionary with 'total_loss' and individual components

        References:
            - PPO paper: https://arxiv.org/abs/1707.06347
        """
        # Compute importance sampling ratio
        ratio = torch.exp(current_log_probs - old_log_probs)

        # Clipped surrogate objective
        clipped_ratio = torch.clamp(
            ratio,
            1.0 - self.config.ppo_clip_range,
            1.0 + self.config.ppo_clip_range,
        )

        # PPO policy loss
        policy_loss_1 = ratio * advantages
        policy_loss_2 = clipped_ratio * advantages
        policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

        loss_dict = {
            'policy_loss': policy_loss,
            'total_loss': policy_loss,
            'ratio_mean': ratio.mean(),
            'ratio_std': ratio.std(),
        }

        # Value function loss (if value model is used)
        if values is not None and returns is not None:
            if self.config.clip_value_loss and old_values is not None:
                # Clipped value loss (PPO style)
                value_pred_clipped = old_values + torch.clamp(
                    values - old_values,
                    -self.config.ppo_clip_range,
                    self.config.ppo_clip_range,
                )
                value_loss_1 = (values - returns).pow(2)
                value_loss_2 = (value_pred_clipped - returns).pow(2)
                value_loss = torch.max(value_loss_1, value_loss_2).mean()
            else:
                # Simple MSE loss
                value_loss = F.mse_loss(values, returns)

            loss_dict['value_loss'] = value_loss
            loss_dict['total_loss'] = (
                loss_dict['total_loss'] + self.config.value_loss_coef * value_loss
            )

        # Entropy bonus
        if entropy is not None:
            entropy_loss = -entropy.mean()
            loss_dict['entropy_loss'] = entropy_loss
            loss_dict['total_loss'] = (
                loss_dict['total_loss'] + self.config.entropy_coef * entropy_loss
            )

        # KL penalty
        if kl_div is not None and self.config.kl_coef > 0:
            kl_loss = kl_div.mean()
            loss_dict['kl_loss'] = kl_loss
            loss_dict['total_loss'] = (
                loss_dict['total_loss'] + self.config.kl_coef * kl_loss
            )

        return loss_dict

    # ============================================================================
    # Model Updates
    # ============================================================================

    def update(
        self,
        loss: torch.Tensor,
        retain_graph: bool = False,
    ) -> Dict[str, float]:
        """
        Perform gradient descent step.

        Args:
            loss: Loss tensor to backpropagate
            retain_graph: Whether to retain computation graph

        Returns:
            metrics: Dictionary with gradient norms and other metrics
        """
        # Zero gradients
        self.optimizer.zero_grad()

        # Backward pass
        loss.backward(retain_graph=retain_graph)

        # Compute gradient norm before clipping
        grad_norm = self._compute_grad_norm()

        # Gradient clipping
        if self.config.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm,
            )

        # Optimizer step
        self.optimizer.step()

        metrics = {
            'grad_norm': grad_norm,
            'loss': loss.item(),
        }

        return metrics

    def update_ref_model(
        self,
        method: Optional[str] = None,
        alpha: Optional[float] = None,
    ) -> None:
        """
        Update reference model.

        Two methods:
        - "copy": Hard copy of current model parameters
        - "ema": Exponential moving average update

        Args:
            method: Update method ("copy" or "ema"), uses config if None
            alpha: EMA coefficient (uses config.ema_alpha if None)
        """
        method = method if method is not None else self.config.ref_update_method
        alpha = alpha if alpha is not None else self.config.ema_alpha

        if method == "copy":
            # Hard copy
            self.ref_model.load_state_dict(self.model.state_dict())
        elif method == "ema":
            # Exponential moving average
            with torch.no_grad():
                for param, ref_param in zip(
                    self.model.parameters(),
                    self.ref_model.parameters(),
                ):
                    ref_param.data = alpha * ref_param.data + (1 - alpha) * param.data
        else:
            raise ValueError(f"Unknown update method: {method}")

        # Keep ref model in eval mode
        self.ref_model.eval()

    # ============================================================================
    # Utilities
    # ============================================================================

    def _to_tensor(
        self,
        data: Union[torch.Tensor, np.ndarray, List],
    ) -> torch.Tensor:
        """Convert data to PyTorch tensor"""
        if isinstance(data, torch.Tensor):
            return data
        elif isinstance(data, np.ndarray):
            return torch.from_numpy(data).float()
        else:
            return torch.tensor(data, dtype=torch.float32)

    def _normalize(self, tensor: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Normalize tensor to zero mean and unit variance"""
        return (tensor - tensor.mean()) / (tensor.std() + eps)

    def _compute_grad_norm(self) -> float:
        """Compute gradient norm across all parameters"""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get current statistics of the backprop module.

        Returns:
            stats: Dictionary with model statistics
        """
        stats = {
            'config': vars(self.config),
            'model_params': sum(p.numel() for p in self.model.parameters()),
            'ref_model_params': sum(p.numel() for p in self.ref_model.parameters()),
        }

        if self.value_model is not None:
            stats['value_model_params'] = sum(
                p.numel() for p in self.value_model.parameters()
            )

        return stats


# ============================================================================
# Helper Functions for Common Patterns
# ============================================================================

def compute_returns(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
) -> torch.Tensor:
    """
    Compute target returns for value function training.

    Returns: R_t = r_t + γ*V(s_{t+1})*(1-done)

    Args:
        rewards: Rewards, shape (batch_size, num_steps)
        values: Value predictions, shape (batch_size, num_steps + 1)
        dones: Done flags, shape (batch_size, num_steps)
        gamma: Discount factor

    Returns:
        returns: Target returns, shape (batch_size, num_steps)
    """
    next_values = values[:, 1:]
    returns = rewards + gamma * next_values * (1 - dones)
    return returns


def compute_entropy(logits: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Compute policy entropy.

    H = -Σ p(a) log p(a)

    Args:
        logits: Action logits, shape (batch_size, num_actions) or (batch_size, seq_len, vocab_size)
        mask: Attention mask (optional)

    Returns:
        entropy: Entropy per sample, shape (batch_size,)
    """
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)

    if mask is not None:
        entropy = entropy * mask
        entropy = entropy.sum(dim=-1) / mask.sum(dim=-1)
    else:
        if entropy.dim() > 1:
            entropy = entropy.mean(dim=-1)

    return entropy
