"""
Verl-Based PPO Training for Password Game Task
================================================

This implements PPO training for the multi-turn password game where:
- Rules accumulate (each turn adds a new rule)
- Model must satisfy ALL previous rules + current rule
- Reward shaped to encourage progress while minimizing password length
- Single H100, no DDP
- Qwen3-0.6B with thinking mode support

Author: Adapted from verl patterns for password game
"""

import os
import sys
import json
import random
import time
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)
import wandb
from tqdm.auto import tqdm

# Add password game to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../tasks/password-game"))
from game import PasswordGame

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class PasswordGamePPOConfig:
    """Configuration for PPO training on password game."""

    # Model
    model_name: str = "Qwen/Qwen2.5-0.6B"
    precision: str = "bfloat16"
    use_flash_attn: bool = True

    # Training
    num_epochs: int = 5
    num_episodes_per_epoch: int = 100  # Episodes (full game runs)
    batch_size: int = 4  # Number of game episodes per batch
    samples_per_state: int = 2  # Multiple responses per game state
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-7
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1

    # PPO hyperparameters
    ppo_epochs: int = 4
    clip_range: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    kl_coef: float = 0.1
    kl_target: float = 0.01  # Target KL for adaptive KL coef
    gamma: float = 0.99  # Discount for multi-turn
    gae_lambda: float = 0.95
    normalize_advantages: bool = True

    # Generation
    max_prompt_length: int = 1024  # Longer for accumulated rules
    max_new_tokens: int = 256
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 50

    # Thinking mode (Qwen-specific)
    enable_thinking: bool = True
    thinking_temperature: float = 0.7
    thinking_top_p: float = 0.95
    parse_thinking: bool = True

    # Password game specific
    max_turns_per_episode: int = 10  # Max rules to attempt per episode
    early_stop_on_failure: bool = False  # Continue even if rule fails
    reward_scale: float = 1.0
    length_penalty_scale: float = 0.1  # Penalty per character
    progress_bonus: float = 2.0  # Bonus for each new rule satisfied

    # Reward shaping
    use_shaped_reward: bool = True
    intermediate_reward_scale: float = 0.5  # Scale for intermediate steps
    final_success_bonus: float = 10.0  # Bonus for completing all rules

    # Logging
    wandb_project: str = "verl-password-game"
    wandb_run_name: Optional[str] = None
    log_interval: int = 5
    eval_interval: int = 25
    save_interval: int = 50
    output_dir: str = f"./verl_password_game_{int(time.time())}"

    # Data
    num_eval_episodes: int = 20

    seed: int = 42

    def __post_init__(self):
        if self.wandb_run_name is None:
            mode = "thinking" if self.enable_thinking else "normal"
            self.wandb_run_name = f"verl_ppo_password_{mode}_{int(time.time())}"
        os.makedirs(self.output_dir, exist_ok=True)

        # Save config
        with open(os.path.join(self.output_dir, "config.json"), "w") as f:
            json.dump(asdict(self), f, indent=2)


# ============================================================================
# Thinking Mode Utilities (Qwen-specific)
# ============================================================================

QWEN_THINK_END_TOKEN = 151668

def parse_thinking_response(output_ids, tokenizer) -> Tuple[str, str]:
    """Parse Qwen thinking mode output into thinking + response."""
    if isinstance(output_ids, torch.Tensor):
        output_ids = output_ids.tolist()

    try:
        # Find last occurrence of thinking end token
        index = len(output_ids) - output_ids[::-1].index(QWEN_THINK_END_TOKEN)
    except ValueError:
        index = 0

    thinking = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip()
    response = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip()
    return thinking, response


def extract_password_from_response(response: str) -> str:
    """Extract password from model response.

    Handles various response formats:
    - Just the password
    - "Password: <password>"
    - "The password is: <password>"
    - Multi-line with password on last line
    """
    response = response.strip()

    # If response is short and looks like a password, return it
    if len(response) < 200 and '\n' not in response:
        # Remove common prefixes
        for prefix in ["password:", "the password is:", "answer:", "here:", "final:"]:
            if response.lower().startswith(prefix):
                response = response[len(prefix):].strip()
                break
        return response

    # Multi-line: try last non-empty line
    lines = [l.strip() for l in response.split('\n') if l.strip()]
    if lines:
        return extract_password_from_response(lines[-1])

    return response


# ============================================================================
# Value Head
# ============================================================================

class ValueHead(nn.Module):
    """Value head for critic network."""

    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, 1)

        # Initialize with small weights for stable training
        nn.init.orthogonal_(self.linear.weight, gain=0.01)
        nn.init.constant_(self.linear.bias, 0.0)

    def forward(self, hidden_states):
        """hidden_states: (batch, seq, hidden_size) -> values: (batch, seq)"""
        hidden_states = self.dropout(hidden_states)
        return self.linear(hidden_states).squeeze(-1)


# ============================================================================
# Password Game Environment Wrapper
# ============================================================================

class PasswordGameEnv:
    """Wrapper around PasswordGame for RL training."""

    def __init__(self, tokenizer, config: PasswordGamePPOConfig):
        self.tokenizer = tokenizer
        self.config = config
        self.game = None
        self.current_password = ""
        self.turn = 0

    def reset(self) -> Tuple[str, Dict]:
        """Reset game and return initial prompt."""
        self.game = PasswordGame()
        self.current_password = ""
        self.turn = 0

        prompt = self._format_game_prompt()
        info = {
            "turn": self.turn,
            "current_rule_index": self.game.current_rule,
            "game_active": self.game.game_active
        }
        return prompt, info

    def step(self, password_response: str) -> Tuple[str, float, bool, Dict]:
        """Execute one step: submit password and get next rule.

        Returns:
            next_prompt: Prompt for next turn (or final state)
            reward: Reward for this step
            done: Whether episode is complete
            info: Additional information
        """
        # Extract password from response
        password = extract_password_from_response(password_response)
        self.current_password = password

        # Get feedback before advancing
        feedback = self.game.get_rule_feedback(password)
        current_rule_index = self.game.current_rule
        rules_passing = feedback['total_passing']

        # Calculate reward
        reward = self._calculate_shaped_reward(
            password=password,
            feedback=feedback,
            previous_turn=self.turn
        )

        # Advance game
        self.game.advance_rule()
        self.turn += 1

        # Check if done
        done = (
            not self.game.game_active or
            self.turn >= self.config.max_turns_per_episode or
            (self.config.early_stop_on_failure and rules_passing < current_rule_index + 1)
        )

        # Get next prompt
        if done:
            next_prompt = "[DONE]"
        else:
            next_prompt = self._format_game_prompt()

        info = {
            "turn": self.turn,
            "password": password,
            "password_length": len(password),
            "rules_passing": rules_passing,
            "current_rule_index": current_rule_index,
            "feedback": feedback,
            "game_active": self.game.game_active,
            "completed_all_rules": not self.game.game_active and self.turn > 0
        }

        return next_prompt, reward, done, info

    def _format_game_prompt(self) -> str:
        """Format current game state as a prompt for the model."""
        system_msg = """You are playing a password game. You will be given rules one at a time.
Each password must satisfy the current rule AND all previous rules.

IMPORTANT: Only respond with the password itself, nothing else.
Think step by step if needed, but your final answer must be just the password string."""

        # Get all rules up to current
        all_rules = self.game.get_all_rules_up_to_current()

        # Format rules
        rules_text = "\n".join([f"{i+1}. {rule}" for i, rule in enumerate(all_rules)])

        user_msg = f"""Current turn: {self.turn + 1}

Rules to satisfy:
{rules_text}

Previous password: {self.current_password if self.current_password else '(none - first turn)'}

What is your password that satisfies all {len(all_rules)} rules above?"""

        # Use chat template if available
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]

        if hasattr(self.tokenizer, 'apply_chat_template'):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.config.enable_thinking
            )
        else:
            return f"System: {system_msg}\n\nUser: {user_msg}\n\nAssistant:"

    def _calculate_shaped_reward(
        self,
        password: str,
        feedback: Dict,
        previous_turn: int
    ) -> float:
        """Calculate shaped reward for password game.

        Reward components:
        1. Base: +1 per passing rule (from game.calculate_reward)
        2. Progress bonus: Extra reward for advancing past previous best
        3. Length penalty: -0.1 per character
        4. Final success bonus: Large bonus for completing all rules
        """
        if not self.config.use_shaped_reward:
            # Use game's built-in reward
            return self.game.calculate_reward(password) * self.config.reward_scale

        reward = 0.0

        # Base reward: points for passing rules
        rules_passing = feedback['total_passing']
        reward += rules_passing * 1.0

        # Progress bonus: reward for getting further than before
        expected_passing = previous_turn + 1  # Should pass all previous + current
        if rules_passing >= expected_passing:
            reward += self.config.progress_bonus

        # Length penalty: encourage shorter passwords
        length_penalty = len(password) * self.config.length_penalty_scale
        reward -= length_penalty

        # Final success bonus: large reward for completing game
        if feedback.get('rules_checked'):
            total_rules = len(feedback['rules_checked'])
            if rules_passing == total_rules and total_rules >= 10:
                reward += self.config.final_success_bonus

        return reward * self.config.reward_scale


# ============================================================================
# Episode Rollout
# ============================================================================

@dataclass
class RolloutStep:
    """Single step in a rollout."""
    prompt: str
    prompt_ids: torch.Tensor
    prompt_mask: torch.Tensor
    response: str
    response_ids: torch.Tensor
    full_ids: torch.Tensor
    full_mask: torch.Tensor
    reward: float
    done: bool
    info: Dict


@dataclass
class Episode:
    """Complete episode (one full game)."""
    steps: List[RolloutStep]
    total_reward: float
    final_turn: int
    final_password: str
    rules_satisfied: int


def run_episode(
    env: PasswordGameEnv,
    policy_model,
    tokenizer,
    config: PasswordGamePPOConfig,
    samples_per_state: int = 1
) -> List[Episode]:
    """Run one or more episodes (full game rollouts).

    Args:
        env: Password game environment
        policy_model: Policy network
        tokenizer: Tokenizer
        config: Configuration
        samples_per_state: Number of parallel samples per state

    Returns:
        List of Episode objects
    """
    policy_model.eval()
    policy_model.config.use_cache = True

    episodes = []

    gen_config = {
        "max_new_tokens": config.max_new_tokens,
        "do_sample": True,
        "temperature": config.thinking_temperature if config.enable_thinking else config.temperature,
        "top_p": config.thinking_top_p if config.enable_thinking else config.top_p,
        "top_k": config.top_k,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    with torch.no_grad():
        for _ in range(samples_per_state):
            # Reset environment
            prompt, info = env.reset()
            steps = []
            episode_reward = 0.0

            # Run episode
            for turn in range(config.max_turns_per_episode):
                # Tokenize prompt
                prompt_inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=config.max_prompt_length
                ).to(DEVICE)

                # Generate response
                outputs = policy_model.generate(**prompt_inputs, **gen_config)

                # Extract generated tokens
                prompt_len = prompt_inputs.input_ids.size(1)
                generated_ids = outputs[:, prompt_len:]

                # Decode response
                if config.enable_thinking and config.parse_thinking:
                    thinking, response = parse_thinking_response(generated_ids[0], tokenizer)
                else:
                    response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

                # Environment step
                next_prompt, reward, done, step_info = env.step(response)
                episode_reward += reward

                # Create attention mask for full sequence
                full_mask = torch.ones_like(outputs)
                full_mask[outputs == tokenizer.pad_token_id] = 0

                # Store step
                step = RolloutStep(
                    prompt=prompt,
                    prompt_ids=prompt_inputs.input_ids[0],
                    prompt_mask=prompt_inputs.attention_mask[0],
                    response=response,
                    response_ids=generated_ids[0],
                    full_ids=outputs[0],
                    full_mask=full_mask[0],
                    reward=reward,
                    done=done,
                    info=step_info
                )
                steps.append(step)

                if done:
                    break

                prompt = next_prompt

            # Create episode
            final_info = steps[-1].info if steps else {}
            episode = Episode(
                steps=steps,
                total_reward=episode_reward,
                final_turn=final_info.get('turn', 0),
                final_password=final_info.get('password', ''),
                rules_satisfied=final_info.get('rules_passing', 0)
            )
            episodes.append(episode)

    return episodes


# ============================================================================
# PPO Utilities
# ============================================================================

def compute_log_probs(model, input_ids, attention_mask, return_values=False, value_head=None):
    """Compute log probabilities (and optionally values) for input sequence."""
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=return_values
    )

    logits = outputs.logits
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)

    # Gather log probs for actual tokens
    token_log_probs = torch.gather(
        log_probs,
        dim=2,
        index=input_ids[:, 1:].unsqueeze(-1)
    ).squeeze(-1)

    # Mask padding
    mask = attention_mask[:, 1:].bool()
    token_log_probs = token_log_probs * mask

    if return_values and value_head is not None:
        hidden_states = outputs.hidden_states[-1]
        values = value_head(hidden_states)
        return token_log_probs, values

    return token_log_probs


def compute_advantages_and_returns(
    rewards: torch.Tensor,
    values: torch.Tensor,
    masks: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE advantages and returns.

    Args:
        rewards: (batch, seq_len)
        values: (batch, seq_len)
        masks: (batch, seq_len)
        gamma: Discount factor
        gae_lambda: GAE lambda

    Returns:
        advantages: (batch, seq_len)
        returns: (batch, seq_len)
    """
    batch_size, seq_len = rewards.shape
    advantages = torch.zeros_like(rewards)

    # Compute GAE
    gae = 0
    for t in reversed(range(seq_len)):
        if t == seq_len - 1:
            next_value = 0
        else:
            next_value = values[:, t + 1]

        delta = rewards[:, t] + gamma * next_value * masks[:, t] - values[:, t]
        gae = delta + gamma * gae_lambda * masks[:, t] * gae
        advantages[:, t] = gae

    returns = advantages + values
    return advantages, returns


def whiten(values: torch.Tensor, mask: torch.Tensor, shift_mean: bool = True) -> torch.Tensor:
    """Whiten (normalize) values."""
    mean = (values * mask).sum() / mask.sum().clamp(min=1)
    var = ((values - mean) ** 2 * mask).sum() / mask.sum().clamp(min=1)
    std = torch.sqrt(var + 1e-8)

    if shift_mean:
        return (values - mean) / std
    else:
        return values / std


# ============================================================================
# Training
# ============================================================================

def train_ppo_password_game(config: PasswordGamePPOConfig):
    """Main training loop for PPO on password game."""

    # Set seed
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    print("="*80)
    print("VERL PPO Training - Password Game")
    print("="*80)
    print(f"Model: {config.model_name}")
    print(f"Thinking mode: {config.enable_thinking}")
    print(f"Episodes per epoch: {config.num_episodes_per_epoch}")
    print(f"Max turns per episode: {config.max_turns_per_episode}")
    print(f"Output: {config.output_dir}")
    print("="*80)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    print(f"✓ Tokenizer loaded (vocab: {len(tokenizer)})")

    # Load models
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32
    }[config.precision]

    policy_model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if config.use_flash_attn else "eager"
    )
    policy_model.config.use_cache = False
    print(f"✓ Policy model loaded: {sum(p.numel() for p in policy_model.parameters())/1e9:.2f}B params")

    reference_model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if config.use_flash_attn else "eager"
    )
    reference_model.eval()
    for param in reference_model.parameters():
        param.requires_grad = False
    print("✓ Reference model loaded and frozen")

    # Value head
    value_head = ValueHead(policy_model.config.hidden_size).to(DEVICE).to(dtype)
    print(f"✓ Value head initialized")

    # Optimizer
    optimizer = torch.optim.AdamW(
        list(policy_model.parameters()) + list(value_head.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    total_steps = config.num_epochs * config.num_episodes_per_epoch
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    print(f"✓ Optimizer: LR={config.learning_rate}, Steps={total_steps}, Warmup={warmup_steps}")

    # WandB
    wandb_run = wandb.init(
        project=config.wandb_project,
        name=config.wandb_run_name,
        config=asdict(config)
    )
    print(f"✓ WandB: {wandb_run.get_url()}")

    # Environment
    env = PasswordGameEnv(tokenizer, config)
    print("✓ Password game environment ready")

    # Baseline evaluation
    print("\n" + "="*80)
    print("BASELINE EVALUATION")
    print("="*80)
    baseline_metrics = evaluate_policy(
        policy_model=policy_model,
        tokenizer=tokenizer,
        config=config,
        num_episodes=min(10, config.num_eval_episodes),
        desc="Baseline"
    )
    print(f"Baseline: {baseline_metrics['mean_reward']:.4f} ± {baseline_metrics['std_reward']:.4f}")
    print(f"Rules satisfied: {baseline_metrics['mean_rules']:.2f} / {baseline_metrics['max_rules']}")
    wandb.log({"baseline/reward": baseline_metrics['mean_reward']})
    print("="*80)

    # Training loop
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80)

    global_step = 0
    best_val_reward = -float('inf')
    kl_coef = config.kl_coef  # Adaptive KL coefficient

    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch+1}/{config.num_epochs}")
        epoch_metrics = defaultdict(list)

        policy_model.train()
        value_head.train()

        for episode_batch_idx in tqdm(range(config.num_episodes_per_epoch // config.batch_size),
                                      desc=f"Epoch {epoch+1}"):

            # Collect rollouts
            all_episodes = []
            for _ in range(config.batch_size):
                episodes = run_episode(
                    env=env,
                    policy_model=policy_model,
                    tokenizer=tokenizer,
                    config=config,
                    samples_per_state=config.samples_per_state
                )
                all_episodes.extend(episodes)

            # Flatten steps from all episodes
            all_steps = []
            for episode in all_episodes:
                all_steps.extend(episode.steps)

            if not all_steps:
                continue

            # Prepare batch tensors
            batch_full_ids = torch.stack([step.full_ids for step in all_steps])
            batch_full_mask = torch.stack([step.full_mask for step in all_steps])
            batch_rewards = torch.tensor([step.reward for step in all_steps],
                                        device=DEVICE, dtype=dtype)

            # Compute old log probs and values
            with torch.no_grad():
                old_log_probs, old_values = compute_log_probs(
                    policy_model,
                    batch_full_ids,
                    batch_full_mask,
                    return_values=True,
                    value_head=value_head
                )

                ref_log_probs = compute_log_probs(
                    reference_model,
                    batch_full_ids,
                    batch_full_mask
                )

            # Find prompt lengths (for masking generated part only)
            prompt_lens = [step.prompt_ids.size(0) for step in all_steps]

            # Create reward tensor per token (assign full reward to last token)
            batch_size_steps, seq_len = batch_full_ids.shape
            reward_per_token = torch.zeros(batch_size_steps, seq_len, device=DEVICE, dtype=dtype)
            for i, (reward, prompt_len) in enumerate(zip(batch_rewards, prompt_lens)):
                gen_len = (batch_full_ids[i, prompt_len:] != tokenizer.pad_token_id).sum()
                if gen_len > 0:
                    # Put reward at last generated token
                    reward_per_token[i, prompt_len + gen_len - 1] = reward

            # Create mask for generated tokens only
            gen_mask = batch_full_mask.clone()
            for i, prompt_len in enumerate(prompt_lens):
                gen_mask[i, :prompt_len] = 0

            # Compute advantages and returns
            advantages, returns = compute_advantages_and_returns(
                rewards=reward_per_token,
                values=old_values,
                masks=gen_mask,
                gamma=config.gamma,
                gae_lambda=config.gae_lambda
            )

            if config.normalize_advantages:
                advantages = whiten(advantages, gen_mask)

            # PPO update
            policy_model.config.use_cache = False
            for ppo_epoch in range(config.ppo_epochs):
                # Forward pass
                curr_log_probs, curr_values = compute_log_probs(
                    policy_model,
                    batch_full_ids,
                    batch_full_mask,
                    return_values=True,
                    value_head=value_head
                )

                # Policy loss (PPO clip)
                log_ratio = curr_log_probs - old_log_probs.detach()
                ratio = torch.exp(log_ratio)

                policy_loss_1 = -advantages.detach() * ratio
                policy_loss_2 = -advantages.detach() * torch.clamp(
                    ratio,
                    1 - config.clip_range,
                    1 + config.clip_range
                )
                policy_loss = torch.max(policy_loss_1, policy_loss_2)
                policy_loss = (policy_loss * gen_mask).sum() / gen_mask.sum().clamp(min=1)

                # Value loss
                value_loss = ((curr_values - returns.detach()) ** 2 * gen_mask).sum() / gen_mask.sum().clamp(min=1)

                # KL divergence
                kl_div = ((curr_log_probs - ref_log_probs.detach()) * gen_mask).sum() / gen_mask.sum().clamp(min=1)

                # Entropy (approximate)
                entropy = -(curr_log_probs * gen_mask).sum() / gen_mask.sum().clamp(min=1)

                # Total loss
                loss = (
                    policy_loss +
                    config.value_loss_coef * value_loss +
                    kl_coef * kl_div -
                    config.entropy_coef * entropy
                )

                # Backward
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    list(policy_model.parameters()) + list(value_head.parameters()),
                    config.max_grad_norm
                )

                # Update
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # Adaptive KL coefficient
            if kl_div.item() > 2 * config.kl_target:
                kl_coef *= 1.5
            elif kl_div.item() < 0.5 * config.kl_target:
                kl_coef *= 0.5
            kl_coef = np.clip(kl_coef, 0.001, 1.0)

            # Logging
            episode_rewards = [ep.total_reward for ep in all_episodes]
            episode_rules = [ep.rules_satisfied for ep in all_episodes]

            epoch_metrics['reward'].extend(episode_rewards)
            epoch_metrics['rules_satisfied'].extend(episode_rules)
            epoch_metrics['loss'].append(loss.item())
            epoch_metrics['policy_loss'].append(policy_loss.item())
            epoch_metrics['value_loss'].append(value_loss.item())
            epoch_metrics['kl_div'].append(kl_div.item())

            if global_step % config.log_interval == 0:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/policy_loss": policy_loss.item(),
                    "train/value_loss": value_loss.item(),
                    "train/kl_div": kl_div.item(),
                    "train/kl_coef": kl_coef,
                    "train/reward": np.mean(episode_rewards),
                    "train/rules_satisfied": np.mean(episode_rules),
                    "train/lr": scheduler.get_last_lr()[0],
                    "epoch": epoch,
                }, step=global_step)

            # Evaluation
            if global_step % config.eval_interval == 0 and global_step > 0:
                eval_metrics = evaluate_policy(
                    policy_model=policy_model,
                    tokenizer=tokenizer,
                    config=config,
                    num_episodes=config.num_eval_episodes,
                    desc=f"Eval@{global_step}"
                )

                wandb.log({
                    "eval/reward": eval_metrics['mean_reward'],
                    "eval/rules_satisfied": eval_metrics['mean_rules'],
                    "eval/success_rate": eval_metrics['success_rate'],
                }, step=global_step)

                print(f"\n[Step {global_step}] Eval: {eval_metrics['mean_reward']:.4f} | "
                      f"Rules: {eval_metrics['mean_rules']:.2f} | "
                      f"Success: {eval_metrics['success_rate']:.2%}")

                # Save best model
                if eval_metrics['mean_reward'] > best_val_reward:
                    best_val_reward = eval_metrics['mean_reward']
                    best_dir = os.path.join(config.output_dir, "best_model")
                    os.makedirs(best_dir, exist_ok=True)
                    policy_model.save_pretrained(best_dir)
                    tokenizer.save_pretrained(best_dir)
                    torch.save(value_head.state_dict(), os.path.join(best_dir, "value_head.pt"))
                    print(f"✓ Saved best model (reward: {best_val_reward:.4f})")

                policy_model.train()
                value_head.train()

            # Checkpointing
            if global_step % config.save_interval == 0 and global_step > 0:
                ckpt_dir = os.path.join(config.output_dir, f"checkpoint-{global_step}")
                os.makedirs(ckpt_dir, exist_ok=True)
                policy_model.save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)
                torch.save(value_head.state_dict(), os.path.join(ckpt_dir, "value_head.pt"))

            global_step += 1
            torch.cuda.empty_cache()

        # Epoch summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Reward: {np.mean(epoch_metrics['reward']):.4f} ± {np.std(epoch_metrics['reward']):.4f}")
        print(f"  Rules: {np.mean(epoch_metrics['rules_satisfied']):.2f}")
        print(f"  Loss: {np.mean(epoch_metrics['loss']):.4f}")

    # Final evaluation
    print("\n" + "="*80)
    print("FINAL EVALUATION")
    print("="*80)
    final_metrics = evaluate_policy(
        policy_model=policy_model,
        tokenizer=tokenizer,
        config=config,
        num_episodes=config.num_eval_episodes * 2,
        desc="Final"
    )
    print(f"Final: {final_metrics['mean_reward']:.4f} ± {final_metrics['std_reward']:.4f}")
    print(f"Rules: {final_metrics['mean_rules']:.2f} / {final_metrics['max_rules']}")
    print(f"Success rate: {final_metrics['success_rate']:.2%}")
    print(f"Improvement: {final_metrics['mean_reward'] - baseline_metrics['mean_reward']:.4f}")

    # Save final model
    final_dir = os.path.join(config.output_dir, "final_model")
    os.makedirs(final_dir, exist_ok=True)
    policy_model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    torch.save(value_head.state_dict(), os.path.join(final_dir, "value_head.pt"))

    # Save summary
    summary = {
        "baseline": baseline_metrics,
        "final": final_metrics,
        "best_val_reward": best_val_reward,
        "improvement": final_metrics['mean_reward'] - baseline_metrics['mean_reward']
    }
    with open(os.path.join(config.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    wandb.finish()
    print("="*80)
    print("TRAINING COMPLETE")
    print("="*80)


def evaluate_policy(
    policy_model,
    tokenizer,
    config: PasswordGamePPOConfig,
    num_episodes: int = 20,
    desc: str = "Eval"
) -> Dict[str, float]:
    """Evaluate policy on password game.

    Returns metrics about performance.
    """
    policy_model.eval()
    policy_model.config.use_cache = True

    env = PasswordGameEnv(tokenizer, config)

    all_rewards = []
    all_rules_satisfied = []
    successes = 0

    with torch.no_grad():
        for _ in tqdm(range(num_episodes), desc=desc):
            episodes = run_episode(
                env=env,
                policy_model=policy_model,
                tokenizer=tokenizer,
                config=config,
                samples_per_state=1
            )

            for episode in episodes:
                all_rewards.append(episode.total_reward)
                all_rules_satisfied.append(episode.rules_satisfied)

                # Success = satisfied at least 5 rules
                if episode.rules_satisfied >= 5:
                    successes += 1

    metrics = {
        "mean_reward": np.mean(all_rewards),
        "std_reward": np.std(all_rewards),
        "mean_rules": np.mean(all_rules_satisfied),
        "max_rules": max(all_rules_satisfied) if all_rules_satisfied else 0,
        "success_rate": successes / len(all_rewards) if all_rewards else 0.0,
        "num_episodes": len(all_rewards)
    }

    return metrics


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.6B")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--episodes-per-epoch", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-7)
    parser.add_argument("--no-thinking", action="store_true")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default="verl-password-game")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = PasswordGamePPOConfig(
        model_name=args.model,
        num_epochs=args.epochs,
        num_episodes_per_epoch=args.episodes_per_epoch,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        enable_thinking=not args.no_thinking,
        wandb_project=args.wandb_project,
        output_dir=args.output_dir or f"./verl_password_game_{int(time.time())}",
        seed=args.seed
    )

    train_ppo_password_game(config)
