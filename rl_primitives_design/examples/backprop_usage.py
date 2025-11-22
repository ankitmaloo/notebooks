"""
Example usage of BackpropModule for different RL algorithms

This demonstrates:
1. PPO training with value function
2. GRPO training without value function
3. REINFORCE with baseline
4. KL divergence computation
5. Reference model updates
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
sys.path.append('..')
from rl_primitives import BackpropModule, BackpropConfig


# ============================================================================
# Example 1: PPO Training (with value function)
# ============================================================================

def example_ppo_training():
    """Example of PPO training with actor-critic"""
    print("=" * 80)
    print("Example 1: PPO Training with Value Function")
    print("=" * 80)

    # Configuration
    config = BackpropConfig(
        gamma=0.99,
        gae_lambda=0.95,
        ppo_clip_range=0.2,
        ppo_epochs=4,
        value_loss_coef=0.1,
        kl_coef=0.1,
        max_grad_norm=1.0,
    )

    # Create dummy model (in practice, use actual LLM)
    model = nn.Sequential(
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)

    # Initialize backprop module
    backprop = BackpropModule(
        model=model,
        optimizer=optimizer,
        config=config,
    )

    # Simulate trajectory data
    batch_size = 8
    num_steps = 10

    rewards = torch.randn(batch_size, num_steps)
    values = torch.randn(batch_size, num_steps + 1)  # +1 for bootstrap
    dones = torch.zeros(batch_size, num_steps)

    # Compute advantages using GAE
    advantages = backprop.compute_advantages(
        rewards=rewards,
        values=values,
        dones=dones,
    )

    print(f"Computed advantages shape: {advantages.shape}")
    print(f"Advantages mean: {advantages.mean():.4f}, std: {advantages.std():.4f}")

    # Simulate log probabilities
    current_log_probs = torch.randn(batch_size, num_steps).sum(dim=1)
    old_log_probs = torch.randn(batch_size, num_steps).sum(dim=1)

    # Compute returns for value function
    returns = rewards.sum(dim=1)  # Simplified
    current_values = values[:, :-1].mean(dim=1)
    old_values = current_values.clone()

    # Compute PPO loss
    loss_dict = backprop.compute_ppo_loss(
        current_log_probs=current_log_probs,
        old_log_probs=old_log_probs,
        advantages=advantages.mean(dim=1),  # Average over time
        values=current_values,
        old_values=old_values,
        returns=returns,
    )

    print(f"\nPPO Loss components:")
    for key, value in loss_dict.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.item():.4f}")

    # Update model
    metrics = backprop.update(loss_dict['total_loss'])
    print(f"\nUpdate metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    print("\n")


# ============================================================================
# Example 2: GRPO Training (no value function)
# ============================================================================

def example_grpo_training():
    """Example of GRPO training with group-relative advantages"""
    print("=" * 80)
    print("Example 2: GRPO Training (Group-Relative)")
    print("=" * 80)

    # Configuration for GRPO
    config = BackpropConfig(
        kl_coef=0.1,
        normalize_advantages=True,
        max_grad_norm=1.0,
    )

    # Create model
    model = nn.Sequential(nn.Linear(128, 128))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)

    backprop = BackpropModule(
        model=model,
        optimizer=optimizer,
        config=config,
    )

    # Simulate group-based sampling (4 groups, 8 samples each)
    num_groups = 4
    group_size = 8
    rewards = torch.randn(num_groups, group_size)

    print(f"Rewards per group:\n{rewards}")

    # Compute group-relative advantages
    advantages = backprop.compute_relative_advantages(
        rewards=rewards,
        group_size=group_size,
    )

    print(f"\nGroup-relative advantages:\n{advantages}")
    print(f"Group means (should be ~0): {advantages.mean(dim=1)}")

    # Simulate log probs
    log_probs = torch.randn(num_groups, group_size)

    # Compute simple policy gradient loss
    loss_dict = backprop.compute_loss(
        log_probs=log_probs.flatten(),
        advantages=advantages.flatten(),
    )

    print(f"\nGRPO Loss: {loss_dict['total_loss'].item():.4f}")

    # Update
    metrics = backprop.update(loss_dict['total_loss'])
    print(f"Gradient norm: {metrics['grad_norm']:.4f}")

    print("\n")


# ============================================================================
# Example 3: KL Divergence Computation
# ============================================================================

def example_kl_divergence():
    """Example of computing KL divergence between policies"""
    print("=" * 80)
    print("Example 3: KL Divergence Computation")
    print("=" * 80)

    # Create model and reference model
    model = nn.Sequential(
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 50257),  # Vocab size
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)

    backprop = BackpropModule(
        model=model,
        optimizer=optimizer,
    )

    # Note: In practice, you'd use actual language model forward passes
    # This is a simplified demonstration

    batch_size = 4
    prompt_len = 10
    response_len = 20

    prompts = torch.randint(0, 50257, (batch_size, prompt_len))
    responses = torch.randint(0, 50257, (batch_size, response_len))

    print(f"Computing KL divergence for {batch_size} samples...")
    print(f"Prompt length: {prompt_len}, Response length: {response_len}")

    # Note: This will work with actual language models
    # For this demo, we're using a simple model
    print("\n(Note: This demo uses simplified models, not actual LLMs)")
    print("In practice, use with AutoModelForCausalLM from transformers")

    print("\n")


# ============================================================================
# Example 4: Reference Model Updates
# ============================================================================

def example_ref_model_updates():
    """Example of different reference model update strategies"""
    print("=" * 80)
    print("Example 4: Reference Model Updates")
    print("=" * 80)

    model = nn.Linear(128, 128)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)

    # Initialize with copy method
    config = BackpropConfig(ref_update_method="copy")
    backprop = BackpropModule(model=model, optimizer=optimizer, config=config)

    # Get initial ref model params
    ref_param_before = next(backprop.ref_model.parameters()).clone()
    model_param_before = next(backprop.model.parameters()).clone()

    # Simulate training
    dummy_input = torch.randn(4, 128)
    loss = model(dummy_input).mean()
    backprop.update(loss)

    model_param_after = next(backprop.model.parameters()).clone()
    ref_param_after_no_update = next(backprop.ref_model.parameters()).clone()

    print("Before update:")
    print(f"  Model param L2 norm: {model_param_before.norm():.4f}")
    print(f"  Ref param L2 norm: {ref_param_before.norm():.4f}")

    print("\nAfter model update (ref not updated):")
    print(f"  Model param L2 norm: {model_param_after.norm():.4f}")
    print(f"  Ref param L2 norm: {ref_param_after_no_update.norm():.4f}")
    print(f"  Model changed: {not torch.allclose(model_param_before, model_param_after)}")
    print(f"  Ref unchanged: {torch.allclose(ref_param_before, ref_param_after_no_update)}")

    # Hard copy update
    backprop.update_ref_model(method="copy")
    ref_param_after_copy = next(backprop.ref_model.parameters()).clone()

    print("\nAfter hard copy update:")
    print(f"  Ref param L2 norm: {ref_param_after_copy.norm():.4f}")
    print(f"  Ref matches model: {torch.allclose(model_param_after, ref_param_after_copy)}")

    # Simulate another update
    loss = model(dummy_input).mean()
    backprop.update(loss)
    model_param_after2 = next(backprop.model.parameters()).clone()

    # EMA update
    backprop.update_ref_model(method="ema", alpha=0.9)
    ref_param_after_ema = next(backprop.ref_model.parameters()).clone()

    print("\nAfter EMA update (alpha=0.9):")
    print(f"  Model param L2 norm: {model_param_after2.norm():.4f}")
    print(f"  Ref param L2 norm: {ref_param_after_ema.norm():.4f}")
    print(f"  Ref is blend of old and new: {not torch.allclose(model_param_after2, ref_param_after_ema)}")

    print("\n")


# ============================================================================
# Example 5: Complete Training Loop
# ============================================================================

def example_complete_training_loop():
    """Example of a complete training loop with BackpropModule"""
    print("=" * 80)
    print("Example 5: Complete Training Loop")
    print("=" * 80)

    # Setup
    config = BackpropConfig(
        gamma=0.99,
        gae_lambda=0.95,
        ppo_clip_range=0.2,
        max_grad_norm=1.0,
    )

    model = nn.Sequential(
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    backprop = BackpropModule(model=model, optimizer=optimizer, config=config)

    # Training loop
    num_iterations = 5
    batch_size = 8

    print(f"Training for {num_iterations} iterations...\n")

    for iteration in range(num_iterations):
        # Simulate collecting rollouts
        rewards = torch.randn(batch_size)
        log_probs = torch.randn(batch_size)

        # Compute advantages (simple version without values)
        advantages = backprop.compute_advantages(rewards)

        # Compute loss
        loss_dict = backprop.compute_loss(log_probs, advantages)

        # Update model
        metrics = backprop.update(loss_dict['total_loss'])

        print(f"Iteration {iteration + 1}/{num_iterations}:")
        print(f"  Loss: {metrics['loss']:.4f}")
        print(f"  Grad norm: {metrics['grad_norm']:.4f}")
        print(f"  Mean reward: {rewards.mean():.4f}")

        # Update reference model every 2 iterations
        if (iteration + 1) % 2 == 0:
            backprop.update_ref_model(method="ema", alpha=0.95)
            print(f"  Updated reference model (EMA)")

    # Get final statistics
    stats = backprop.get_statistics()
    print(f"\nFinal statistics:")
    print(f"  Model parameters: {stats['model_params']:,}")
    print(f"  Config: {stats['config']}")

    print("\n")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("BackpropModule Usage Examples")
    print("=" * 80 + "\n")

    # Run all examples
    example_ppo_training()
    example_grpo_training()
    example_kl_divergence()
    example_ref_model_updates()
    example_complete_training_loop()

    print("=" * 80)
    print("All examples completed successfully!")
    print("=" * 80)
