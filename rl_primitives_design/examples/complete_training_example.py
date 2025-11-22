#!/usr/bin/env python3
"""
Complete End-to-End RL Training Examples
=========================================

This script demonstrates how to use all RL primitives components together
for complete training pipelines with different algorithms and environments.

Examples:
1. PPO training with simple text environment
2. GRPO training with SVRL environment
3. Multi-objective Pareto optimization
4. Custom environment and reward computer
"""

import torch
from torch.optim import Adam
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import all RL primitives components
from rl_primitives import (
    # Inference
    InferenceModule,
    # Environment
    SimpleTextEnvironment,
    SVRLEnvironment,
    create_environment,
    # Rollout Management
    RolloutManager,
    RolloutConfig,
    # Rewards
    SVRLRewardComputer,
    ParetoRewardComputer,
    create_reward_computer,
    # Backprop
    BackpropModule,
    BackpropConfig,
    # Algorithms
    PPOAlgorithm,
    GRPOAlgorithm,
    AlgorithmConfig,
    create_algorithm,
)


def example1_ppo_simple_text():
    """
    Example 1: PPO Training on Simple Text Generation

    This example shows basic PPO training on a simple text completion task.
    """
    print("\n" + "="*80)
    print("Example 1: PPO Training on Simple Text Generation")
    print("="*80 + "\n")

    # 1. Setup model and inference
    print("Loading model...")
    model_name = "gpt2"  # Use small model for demo
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    inference = InferenceModule(
        model=model,
        tokenizer=tokenizer,
        backend="huggingface",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # 2. Create environment
    print("Creating environment...")
    prompts = [
        "The future of artificial intelligence is",
        "The most important invention in history was",
        "Climate change can be addressed by",
        "The key to happiness is",
    ]

    env = SimpleTextEnvironment(
        inference=inference,
        prompts=prompts,
        max_steps=1,  # Single generation per episode
        system_prompt="Generate thoughtful, informative responses."
    )

    # 3. Setup rollout manager
    rollout_config = RolloutConfig(
        num_parallel=8,
        verbose=True,
        enable_progress_bar=True
    )
    rollout_manager = RolloutManager(env, rollout_config)

    # 4. Setup reward computer (simple quality scoring)
    class SimpleRewardComputer:
        def compute_rewards(self, trajectories, method="absolute"):
            import numpy as np
            # Reward longer, more informative responses
            rewards = []
            for traj in trajectories:
                # Simple heuristic: length + presence of key words
                response = traj.response
                length_reward = min(len(response.split()), 50) / 50.0
                quality_bonus = 0.1 if any(word in response.lower()
                                          for word in ['because', 'therefore', 'however']) else 0
                rewards.append(length_reward + quality_bonus)

            if method == "relative":
                rewards = np.array(rewards)
                return (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            return np.array(rewards)

    reward_computer = SimpleRewardComputer()

    # 5. Setup backprop module
    backprop_config = BackpropConfig(
        gamma=0.99,
        gae_lambda=0.95,
        ppo_clip_range=0.2,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        normalize_advantages=True,
        max_grad_norm=1.0
    )

    optimizer = Adam(model.parameters(), lr=1e-5)
    backprop = BackpropModule(
        model=model,
        ref_model=None,  # Will create copy internally
        optimizer=optimizer,
        config=backprop_config
    )

    # 6. Create PPO algorithm
    algo_config = AlgorithmConfig(
        batch_size=16,
        ppo_epochs=4,
        learning_rate=1e-5,
        kl_coef=0.1,
        checkpoint_dir="./checkpoints/ppo_simple",
        save_every=10,
        use_wandb=False,  # Set to True for logging
    )

    algo = PPOAlgorithm(
        env=env,
        rollout_manager=rollout_manager,
        reward_computer=reward_computer,
        backprop=backprop,
        config=algo_config
    )

    # 7. Train!
    print("\nStarting training...")
    metrics = algo.train(num_iterations=5, progress_bar=True)

    print(f"\nTraining complete!")
    print(f"Final average reward: {metrics[-1].reward_mean:.3f}")
    print(f"Final policy loss: {metrics[-1].loss_policy:.3f}")

    return algo, metrics


def example2_grpo_svrl():
    """
    Example 2: GRPO Training on Self-Verification RL Environment

    This shows GRPO (simpler than PPO) on a task where the model
    decides when to use expensive verification tools.
    """
    print("\n" + "="*80)
    print("Example 2: GRPO Training on Self-Verification RL")
    print("="*80 + "\n")

    # Setup (similar to example 1)
    model_name = "gpt2"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    inference = InferenceModule(
        model=model,
        tokenizer=tokenizer,
        backend="huggingface",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Define verification tools
    def mock_calculator(expression):
        """Mock calculator tool"""
        try:
            return {"result": eval(expression), "cost": 1.0}
        except:
            return {"result": "error", "cost": 1.0}

    def mock_web_search(query):
        """Mock web search tool"""
        return {"results": f"Mock results for: {query}", "cost": 2.0}

    verification_tools = {
        "calculator": {"cost": 1.0, "fn": mock_calculator},
        "web_search": {"cost": 2.0, "fn": mock_web_search}
    }

    # Create SVRL environment
    env = SVRLEnvironment(
        inference=inference,
        task_prompts=[
            "Calculate 123 * 456",
            "What is the capital of France?",
            "Solve: 2x + 5 = 13",
        ],
        verification_tools=verification_tools,
        initial_budget=10.0,
        max_steps=5
    )

    # Rollout manager
    rollout_manager = RolloutManager(
        env,
        RolloutConfig(num_parallel=16, verbose=True)
    )

    # SVRL reward computer (task quality vs verification cost)
    reward_computer = SVRLRewardComputer(
        task_weight=1.0,
        efficiency_weight=0.3,
        speed_weight=0.1
    )

    # Backprop (GRPO doesn't need value function)
    optimizer = Adam(model.parameters(), lr=1e-5)
    backprop = BackpropModule(
        model=model,
        optimizer=optimizer,
        config=BackpropConfig(gamma=0.99)
    )

    # GRPO algorithm (simpler than PPO)
    algo = GRPOAlgorithm(
        env=env,
        rollout_manager=rollout_manager,
        reward_computer=reward_computer,
        backprop=backprop,
        config=AlgorithmConfig(
            batch_size=32,
            grpo_top_percentile=50,  # Only update on top 50%
            learning_rate=1e-5,
            checkpoint_dir="./checkpoints/grpo_svrl"
        )
    )

    # Train
    print("\nStarting GRPO training...")
    metrics = algo.train(num_iterations=5, progress_bar=True)

    print(f"\nTraining complete!")
    print(f"Final average reward: {metrics[-1].reward_mean:.3f}")

    return algo, metrics


def example3_pareto_multiobjective():
    """
    Example 3: Multi-Objective Optimization with Pareto Rewards

    Optimize for multiple objectives simultaneously using Pareto ranking.
    """
    print("\n" + "="*80)
    print("Example 3: Multi-Objective Pareto Optimization")
    print("="*80 + "\n")

    # Setup
    model_name = "gpt2"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    inference = InferenceModule(
        model=model,
        tokenizer=tokenizer,
        backend="huggingface",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Environment
    env = SimpleTextEnvironment(
        inference=inference,
        prompts=["Explain quantum computing", "Describe photosynthesis"],
        max_steps=1
    )

    # Pareto reward computer with 3 objectives
    def accuracy_score(traj):
        # Mock accuracy (in practice, use a verifier model)
        return len(traj.response.split()) / 100.0

    def efficiency_score(traj):
        # Reward shorter, more concise answers
        return 1.0 - min(traj.step_count / 10.0, 1.0)

    def informativeness_score(traj):
        # Count informative words
        keywords = ['because', 'therefore', 'which', 'when', 'where']
        count = sum(1 for word in keywords if word in traj.response.lower())
        return count / 5.0

    reward_computer = ParetoRewardComputer(
        objective_functions=[accuracy_score, efficiency_score, informativeness_score],
        objective_names=['accuracy', 'efficiency', 'informativeness']
    )

    # Rollout and backprop
    rollout_manager = RolloutManager(env, RolloutConfig(num_parallel=16))

    optimizer = Adam(model.parameters(), lr=1e-5)
    backprop = BackpropModule(model=model, optimizer=optimizer)

    # PPO with Pareto rewards
    algo = PPOAlgorithm(
        env=env,
        rollout_manager=rollout_manager,
        reward_computer=reward_computer,
        backprop=backprop,
        config=AlgorithmConfig(
            batch_size=32,
            ppo_epochs=4,
            checkpoint_dir="./checkpoints/pareto_multi"
        )
    )

    print("\nStarting multi-objective training...")
    metrics = algo.train(num_iterations=5, progress_bar=True)

    print(f"\nTraining complete!")
    print(f"Pareto frontier evolved over training")

    return algo, metrics


def example4_custom_components():
    """
    Example 4: Using Custom Environment and Reward Computer

    Shows how to extend base classes for custom use cases.
    """
    print("\n" + "="*80)
    print("Example 4: Custom Environment and Reward Computer")
    print("="*80 + "\n")

    from rl_primitives import BaseEnvironment, State, RewardComputer
    import numpy as np

    # Custom environment: Count to 10 task
    class CountingEnvironment(BaseEnvironment):
        def __init__(self, inference):
            super().__init__(inference)
            self.target = 10

        def reset(self) -> State:
            return State(
                prompt="Count from 1 to 10:",
                metadata={"target": self.target}
            )

        def step(self, state: State) -> State:
            prompt = self.build_prompt(state)
            response = self.inference.generate([prompt])[0]

            new_state = state.copy()
            new_state.response = response
            new_state.step_count += 1
            return new_state

        def is_terminal(self, state: State) -> bool:
            return state.step_count >= 1  # Single step

        def build_prompt(self, state: State) -> str:
            return state.prompt

        def update_state(self, state: State, response: str) -> State:
            state.response = response
            return state

    # Custom reward computer
    class CountingRewardComputer(RewardComputer):
        def score_trajectory(self, trajectory: State) -> float:
            # Check if response contains numbers 1-10
            response = trajectory.response
            score = 0
            for i in range(1, 11):
                if str(i) in response:
                    score += 0.1
            return score

    # Setup
    model_name = "gpt2"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    inference = InferenceModule(
        model=model,
        tokenizer=tokenizer,
        backend="huggingface",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Use custom environment and reward
    env = CountingEnvironment(inference)
    reward_computer = CountingRewardComputer()

    # Standard components
    rollout_manager = RolloutManager(env, RolloutConfig(num_parallel=8))
    optimizer = Adam(model.parameters(), lr=1e-5)
    backprop = BackpropModule(model=model, optimizer=optimizer)

    # Train with GRPO
    algo = GRPOAlgorithm(
        env=env,
        rollout_manager=rollout_manager,
        reward_computer=reward_computer,
        backprop=backprop,
        config=AlgorithmConfig(batch_size=16)
    )

    print("\nTraining custom counting task...")
    metrics = algo.train(num_iterations=3, progress_bar=True)

    print(f"\nCustom training complete!")

    return algo, metrics


def main():
    """Run all examples"""
    print("\n" + "="*80)
    print("RL PRIMITIVES - COMPLETE TRAINING EXAMPLES")
    print("="*80)

    # You can run individual examples:

    # Example 1: PPO on simple text
    # algo1, metrics1 = example1_ppo_simple_text()

    # Example 2: GRPO on SVRL
    # algo2, metrics2 = example2_grpo_svrl()

    # Example 3: Multi-objective Pareto
    # algo3, metrics3 = example3_pareto_multiobjective()

    # Example 4: Custom components
    # algo4, metrics4 = example4_custom_components()

    print("\n" + "="*80)
    print("All examples defined! Uncomment in main() to run specific examples.")
    print("="*80 + "\n")

    print("Quick start guide:")
    print("1. example1_ppo_simple_text() - Basic PPO training")
    print("2. example2_grpo_svrl() - GRPO with verification tools")
    print("3. example3_pareto_multiobjective() - Multi-objective optimization")
    print("4. example4_custom_components() - Custom environment/rewards")
    print("\nTo run: Uncomment the desired example in main() and execute this script.")


if __name__ == "__main__":
    main()
