#!/usr/bin/env python3
"""
verl PPO Training Example with Modular RL Primitives
====================================================

This example demonstrates how to use verl (a production-ready RL framework)
while maintaining the modular primitives design from this repository.

verl advantages:
- Production-scale distributed training (FSDP, Megatron-LM)
- Optimized actor-critic separation
- Built-in vLLM/SGLang inference backends
- Hybrid Engine for efficient generation

This script shows how to:
1. Use verl's optimized components where beneficial
2. Maintain modular design for customization
3. Integrate with existing reward computers and environments
"""

import os
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import torch
import torch.distributed as dist
from torch.optim import Adam
from transformers import AutoTokenizer

# verl imports
try:
    from verl import DataProto
    from verl.trainer.ppo import (
        core_algos,
        ray_trainer,
    )
    from verl.utils.model import (
        update_model_config,
        print_model_size
    )
    from verl.utils.reward_score import gsm8k_rule_based_reward
    from single_controller.base import Worker, WorkerGroup
    from single_controller.ray import RayResourcePool, RayClassWithInitArgs
    VERL_AVAILABLE = True
except ImportError:
    VERL_AVAILABLE = False
    print("Warning: verl not installed. Install with: pip install verl")


@dataclass
class VerlPPOConfig:
    """Configuration for verl PPO training"""

    # Model configuration
    model_name: str = "meta-llama/Llama-2-7b-hf"
    ref_model_name: Optional[str] = None  # Uses model_name if None

    # Training configuration
    total_epochs: int = 10
    rollout_batch_size: int = 512
    train_batch_size: int = 256
    ppo_epochs: int = 4

    # PPO hyperparameters
    learning_rate: float = 1e-6
    gamma: float = 1.0  # No discounting for language
    lam: float = 0.95   # GAE lambda
    clip_range: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    kl_coef: float = 0.05
    max_grad_norm: float = 1.0

    # Generation configuration
    max_prompt_length: int = 512
    max_response_length: int = 512
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0

    # verl-specific: inference backend
    actor_inference_backend: str = "vllm"  # "vllm", "sglang", or "hf"
    vllm_tensor_parallel_size: int = 1
    vllm_gpu_memory_utilization: float = 0.4

    # Distributed training
    fsdp_enabled: bool = True
    fsdp_sharding_strategy: str = "FULL_SHARD"  # or "SHARD_GRAD_OP"

    # Paths
    data_path: str = "./data/prompts.jsonl"
    output_dir: str = "./checkpoints/verl_ppo"
    log_dir: str = "./logs/verl_ppo"

    # Logging
    log_interval: int = 10
    save_interval: int = 100
    eval_interval: int = 50

    # Device
    local_rank: int = 0
    device: str = "cuda"


class VerlActorWrapper:
    """
    Wrapper around verl's Actor for compatibility with our primitives.

    verl's Actor handles:
    - Model loading with FSDP
    - Efficient generation via vLLM/SGLang
    - PPO training step with value function
    """

    def __init__(self, config: VerlPPOConfig):
        self.config = config
        self.device = config.device

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Initialize verl components (simplified - actual verl setup is more complex)
        self._setup_verl_components()

    def _setup_verl_components(self):
        """Setup verl's distributed components"""
        if not VERL_AVAILABLE:
            raise RuntimeError("verl is required for this example")

        # In production verl setup:
        # 1. Create RayResourcePool for distributed workers
        # 2. Setup Actor workers with FSDP
        # 3. Setup Critic workers (separate for efficiency)
        # 4. Setup vLLM/SGLang inference backend
        # 5. Setup reference model for KL penalty

        # For this example, we'll use simplified local setup
        print(f"Setting up verl components with {self.config.actor_inference_backend} backend...")

        # In real implementation, you would:
        # self.actor_pool = RayResourcePool(...)
        # self.critic_pool = RayResourcePool(...)
        # self.ref_pool = RayResourcePool(...)

    def generate_rollouts(
        self,
        prompts: List[str],
        batch_size: int
    ) -> DataProto:
        """
        Generate rollouts using verl's optimized generation.

        Returns DataProto with:
        - prompts: input prompts
        - responses: generated text
        - prompt_ids: tokenized prompts
        - response_ids: tokenized responses
        - values: value function predictions
        - old_log_probs: log probs from generation
        """
        # Tokenize prompts
        tokenized = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.config.max_prompt_length,
            return_tensors="pt"
        )

        # In real verl:
        # data = self.actor_pool.submit_sync(
        #     fn=generate_sequences,
        #     prompts=prompts,
        #     max_new_tokens=self.config.max_response_length,
        #     temperature=self.config.temperature,
        #     top_p=self.config.top_p,
        #     do_sample=True
        # )

        # For this example, return mock DataProto structure
        data = DataProto(
            prompts=prompts,
            responses=["Generated response placeholder"] * len(prompts),
            # Additional fields would be filled by verl's generation
        )

        return data

    def compute_advantages(
        self,
        data: DataProto,
        rewards: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute GAE advantages using verl's implementation.

        verl provides optimized advantage computation with:
        - Proper GAE calculation
        - Efficient batching
        - Distributed synchronization
        """
        # In real verl:
        # advantages = core_algos.compute_gae_advantage(
        #     values=data.values,
        #     rewards=rewards,
        #     gamma=self.config.gamma,
        #     lam=self.config.lam
        # )

        # Simplified GAE for this example
        advantages = rewards - rewards.mean()
        return advantages

    def ppo_update(
        self,
        data: DataProto,
        advantages: torch.Tensor,
        returns: torch.Tensor
    ) -> Dict[str, float]:
        """
        Execute PPO update using verl's optimized implementation.

        verl's PPO implementation includes:
        - Multiple epochs over same data
        - Clipped surrogate loss
        - Value function loss
        - Entropy regularization
        - KL penalty vs reference model
        - Gradient clipping and normalization
        """
        metrics = {}

        # In real verl:
        # for ppo_epoch in range(self.config.ppo_epochs):
        #     batch_metrics = self.actor_pool.submit_sync(
        #         fn=ppo_step,
        #         data=data,
        #         advantages=advantages,
        #         returns=returns,
        #         clip_range=self.config.clip_range,
        #         vf_coef=self.config.vf_coef,
        #         ent_coef=self.config.ent_coef
        #     )
        #
        #     # Also compute KL vs reference model
        #     kl_div = self.ref_pool.submit_sync(
        #         fn=compute_kl,
        #         data=data
        #     )
        #
        #     metrics['policy_loss'] = batch_metrics['policy_loss']
        #     metrics['value_loss'] = batch_metrics['value_loss']
        #     metrics['entropy'] = batch_metrics['entropy']
        #     metrics['kl_div'] = kl_div

        # Placeholder metrics
        metrics = {
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy': 0.0,
            'kl_div': 0.0,
        }

        return metrics


class VerlRewardComputer:
    """
    Reward computation compatible with verl's DataProto format.

    This bridges our modular reward design with verl's data structures.
    """

    def __init__(self, reward_fn_type: str = "rule_based"):
        self.reward_fn_type = reward_fn_type

    def compute_rewards(
        self,
        data: DataProto,
        task_type: str = "general"
    ) -> torch.Tensor:
        """
        Compute rewards for verl DataProto.

        Args:
            data: DataProto with prompts and responses
            task_type: Type of task ("general", "math", "code")

        Returns:
            Tensor of rewards, shape [batch_size]
        """
        rewards = []

        for prompt, response in zip(data.prompts, data.responses):
            if task_type == "math" and VERL_AVAILABLE:
                # Use verl's GSM8K rule-based reward
                reward = gsm8k_rule_based_reward(prompt, response)
            else:
                # Custom reward function
                reward = self._compute_general_reward(prompt, response)

            rewards.append(reward)

        return torch.tensor(rewards, dtype=torch.float32)

    def _compute_general_reward(self, prompt: str, response: str) -> float:
        """
        TODO: Implement your custom reward function

        Examples:
        - Call a reward model (e.g., OpenAssistant reward model)
        - Rule-based scoring (length, keywords, format)
        - Verification (code execution, math verification)
        """
        # Placeholder: reward longer, more detailed responses
        length_score = min(len(response.split()) / 100.0, 1.0)

        # Reward informative words
        informative_words = ['because', 'therefore', 'however', 'which']
        info_score = sum(0.1 for word in informative_words
                        if word.lower() in response.lower())

        return length_score + info_score


def load_prompts(data_path: str) -> List[str]:
    """Load prompts from JSONL file"""
    import json
    prompts = []

    if not os.path.exists(data_path):
        # Return sample prompts if file doesn't exist
        return [
            "Explain how photosynthesis works.",
            "What are the key principles of machine learning?",
            "Describe the water cycle in detail.",
            "How does a computer processor work?",
        ]

    with open(data_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            prompts.append(data.get('prompt', data.get('text', '')))

    return prompts


def train_verl_ppo(config: VerlPPOConfig):
    """
    Main training loop using verl with modular primitives.

    This demonstrates how to:
    1. Use verl's optimized components (Actor, Critic, vLLM generation)
    2. Maintain modularity with custom reward functions
    3. Leverage verl's distributed training capabilities
    """

    print("="*80)
    print("verl PPO Training with Modular Primitives")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Model: {config.model_name}")
    print(f"  Inference backend: {config.actor_inference_backend}")
    print(f"  FSDP enabled: {config.fsdp_enabled}")
    print(f"  Rollout batch size: {config.rollout_batch_size}")
    print(f"  Training batch size: {config.train_batch_size}")
    print(f"  Total epochs: {config.total_epochs}")
    print("="*80)

    # Setup
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)

    # Initialize verl actor wrapper
    actor = VerlActorWrapper(config)

    # Initialize reward computer (modular!)
    reward_computer = VerlRewardComputer(reward_fn_type="rule_based")

    # Load training prompts
    prompts = load_prompts(config.data_path)
    print(f"\nLoaded {len(prompts)} training prompts")

    # Training loop
    global_step = 0

    for epoch in range(config.total_epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1}/{config.total_epochs}")
        print(f"{'='*80}")

        # Sample batch of prompts
        import random
        batch_prompts = random.sample(prompts, min(config.rollout_batch_size, len(prompts)))

        # 1. Generate rollouts using verl's optimized generation
        print(f"\n[Step 1/5] Generating {len(batch_prompts)} rollouts...")
        data = actor.generate_rollouts(
            prompts=batch_prompts,
            batch_size=config.rollout_batch_size
        )

        # 2. Compute rewards (modular reward function!)
        print(f"[Step 2/5] Computing rewards...")
        rewards = reward_computer.compute_rewards(data, task_type="general")

        print(f"  Reward stats: mean={rewards.mean():.3f}, "
              f"std={rewards.std():.3f}, "
              f"min={rewards.min():.3f}, "
              f"max={rewards.max():.3f}")

        # 3. Compute advantages using GAE
        print(f"[Step 3/5] Computing advantages with GAE...")
        advantages = actor.compute_advantages(data, rewards)

        # Compute returns (for value function training)
        returns = advantages + rewards.mean()  # Simplified

        # 4. PPO update (multiple epochs)
        print(f"[Step 4/5] Running PPO update ({config.ppo_epochs} epochs)...")
        metrics = actor.ppo_update(
            data=data,
            advantages=advantages,
            returns=returns
        )

        # 5. Logging
        print(f"[Step 5/5] Logging metrics...")
        print(f"\n  Training metrics:")
        print(f"    Policy loss: {metrics['policy_loss']:.4f}")
        print(f"    Value loss:  {metrics['value_loss']:.4f}")
        print(f"    Entropy:     {metrics['entropy']:.4f}")
        print(f"    KL div:      {metrics['kl_div']:.4f}")

        global_step += 1

        # Save checkpoint
        if global_step % config.save_interval == 0:
            checkpoint_path = os.path.join(
                config.output_dir,
                f"checkpoint_step_{global_step}.pt"
            )
            print(f"\n  Saving checkpoint to {checkpoint_path}")
            # In real implementation: save model state

        # Evaluation
        if global_step % config.eval_interval == 0:
            print(f"\n  Running evaluation...")
            # In real implementation: run eval on held-out set

    print("\n" + "="*80)
    print("Training complete!")
    print("="*80)


def example_with_custom_reward():
    """
    Example showing how to use custom reward function with verl.

    This demonstrates the key advantage of our modular design:
    - verl handles the heavy lifting (distributed training, efficient generation)
    - You maintain full control over rewards, data, and training logic
    """

    # Create custom reward computer
    class CustomRewardComputer(VerlRewardComputer):
        def __init__(self, reward_model_name: str = "OpenAssistant/reward-model-deberta-v3-large"):
            super().__init__()
            # Load your reward model
            # self.reward_model = AutoModelForSequenceClassification.from_pretrained(
            #     reward_model_name
            # )
            print(f"Initialized with reward model: {reward_model_name}")

        def _compute_general_reward(self, prompt: str, response: str) -> float:
            # Use your reward model
            # inputs = self.tokenizer(prompt + response, return_tensors="pt")
            # score = self.reward_model(**inputs).logits[0].item()
            # return score

            # Placeholder
            return 1.0

    # Setup config
    config = VerlPPOConfig(
        model_name="meta-llama/Llama-2-7b-hf",
        total_epochs=5,
        rollout_batch_size=128,
        actor_inference_backend="vllm",  # Use vLLM for fast generation
    )

    # Train with custom reward
    # train_verl_ppo(config)

    print("\nCustom reward example configured!")
    print("Uncomment train_verl_ppo(config) to run.")


if __name__ == "__main__":
    """
    Example usage of verl PPO training with modular primitives.

    Prerequisites:
    1. Install verl: pip install verl
    2. Install vLLM: pip install vllm
    3. Prepare your dataset in JSONL format

    Run:
        python verl_ppo_training.py
    """

    print("\n" + "="*80)
    print("verl PPO Training Example")
    print("="*80)
    print("\nThis example shows how to use verl's production-ready components")
    print("while maintaining the modular primitives design.\n")

    print("verl advantages:")
    print("  ✓ Distributed training with FSDP/Megatron")
    print("  ✓ Fast generation with vLLM/SGLang")
    print("  ✓ Optimized actor-critic separation")
    print("  ✓ Production-tested PPO implementation")

    print("\nModular primitives advantages:")
    print("  ✓ Custom reward functions")
    print("  ✓ Flexible data loading")
    print("  ✓ Easy experimentation")
    print("  ✓ Full control over training logic")

    print("\n" + "="*80)
    print("Best of both worlds!")
    print("="*80)

    if not VERL_AVAILABLE:
        print("\n⚠ Warning: verl not installed")
        print("Install with: pip install verl")
        print("\nThis script will show the structure but won't run training.")

    # Example 1: Basic verl PPO training
    print("\n\nExample 1: Basic verl PPO Training")
    print("-" * 80)

    config = VerlPPOConfig(
        model_name="meta-llama/Llama-2-7b-hf",
        total_epochs=3,
        rollout_batch_size=64,
        train_batch_size=32,
        actor_inference_backend="vllm",
        data_path="./data/prompts.jsonl",
        output_dir="./checkpoints/verl_ppo_example"
    )

    # Uncomment to run:
    # train_verl_ppo(config)

    print("Configuration created. Uncomment train_verl_ppo(config) to run.")

    # Example 2: Custom reward function
    print("\n\nExample 2: verl with Custom Reward Function")
    print("-" * 80)
    # example_with_custom_reward()

    print("\n" + "="*80)
    print("Examples configured!")
    print("Uncomment the function calls to run training.")
    print("="*80)
