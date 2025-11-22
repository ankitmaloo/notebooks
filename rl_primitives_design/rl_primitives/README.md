# RL Primitives

**Modular, Production-Ready Components for Reinforcement Learning of LLMs**

A clean, extensible architecture for RL training based on the design in `../v2.md`.

## 🎯 Overview

RL Primitives provides a set of reusable components that separate concerns in RL training:

1. **InferenceModule** - Handles model inference (vLLM, HuggingFace)
2. **Environment** - Defines interaction logic (no reward computation)
3. **RolloutManager** - Manages parallel trajectory collection
4. **RewardComputer** - Computes rewards after collection (absolute, relative, Pareto)
5. **BackpropModule** - Handles gradients and model updates
6. **Algorithms** - Orchestrates everything (PPO, GRPO, REINFORCE)

## 🚀 Quick Start

### Installation

```bash
# From the rl_primitives_design directory
pip install -e .
```

### Basic Usage

```python
from rl_primitives import (
    InferenceModule,
    SimpleTextEnvironment,
    RolloutManager,
    BackpropModule,
    PPOAlgorithm,
    AlgorithmConfig,
)
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 1. Setup inference
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
inference = InferenceModule(model=model, tokenizer=tokenizer)

# 2. Create environment
env = SimpleTextEnvironment(
    inference=inference,
    prompts=["Explain AI:", "What is ML?"],
    max_steps=1
)

# 3. Setup training components
rollout_manager = RolloutManager(env, num_parallel=16)
backprop = BackpropModule(model=model, optimizer=torch.optim.Adam(model.parameters()))

# 4. Define rewards
class MyRewardComputer:
    def compute_rewards(self, trajectories, method="absolute"):
        return [len(t.response.split()) / 50.0 for t in trajectories]

# 5. Create algorithm and train
algo = PPOAlgorithm(
    env=env,
    rollout_manager=rollout_manager,
    reward_computer=MyRewardComputer(),
    backprop=backprop,
    config=AlgorithmConfig(batch_size=32)
)

metrics = algo.train(num_iterations=100, progress_bar=True)
```

## 📦 Components

### 1. InferenceModule

Multi-backend inference with generation, logits, and tool calling support.

**Backends:**
- `HuggingFaceBackend` - Full features, slower
- `VLLMBackend` - High performance, production use

**Key Methods:**
```python
inference = InferenceModule("meta-llama/Llama-2-7b-hf", backend="vllm")

# Generate
results = inference.generate(prompts, max_new_tokens=100)

# Get logprobs for policy gradient
logprobs = inference.get_logprobs(prompts, responses)

# Get logits for KL divergence
logits = inference.get_logits(prompts, responses)

# Generate with tools
responses, tool_calls = inference.batch_generate_with_tools(
    prompts,
    available_tools=['calculator', 'search']
)
```

### 2. Environment

Pure interaction logic - no reward computation.

**Built-in Environments:**
- `SimpleTextEnvironment` - Basic text generation
- `SVRLEnvironment` - Self-verification with tool budget
- `ConversationEnvironment` - Multi-turn dialogue

**Custom Environment:**
```python
from rl_primitives import BaseEnvironment, State

class MyEnvironment(BaseEnvironment):
    def reset(self) -> State:
        return State(prompt="Your task...")

    def step(self, state: State) -> State:
        response = self.inference.generate([state.prompt])[0]
        state.response = response
        state.step_count += 1
        return state

    def is_terminal(self, state: State) -> bool:
        return state.step_count >= 10 or "DONE" in state.response

    def build_prompt(self, state: State) -> str:
        return state.prompt

    def update_state(self, state: State, response: str) -> State:
        state.response = response
        return state
```

### 3. RolloutManager

Manages parallel rollout collection with different trajectory lengths.

```python
from rl_primitives import RolloutManager, RolloutConfig

config = RolloutConfig(
    num_parallel=32,
    maintain_parallel_count=False,
    verbose=True,
    enable_progress_bar=True
)

manager = RolloutManager(env, config)

# Collect trajectories
trajectories = manager.collect_rollouts(
    min_trajectories=100,
    max_steps_per_rollout=50
)

# Get statistics
stats = manager.get_statistics()
print(f"Avg length: {stats['avg_trajectory_length']}")
```

### 4. RewardComputer

Compute rewards after trajectory collection (not during).

**Built-in Computers:**
- `SVRLRewardComputer` - Task quality vs verification cost
- `ParetoRewardComputer` - Multi-objective Pareto ranking

**Reward Methods:**
- `absolute` - Independent trajectory scoring
- `relative` - GRPO-style ranking (normalized ranks)
- `pareto` - Multi-objective Pareto frontier

```python
from rl_primitives import ParetoRewardComputer

reward_computer = ParetoRewardComputer(
    objective_functions=[accuracy_fn, efficiency_fn, safety_fn],
    objective_names=['accuracy', 'efficiency', 'safety']
)

rewards = reward_computer.compute_rewards(trajectories, method="pareto")
```

### 5. BackpropModule

Gradient computation and model updates.

```python
from rl_primitives import BackpropModule, BackpropConfig

config = BackpropConfig(
    gamma=0.99,
    gae_lambda=0.95,
    ppo_clip_range=0.2,
    kl_coef=0.1,
    normalize_advantages=True
)

backprop = BackpropModule(
    model=model,
    ref_model=ref_model,  # Optional
    optimizer=optimizer,
    config=config
)

# Compute advantages
advantages = backprop.compute_advantages(rewards, values)

# Compute PPO loss
loss = backprop.compute_ppo_loss(trajectories, advantages)

# Update model
backprop.update(loss)

# Update reference model
backprop.update_ref_model(method="ema")
```

### 6. Algorithms

Complete training loops with checkpointing and logging.

**Available Algorithms:**
- `PPOAlgorithm` - Proximal Policy Optimization
- `GRPOAlgorithm` - Group Relative Policy Optimization
- `REINFORCEAlgorithm` - Vanilla policy gradient

```python
from rl_primitives import PPOAlgorithm, AlgorithmConfig

config = AlgorithmConfig(
    batch_size=32,
    ppo_epochs=4,
    learning_rate=1e-6,
    checkpoint_dir="./checkpoints",
    save_every=10,
    use_wandb=True,
    wandb_project="my-rl-project"
)

algo = PPOAlgorithm(
    env=env,
    rollout_manager=rollout_manager,
    reward_computer=reward_computer,
    backprop=backprop,
    config=config
)

# Train
metrics = algo.train(num_iterations=1000, progress_bar=True)

# Save checkpoint
algo.save_checkpoint(iteration=1000, best=True)
```

## 🎓 Examples

See `../examples/` for complete examples:

1. **`complete_training_example.py`** - End-to-end training pipelines
   - PPO on simple text generation
   - GRPO on self-verification RL
   - Multi-objective Pareto optimization
   - Custom environments and rewards

2. **`inference_examples.py`** - InferenceModule usage
3. **`backprop_usage.py`** - BackpropModule examples

## 🏗️ Architecture

Based on the design in `../v2.md`, this architecture provides:

✅ **Clean Separation of Concerns**
- Environment = interaction logic only
- RolloutManager = parallel collection
- RewardComputer = scoring after collection
- BackpropModule = gradients and updates
- Algorithm = orchestration

✅ **Variable-Length Trajectories**
- Natural handling of different completion times
- Efficient batched inference
- No padding waste

✅ **Multiple Reward Strategies**
- Absolute: Independent scoring
- Relative: GRPO-style ranking
- Pareto: Multi-objective optimization

✅ **Production-Ready**
- Full type annotations
- Comprehensive docstrings
- Checkpointing and logging
- Progress tracking
- Wandb integration

✅ **Extensible**
- Easy to add new environments
- Easy to add new reward functions
- Easy to add new algorithms

## 📚 Documentation

- **InferenceModule**: See `../docs/INFERENCE_MODULE.md`
- **BackpropModule**: See `BACKPROP_README.md`
- **RewardComputer**: See `examples/README_reward_computer.md`

## 🔧 Development

```bash
# Run tests (when available)
pytest tests/

# Run examples
python examples/complete_training_example.py

# Check specific components
python -m rl_primitives.environment  # Demo environment
python -m rl_primitives.inference    # Demo inference
```

## 📝 Design Principles

1. **Separation**: Each component has one clear responsibility
2. **Reusability**: Components work across different experiments
3. **Simplicity**: Minimal configuration, sensible defaults
4. **Flexibility**: Easy to customize and extend
5. **Performance**: Efficient batching and parallelization

## 🎯 Use Cases

### Monday: Pareto Multi-Objective RL
```python
env = MyParetoEnv(inference)
reward_computer = ParetoRewardComputer(objectives=[acc, eff, safety])
algo = PPOAlgorithm(env, rollout_mgr, reward_computer, backprop)
```

### Tuesday: Self-Verification RL
```python
env = SVRLEnvironment(inference, tools, budget=10.0)
reward_computer = SVRLRewardComputer()
algo = GRPOAlgorithm(env, rollout_mgr, reward_computer, backprop)
```

### Wednesday: Combine Them!
```python
class CombinedEnv(MyParetoEnv, SVRLEnvironment):
    pass  # Mix both capabilities
```

## 🚀 Performance Tips

1. **Use vLLM backend** for production (10x faster)
2. **Increase num_parallel** for better GPU utilization
3. **Use GRPO** for faster prototyping (no value function)
4. **Enable AMP** for mixed precision training
5. **Use relative rewards** for better stability

## 📄 License

See main repository license.

## 🤝 Contributing

Contributions welcome! This is a research-focused modular design.

## 📞 Support

See documentation in each module and check `../v2.md` for architecture details.
