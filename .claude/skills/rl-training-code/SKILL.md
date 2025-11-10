---
name: rl-training-code
description: "Comprehensive RL training code generator for LLM post-training. Use when the user wants to: (1) Write training code for RL post-training of LLMs using verl or similar frameworks, (2) Implement specific RL algorithms like PPO, GRPO, or on-policy distillation, (3) Create training scripts for models loaded from HuggingFace with PyTorch, (4) Set up training environments with Dockerfiles, (5) Implement data loaders for various RL training formats, (6) Create evaluation harnesses, (7) Generate quick test notebooks for prototyping ideas from papers, or (8) Work with bf16/fp16/fp32 precision training."
---

# RL Training Code Generator

Generate production-ready RL training code for LLM post-training, inspired by verl and nanochat patterns.

## Overview

This skill provides complete, modular training code for RL post-training of LLMs. It includes training scripts (PPO, GRPO, on-policy distillation), data loaders (prompts, conversations, math, code), evaluation harnesses, Dockerfile generation, notebook templates, and reference documentation for verl patterns, RL algorithms, and nanochat insights.

## Core Capabilities

### 1. Generate Training Scripts

Read and adapt the training script templates from `scripts/`:

- `ppo_train.py` - Full PPO implementation with value function
- `on_policy_distillation.py` - Teacher-student distillation
- `data_loader.py` - Flexible data loading utilities
- `evaluation.py` - Multi-task evaluation harness
- `generate_dockerfile.py` - Environment setup automation

### 2. Create Modular, Production-Ready Code

All scripts support bf16/fp16/fp32 precision, HuggingFace model loading, pure PyTorch implementation, modular design, wandb/tensorboard logging, and automatic checkpointing.

### 3. Support Multiple RL Algorithms

Implementations provided for PPO (standard RLHF with value function), GRPO (simplified variant without value function), and on-policy distillation (teacher-student learning).

### 4. Flexible Data Loading

Support for prompts (JSONL, JSON, TXT), conversations (multi-turn), math problems (GSM8K-style), code generation, and custom formats.

### 5. Complete Evaluation Suite

Evaluate generation quality, math accuracy (GSM8K-style), code generation, perplexity, and custom metrics.

## Usage Patterns

### Pattern 1: Generate Full Training Script

User: "Write training code for PPO with Llama-2-7b on my dataset"

Process:
1. Read `scripts/ppo_train.py` template
2. Adapt configuration for user's model and data
3. Customize reward function if specified
4. Generate complete, runnable script

### Pattern 2: Create Quick Notebook for Testing

User: "I want to test this paper's idea quickly with a notebook"

Process:
1. Read `assets/notebook_template.ipynb`
2. Adapt for user's specific paper/idea
3. Customize reward function section
4. Provide ready-to-run notebook

### Pattern 3: Setup Training Environment

User: "Create a Docker image for verl training"

Process:
1. Use `scripts/generate_dockerfile.py`
2. Select appropriate style (base, verl, nanochat)
3. Configure CUDA version and dependencies
4. Generate Dockerfile with build instructions

### Pattern 4: Implement New RL Algorithm

User: "Implement [algorithm from paper] for LLM training"

Process:
1. Read `references/rl_algorithms.md` for patterns
2. Read `references/nanochat_patterns.md` for simplicity lessons
3. Adapt existing script template
4. Implement paper's specific modifications
5. Add evaluation if paper includes it

### Pattern 5: Create Custom Data Loader

User: "Create a data loader for my dataset format"

Process:
1. Read `scripts/data_loader.py` for patterns
2. Identify closest existing format
3. Create custom Dataset class
4. Add to factory function
5. Provide usage example

## Using the Resources

### Scripts Directory

**When to use each script:**

- **ppo_train.py**: Standard RLHF training with value function. Use when you need full PPO with critic.
  
- **on_policy_distillation.py**: Distilling larger model into smaller one. Use when compressing models or when you have a teacher model.
  
- **data_loader.py**: Creating custom data loading. Always read this when implementing new data formats.
  
- **evaluation.py**: Evaluating trained models. Use when setting up eval pipelines.
  
- **generate_dockerfile.py**: Creating training environments. Use when user needs Docker setup.

### References Directory

**Read these for context and patterns:**

- **verl_patterns.md**: Read when user mentions verl, needs verl installation, or wants production-scale training.
  
- **rl_algorithms.md**: Read when implementing new RL algorithms or comparing approaches.
  
- **nanochat_patterns.md**: Read when user wants minimal/simple implementations or mentions nanochat.

### Assets Directory

- **training_config.yaml**: Template configuration file. Use as reference for config structure.
  
- **notebook_template.ipynb**: Quick testing notebook. Use when user wants rapid prototyping.

## Important Implementation Notes

### Precision Handling

Always support configurable precision:

```python
if precision == "bf16":
    dtype = torch.bfloat16
elif precision == "fp16":
    dtype = torch.float16
else:
    dtype = torch.float32

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=dtype,
    device_map="auto"
)
```

### Reward Functions

Reward functions are task-specific. Provide placeholder that user must implement:

```python
def compute_reward(response: str, prompt: str) -> float:
    """
    TODO: Implement your actual reward function
    
    Examples:
    - Call reward model
    - Rule-based scoring
    - Verifiable metrics (math/code correctness)
    """
    # Placeholder implementation
    return 0.0
```

### Training Loop Structure

Follow this structure for consistency:

```python
for epoch in range(total_epochs):
    for batch in dataloader:
        # 1. Generate responses
        responses = generate_responses(batch['prompts'])
        
        # 2. Compute rewards
        rewards = compute_rewards(prompts, responses)
        
        # 3. Compute loss (algorithm-specific)
        loss = compute_loss(...)
        
        # 4. Backward pass
        loss.backward()
        
        # 5. Optimize
        if step % grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        # 6. Log metrics
        if step % log_interval == 0:
            log_metrics(loss, reward, etc.)
        
        # 7. Checkpoint
        if step % save_interval == 0:
            save_checkpoint()
```

### File Organization

When generating code, create well-organized structure:

```
training_project/
├── train.py              # Main training script
├── config.yaml           # Configuration
├── data/                 # Data directory
├── checkpoints/          # Model checkpoints
├── logs/                 # Training logs
└── evaluation/           # Evaluation results
```

## Common Workflows

### Workflow 1: Standard PPO Training

1. Read `scripts/ppo_train.py`
2. Adapt model name and data paths
3. Implement reward function
4. Generate script with proper config
5. Add evaluation if requested

### Workflow 2: Quick Paper Prototype

1. Read `assets/notebook_template.ipynb`
2. Read `references/rl_algorithms.md` for relevant algorithm
3. Identify paper's core contribution
4. Modify reward function section
5. Add paper-specific modifications
6. Provide ready-to-run notebook

### Workflow 3: verl Integration

1. Read `references/verl_patterns.md` thoroughly
2. Select appropriate backend (FSDP/Megatron)
3. Adapt script to use verl components
4. Generate Dockerfile with verl if needed
5. Provide installation instructions

### Workflow 4: Custom Dataset

1. Read `scripts/data_loader.py`
2. Understand user's data format
3. Create custom Dataset class
4. Add to create_dataloader factory
5. Provide usage example
6. Include data format validation

## Best Practices

1. **Always Make Code Runnable**: Generated code should work out of the box (after user fills in TODOs)

2. **Clear TODOs**: Mark what user must implement with clear comments

3. **Type Hints**: Include type hints for better code clarity

4. **Error Handling**: Include basic error handling and validation

5. **Documentation**: Add docstrings to all functions

6. **Configuration**: Use config objects, not hardcoded values

7. **Logging**: Include comprehensive logging

8. **Checkpointing**: Always save models during training

## Example Scenarios

### Scenario: "Write code for PPO with my model and dataset"

Read ppo_train.py template and adapt MODEL_NAME to user's model, data loading to user's format, reward function placeholder, and any specific user requirements. Output complete, runnable script.

### Scenario: "Create a notebook to test GRPO quickly"

Read notebook_template.ipynb and references/rl_algorithms.md for GRPO. Modify training loop to use GRPO instead of PPO, simplify (remove value function), add GRPO-specific hyperparameters. Output notebook with clear TODOs.

### Scenario: "Set up verl training environment"

Read references/verl_patterns.md. Use scripts/generate_dockerfile.py to generate verl-style Dockerfile. Add installation instructions and provide example training command.

### Scenario: "Implement [paper]'s algorithm"

Read references/rl_algorithms.md for base understanding and references/nanochat_patterns.md for simplification ideas. Identify closest existing algorithm and read that algorithm's script. Implement paper's modifications, add comments explaining changes, and provide usage example.

## Notes

- **Modular Design**: Scripts are designed to be easily customizable
- **No Black Boxes**: All implementations are transparent and hackable
- **Production-Ready**: Code includes logging, checkpointing, evaluation
- **Well-Tested Patterns**: Based on verl and nanochat proven approaches
- **Precision Aware**: Always support bf16/fp16/fp32
- **Device Agnostic**: Code should work on CPU/single GPU/multi-GPU
