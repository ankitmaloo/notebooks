# RL Training Primitives: Complete Working Examples

This document provides complete, runnable examples showing how to use the functional primitives architecture.

---

## Example 1: Basic GRPO Training Loop

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List
import math

# ============================================================================
# Setup
# ============================================================================

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-8B",
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8B")

# Create reference model (frozen copy)
import copy
ref_model = copy.deepcopy(model)
ref_model.eval()
for param in ref_model.parameters():
    param.requires_grad = False

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# ============================================================================
# Define Custom Reward Function
# ============================================================================

def correctness_reward(rollout: Rollout) -> float:
    """Check if generation contains correct answer"""
    text = rollout.generation.text.lower()
    
    # Simple heuristic: check for key phrases
    if "correct" in text or "yes" in text:
        return 1.0
    elif "incorrect" in text or "no" in text:
        return 0.0
    else:
        return 0.5

# ============================================================================
# Training Configuration
# ============================================================================

gen_config = GenerationConfig(
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)

algorithm_config = create_algorithm_config(
    algorithm="grpo",
    params={"kl_coef": 0.1}
)

initial_state = TrainingState(
    iteration=0,
    global_step=0,
    model=model,
    optimizer=optimizer,
    ref_model=ref_model,
    stats_history=[]
)

# ============================================================================
# Prompt Dataset
# ============================================================================

training_prompts = [
    "What is 2+2? Explain your answer.",
    "Is Paris the capital of France?",
    "What color is the sky?",
    # ... more prompts
] * 10  # 30 total prompts

def load_prompts(iteration: int) -> List[str]:
    """Load prompts for iteration"""
    batch_size = 4
    start_idx = (iteration * batch_size) % len(training_prompts)
    end_idx = start_idx + batch_size
    return training_prompts[start_idx:end_idx]

# ============================================================================
# Run Training
# ============================================================================

final_state = train(
    initial_state=initial_state,
    prompt_loader=load_prompts,
    num_iterations=50,
    gen_config=gen_config,
    algorithm_config=algorithm_config,
    reward_fn=correctness_reward,
    logger=log_to_console,
    eval_every=10
)

print(f"Training complete! Final iteration: {final_state.iteration}")
```

---

## Example 2: Vector Rewards with Custom Aggregation

```python
# ============================================================================
# Multi-Objective Reward Functions
# ============================================================================

def correctness_reward(rollout: Rollout) -> float:
    """Evaluate correctness (0-1)"""
    # Could use an LLM judge here
    text = rollout.generation.text
    # Simplified: check length and keywords
    return 1.0 if len(text) > 50 and "because" in text else 0.5

def helpfulness_reward(rollout: Rollout) -> float:
    """Evaluate helpfulness (0-1)"""
    text = rollout.generation.text
    # Check for helpful markers
    helpful_markers = ["here's how", "you can", "i recommend", "step"]
    score = sum(1 for marker in helpful_markers if marker in text.lower())
    return min(score / 4, 1.0)

def safety_reward(rollout: Rollout) -> float:
    """Evaluate safety (0-1)"""
    text = rollout.generation.text.lower()
    # Check for unsafe content
    unsafe_markers = ["hack", "illegal", "dangerous"]
    if any(marker in text for marker in unsafe_markers):
        return 0.0
    return 1.0

def conciseness_reward(rollout: Rollout) -> float:
    """Reward conciseness (inverse of length, normalized)"""
    length = len(rollout.generation.text)
    # Prefer 100-300 characters
    if 100 <= length <= 300:
        return 1.0
    elif length < 100:
        return length / 100
    else:
        return max(300 / length, 0.3)

# ============================================================================
# Compose Vector Reward Function
# ============================================================================

def multi_objective_reward(rollout: Rollout) -> List[float]:
    """Compute all reward objectives"""
    return [
        correctness_reward(rollout),
        helpfulness_reward(rollout),
        safety_reward(rollout),
        conciseness_reward(rollout)
    ]

# ============================================================================
# Custom Aggregation Strategies
# ============================================================================

def weighted_sum_aggregator(rewards: List[float]) -> float:
    """Weighted sum of objectives"""
    weights = [0.4, 0.3, 0.2, 0.1]  # correctness, helpfulness, safety, conciseness
    return sum(w * r for w, r in zip(weights, rewards))

def min_threshold_aggregator(rewards: List[float], threshold: float = 0.5) -> float:
    """Product of rewards above threshold, 0 if any below"""
    if any(r < threshold for r in rewards):
        return 0.0
    return math.prod(rewards)

def lexicographic_aggregator(rewards: List[float]) -> float:
    """Lexicographic ordering: safety > correctness > helpfulness > conciseness"""
    safety, correctness, helpfulness, conciseness = rewards[2], rewards[0], rewards[1], rewards[3]
    
    # Safety is paramount
    if safety < 0.9:
        return safety * 0.1  # Heavy penalty
    
    # Then correctness
    if correctness < 0.7:
        return safety * 0.5 + correctness * 0.3
    
    # Then consider all
    return weighted_sum_aggregator(rewards)

# ============================================================================
# Use in Training
# ============================================================================

def final_reward_fn(rollout: Rollout) -> float:
    """Compute final scalar reward from vector"""
    vector_rewards = multi_objective_reward(rollout)
    
    # Store vector in metadata for analysis
    rollout = rollout._replace(
        reward_metadata={"vector_rewards": vector_rewards}
    )
    
    # Aggregate to scalar
    return weighted_sum_aggregator(vector_rewards)

# Train with multi-objective reward
final_state = train(
    initial_state=initial_state,
    prompt_loader=load_prompts,
    num_iterations=100,
    gen_config=gen_config,
    algorithm_config=algorithm_config,
    reward_fn=final_reward_fn,
    logger=log_to_console
)
```

---

## Example 3: Algorithm Switching Mid-Training

```python
# ============================================================================
# Dynamic Algorithm Switching
# ============================================================================

class AlgorithmScheduler:
    """Schedule algorithm changes during training"""
    
    def __init__(self):
        self.phase_thresholds = {
            0: ("grpo", {"kl_coef": 0.2}),      # Iterations 0-29: GRPO with high KL
            30: ("grpo", {"kl_coef": 0.05}),    # Iterations 30-59: GRPO with low KL
            60: ("dapo", {"beta": 0.1, "pair_selector": self.top_bottom_pairs})  # 60+: DAPO
        }
    
    def top_bottom_pairs(self, batch: Batch) -> List[tuple[Rollout, Rollout]]:
        """Select top vs bottom pairs"""
        sorted_rollouts = sorted(
            batch.rollouts,
            key=lambda r: r.rewards if isinstance(r.rewards, float) else sum(r.rewards),
            reverse=True
        )
        
        pairs = []
        n = len(sorted_rollouts)
        for i in range(n // 2):
            preferred = sorted_rollouts[i]
            dispreferred = sorted_rollouts[n - 1 - i]
            pairs.append((preferred, dispreferred))
        
        return pairs
    
    def get_config(self, iteration: int) -> Dict[str, Any]:
        """Get algorithm config for iteration"""
        # Find the most recent threshold
        applicable_threshold = 0
        for threshold in sorted(self.phase_thresholds.keys(), reverse=True):
            if iteration >= threshold:
                applicable_threshold = threshold
                break
        
        algo, params = self.phase_thresholds[applicable_threshold]
        return create_algorithm_config(algo, params)

# ============================================================================
# Custom Training Loop with Algorithm Switching
# ============================================================================

def train_with_algorithm_switching(
    initial_state: TrainingState,
    prompt_loader: Callable[[int], List[str]],
    num_iterations: int,
    gen_config: GenerationConfig,
    algorithm_scheduler: AlgorithmScheduler,
    reward_fn: Callable,
    logger: Callable
) -> TrainingState:
    """Training loop with dynamic algorithm switching"""
    
    state = initial_state
    
    for iteration in range(num_iterations):
        # Get algorithm for this iteration
        algorithm_config = algorithm_scheduler.get_config(iteration)
        
        # Log algorithm if it changed
        if iteration == 0 or algorithm_config != algorithm_scheduler.get_config(iteration - 1):
            print(f"\n=== Switching to {algorithm_config['algorithm'].upper()} at iteration {iteration} ===\n")
        
        # Load prompts
        prompts = prompt_loader(iteration)
        
        # Training iteration
        state, batch, stats = training_iteration(
            state,
            prompts,
            state.model,
            tokenizer,
            gen_config,
            algorithm_config,
            reward_fn
        )
        
        # Log
        logger(state, stats)
    
    return state

# ============================================================================
# Run Training with Switching
# ============================================================================

scheduler = AlgorithmScheduler()

final_state = train_with_algorithm_switching(
    initial_state=initial_state,
    prompt_loader=load_prompts,
    num_iterations=100,
    gen_config=gen_config,
    algorithm_scheduler=scheduler,
    reward_fn=correctness_reward,
    logger=log_to_console
)
```

---

## Example 4: Tool Call Tracking with Multi-Turn Rollouts

```python
# ============================================================================
# Tool Definitions
# ============================================================================

AVAILABLE_TOOLS = {
    "calculator": {
        "description": "Perform arithmetic calculations",
        "parameters": {"expression": "string"}
    },
    "search": {
        "description": "Search for information",
        "parameters": {"query": "string"}
    },
    "python": {
        "description": "Execute Python code",
        "parameters": {"code": "string"}
    }
}

# ============================================================================
# Tool Parsing and Execution
# ============================================================================

def parse_tool_calls_xml(text: str) -> List[Dict[str, Any]]:
    """Parse XML-style tool calls"""
    import re
    
    pattern = r'<tool_call name="([^"]+)">(.*?)</tool_call>'
    matches = re.findall(pattern, text, re.DOTALL)
    
    tool_calls = []
    for name, args_text in matches:
        if name in AVAILABLE_TOOLS:
            tool_calls.append({
                "id": f"call_{len(tool_calls)}",
                "name": name,
                "arguments": {"input": args_text.strip()}
            })
    
    return tool_calls

def execute_tool_mock(tool_call: Dict[str, Any]) -> Dict[str, Any]:
    """Mock tool execution"""
    name = tool_call["name"]
    args = tool_call["arguments"]
    
    # Simulate execution
    if name == "calculator":
        try:
            result = eval(args["input"])
            return {
                "id": tool_call["id"],
                "output": str(result),
                "success": True
            }
        except:
            return {
                "id": tool_call["id"],
                "output": "Error evaluating expression",
                "success": False
            }
    
    elif name == "search":
        return {
            "id": tool_call["id"],
            "output": f"Search results for: {args['input']}",
            "success": True
        }
    
    elif name == "python":
        return {
            "id": tool_call["id"],
            "output": "Code executed successfully",
            "success": True
        }
    
    return {
        "id": tool_call["id"],
        "output": "Unknown tool",
        "success": False
    }

# ============================================================================
# Multi-Turn Tool Reward
# ============================================================================

def tool_usage_reward(rollout: Rollout) -> float:
    """Reward successful tool usage"""
    if not rollout.tool_calls:
        return 0.0
    
    # Base reward for using tools
    base_reward = 0.5
    
    # Bonus for successful execution
    if rollout.tool_results:
        success_rate = sum(r.get("success", False) for r in rollout.tool_results) / len(rollout.tool_results)
        base_reward += success_rate * 0.5
    
    # Bonus for diversity
    unique_tools = len(set(call["name"] for call in rollout.tool_calls))
    diversity_bonus = min(unique_tools * 0.1, 0.3)
    
    # Bonus for multi-turn (more interaction)
    turn_bonus = min(len(rollout.conversation.messages) * 0.05, 0.3)
    
    return base_reward + diversity_bonus + turn_bonus

# ============================================================================
# Create Multi-Turn Tool Rollout
# ============================================================================

def create_tool_rollout(
    model: Any,
    tokenizer: Any,
    initial_prompt: str,
    max_turns: int = 5
) -> Rollout:
    """Create rollout with tool use"""
    
    initial_conversation = Conversation(messages=[
        Message(role="user", content=initial_prompt)
    ])
    
    # Stop when model outputs <done>
    def stop_condition(conv: Conversation) -> bool:
        if not conv.messages:
            return False
        return "<done>" in conv.messages[-1].content.lower()
    
    rollout = create_multiturn_rollout(
        model=model,
        tokenizer=tokenizer,
        initial_conversation=initial_conversation,
        gen_config=GenerationConfig(
            max_new_tokens=256,
            temperature=0.7,
            stop_strings=["<done>"]
        ),
        num_turns=max_turns,
        tool_parser=parse_tool_calls_xml,
        tool_executor=execute_tool_mock,
        stop_condition=stop_condition
    )
    
    # Compute reward
    reward = tool_usage_reward(rollout)
    rollout = rollout._replace(rewards=reward)
    
    return rollout

# ============================================================================
# Training with Tool Use
# ============================================================================

tool_prompts = [
    "Calculate the sum of 123 and 456, then multiply by 2.",
    "Search for information about Python programming and summarize it.",
    "Write Python code to sort a list [3, 1, 4, 1, 5, 9, 2, 6].",
    # ... more tool-use prompts
]

def load_tool_prompts(iteration: int) -> List[str]:
    batch_size = 4
    start_idx = (iteration * batch_size) % len(tool_prompts)
    end_idx = start_idx + batch_size
    return tool_prompts[start_idx:end_idx]

# Custom rollout creation for tool use
def create_tool_batch(prompts: List[str]) -> Batch:
    rollouts = [
        create_tool_rollout(model, tokenizer, prompt)
        for prompt in prompts
    ]
    return Batch(rollouts=rollouts)

# Train (simplified - would need integration with main training loop)
for iteration in range(50):
    prompts = load_tool_prompts(iteration)
    batch = create_tool_batch(prompts)
    
    # Compute advantages and loss
    advantages = compute_grpo_advantages(batch)
    loss = compute_grpo_loss(batch, advantages, kl_coef=0.1)
    
    # Optimization step
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    # Log
    print(f"Iteration {iteration}: Loss={loss.item():.4f}, "
          f"Avg Tools Used={sum(len(r.tool_calls) for r in batch.rollouts) / len(batch):.2f}")
```

---

## Example 5: Custom KL Divergence with Reference Model Updates

```python
# ============================================================================
# Custom KL Computation
# ============================================================================

def adaptive_kl(logprobs: List[float], ref_logprobs: List[float]) -> float:
    """
    Adaptive KL that focuses on high-probability tokens.
    Only penalize divergence on tokens where policy has high confidence.
    """
    import math
    
    kl = 0.0
    for lp, ref_lp in zip(logprobs, ref_logprobs):
        prob = math.exp(lp)
        
        # Only count KL for high-confidence predictions (prob > 0.1)
        if prob > 0.1:
            token_kl = prob * (lp - ref_lp)
            kl += token_kl
    
    return kl

def truncated_kl(logprobs: List[float], ref_logprobs: List[float], threshold: float = 1.0) -> float:
    """
    Truncated KL: clip individual token KL values.
    Prevents extreme penalties on single tokens.
    """
    import math
    
    kl = 0.0
    for lp, ref_lp in zip(logprobs, ref_logprobs):
        prob = math.exp(lp)
        token_kl = prob * (lp - ref_lp)
        
        # Clip to threshold
        token_kl = min(token_kl, threshold)
        kl += token_kl
    
    return kl

# ============================================================================
# Reference Model Update Strategies
# ============================================================================

def ema_update(ref_model: Any, policy_model: Any, decay: float = 0.999) -> Any:
    """Exponential moving average update"""
    import torch
    
    with torch.no_grad():
        for ref_param, policy_param in zip(ref_model.parameters(), policy_model.parameters()):
            ref_param.data.mul_(decay).add_(policy_param.data, alpha=1 - decay)
    
    return ref_model

def periodic_reset(
    ref_model: Any,
    policy_model: Any,
    iteration: int,
    reset_every: int = 20
) -> Any:
    """Periodically reset reference model to current policy"""
    if iteration % reset_every == 0:
        print(f"Resetting reference model at iteration {iteration}")
        import copy
        ref_model = copy.deepcopy(policy_model)
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False
    
    return ref_model

def no_update(ref_model: Any, policy_model: Any) -> Any:
    """Keep reference model fixed"""
    return ref_model

# ============================================================================
# Create Update Function with Context
# ============================================================================

def create_ref_update_fn(strategy: str, **kwargs):
    """Factory for reference model update functions"""
    
    if strategy == "ema":
        decay = kwargs.get("decay", 0.999)
        return lambda ref, policy: ema_update(ref, policy, decay)
    
    elif strategy == "periodic":
        reset_every = kwargs.get("reset_every", 20)
        iteration_counter = {"count": 0}
        
        def update_fn(ref, policy):
            iteration_counter["count"] += 1
            return periodic_reset(ref, policy, iteration_counter["count"], reset_every)
        
        return update_fn
    
    elif strategy == "none":
        return no_update
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

# ============================================================================
# Training with Custom KL and Reference Updates
# ============================================================================

# Use truncated KL
def create_rollout_with_custom_kl(
    model: Any,
    tokenizer: Any,
    prompt: str,
    gen_config: GenerationConfig,
    ref_model: Any
) -> Rollout:
    """Create rollout with custom KL computation"""
    
    # Generate
    generation = generate_with_logprobs(model, tokenizer, prompt, gen_config)
    
    # Get reference logprobs
    ref_logprobs = get_reference_logprobs(
        ref_model, tokenizer, prompt, generation.token_ids
    )
    
    # Compute custom KL
    kl_div = truncated_kl(generation.logprobs, ref_logprobs, threshold=1.0)
    per_token_kl = compute_per_token_kl(generation.logprobs, ref_logprobs)
    
    rollout = Rollout(
        prompt=prompt,
        conversation=Conversation(messages=[]),
        generation=generation,
        token_ids=generation.token_ids,
        logprobs=generation.logprobs,
        ref_logprobs=ref_logprobs,
        rewards=0.0,
        kl_divergence=kl_div,
        per_token_kl=per_token_kl
    )
    
    return rollout

# Create EMA update function
ref_update_fn = create_ref_update_fn("ema", decay=0.995)

# Train with custom KL and EMA updates
final_state = train(
    initial_state=initial_state,
    prompt_loader=load_prompts,
    num_iterations=100,
    gen_config=gen_config,
    algorithm_config=algorithm_config,
    reward_fn=correctness_reward,
    ref_model_update_fn=ref_update_fn,  # EMA updates
    logger=log_to_console
)
```

---

## Example 6: Complete Custom Pipeline

This example shows a complete custom training pipeline with all extension points used.

```python
# ============================================================================
# Complete Custom Training Pipeline
# ============================================================================

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Any, Callable
import wandb

# ============================================================================
# 1. Setup Models
# ============================================================================

def setup_models(model_name: str):
    """Setup policy and reference models"""
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        use_flash_attention_2=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create reference model with EMA
    import copy
    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    
    return model, tokenizer, ref_model

# ============================================================================
# 2. Custom Reward Model
# ============================================================================

class RewardEnsemble:
    """Ensemble of reward functions"""
    
    def __init__(self, reward_model_path: str = None):
        self.rm = None
        if reward_model_path:
            self.rm = AutoModelForCausalLM.from_pretrained(
                reward_model_path,
                torch_dtype=torch.bfloat16,
                device_map="cuda"
            )
    
    def rule_based_reward(self, rollout: Rollout) -> float:
        """Simple rule-based reward"""
        text = rollout.generation.text
        score = 0.0
        
        # Length reward (prefer 100-500 chars)
        length = len(text)
        if 100 <= length <= 500:
            score += 0.3
        
        # Structure reward (has proper formatting)
        if any(marker in text for marker in ["\n\n", "1.", "2.", "First", "Second"]):
            score += 0.2
        
        # Keyword reward
        if any(kw in text.lower() for kw in ["because", "therefore", "explanation"]):
            score += 0.2
        
        # Completeness (ends properly)
        if text.strip().endswith((".", "!", "?")):
            score += 0.1
        
        return score
    
    def model_based_reward(self, rollout: Rollout) -> float:
        """Learned reward model"""
        if self.rm is None:
            return 0.0
        
        # Use reward model to score
        score = compute_reward_model_score(
            self.rm,
            tokenizer,
            rollout.prompt,
            rollout.generation.text
        )
        return torch.sigmoid(torch.tensor(score)).item()
    
    def combined_reward(self, rollout: Rollout) -> float:
        """Combine multiple reward signals"""
        rule_score = self.rule_based_reward(rollout)
        model_score = self.model_based_reward(rollout)
        
        # Weighted combination
        if self.rm is not None:
            return 0.3 * rule_score + 0.7 * model_score
        else:
            return rule_score

# ============================================================================
# 3. Advanced Algorithm Configuration
# ============================================================================

class AdaptiveAlgorithmConfig:
    """Adaptive algorithm configuration based on training progress"""
    
    def __init__(self):
        self.iteration = 0
        self.reward_history = []
    
    def update(self, stats: TrainingStats):
        """Update based on training statistics"""
        self.iteration += 1
        self.reward_history.append(stats.rewards_mean)
    
    def get_config(self) -> Dict[str, Any]:
        """Get algorithm config for current state"""
        
        # Early training: GRPO with high KL coefficient
        if self.iteration < 30:
            return create_algorithm_config(
                "grpo",
                {"kl_coef": 0.2}
            )
        
        # Mid training: GRPO with adaptive KL
        elif self.iteration < 60:
            # Reduce KL coefficient if rewards are improving
            if len(self.reward_history) >= 10:
                recent_improvement = self.reward_history[-1] - self.reward_history[-10]
                if recent_improvement > 0.1:
                    kl_coef = 0.05  # Low KL when improving
                else:
                    kl_coef = 0.15  # Higher KL when plateaued
            else:
                kl_coef = 0.1
            
            return create_algorithm_config(
                "grpo",
                {"kl_coef": kl_coef}
            )
        
        # Late training: Switch to DAPO for fine-tuning
        else:
            return create_algorithm_config(
                "dapo",
                {
                    "beta": 0.1,
                    "pair_selector": self.select_pairs
                }
            )
    
    def select_pairs(self, batch: Batch) -> List[tuple[Rollout, Rollout]]:
        """Select pairs for DAPO"""
        sorted_rollouts = sorted(
            batch.rollouts,
            key=lambda r: r.rewards if isinstance(r.rewards, float) else sum(r.rewards),
            reverse=True
        )
        
        pairs = []
        n = len(sorted_rollouts)
        
        # Top 25% vs bottom 25%
        top_k = n // 4
        for i in range(top_k):
            preferred = sorted_rollouts[i]
            dispreferred = sorted_rollouts[n - 1 - i]
            pairs.append((preferred, dispreferred))
        
        return pairs

# ============================================================================
# 4. Custom Training Loop with All Extensions
# ============================================================================

def advanced_training_loop(
    model_name: str,
    prompts_dataset: List[str],
    num_iterations: int,
    batch_size: int = 8,
    use_wandb: bool = True
):
    """Complete training loop with all custom extensions"""
    
    # Setup
    model, tokenizer, ref_model = setup_models(model_name)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iterations)
    
    # Custom components
    reward_ensemble = RewardEnsemble()  # Could pass reward model path
    algo_config = AdaptiveAlgorithmConfig()
    
    # EMA reference model update
    def ema_ref_update(ref, policy):
        return ema_update(ref, policy, decay=0.999)
    
    # WandB logging
    if use_wandb:
        wandb.init(project="rl-primitives", name="custom-training")
    
    # Training state
    state = TrainingState(
        iteration=0,
        global_step=0,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        ref_model=ref_model,
        stats_history=[]
    )
    
    # Generation config
    gen_config = GenerationConfig(
        max_new_tokens=384,
        temperature=0.8,
        top_p=0.95,
        do_sample=True
    )
    
    # Prompt loader with shuffling
    import random
    def load_prompts(iteration: int) -> List[str]:
        random.seed(iteration)
        shuffled = random.sample(prompts_dataset, min(batch_size, len(prompts_dataset)))
        return shuffled
    
    # Custom logger
    def custom_logger(state: TrainingState, stats: TrainingStats):
        # Console
        print(f"[Iter {state.iteration:3d}] "
              f"Loss: {stats.loss:.4f} | "
              f"Reward: {stats.rewards_mean:.4f}±{stats.rewards_std:.4f} | "
              f"KL: {stats.kl_divergence:.4f} | "
              f"LR: {stats.learning_rate:.2e}")
        
        # WandB
        if use_wandb:
            wandb.log({
                "iteration": state.iteration,
                "loss": stats.loss,
                "reward_mean": stats.rewards_mean,
                "reward_std": stats.rewards_std,
                "kl_divergence": stats.kl_divergence,
                "learning_rate": stats.learning_rate,
                "grad_norm": stats.grad_norm
            })
    
    # Checkpointing
    def checkpoint_fn(state: TrainingState):
        save_checkpoint(state, f"checkpoints/iter_{state.iteration}")
    
    # Evaluation
    def eval_fn(state: TrainingState) -> Dict[str, float]:
        # Run on held-out prompts
        eval_prompts = ["Explain quantum computing in simple terms.",
                       "What are the benefits of exercise?"]
        
        eval_batch = create_batch_rollouts(
            state.model, tokenizer, eval_prompts, gen_config,
            ref_model=state.ref_model,
            reward_fn=reward_ensemble.combined_reward
        )
        
        rewards = [r.rewards for r in eval_batch.rollouts]
        return {
            "eval_reward_mean": sum(rewards) / len(rewards),
            "eval_reward_max": max(rewards),
            "eval_reward_min": min(rewards)
        }
    
    # Main training loop
    for iteration in range(num_iterations):
        # Get adaptive algorithm config
        algorithm_config = algo_config.get_config()
        
        # Load prompts
        prompts = load_prompts(iteration)
        
        # Create rollouts
        batch = create_batch_rollouts(
            model, tokenizer, prompts, gen_config,
            ref_model=ref_model,
            reward_fn=reward_ensemble.combined_reward,
            batch_size=batch_size
        )
        
        # Training step
        state, stats = training_step(state, batch, algorithm_config)
        
        # Update algorithm config
        algo_config.update(stats)
        
        # Update reference model (EMA)
        new_ref_model = ema_ref_update(ref_model, model)
        ref_model = new_ref_model
        state = state._replace(ref_model=ref_model)
        
        # Update iteration
        state = state._replace(iteration=iteration + 1)
        
        # Logging
        custom_logger(state, stats)
        
        # Periodic evaluation
        if iteration % 10 == 0:
            eval_results = eval_fn(state)
            print(f"  Eval: {eval_results}")
            if use_wandb:
                wandb.log(eval_results)
        
        # Periodic checkpointing
        if iteration % 20 == 0:
            checkpoint_fn(state)
    
    # Final checkpoint
    checkpoint_fn(state)
    
    if use_wandb:
        wandb.finish()
    
    return state

# ============================================================================
# Run Complete Pipeline
# ============================================================================

if __name__ == "__main__":
    # Dataset of prompts
    training_prompts = [
        "Explain how neural networks work.",
        "What is the capital of France and why?",
        "Write a short story about a robot learning to paint.",
        # ... more prompts
    ] * 10
    
    final_state = advanced_training_loop(
        model_name="meta-llama/Llama-3-8B",
        prompts_dataset=training_prompts,
        num_iterations=100,
        batch_size=8,
        use_wandb=True
    )
    
    print("\nTraining complete!")
    print(f"Final iteration: {final_state.iteration}")
    print(f"Total steps: {final_state.global_step}")
```

---

## Key Takeaways

1. **Pure Functions**: All components are functions that can be swapped independently
2. **No Hidden State**: All data flows through explicit parameters and return values
3. **Composability**: Functions naturally compose to build complex behavior
4. **Extension Points**: Every component has clear extension points via function parameters
5. **Type Safety**: Using NamedTuples provides clear data structure contracts
6. **Testability**: Each function can be tested in isolation

This functional architecture makes it easy to:
- Swap reward functions
- Change KL computation methods
- Switch algorithms mid-training
- Update reference models differently
- Add custom logging/checkpointing
- Integrate new backends (vLLM, SGLang, etc.)
- Handle tool calls and multi-turn conversations

All without modifying the core primitives!
