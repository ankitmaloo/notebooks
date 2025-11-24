# RL Training Primitives: Functional Architecture

## Design Philosophy

This architecture is built on **functions, not classes** to maximize:
- **Hackability**: Swap any component by passing different functions
- **Modularity**: Each primitive is independent and composable
- **Transparency**: All data flows explicitly through function parameters
- **Testability**: Pure functions where possible, side effects isolated

### Core Principles

1. **Functions as first-class citizens**: Pass functions as arguments for extension points
2. **Explicit data flow**: Use dicts/namedtuples, no hidden state
3. **Composition over inheritance**: Combine functions to build complex behavior
4. **Dependency injection**: Pass dependencies as function arguments
5. **Immutability**: Return new structures instead of mutating

---

## Data Structures

```python
from typing import NamedTuple, Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass
import torch

# ============================================================================
# Core Data Structures
# ============================================================================

class ModelConfig(NamedTuple):
    """Configuration for model loading"""
    model_name_or_path: str
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16
    use_flash_attention: bool = True
    tensor_parallel_size: int = 1
    load_in_8bit: bool = False
    trust_remote_code: bool = False
    custom_config: Dict[str, Any] = {}


class GenerationConfig(NamedTuple):
    """Configuration for generation"""
    max_new_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    do_sample: bool = True
    num_return_sequences: int = 1
    stop_strings: List[str] = []
    logprobs: bool = False
    custom_params: Dict[str, Any] = {}


class Message(NamedTuple):
    """Single conversation message"""
    role: str  # "system", "user", "assistant", "tool"
    content: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    metadata: Dict[str, Any] = {}


class Conversation(NamedTuple):
    """Multi-turn conversation state"""
    messages: List[Message]
    metadata: Dict[str, Any] = {}
    
    def add_message(self, message: Message) -> 'Conversation':
        """Return new conversation with added message (immutable)"""
        return Conversation(
            messages=self.messages + [message],
            metadata=self.metadata
        )


class Generation(NamedTuple):
    """Single generation output"""
    text: str
    logprobs: Optional[List[float]] = None
    token_ids: Optional[List[int]] = None
    finish_reason: str = "length"
    metadata: Dict[str, Any] = {}


class Rollout(NamedTuple):
    """Complete rollout data for one sample"""
    prompt: str
    conversation: Conversation
    generation: Generation
    
    # Per-token data
    token_ids: List[int]
    logprobs: List[float]
    ref_logprobs: Optional[List[float]] = None
    
    # Rewards
    rewards: Union[float, List[float]]  # scalar or vector
    reward_metadata: Dict[str, Any] = {}
    
    # Tool tracking
    tool_calls: List[Dict[str, Any]] = []
    tool_results: List[Dict[str, Any]] = []
    
    # KL divergence
    kl_divergence: Optional[float] = None
    per_token_kl: Optional[List[float]] = None
    
    metadata: Dict[str, Any] = {}


class Batch(NamedTuple):
    """Batch of rollouts"""
    rollouts: List[Rollout]
    batch_metadata: Dict[str, Any] = {}
    
    def __len__(self):
        return len(self.rollouts)
    
    def filter(self, predicate: Callable[[Rollout], bool]) -> 'Batch':
        """Return new batch with filtered rollouts"""
        return Batch(
            rollouts=[r for r in self.rollouts if predicate(r)],
            batch_metadata=self.batch_metadata
        )


class TrainingStats(NamedTuple):
    """Statistics from training step"""
    loss: float
    grad_norm: float
    learning_rate: float
    kl_divergence: float
    rewards_mean: float
    rewards_std: float
    custom_metrics: Dict[str, float] = {}


class TrainingState(NamedTuple):
    """Complete training state"""
    iteration: int
    global_step: int
    model: Any  # PyTorch model
    optimizer: Any
    scheduler: Optional[Any] = None
    ref_model: Optional[Any] = None
    stats_history: List[TrainingStats] = []
    metadata: Dict[str, Any] = {}
```

---

## 1. Inference/Generation Primitives

### Core Functions

```python
# ============================================================================
# Model Loading
# ============================================================================

def load_model(
    config: ModelConfig,
    model_loader: Optional[Callable[[ModelConfig], Any]] = None
) -> Any:
    """
    Load a model with optional custom loader.
    
    Extension point: Pass custom model_loader for different backends
    (vLLM, SGLang, HF Transformers, etc.)
    
    Args:
        config: Model configuration
        model_loader: Custom loading function (default: HF transformers)
    
    Returns:
        Loaded model
    """
    if model_loader is not None:
        return model_loader(config)
    
    # Default: HuggingFace Transformers
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        torch_dtype=config.dtype,
        device_map=config.device,
        trust_remote_code=config.trust_remote_code,
        **config.custom_config
    )


def load_tokenizer(
    config: ModelConfig,
    tokenizer_loader: Optional[Callable[[ModelConfig], Any]] = None
) -> Any:
    """
    Load tokenizer with optional custom loader.
    
    Extension point: Pass custom tokenizer_loader
    """
    if tokenizer_loader is not None:
        return tokenizer_loader(config)
    
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=config.trust_remote_code
    )


# ============================================================================
# Generation
# ============================================================================

def generate(
    model: Any,
    tokenizer: Any,
    prompt: str,
    config: GenerationConfig,
    generator: Optional[Callable] = None
) -> Generation:
    """
    Generate text from prompt.
    
    Extension point: Pass custom generator function for different backends
    
    Args:
        model: Model to use
        tokenizer: Tokenizer
        prompt: Input prompt
        config: Generation configuration
        generator: Custom generation function
    
    Returns:
        Generation output
    """
    if generator is not None:
        return generator(model, tokenizer, prompt, config)
    
    # Default: HuggingFace generate
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
        do_sample=config.do_sample,
        num_return_sequences=config.num_return_sequences,
        output_scores=config.logprobs,
        return_dict_in_generate=True,
        **config.custom_params
    )
    
    text = tokenizer.decode(outputs.sequences[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    return Generation(
        text=text,
        token_ids=outputs.sequences[0].tolist(),
        finish_reason="length",
        metadata={}
    )


def generate_with_logprobs(
    model: Any,
    tokenizer: Any,
    prompt: str,
    config: GenerationConfig,
    logprob_extractor: Optional[Callable] = None
) -> Generation:
    """
    Generate with token-level logprobs.
    
    Extension point: Pass custom logprob_extractor for different backends
    
    Returns:
        Generation with logprobs populated
    """
    if logprob_extractor is not None:
        return logprob_extractor(model, tokenizer, prompt, config)
    
    # Default implementation
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        do_sample=config.do_sample,
        output_scores=True,
        return_dict_in_generate=True,
        **config.custom_params
    )
    
    # Extract logprobs from scores
    import torch.nn.functional as F
    logprobs = []
    for score, token_id in zip(outputs.scores, outputs.sequences[0][inputs.input_ids.shape[1]:]):
        log_probs = F.log_softmax(score[0], dim=-1)
        logprobs.append(log_probs[token_id].item())
    
    text = tokenizer.decode(outputs.sequences[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    return Generation(
        text=text,
        logprobs=logprobs,
        token_ids=outputs.sequences[0].tolist(),
        finish_reason="length"
    )


def batch_generate(
    model: Any,
    tokenizer: Any,
    prompts: List[str],
    config: GenerationConfig,
    batch_size: int = 8,
    generator: Optional[Callable] = None
) -> List[Generation]:
    """
    Generate for multiple prompts with batching.
    
    Extension point: Pass custom generator for batched generation
    """
    if generator is not None:
        return generator(model, tokenizer, prompts, config, batch_size)
    
    # Default: batch and generate
    results = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        batch_results = [
            generate(model, tokenizer, p, config)
            for p in batch_prompts
        ]
        results.extend(batch_results)
    
    return results


# ============================================================================
# Conversation Handling
# ============================================================================

def apply_chat_template(
    conversation: Conversation,
    tokenizer: Any,
    template_fn: Optional[Callable[[Conversation, Any], str]] = None
) -> str:
    """
    Convert conversation to prompt string using chat template.
    
    Extension point: Pass custom template_fn for different formats
    
    Args:
        conversation: Conversation to format
        tokenizer: Tokenizer (may have built-in chat template)
        template_fn: Custom template function
    
    Returns:
        Formatted prompt string
    """
    if template_fn is not None:
        return template_fn(conversation, tokenizer)
    
    # Default: use tokenizer's chat template if available
    if hasattr(tokenizer, 'apply_chat_template'):
        messages = [
            {"role": m.role, "content": m.content}
            for m in conversation.messages
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Fallback: simple concatenation
    parts = []
    for msg in conversation.messages:
        parts.append(f"{msg.role}: {msg.content}")
    parts.append("assistant:")
    return "\n".join(parts)


def generate_conversation_turn(
    model: Any,
    tokenizer: Any,
    conversation: Conversation,
    config: GenerationConfig,
    template_fn: Optional[Callable] = None,
    generator: Optional[Callable] = None
) -> tuple[Conversation, Generation]:
    """
    Generate next turn in conversation.
    
    Returns:
        Updated conversation and generation
    """
    prompt = apply_chat_template(conversation, tokenizer, template_fn)
    generation = generate(model, tokenizer, prompt, config, generator)
    
    new_message = Message(role="assistant", content=generation.text)
    new_conversation = conversation.add_message(new_message)
    
    return new_conversation, generation


def extract_tool_calls(
    generation: Generation,
    tool_parser: Callable[[str], List[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    """
    Extract tool calls from generation.
    
    Extension point: Pass custom tool_parser for different formats
    (function calling, ReAct, JSON, etc.)
    
    Args:
        generation: Generated text
        tool_parser: Function to parse tool calls from text
    
    Returns:
        List of tool call dictionaries
    """
    return tool_parser(generation.text)


def execute_tools(
    tool_calls: List[Dict[str, Any]],
    tool_executor: Callable[[Dict[str, Any]], Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Execute tool calls and return results.
    
    Extension point: Pass custom tool_executor
    
    Args:
        tool_calls: List of tool calls
        tool_executor: Function to execute a single tool call
    
    Returns:
        List of tool results
    """
    return [tool_executor(call) for call in tool_calls]
```

---

## 2. Reward Primitives

### Core Functions

```python
# ============================================================================
# Reward Computation
# ============================================================================

def compute_scalar_reward(
    rollout: Rollout,
    reward_fn: Callable[[Rollout], float]
) -> float:
    """
    Compute scalar reward for rollout.
    
    Extension point: Pass custom reward_fn
    
    Examples of reward_fn:
    - Length-based: lambda r: len(r.generation.text)
    - Tool success: lambda r: 1.0 if successful_tool_use(r) else 0.0
    - LLM-as-judge: lambda r: llm_judge_score(r)
    """
    return reward_fn(rollout)


def compute_vector_reward(
    rollout: Rollout,
    reward_fns: List[Callable[[Rollout], float]]
) -> List[float]:
    """
    Compute vector of rewards (multi-objective).
    
    Extension point: Pass list of reward_fns for different objectives
    
    Example:
        reward_fns = [
            lambda r: correctness_score(r),
            lambda r: helpfulness_score(r),
            lambda r: safety_score(r)
        ]
    """
    return [fn(rollout) for fn in reward_fns]


def aggregate_vector_rewards(
    vector_rewards: List[float],
    aggregator: Callable[[List[float]], float]
) -> float:
    """
    Aggregate vector rewards to scalar.
    
    Extension point: Pass custom aggregator
    
    Examples:
    - Sum: lambda v: sum(v)
    - Weighted sum: lambda v: 0.5*v[0] + 0.3*v[1] + 0.2*v[2]
    - Min: lambda v: min(v)
    - Product: lambda v: prod(v)
    """
    return aggregator(vector_rewards)


def compute_batch_rewards(
    batch: Batch,
    reward_fn: Callable[[Rollout], Union[float, List[float]]],
    is_vector: bool = False
) -> List[Union[float, List[float]]]:
    """
    Compute rewards for entire batch.
    
    Args:
        batch: Batch of rollouts
        reward_fn: Reward function (returns scalar or vector)
        is_vector: Whether rewards are vectors
    
    Returns:
        List of rewards (scalar or vector per rollout)
    """
    return [reward_fn(rollout) for rollout in batch.rollouts]


def normalize_rewards(
    rewards: List[float],
    normalizer: Callable[[List[float]], List[float]]
) -> List[float]:
    """
    Normalize rewards across batch.
    
    Extension point: Pass custom normalizer
    
    Examples:
    - Z-score: lambda r: (r - mean(r)) / std(r)
    - Min-max: lambda r: (r - min(r)) / (max(r) - min(r))
    - Rank-based: lambda r: rankdata(r)
    """
    return normalizer(rewards)


def apply_reward_shaping(
    rollout: Rollout,
    base_reward: float,
    shaping_fn: Callable[[Rollout, float], float]
) -> float:
    """
    Apply reward shaping to base reward.
    
    Extension point: Pass custom shaping_fn
    
    Examples:
    - KL penalty: lambda r, reward: reward - kl_coef * r.kl_divergence
    - Length bonus: lambda r, reward: reward + length_bonus * len(r.token_ids)
    - Entropy bonus: lambda r, reward: reward + entropy_coef * entropy(r.logprobs)
    """
    return shaping_fn(rollout, base_reward)


# ============================================================================
# Reward Model Integration
# ============================================================================

def load_reward_model(
    config: ModelConfig,
    loader: Optional[Callable[[ModelConfig], Any]] = None
) -> Any:
    """
    Load reward model.
    
    Extension point: Pass custom loader for different RM architectures
    """
    if loader is not None:
        return loader(config)
    
    # Default: load as classifier
    from transformers import AutoModelForSequenceClassification
    return AutoModelForSequenceClassification.from_pretrained(
        config.model_name_or_path,
        torch_dtype=config.dtype,
        device_map=config.device
    )


def compute_reward_model_score(
    reward_model: Any,
    tokenizer: Any,
    prompt: str,
    completion: str,
    scorer: Optional[Callable] = None
) -> float:
    """
    Compute reward model score for prompt-completion pair.
    
    Extension point: Pass custom scorer for different RM types
    """
    if scorer is not None:
        return scorer(reward_model, tokenizer, prompt, completion)
    
    # Default: score with sequence classifier
    text = prompt + completion
    inputs = tokenizer(text, return_tensors="pt").to(reward_model.device)
    outputs = reward_model(**inputs)
    return outputs.logits[0, 0].item()
```

---

## 3. KL Divergence Primitives

### Core Functions

```python
# ============================================================================
# KL Computation
# ============================================================================

def compute_forward_kl(
    logprobs: List[float],
    ref_logprobs: List[float],
    kl_fn: Optional[Callable[[List[float], List[float]], float]] = None
) -> float:
    """
    Compute forward KL divergence: KL(policy || ref).
    
    Extension point: Pass custom kl_fn for different computation methods
    
    Args:
        logprobs: Log probabilities from policy
        ref_logprobs: Log probabilities from reference
        kl_fn: Custom KL computation function
    
    Returns:
        KL divergence (scalar)
    """
    if kl_fn is not None:
        return kl_fn(logprobs, ref_logprobs)
    
    # Default: sum of per-token KL
    import math
    kl = 0.0
    for lp, ref_lp in zip(logprobs, ref_logprobs):
        prob = math.exp(lp)
        kl += prob * (lp - ref_lp)
    return kl


def compute_reverse_kl(
    logprobs: List[float],
    ref_logprobs: List[float],
    kl_fn: Optional[Callable[[List[float], List[float]], float]] = None
) -> float:
    """
    Compute reverse KL divergence: KL(ref || policy).
    
    Extension point: Pass custom kl_fn
    """
    if kl_fn is not None:
        return kl_fn(ref_logprobs, logprobs)
    
    # Default: reverse of forward KL
    return compute_forward_kl(ref_logprobs, logprobs)


def compute_per_token_kl(
    logprobs: List[float],
    ref_logprobs: List[float],
    kl_fn: Optional[Callable[[float, float], float]] = None
) -> List[float]:
    """
    Compute per-token KL divergence.
    
    Extension point: Pass custom kl_fn for per-token computation
    
    Returns:
        List of KL values (one per token)
    """
    if kl_fn is not None:
        return [kl_fn(lp, ref_lp) for lp, ref_lp in zip(logprobs, ref_logprobs)]
    
    # Default: simple difference
    import math
    per_token_kl = []
    for lp, ref_lp in zip(logprobs, ref_logprobs):
        prob = math.exp(lp)
        kl = prob * (lp - ref_lp)
        per_token_kl.append(kl)
    
    return per_token_kl


def compute_batch_kl(
    batch: Batch,
    kl_fn: Callable[[List[float], List[float]], float]
) -> List[float]:
    """
    Compute KL divergence for entire batch.
    
    Args:
        batch: Batch of rollouts (must have logprobs and ref_logprobs)
        kl_fn: KL computation function
    
    Returns:
        List of KL values (one per rollout)
    """
    return [
        kl_fn(r.logprobs, r.ref_logprobs)
        for r in batch.rollouts
        if r.ref_logprobs is not None
    ]


# ============================================================================
# Reference Model Handling
# ============================================================================

def create_reference_model(
    model: Any,
    ref_model_fn: Optional[Callable[[Any], Any]] = None
) -> Any:
    """
    Create reference model from policy model.
    
    Extension point: Pass custom ref_model_fn for different strategies
    
    Examples:
    - Deep copy: lambda m: copy.deepcopy(m)
    - Shared parameters: lambda m: m (same model)
    - EMA: lambda m: create_ema_model(m, decay=0.999)
    """
    if ref_model_fn is not None:
        return ref_model_fn(model)
    
    # Default: deep copy
    import copy
    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    return ref_model


def update_reference_model(
    ref_model: Any,
    policy_model: Any,
    update_fn: Callable[[Any, Any], Any]
) -> Any:
    """
    Update reference model based on policy model.
    
    Extension point: Pass custom update_fn for different update strategies
    
    Examples:
    - Full update: lambda ref, policy: copy_params(policy, ref)
    - EMA update: lambda ref, policy: ema_update(ref, policy, decay=0.999)
    - No update: lambda ref, policy: ref
    - Periodic reset: lambda ref, policy: reset_if_needed(ref, policy, iteration)
    
    Args:
        ref_model: Reference model to update
        policy_model: Current policy model
        update_fn: Update strategy function
    
    Returns:
        Updated reference model
    """
    return update_fn(ref_model, policy_model)


def get_reference_logprobs(
    ref_model: Any,
    tokenizer: Any,
    prompt: str,
    token_ids: List[int],
    logprob_extractor: Optional[Callable] = None
) -> List[float]:
    """
    Get reference model logprobs for given tokens.
    
    Extension point: Pass custom logprob_extractor
    
    Args:
        ref_model: Reference model
        tokenizer: Tokenizer
        prompt: Input prompt
        token_ids: Token IDs to get logprobs for
        logprob_extractor: Custom extraction function
    
    Returns:
        List of logprobs (one per token)
    """
    if logprob_extractor is not None:
        return logprob_extractor(ref_model, tokenizer, prompt, token_ids)
    
    # Default: forward pass and extract logprobs
    import torch
    import torch.nn.functional as F
    
    with torch.no_grad():
        inputs = tokenizer(prompt, return_tensors="pt").to(ref_model.device)
        full_ids = torch.cat([inputs.input_ids, torch.tensor([token_ids]).to(ref_model.device)], dim=1)
        
        outputs = ref_model(full_ids)
        logits = outputs.logits
        
        logprobs = []
        for i, token_id in enumerate(token_ids):
            log_probs = F.log_softmax(logits[0, inputs.input_ids.shape[1] + i - 1], dim=-1)
            logprobs.append(log_probs[token_id].item())
        
        return logprobs
```

---

## 4. Rollout Primitives

### Core Functions

```python
# ============================================================================
# Rollout Generation
# ============================================================================

def create_rollout(
    model: Any,
    tokenizer: Any,
    prompt: str,
    gen_config: GenerationConfig,
    ref_model: Optional[Any] = None,
    reward_fn: Optional[Callable[[Rollout], Union[float, List[float]]]] = None,
    tool_parser: Optional[Callable[[str], List[Dict]]] = None,
    conversation: Optional[Conversation] = None
) -> Rollout:
    """
    Create a complete rollout (generation + rewards + KL).
    
    Extension points:
    - reward_fn: Custom reward computation
    - tool_parser: Extract tool calls from generation
    - ref_model: Compute KL divergence
    
    Args:
        model: Policy model
        tokenizer: Tokenizer
        prompt: Input prompt
        gen_config: Generation configuration
        ref_model: Reference model for KL (optional)
        reward_fn: Reward function (optional)
        tool_parser: Tool call parser (optional)
        conversation: Conversation context (optional)
    
    Returns:
        Complete rollout
    """
    # Generate with logprobs
    generation = generate_with_logprobs(model, tokenizer, prompt, gen_config)
    
    # Extract tool calls if parser provided
    tool_calls = []
    if tool_parser is not None:
        tool_calls = extract_tool_calls(generation, tool_parser)
    
    # Get reference logprobs if ref_model provided
    ref_logprobs = None
    kl_div = None
    per_token_kl = None
    
    if ref_model is not None and generation.logprobs is not None:
        ref_logprobs = get_reference_logprobs(
            ref_model, tokenizer, prompt, generation.token_ids
        )
        kl_div = compute_forward_kl(generation.logprobs, ref_logprobs)
        per_token_kl = compute_per_token_kl(generation.logprobs, ref_logprobs)
    
    # Create initial rollout
    rollout = Rollout(
        prompt=prompt,
        conversation=conversation or Conversation(messages=[]),
        generation=generation,
        token_ids=generation.token_ids,
        logprobs=generation.logprobs or [],
        ref_logprobs=ref_logprobs,
        rewards=0.0,  # Will be computed below
        tool_calls=tool_calls,
        tool_results=[],
        kl_divergence=kl_div,
        per_token_kl=per_token_kl
    )
    
    # Compute rewards if reward_fn provided
    if reward_fn is not None:
        rewards = reward_fn(rollout)
        rollout = rollout._replace(rewards=rewards)
    
    return rollout


def create_batch_rollouts(
    model: Any,
    tokenizer: Any,
    prompts: List[str],
    gen_config: GenerationConfig,
    ref_model: Optional[Any] = None,
    reward_fn: Optional[Callable] = None,
    tool_parser: Optional[Callable] = None,
    batch_size: int = 8
) -> Batch:
    """
    Create rollouts for batch of prompts.
    
    Args:
        model: Policy model
        tokenizer: Tokenizer
        prompts: List of prompts
        gen_config: Generation configuration
        ref_model: Reference model (optional)
        reward_fn: Reward function (optional)
        tool_parser: Tool parser (optional)
        batch_size: Batch size for generation
    
    Returns:
        Batch of rollouts
    """
    rollouts = []
    
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        
        for prompt in batch_prompts:
            rollout = create_rollout(
                model, tokenizer, prompt, gen_config,
                ref_model=ref_model,
                reward_fn=reward_fn,
                tool_parser=tool_parser
            )
            rollouts.append(rollout)
    
    return Batch(rollouts=rollouts)


# ============================================================================
# Tool Call Tracking
# ============================================================================

def track_tool_calls(
    rollout: Rollout,
    tool_tracker: Callable[[List[Dict]], Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Track tool usage statistics from rollout.
    
    Extension point: Pass custom tool_tracker
    
    Examples:
    - Count calls: lambda calls: {"count": len(calls)}
    - Track types: lambda calls: {"types": [c["name"] for c in calls]}
    - Track success: lambda calls: {"success_rate": success_rate(calls)}
    
    Args:
        rollout: Rollout with tool calls
        tool_tracker: Function to extract tracking info
    
    Returns:
        Tool tracking metadata
    """
    return tool_tracker(rollout.tool_calls)


def execute_and_track_tools(
    rollout: Rollout,
    tool_executor: Callable[[Dict], Dict],
    result_tracker: Optional[Callable[[List[Dict]], Dict]] = None
) -> Rollout:
    """
    Execute tools in rollout and track results.
    
    Extension points:
    - tool_executor: How to execute each tool
    - result_tracker: How to track execution results
    
    Returns:
        Rollout with tool_results populated
    """
    tool_results = execute_tools(rollout.tool_calls, tool_executor)
    
    tracking_metadata = {}
    if result_tracker is not None:
        tracking_metadata = result_tracker(tool_results)
    
    return rollout._replace(
        tool_results=tool_results,
        metadata={**rollout.metadata, "tool_tracking": tracking_metadata}
    )


# ============================================================================
# Multi-turn Rollouts
# ============================================================================

def create_multiturn_rollout(
    model: Any,
    tokenizer: Any,
    initial_conversation: Conversation,
    gen_config: GenerationConfig,
    num_turns: int,
    ref_model: Optional[Any] = None,
    reward_fn: Optional[Callable] = None,
    tool_parser: Optional[Callable] = None,
    tool_executor: Optional[Callable] = None,
    stop_condition: Optional[Callable[[Conversation], bool]] = None
) -> Rollout:
    """
    Create multi-turn conversational rollout.
    
    Extension points:
    - reward_fn: Reward entire conversation
    - tool_parser: Parse tool calls
    - tool_executor: Execute tools between turns
    - stop_condition: Early stopping logic
    
    Args:
        model: Policy model
        tokenizer: Tokenizer
        initial_conversation: Starting conversation
        gen_config: Generation config
        num_turns: Maximum number of turns
        ref_model: Reference model (optional)
        reward_fn: Reward function (optional)
        tool_parser: Tool parser (optional)
        tool_executor: Tool executor (optional)
        stop_condition: Function to check if conversation should stop
    
    Returns:
        Rollout with full conversation
    """
    conversation = initial_conversation
    all_tool_calls = []
    all_tool_results = []
    all_logprobs = []
    all_token_ids = []
    
    for turn in range(num_turns):
        # Check stop condition
        if stop_condition is not None and stop_condition(conversation):
            break
        
        # Generate turn
        prompt = apply_chat_template(conversation, tokenizer)
        generation = generate_with_logprobs(model, tokenizer, prompt, gen_config)
        
        # Update conversation
        new_message = Message(role="assistant", content=generation.text)
        conversation = conversation.add_message(new_message)
        
        # Track logprobs and tokens
        if generation.logprobs:
            all_logprobs.extend(generation.logprobs)
        if generation.token_ids:
            all_token_ids.extend(generation.token_ids)
        
        # Handle tool calls
        if tool_parser is not None:
            tool_calls = extract_tool_calls(generation, tool_parser)
            all_tool_calls.extend(tool_calls)
            
            if tool_executor is not None and tool_calls:
                tool_results = execute_tools(tool_calls, tool_executor)
                all_tool_results.extend(tool_results)
                
                # Add tool results to conversation
                for result in tool_results:
                    tool_msg = Message(
                        role="tool",
                        content=str(result.get("output", "")),
                        tool_call_id=result.get("id")
                    )
                    conversation = conversation.add_message(tool_msg)
    
    # Compute reference logprobs and KL if ref_model provided
    ref_logprobs = None
    kl_div = None
    
    if ref_model is not None and all_logprobs:
        # For multi-turn, we'd need to recompute for full conversation
        # This is simplified - in practice you'd handle this more carefully
        pass
    
    # Create final generation representing entire conversation
    final_generation = Generation(
        text=conversation.messages[-1].content if conversation.messages else "",
        logprobs=all_logprobs,
        token_ids=all_token_ids
    )
    
    # Create rollout
    rollout = Rollout(
        prompt=apply_chat_template(initial_conversation, tokenizer),
        conversation=conversation,
        generation=final_generation,
        token_ids=all_token_ids,
        logprobs=all_logprobs,
        ref_logprobs=ref_logprobs,
        rewards=0.0,
        tool_calls=all_tool_calls,
        tool_results=all_tool_results,
        kl_divergence=kl_div
    )
    
    # Compute rewards
    if reward_fn is not None:
        rewards = reward_fn(rollout)
        rollout = rollout._replace(rewards=rewards)
    
    return rollout
```

---

## 5. Algorithm Primitives

### Core Functions

```python
# ============================================================================
# GRPO (Group Relative Policy Optimization)
# ============================================================================

def compute_grpo_advantages(
    batch: Batch,
    advantage_fn: Optional[Callable[[List[float]], List[float]]] = None
) -> List[float]:
    """
    Compute GRPO advantages (group-relative).
    
    Extension point: Pass custom advantage_fn
    
    Default: advantages = rewards - mean(rewards)
    
    Args:
        batch: Batch of rollouts
        advantage_fn: Custom advantage computation
    
    Returns:
        List of advantages (one per rollout)
    """
    rewards = [r.rewards if isinstance(r.rewards, float) else sum(r.rewards) 
               for r in batch.rollouts]
    
    if advantage_fn is not None:
        return advantage_fn(rewards)
    
    # Default: group-relative (mean baseline)
    mean_reward = sum(rewards) / len(rewards)
    return [r - mean_reward for r in rewards]


def compute_grpo_loss(
    batch: Batch,
    advantages: List[float],
    kl_coef: float = 0.1,
    loss_fn: Optional[Callable] = None
) -> torch.Tensor:
    """
    Compute GRPO loss.
    
    Extension point: Pass custom loss_fn
    
    Default: -mean(advantages * log_probs) + kl_coef * mean(kl)
    
    Args:
        batch: Batch of rollouts
        advantages: Advantages for each rollout
        kl_coef: KL penalty coefficient
        loss_fn: Custom loss function
    
    Returns:
        Loss tensor
    """
    if loss_fn is not None:
        return loss_fn(batch, advantages, kl_coef)
    
    # Default GRPO loss
    import torch
    
    policy_loss = 0.0
    kl_penalty = 0.0
    
    for rollout, adv in zip(batch.rollouts, advantages):
        # Policy gradient term
        log_prob_sum = sum(rollout.logprobs) if rollout.logprobs else 0.0
        policy_loss -= adv * log_prob_sum
        
        # KL penalty
        if rollout.kl_divergence is not None:
            kl_penalty += rollout.kl_divergence
    
    policy_loss /= len(batch)
    kl_penalty /= len(batch)
    
    total_loss = policy_loss + kl_coef * kl_penalty
    
    return torch.tensor(total_loss, requires_grad=True)


# ============================================================================
# DAPO (Direct Alignment from Preferences)
# ============================================================================

def compute_dapo_pairs(
    batch: Batch,
    pair_selector: Callable[[Batch], List[tuple[Rollout, Rollout]]]
) -> List[tuple[Rollout, Rollout]]:
    """
    Select preference pairs for DAPO.
    
    Extension point: Pass custom pair_selector
    
    Examples:
    - Top vs bottom: Select highest reward vs lowest reward
    - Random pairs: Random selection
    - All pairs: All combinations
    
    Args:
        batch: Batch of rollouts
        pair_selector: Function to select pairs
    
    Returns:
        List of (preferred, dispreferred) rollout pairs
    """
    return pair_selector(batch)


def compute_dapo_loss(
    pairs: List[tuple[Rollout, Rollout]],
    beta: float = 0.1,
    loss_fn: Optional[Callable] = None
) -> torch.Tensor:
    """
    Compute DAPO loss (DPO-style).
    
    Extension point: Pass custom loss_fn
    
    Default: -log(sigmoid(beta * (log_ratio_preferred - log_ratio_dispreferred)))
    
    Args:
        pairs: List of (preferred, dispreferred) pairs
        beta: Temperature parameter
        loss_fn: Custom loss function
    
    Returns:
        Loss tensor
    """
    if loss_fn is not None:
        return loss_fn(pairs, beta)
    
    # Default DPO-style loss
    import torch
    import torch.nn.functional as F
    
    losses = []
    
    for preferred, dispreferred in pairs:
        # Compute log ratios
        pref_log_ratio = sum(preferred.logprobs) - sum(preferred.ref_logprobs or preferred.logprobs)
        dispref_log_ratio = sum(dispreferred.logprobs) - sum(dispreferred.ref_logprobs or dispreferred.logprobs)
        
        # DPO loss
        logits = beta * (pref_log_ratio - dispref_log_ratio)
        loss = -F.logsigmoid(logits)
        losses.append(loss)
    
    return torch.stack(losses).mean()


# ============================================================================
# PPO (Proximal Policy Optimization)
# ============================================================================

def compute_ppo_advantages(
    batch: Batch,
    value_fn: Callable[[Rollout], float],
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    advantage_fn: Optional[Callable] = None
) -> tuple[List[float], List[float]]:
    """
    Compute PPO advantages using GAE.
    
    Extension point: Pass custom advantage_fn
    
    Args:
        batch: Batch of rollouts
        value_fn: Value function V(s)
        gamma: Discount factor
        gae_lambda: GAE lambda
        advantage_fn: Custom advantage computation
    
    Returns:
        Tuple of (advantages, returns)
    """
    if advantage_fn is not None:
        return advantage_fn(batch, value_fn, gamma, gae_lambda)
    
    # Default: GAE
    advantages = []
    returns = []
    
    for rollout in batch.rollouts:
        value = value_fn(rollout)
        reward = rollout.rewards if isinstance(rollout.rewards, float) else sum(rollout.rewards)
        
        # Simplified: assuming single-step episodes
        advantage = reward - value
        ret = reward
        
        advantages.append(advantage)
        returns.append(ret)
    
    return advantages, returns


def compute_ppo_loss(
    batch: Batch,
    old_logprobs: List[List[float]],
    advantages: List[float],
    clip_epsilon: float = 0.2,
    value_loss_coef: float = 0.5,
    entropy_coef: float = 0.01,
    loss_fn: Optional[Callable] = None
) -> torch.Tensor:
    """
    Compute PPO clipped loss.
    
    Extension point: Pass custom loss_fn
    
    Args:
        batch: Batch of rollouts with new logprobs
        old_logprobs: Old logprobs (before update)
        advantages: Advantages
        clip_epsilon: PPO clip parameter
        value_loss_coef: Value loss coefficient
        entropy_coef: Entropy bonus coefficient
        loss_fn: Custom loss function
    
    Returns:
        Loss tensor
    """
    if loss_fn is not None:
        return loss_fn(batch, old_logprobs, advantages, clip_epsilon, value_loss_coef, entropy_coef)
    
    # Default: PPO clipped loss
    import torch
    import torch.nn.functional as F
    
    policy_losses = []
    
    for rollout, old_lp, adv in zip(batch.rollouts, old_logprobs, advantages):
        # Compute ratio
        new_lp_sum = sum(rollout.logprobs)
        old_lp_sum = sum(old_lp)
        ratio = torch.exp(torch.tensor(new_lp_sum - old_lp_sum))
        
        # Clipped surrogate
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * adv
        policy_loss = -torch.min(surr1, surr2)
        
        policy_losses.append(policy_loss)
    
    return torch.stack(policy_losses).mean()


# ============================================================================
# Algorithm Switching
# ============================================================================

def create_algorithm_config(
    algorithm: str,
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create algorithm configuration.
    
    Args:
        algorithm: Algorithm name ("grpo", "dapo", "ppo")
        params: Algorithm-specific parameters
    
    Returns:
        Algorithm configuration dict
    """
    return {
        "algorithm": algorithm,
        "params": params
    }


def compute_loss_by_algorithm(
    batch: Batch,
    algorithm_config: Dict[str, Any],
    **kwargs
) -> torch.Tensor:
    """
    Dispatch loss computation based on algorithm.
    
    Extension point: Add new algorithms to dispatch table
    
    Args:
        batch: Batch of rollouts
        algorithm_config: Algorithm configuration
        **kwargs: Additional arguments for loss computation
    
    Returns:
        Loss tensor
    """
    algo = algorithm_config["algorithm"]
    params = algorithm_config["params"]
    
    if algo == "grpo":
        advantages = compute_grpo_advantages(batch)
        return compute_grpo_loss(batch, advantages, **params)
    
    elif algo == "dapo":
        pairs = compute_dapo_pairs(batch, params["pair_selector"])
        return compute_dapo_loss(pairs, **params)
    
    elif algo == "ppo":
        advantages, returns = compute_ppo_advantages(batch, params["value_fn"])
        return compute_ppo_loss(batch, kwargs["old_logprobs"], advantages, **params)
    
    else:
        raise ValueError(f"Unknown algorithm: {algo}")
```

---

## 6. Training Loop Primitives

### Core Functions

```python
# ============================================================================
# Training Iteration
# ============================================================================

def training_step(
    state: TrainingState,
    batch: Batch,
    algorithm_config: Dict[str, Any],
    optimizer_step_fn: Optional[Callable] = None,
    **kwargs
) -> tuple[TrainingState, TrainingStats]:
    """
    Execute single training step.
    
    Extension point: Pass custom optimizer_step_fn
    
    Args:
        state: Current training state
        batch: Batch of rollouts
        algorithm_config: Algorithm configuration
        optimizer_step_fn: Custom optimizer step function
        **kwargs: Additional arguments
    
    Returns:
        Updated state and stats
    """
    # Compute loss
    loss = compute_loss_by_algorithm(batch, algorithm_config, **kwargs)
    
    # Optimizer step
    if optimizer_step_fn is not None:
        grad_norm = optimizer_step_fn(state.optimizer, loss)
    else:
        state.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(state.model.parameters(), 1.0)
        state.optimizer.step()
        if state.scheduler is not None:
            state.scheduler.step()
    
    # Compute stats
    rewards = [r.rewards if isinstance(r.rewards, float) else sum(r.rewards) 
               for r in batch.rollouts]
    kl_divs = [r.kl_divergence for r in batch.rollouts if r.kl_divergence is not None]
    
    stats = TrainingStats(
        loss=loss.item(),
        grad_norm=grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm,
        learning_rate=state.optimizer.param_groups[0]['lr'],
        kl_divergence=sum(kl_divs) / len(kl_divs) if kl_divs else 0.0,
        rewards_mean=sum(rewards) / len(rewards),
        rewards_std=(sum((r - sum(rewards)/len(rewards))**2 for r in rewards) / len(rewards)) ** 0.5
    )
    
    # Update state
    new_state = state._replace(
        global_step=state.global_step + 1,
        stats_history=state.stats_history + [stats]
    )
    
    return new_state, stats


def training_iteration(
    state: TrainingState,
    prompts: List[str],
    model: Any,
    tokenizer: Any,
    gen_config: GenerationConfig,
    algorithm_config: Dict[str, Any],
    reward_fn: Callable[[Rollout], Union[float, List[float]]],
    ref_model_update_fn: Optional[Callable] = None,
    iteration_hook: Optional[Callable[[TrainingState, Batch], None]] = None
) -> tuple[TrainingState, Batch, TrainingStats]:
    """
    Execute full training iteration.
    
    Extension points:
    - reward_fn: Custom reward computation
    - ref_model_update_fn: How to update reference model
    - iteration_hook: Custom logic to run each iteration
    
    Args:
        state: Current training state
        prompts: Prompts for this iteration
        model: Policy model
        tokenizer: Tokenizer
        gen_config: Generation config
        algorithm_config: Algorithm config
        reward_fn: Reward function
        ref_model_update_fn: Reference model update function
        iteration_hook: Custom hook function
    
    Returns:
        Tuple of (new_state, batch, stats)
    """
    # Generate rollouts
    batch = create_batch_rollouts(
        model, tokenizer, prompts, gen_config,
        ref_model=state.ref_model,
        reward_fn=reward_fn
    )
    
    # Run iteration hook if provided
    if iteration_hook is not None:
        iteration_hook(state, batch)
    
    # Training step
    new_state, stats = training_step(state, batch, algorithm_config)
    
    # Update reference model if needed
    if ref_model_update_fn is not None and state.ref_model is not None:
        new_ref_model = update_reference_model(
            state.ref_model, model, ref_model_update_fn
        )
        new_state = new_state._replace(ref_model=new_ref_model)
    
    # Increment iteration
    new_state = new_state._replace(iteration=state.iteration + 1)
    
    return new_state, batch, stats


# ============================================================================
# Training Loop
# ============================================================================

def train(
    initial_state: TrainingState,
    prompt_loader: Callable[[int], List[str]],
    num_iterations: int,
    gen_config: GenerationConfig,
    algorithm_config: Dict[str, Any],
    reward_fn: Callable[[Rollout], Union[float, List[float]]],
    ref_model_update_fn: Optional[Callable] = None,
    checkpoint_fn: Optional[Callable[[TrainingState], None]] = None,
    eval_fn: Optional[Callable[[TrainingState], Dict[str, float]]] = None,
    eval_every: int = 10,
    logger: Optional[Callable[[TrainingState, TrainingStats], None]] = None
) -> TrainingState:
    """
    Main training loop.
    
    Extension points:
    - prompt_loader: How to load prompts for each iteration
    - reward_fn: Reward computation
    - ref_model_update_fn: Reference model updates
    - checkpoint_fn: How to save checkpoints
    - eval_fn: Evaluation logic
    - logger: Logging logic
    
    Args:
        initial_state: Initial training state
        prompt_loader: Function to load prompts for iteration
        num_iterations: Number of training iterations
        gen_config: Generation config
        algorithm_config: Algorithm config
        reward_fn: Reward function
        ref_model_update_fn: Reference model update function
        checkpoint_fn: Checkpoint save function
        eval_fn: Evaluation function
        eval_every: Evaluate every N iterations
        logger: Logging function
    
    Returns:
        Final training state
    """
    state = initial_state
    
    for iteration in range(num_iterations):
        # Load prompts for this iteration
        prompts = prompt_loader(iteration)
        
        # Training iteration
        state, batch, stats = training_iteration(
            state,
            prompts,
            state.model,
            None,  # tokenizer would be passed in real implementation
            gen_config,
            algorithm_config,
            reward_fn,
            ref_model_update_fn
        )
        
        # Logging
        if logger is not None:
            logger(state, stats)
        
        # Evaluation
        if eval_fn is not None and iteration % eval_every == 0:
            eval_results = eval_fn(state)
            print(f"Iteration {iteration} eval: {eval_results}")
        
        # Checkpointing
        if checkpoint_fn is not None and iteration % eval_every == 0:
            checkpoint_fn(state)
    
    return state


# ============================================================================
# Logging Primitives
# ============================================================================

def log_to_console(
    state: TrainingState,
    stats: TrainingStats,
    log_fn: Optional[Callable[[str], None]] = None
) -> None:
    """
    Log training stats to console.
    
    Extension point: Pass custom log_fn
    """
    msg = f"[Iter {state.iteration}] Loss: {stats.loss:.4f}, Reward: {stats.rewards_mean:.4f}, KL: {stats.kl_divergence:.4f}"
    
    if log_fn is not None:
        log_fn(msg)
    else:
        print(msg)


def log_to_wandb(
    state: TrainingState,
    stats: TrainingStats,
    project: str,
    entity: Optional[str] = None
) -> None:
    """
    Log training stats to Weights & Biases.
    
    Args:
        state: Training state
        stats: Training stats
        project: W&B project name
        entity: W&B entity (optional)
    """
    import wandb
    
    if not wandb.run:
        wandb.init(project=project, entity=entity)
    
    wandb.log({
        "iteration": state.iteration,
        "loss": stats.loss,
        "grad_norm": stats.grad_norm,
        "learning_rate": stats.learning_rate,
        "kl_divergence": stats.kl_divergence,
        "rewards_mean": stats.rewards_mean,
        "rewards_std": stats.rewards_std,
        **stats.custom_metrics
    })


# ============================================================================
# Checkpointing
# ============================================================================

def save_checkpoint(
    state: TrainingState,
    checkpoint_dir: str,
    saver: Optional[Callable[[TrainingState, str], None]] = None
) -> None:
    """
    Save training checkpoint.
    
    Extension point: Pass custom saver
    
    Args:
        state: Training state to save
        checkpoint_dir: Directory to save checkpoint
        saver: Custom save function
    """
    if saver is not None:
        saver(state, checkpoint_dir)
        return
    
    # Default: save PyTorch checkpoint
    import torch
    import os
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        "iteration": state.iteration,
        "global_step": state.global_step,
        "model_state_dict": state.model.state_dict(),
        "optimizer_state_dict": state.optimizer.state_dict(),
        "scheduler_state_dict": state.scheduler.state_dict() if state.scheduler else None,
        "stats_history": state.stats_history,
        "metadata": state.metadata
    }
    
    torch.save(checkpoint, os.path.join(checkpoint_dir, f"checkpoint_{state.iteration}.pt"))


def load_checkpoint(
    checkpoint_path: str,
    model: Any,
    optimizer: Any,
    scheduler: Optional[Any] = None,
    loader: Optional[Callable[[str], Dict]] = None
) -> TrainingState:
    """
    Load training checkpoint.
    
    Extension point: Pass custom loader
    
    Args:
        checkpoint_path: Path to checkpoint
        model: Model to load state into
        optimizer: Optimizer to load state into
        scheduler: Scheduler to load state into (optional)
        loader: Custom load function
    
    Returns:
        Training state
    """
    if loader is not None:
        checkpoint = loader(checkpoint_path)
    else:
        import torch
        checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and checkpoint["scheduler_state_dict"]:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    return TrainingState(
        iteration=checkpoint["iteration"],
        global_step=checkpoint["global_step"],
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        stats_history=checkpoint["stats_history"],
        metadata=checkpoint["metadata"]
    )
```

---

## Composition Patterns

### Pattern 1: Basic Training Loop

```python
# Setup
model_config = ModelConfig(model_name_or_path="meta-llama/Llama-3-8B")
gen_config = GenerationConfig(max_new_tokens=512, temperature=0.7)

model = load_model(model_config)
tokenizer = load_tokenizer(model_config)
ref_model = create_reference_model(model)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# Define reward function
def simple_reward(rollout: Rollout) -> float:
    return len(rollout.generation.text)  # Length-based reward

# Algorithm config
algorithm_config = create_algorithm_config(
    algorithm="grpo",
    params={"kl_coef": 0.1}
)

# Initial state
state = TrainingState(
    iteration=0,
    global_step=0,
    model=model,
    optimizer=optimizer,
    ref_model=ref_model
)

# Prompt loader
def load_prompts(iteration: int) -> List[str]:
    return ["Write a story about a robot"] * 32

# Train
final_state = train(
    initial_state=state,
    prompt_loader=load_prompts,
    num_iterations=100,
    gen_config=gen_config,
    algorithm_config=algorithm_config,
    reward_fn=simple_reward,
    logger=log_to_console
)
```

### Pattern 2: Switching Algorithms Mid-Training

```python
def adaptive_algorithm_loader(iteration: int) -> Dict[str, Any]:
    """Switch from GRPO to DAPO after 50 iterations"""
    if iteration < 50:
        return create_algorithm_config("grpo", {"kl_coef": 0.1})
    else:
        return create_algorithm_config(
            "dapo",
            {
                "beta": 0.1,
                "pair_selector": lambda batch: select_top_bottom_pairs(batch)
            }
        )

# Use in training loop with custom iteration function
for iteration in range(100):
    algorithm_config = adaptive_algorithm_loader(iteration)
    # ... rest of training iteration
```

### Pattern 3: Custom Reward Function (Vector Rewards)

```python
def multi_objective_reward(rollout: Rollout) -> List[float]:
    """Compute multiple reward objectives"""
    correctness = check_correctness(rollout.generation.text)
    helpfulness = score_helpfulness(rollout.generation.text)
    safety = check_safety(rollout.generation.text)
    
    return [correctness, helpfulness, safety]

def weighted_aggregator(rewards: List[float]) -> float:
    """Aggregate with weights"""
    weights = [0.5, 0.3, 0.2]
    return sum(w * r for w, r in zip(weights, rewards))

# Use in rollout creation
def reward_with_aggregation(rollout: Rollout) -> float:
    vector_rewards = multi_objective_reward(rollout)
    return weighted_aggregator(vector_rewards)
```

### Pattern 4: Custom KL Divergence

```python
def custom_kl_fn(logprobs: List[float], ref_logprobs: List[float]) -> float:
    """Custom KL that only penalizes large divergences"""
    import math
    kl = 0.0
    for lp, ref_lp in zip(logprobs, ref_logprobs):
        prob = math.exp(lp)
        token_kl = prob * (lp - ref_lp)
        # Only count if exceeds threshold
        if token_kl > 0.1:
            kl += token_kl
    return kl

# Use in rollout creation
rollout = create_rollout(
    model, tokenizer, prompt, gen_config,
    ref_model=ref_model
)
# Override KL with custom computation
custom_kl = custom_kl_fn(rollout.logprobs, rollout.ref_logprobs)
rollout = rollout._replace(kl_divergence=custom_kl)
```

### Pattern 5: EMA Reference Model Updates

```python
def ema_update_fn(ref_model: Any, policy_model: Any, decay: float = 0.999) -> Any:
    """Update reference model using EMA"""
    with torch.no_grad():
        for ref_param, policy_param in zip(ref_model.parameters(), policy_model.parameters()):
            ref_param.data = decay * ref_param.data + (1 - decay) * policy_param.data
    return ref_model

# Use in training loop
final_state = train(
    initial_state=state,
    # ...
    ref_model_update_fn=lambda ref, policy: ema_update_fn(ref, policy, decay=0.999)
)
```

### Pattern 6: Tool Call Tracking

```python
def parse_function_calls(text: str) -> List[Dict[str, Any]]:
    """Parse function calls from text"""
    import re
    pattern = r'<function_call>(.*?)</function_call>'
    matches = re.findall(pattern, text, re.DOTALL)
    return [{"name": m.strip(), "args": {}} for m in matches]

def execute_function(tool_call: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a function call"""
    # Mock execution
    return {
        "id": "call_123",
        "output": f"Result of {tool_call['name']}",
        "success": True
    }

def track_tool_stats(tool_calls: List[Dict]) -> Dict[str, Any]:
    """Track tool usage statistics"""
    return {
        "num_calls": len(tool_calls),
        "unique_tools": len(set(c["name"] for c in tool_calls))
    }

# Use in rollout creation
rollout = create_rollout(
    model, tokenizer, prompt, gen_config,
    tool_parser=parse_function_calls
)

rollout_with_execution = execute_and_track_tools(
    rollout,
    tool_executor=execute_function,
    result_tracker=lambda results: {"success_rate": sum(r["success"] for r in results) / len(results)}
)
```

---

## Extension Scenarios

### Scenario 1: Adding a New Algorithm (REINFORCE)

```python
def compute_reinforce_loss(
    batch: Batch,
    baseline_fn: Optional[Callable[[Rollout], float]] = None
) -> torch.Tensor:
    """REINFORCE algorithm with optional baseline"""
    import torch
    
    losses = []
    for rollout in batch.rollouts:
        reward = rollout.rewards if isinstance(rollout.rewards, float) else sum(rollout.rewards)
        
        # Subtract baseline if provided
        if baseline_fn is not None:
            reward = reward - baseline_fn(rollout)
        
        # REINFORCE: -reward * log_prob
        log_prob_sum = sum(rollout.logprobs)
        loss = -reward * log_prob_sum
        losses.append(loss)
    
    return torch.stack(losses).mean()

# Register in algorithm dispatcher
def compute_loss_by_algorithm_extended(batch: Batch, algorithm_config: Dict[str, Any], **kwargs):
    """Extended version with REINFORCE"""
    algo = algorithm_config["algorithm"]
    
    if algo == "reinforce":
        return compute_reinforce_loss(batch, **algorithm_config["params"])
    else:
        return compute_loss_by_algorithm(batch, algorithm_config, **kwargs)
```

### Scenario 2: Custom Generation Backend (vLLM)

```python
def vllm_generator(model, tokenizer, prompt: str, config: GenerationConfig) -> Generation:
    """Custom generator using vLLM"""
    from vllm import SamplingParams
    
    sampling_params = SamplingParams(
        max_tokens=config.max_new_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        logprobs=1 if config.logprobs else None
    )
    
    outputs = model.generate([prompt], sampling_params)
    output = outputs[0]
    
    logprobs = None
    if output.outputs[0].logprobs:
        logprobs = [lp[token_id] for lp, token_id in zip(
            output.outputs[0].logprobs,
            output.outputs[0].token_ids
        )]
    
    return Generation(
        text=output.outputs[0].text,
        logprobs=logprobs,
        token_ids=output.outputs[0].token_ids,
        finish_reason=output.outputs[0].finish_reason
    )

# Use in rollout creation
rollout = create_rollout(
    vllm_model, tokenizer, prompt, gen_config,
    generator=vllm_generator  # Swap generator
)
```

### Scenario 3: Reward Model Integration

```python
def create_rm_reward_fn(reward_model, tokenizer):
    """Create reward function using reward model"""
    def reward_fn(rollout: Rollout) -> float:
        score = compute_reward_model_score(
            reward_model,
            tokenizer,
            rollout.prompt,
            rollout.generation.text
        )
        return score
    return reward_fn

# Use in training
rm_config = ModelConfig(model_name_or_path="reward-model-path")
reward_model = load_reward_model(rm_config)
reward_fn = create_rm_reward_fn(reward_model, tokenizer)

final_state = train(
    initial_state=state,
    # ...
    reward_fn=reward_fn
)
```

### Scenario 4: Multi-Turn Tool Use

```python
def multi_turn_tool_reward(rollout: Rollout) -> float:
    """Reward based on successful multi-turn tool use"""
    if not rollout.tool_calls:
        return 0.0
    
    # Check if tools were used correctly
    successful_tools = sum(
        1 for result in rollout.tool_results
        if result.get("success", False)
    )
    
    # Reward proportional to successful tool use
    success_rate = successful_tools / len(rollout.tool_calls)
    
    # Bonus for multi-turn
    turn_bonus = len(rollout.conversation.messages) * 0.1
    
    return success_rate + turn_bonus

# Create multi-turn rollout with tools
rollout = create_multiturn_rollout(
    model, tokenizer,
    initial_conversation=Conversation(messages=[
        Message(role="user", content="Help me debug this code")
    ]),
    gen_config=gen_config,
    num_turns=5,
    tool_parser=parse_function_calls,
    tool_executor=execute_function,
    stop_condition=lambda conv: "DONE" in conv.messages[-1].content
)

reward = multi_turn_tool_reward(rollout)
```

### Scenario 5: Dynamic KL Coefficient Scheduling

```python
def create_kl_schedule(
    initial_kl: float = 0.1,
    final_kl: float = 0.01,
    num_iterations: int = 100
):
    """Create KL coefficient schedule"""
    def kl_coef_fn(iteration: int) -> float:
        # Linear decay
        progress = iteration / num_iterations
        return initial_kl + (final_kl - initial_kl) * progress
    return kl_coef_fn

kl_schedule = create_kl_schedule()

# Use in training loop
for iteration in range(100):
    current_kl_coef = kl_schedule(iteration)
    
    algorithm_config = create_algorithm_config(
        "grpo",
        {"kl_coef": current_kl_coef}
    )
    
    # ... training iteration
```

---

## Summary

This functional architecture provides:

1. **Maximum Hackability**
   - Every component accepts function arguments for customization
   - Easy to swap algorithms, reward functions, KL methods, etc.
   - No class hierarchies to navigate

2. **Clear Extension Points**
   - Model loading: `model_loader`, `tokenizer_loader`
   - Generation: `generator`, `logprob_extractor`, `tool_parser`
   - Rewards: `reward_fn`, `aggregator`, `shaping_fn`
   - KL: `kl_fn`, `ref_model_fn`, `update_fn`
   - Algorithms: `advantage_fn`, `loss_fn`, `pair_selector`
   - Training: `checkpoint_fn`, `eval_fn`, `logger`

3. **Composability**
   - Functions compose naturally
   - Data flows explicitly through NamedTuples
   - No hidden state or side effects (except model training)

4. **Flexibility**
   - Switch algorithms mid-training
   - Update reference model with different strategies
   - Combine multiple reward objectives
   - Track tool calls in rollouts
   - Use different generation backends

All primitives are independent, testable, and can be mixed and matched to build custom training pipelines.
