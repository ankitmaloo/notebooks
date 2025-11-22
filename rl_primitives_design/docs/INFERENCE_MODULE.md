# InferenceModule Documentation

## Overview

The `InferenceModule` is a production-ready component for LLM inference in RL training scenarios. It provides a unified interface across multiple backends (vLLM, HuggingFace Transformers) with support for generation, logit computation, log probability extraction, and tool calling.

## Features

### ✅ Multi-Backend Support
- **HuggingFace Transformers**: Full-featured backend with logits/logprobs support
- **vLLM**: High-performance backend for efficient serving
- Automatic backend selection

### ✅ Comprehensive Generation API
- Single and batch generation
- Customizable sampling parameters (temperature, top-p, top-k)
- Return logits and log probabilities
- Generation caching for efficiency

### ✅ RL Training Support
- `get_logits()`: For KL divergence calculations
- `get_logprobs()`: For policy gradient computation
- Efficient batching for rollouts

### ✅ Tool Calling
- Multiple parsing formats (JSON, XML, function-style)
- Automatic prompt augmentation with tool descriptions
- Structured tool call extraction

### ✅ Production-Ready
- Comprehensive error handling
- Type hints throughout
- Extensive documentation
- Configurable caching
- Performance optimizations

## Installation

### Basic Installation

```bash
pip install torch transformers
```

### With vLLM Support (Recommended for Production)

```bash
pip install torch transformers vllm
```

### With All Features

```bash
pip install torch transformers vllm accelerate
```

## Quick Start

### Basic Usage

```python
from rl_primitives.inference import InferenceModule

# Initialize with HuggingFace backend
inference = InferenceModule(
    model_name="gpt2",
    backend="huggingface",
    device="cuda"
)

# Generate text
result = inference.generate(
    "The future of AI is",
    max_new_tokens=50,
    temperature=0.8
)

print(result.texts[0])
```

### Factory Function (Recommended)

```python
from rl_primitives.inference import create_inference_module

# Auto-select best available backend
inference = create_inference_module(
    "meta-llama/Llama-2-7b-hf",
    backend="auto",  # Uses vLLM if available, else HuggingFace
    dtype="bfloat16"
)
```

## API Reference

### InferenceModule

#### Constructor

```python
InferenceModule(
    model_name: str,
    backend: str = "huggingface",
    cache_size: int = 1000,
    tool_parser: str = "json",
    **backend_kwargs
)
```

**Parameters:**
- `model_name`: HuggingFace model identifier
- `backend`: Backend to use ("huggingface" or "vllm")
- `cache_size`: Number of generations to cache (0 to disable)
- `tool_parser`: Tool parsing format ("json", "xml", "function")
- `**backend_kwargs`: Backend-specific parameters

**Backend-Specific Parameters:**

For **HuggingFace**:
- `device`: Device placement ("auto", "cuda", "cpu")
- `dtype`: Model dtype ("auto", "float16", "bfloat16", "float32")
- `trust_remote_code`: Whether to trust remote code (default: False)

For **vLLM**:
- `tensor_parallel_size`: Number of GPUs for tensor parallelism
- `dtype`: Model dtype
- `gpu_memory_utilization`: GPU memory ratio (default: 0.9)

#### Methods

##### generate()

Core generation method with full control over sampling.

```python
def generate(
    self,
    prompts: Union[str, List[str]],
    use_cache: bool = True,
    **kwargs
) -> GenerationResult
```

**Parameters:**
- `prompts`: Single prompt or list of prompts
- `use_cache`: Whether to use generation cache
- `max_new_tokens`: Maximum tokens to generate (default: 256)
- `temperature`: Sampling temperature (default: 1.0)
- `top_p`: Nucleus sampling parameter (default: 1.0)
- `top_k`: Top-k sampling parameter (default: -1)
- `do_sample`: Whether to use sampling (default: True)
- `num_return_sequences`: Number of sequences per prompt (default: 1)
- `return_logits`: Whether to return logits (default: False)
- `return_logprobs`: Whether to return log probabilities (default: False)

**Returns:** `GenerationResult` object

**Example:**
```python
result = inference.generate(
    ["What is AI?", "Explain ML"],
    max_new_tokens=100,
    temperature=0.7,
    top_p=0.9,
    return_logprobs=True
)

for text in result.texts:
    print(text)
```

##### get_logits()

Get logits for prompt-response pairs (for KL divergence calculation).

```python
def get_logits(
    self,
    prompts: Union[str, List[str]],
    responses: Union[str, List[str]]
) -> torch.Tensor
```

**Parameters:**
- `prompts`: Single prompt or list of prompts
- `responses`: Corresponding responses

**Returns:** Tensor of shape `[batch_size, seq_len, vocab_size]`

**Example:**
```python
import torch.nn.functional as F

# Policy and reference models
policy_logits = policy.get_logits(prompts, responses)
ref_logits = reference.get_logits(prompts, responses)

# Calculate KL divergence
kl_div = F.kl_div(
    F.log_softmax(policy_logits, dim=-1),
    F.softmax(ref_logits, dim=-1),
    reduction='batchmean'
)
```

##### get_logprobs()

Get log probabilities for prompt-response pairs (for policy gradients).

```python
def get_logprobs(
    self,
    prompts: Union[str, List[str]],
    responses: Union[str, List[str]]
) -> torch.Tensor
```

**Parameters:**
- `prompts`: Single prompt or list of prompts
- `responses`: Corresponding responses

**Returns:** Tensor of shape `[batch_size, seq_len]`

**Example:**
```python
# Get log probabilities for rewards weighting
logprobs = inference.get_logprobs(prompts, responses)

# Policy gradient loss
advantages = compute_advantages(rewards)
policy_loss = -(logprobs * advantages).mean()
```

##### batch_generate_with_tools()

Generate with tool calling support.

```python
def batch_generate_with_tools(
    self,
    prompts: Union[str, List[str]],
    available_tools: Optional[List[Dict[str, Any]]] = None,
    max_new_tokens: int = 512,
    **kwargs
) -> Tuple[GenerationResult, List[List[ToolCall]]]
```

**Parameters:**
- `prompts`: Single prompt or list of prompts
- `available_tools`: List of tool definitions
- `max_new_tokens`: Maximum tokens to generate
- `**kwargs`: Additional generation parameters

**Returns:** Tuple of (GenerationResult, List of ToolCalls per prompt)

**Example:**
```python
tools = [
    {
        "name": "search",
        "description": "Search the web",
        "parameters": {"query": "string"}
    }
]

result, tool_calls = inference.batch_generate_with_tools(
    "Find the latest AI news",
    available_tools=tools
)

for tc in tool_calls[0]:
    print(f"Tool: {tc.name}, Args: {tc.arguments}")
```

##### parse_tool_calls()

Parse tool calls from generated text.

```python
def parse_tool_calls(self, text: str) -> List[ToolCall]
```

**Parameters:**
- `text`: Generated text potentially containing tool calls

**Returns:** List of `ToolCall` objects

##### clear_cache()

Clear the generation cache.

```python
def clear_cache(self) -> None
```

### Data Structures

#### GenerationResult

```python
@dataclass
class GenerationResult:
    texts: List[str]
    token_ids: Optional[List[List[int]]] = None
    logprobs: Optional[List[torch.Tensor]] = None
    logits: Optional[List[torch.Tensor]] = None
    metadata: Optional[Dict[str, Any]] = None
```

#### ToolCall

```python
@dataclass
class ToolCall:
    name: str
    arguments: Dict[str, Any]
    raw_text: str
    start_pos: int
    end_pos: int
```

## Advanced Usage

### RL Training Integration

#### PPO Training Loop

```python
from rl_primitives.inference import InferenceModule
import torch.nn.functional as F

# Initialize
policy = InferenceModule("gpt2", backend="huggingface")
reference = InferenceModule("gpt2", backend="huggingface")
# Freeze reference model
reference.backend.model.requires_grad_(False)

# Training loop
for batch in dataloader:
    # Generate responses
    result = policy.generate(
        batch['prompts'],
        max_new_tokens=128,
        temperature=0.8
    )

    # Compute rewards (task-specific)
    rewards = compute_rewards(batch['prompts'], result.texts)

    # Get log probabilities
    logprobs = policy.get_logprobs(batch['prompts'], result.texts)

    # Compute KL penalty
    policy_logits = policy.get_logits(batch['prompts'], result.texts)
    ref_logits = reference.get_logits(batch['prompts'], result.texts)
    kl_div = F.kl_div(
        F.log_softmax(policy_logits, dim=-1),
        F.softmax(ref_logits, dim=-1),
        reduction='batchmean'
    )

    # Compute loss
    advantages = compute_advantages(rewards)
    policy_loss = -(logprobs * advantages).mean()
    total_loss = policy_loss + 0.1 * kl_div

    # Update
    total_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

#### GRPO Training Loop

```python
# GRPO doesn't need value function, just relative rewards
for batch in dataloader:
    # Generate multiple responses per prompt
    results = []
    for prompt in batch['prompts']:
        result = policy.generate(
            [prompt] * 8,  # 8 samples per prompt
            max_new_tokens=128,
            temperature=0.8
        )
        results.append(result)

    # Compute relative rewards (rank-based)
    all_rewards = compute_rewards_batch(results)
    relative_rewards = rank_normalize(all_rewards)

    # Update on better-than-average samples
    for prompt, responses, rel_rewards in zip(
        batch['prompts'], results, relative_rewards
    ):
        # Filter good samples
        good_mask = rel_rewards > 0
        good_responses = [r for r, m in zip(responses, good_mask) if m]
        good_rewards = rel_rewards[good_mask]

        # Compute loss
        logprobs = policy.get_logprobs(
            [prompt] * len(good_responses),
            good_responses
        )
        loss = -(logprobs * good_rewards).mean()

        loss.backward()

    optimizer.step()
    optimizer.zero_grad()
```

### Tool-Augmented Environments

```python
class ToolEnvironment:
    def __init__(self, inference: InferenceModule):
        self.inference = inference
        self.tools = self.define_tools()

    def define_tools(self):
        return [
            {
                "name": "verify",
                "description": "Verify mathematical calculation",
                "parameters": {"expression": "string"}
            },
            {
                "name": "search",
                "description": "Search for information",
                "parameters": {"query": "string"}
            }
        ]

    def step(self, state):
        # Generate with tools
        result, tool_calls = self.inference.batch_generate_with_tools(
            state.prompt,
            available_tools=self.tools
        )

        # Execute tool calls
        for tc in tool_calls[0]:
            if tc.name == "verify":
                result = self.execute_verify(tc.arguments)
                state.add_observation(result)
            elif tc.name == "search":
                result = self.execute_search(tc.arguments)
                state.add_observation(result)

        state.response = result.texts[0]
        return state
```

### Caching Strategies

```python
# Disable caching for exploration
inference = InferenceModule("gpt2", cache_size=0)

# Large cache for evaluation
inference = InferenceModule("gpt2", cache_size=10000)

# Manual cache management
inference.generate(prompts, use_cache=True)  # Use cache
inference.generate(prompts, use_cache=False)  # Bypass cache
inference.clear_cache()  # Clear all cached results
```

### Multi-GPU Inference with vLLM

```python
# vLLM with tensor parallelism
inference = InferenceModule(
    "meta-llama/Llama-2-70b-hf",
    backend="vllm",
    tensor_parallel_size=4,  # Use 4 GPUs
    dtype="bfloat16",
    gpu_memory_utilization=0.95
)

# Efficient batch generation
prompts = ["prompt"] * 100  # Large batch
result = inference.generate(
    prompts,
    max_new_tokens=256,
    temperature=0.8
)
```

## Tool Call Formats

### JSON Format

```json
{
    "tool": "search",
    "arguments": {
        "query": "Python tutorials",
        "max_results": 5
    }
}
```

### XML Format (Anthropic-style)

```xml
<function_calls>
<invoke>
<tool_name>search</tool_name>
<parameters>
<query>Python tutorials</query>
<max_results>5</max_results>
</parameters>
</invoke>
</function_calls>
```

### Function Style

```python
search(query="Python tutorials", max_results=5)
```

## Performance Considerations

### Backend Selection

| Backend | Use Case | Pros | Cons |
|---------|----------|------|------|
| **HuggingFace** | Small models, development, logits needed | Full features, easy debugging | Slower for large batches |
| **vLLM** | Production, large models, high throughput | Very fast, efficient | Limited logits support |

### Optimization Tips

1. **Use vLLM for inference-heavy workloads**
   ```python
   inference = InferenceModule("model", backend="vllm")
   ```

2. **Enable caching for repeated prompts**
   ```python
   inference = InferenceModule("model", cache_size=5000)
   ```

3. **Batch requests when possible**
   ```python
   # Good: Single batched call
   result = inference.generate(prompts)

   # Bad: Multiple individual calls
   results = [inference.generate(p) for p in prompts]
   ```

4. **Use appropriate precision**
   ```python
   # bfloat16 is ideal for most modern GPUs
   inference = InferenceModule("model", dtype="bfloat16")
   ```

5. **Adjust GPU memory utilization for vLLM**
   ```python
   # Leave some memory for other operations
   inference = InferenceModule(
       "model",
       backend="vllm",
       gpu_memory_utilization=0.85
   )
   ```

## Troubleshooting

### Issue: Out of Memory (OOM)

**Solutions:**
1. Reduce batch size
2. Use smaller model
3. Enable gradient checkpointing (HuggingFace)
4. Lower GPU memory utilization (vLLM)
5. Use quantization

```python
# Enable gradient checkpointing
inference = InferenceModule(
    "model",
    backend="huggingface",
    gradient_checkpointing=True
)
```

### Issue: Slow Generation

**Solutions:**
1. Switch to vLLM backend
2. Enable caching
3. Use larger batch sizes
4. Reduce max_new_tokens

### Issue: vLLM Not Available

**Solution:**
```bash
# Install vLLM
pip install vllm

# Or use HuggingFace backend
inference = InferenceModule("model", backend="huggingface")
```

### Issue: Tool Calls Not Detected

**Solutions:**
1. Check tool parser format matches output format
2. Try different parsers ("json", "xml", "function")
3. Inspect raw generated text

```python
result, tool_calls = inference.batch_generate_with_tools(
    prompt,
    available_tools=tools
)

# Debug: Check raw output
print("Raw output:", result.texts[0])

# Try different parsers
inference.tool_parser_type = "xml"
tool_calls = inference.parse_tool_calls(result.texts[0])
```

## Best Practices

1. **Use factory function for flexibility**
   ```python
   inference = create_inference_module("model", backend="auto")
   ```

2. **Freeze reference models in RL training**
   ```python
   reference.backend.model.requires_grad_(False)
   ```

3. **Batch operations when possible**
   ```python
   # Batch prompts, responses, and logprob computation
   logprobs = inference.get_logprobs(prompts, responses)
   ```

4. **Cache expensive computations**
   ```python
   inference = InferenceModule("model", cache_size=1000)
   ```

5. **Handle errors gracefully**
   ```python
   try:
       result = inference.generate(prompts)
   except Exception as e:
       logger.error(f"Generation failed: {e}")
       # Fallback strategy
   ```

## Examples

See `examples/inference_examples.py` for comprehensive usage examples including:
- Basic generation
- Logits and logprobs
- Tool calling
- Caching
- RL training integration
- Multi-backend comparison

## Related Components

- **BackpropModule**: Gradient computation and optimization
- **RolloutManager**: Parallel trajectory collection
- **Environment**: Task-specific interaction logic
- **RewardComputer**: Reward calculation strategies

## References

- [HuggingFace Transformers Documentation](https://huggingface.co/docs/transformers)
- [vLLM Documentation](https://docs.vllm.ai)
- Architecture document: `v2.md`

## License

MIT License
