"""
InferenceModule Usage Examples

This file demonstrates various usage patterns for the InferenceModule component.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rl_primitives.inference import (
    InferenceModule,
    create_inference_module,
    GenerationResult,
    ToolCall
)


# ============================================================================
# Example 1: Basic Generation with HuggingFace Backend
# ============================================================================

def example_basic_generation():
    """Demonstrate basic text generation."""
    print("=" * 80)
    print("Example 1: Basic Generation")
    print("=" * 80)

    # Initialize inference module
    inference = InferenceModule(
        model_name="gpt2",
        backend="huggingface",
        device="cpu",  # Change to "cuda" if GPU available
        dtype="float32"
    )

    # Single prompt generation
    result = inference.generate(
        "The future of artificial intelligence is",
        max_new_tokens=50,
        temperature=0.8,
        do_sample=True
    )

    print(f"\nPrompt: The future of artificial intelligence is")
    print(f"Generated: {result.texts[0]}\n")

    # Batch generation
    prompts = [
        "Explain quantum computing in simple terms:",
        "What is the meaning of life?",
        "How does a neural network work?"
    ]

    result = inference.generate(
        prompts,
        max_new_tokens=100,
        temperature=0.7
    )

    print("\nBatch Generation:")
    for i, (prompt, text) in enumerate(zip(prompts, result.texts)):
        print(f"\n{i+1}. {prompt}")
        print(f"   → {text[:150]}...")


# ============================================================================
# Example 2: Generation with Logits and Logprobs
# ============================================================================

def example_logits_and_logprobs():
    """Demonstrate getting logits and log probabilities."""
    print("\n" + "=" * 80)
    print("Example 2: Logits and Log Probabilities")
    print("=" * 80)

    inference = InferenceModule(
        model_name="gpt2",
        backend="huggingface",
        device="cpu"
    )

    # Generate with logits
    result = inference.generate(
        "The capital of France is",
        max_new_tokens=10,
        return_logits=True,
        return_logprobs=True,
        do_sample=False  # Greedy decoding for deterministic output
    )

    print(f"\nGenerated: {result.texts[0]}")

    if result.logits is not None:
        print(f"Logits shape: {result.logits.shape}")
        print(f"Logits (first token, first 10 vocab items): {result.logits[0, 0, :10]}")

    if result.logprobs is not None:
        print(f"Logprobs shape: {result.logprobs.shape}")
        print(f"Logprobs (first 5 tokens): {result.logprobs[0, :5]}")
        print(f"Average log probability: {result.logprobs[0].mean().item():.4f}")

    # Get logprobs for specific prompt-response pairs (for RL training)
    prompts = ["What is 2+2?", "What is the sky?"]
    responses = [" 4", " blue"]

    logprobs = inference.get_logprobs(prompts, responses)
    print(f"\n\nLogprobs for prompt-response pairs:")
    print(f"Shape: {logprobs.shape}")

    for i, (p, r, lp) in enumerate(zip(prompts, responses, logprobs)):
        print(f"{i+1}. '{p}' → '{r}'")
        print(f"   Mean logprob: {lp.mean().item():.4f}")


# ============================================================================
# Example 3: Tool Calling
# ============================================================================

def example_tool_calling():
    """Demonstrate generation with tool calling support."""
    print("\n" + "=" * 80)
    print("Example 3: Tool Calling")
    print("=" * 80)

    inference = InferenceModule(
        model_name="gpt2",
        backend="huggingface",
        device="cpu",
        tool_parser="json"  # Can be "json", "xml", or "function"
    )

    # Define available tools
    tools = [
        {
            "name": "search",
            "description": "Search the web for information",
            "parameters": {
                "query": "string",
                "max_results": "integer"
            }
        },
        {
            "name": "calculator",
            "description": "Perform mathematical calculations",
            "parameters": {
                "expression": "string"
            }
        }
    ]

    # Generate with tools
    prompt = "I need to calculate 25 * 37 and search for Python tutorials."

    result, tool_calls = inference.batch_generate_with_tools(
        prompt,
        available_tools=tools,
        max_new_tokens=200,
        temperature=0.7
    )

    print(f"\nPrompt: {prompt}")
    print(f"\nGenerated response:\n{result.texts[0]}\n")

    print(f"Tool calls detected: {len(tool_calls[0])}")
    for tc in tool_calls[0]:
        print(f"\n  Tool: {tc.name}")
        print(f"  Arguments: {tc.arguments}")
        print(f"  Raw text: {tc.raw_text[:100]}...")


# ============================================================================
# Example 4: Caching for Efficiency
# ============================================================================

def example_caching():
    """Demonstrate generation caching."""
    print("\n" + "=" * 80)
    print("Example 4: Generation Caching")
    print("=" * 80)

    inference = InferenceModule(
        model_name="gpt2",
        backend="huggingface",
        device="cpu",
        cache_size=100  # Cache up to 100 generations
    )

    import time

    prompt = "The quick brown fox"

    # First generation (not cached)
    start = time.time()
    result1 = inference.generate(prompt, max_new_tokens=20, temperature=0.7)
    time1 = time.time() - start

    # Second generation (cached)
    start = time.time()
    result2 = inference.generate(prompt, max_new_tokens=20, temperature=0.7)
    time2 = time.time() - start

    print(f"\nPrompt: {prompt}")
    print(f"\nFirst generation (uncached): {time1:.4f}s")
    print(f"  Result: {result1.texts[0]}")
    print(f"\nSecond generation (cached): {time2:.4f}s")
    print(f"  Result: {result2.texts[0]}")
    print(f"\nSpeedup: {time1/time2:.2f}x")

    # Clear cache
    inference.clear_cache()
    print("\nCache cleared!")


# ============================================================================
# Example 5: Using Factory Function with Auto Backend Selection
# ============================================================================

def example_factory_function():
    """Demonstrate using the factory function."""
    print("\n" + "=" * 80)
    print("Example 5: Factory Function with Auto Backend")
    print("=" * 80)

    # Auto-select backend (will use vLLM if available, else HuggingFace)
    inference = create_inference_module(
        "gpt2",
        backend="auto",
        device="cpu",
        cache_size=50
    )

    print(f"\nInferenceModule created: {inference}")

    result = inference.generate(
        "Hello, world!",
        max_new_tokens=30
    )

    print(f"\nGenerated: {result.texts[0]}")


# ============================================================================
# Example 6: Advanced - KL Divergence Calculation
# ============================================================================

def example_kl_divergence():
    """Demonstrate KL divergence calculation (useful for RL training)."""
    print("\n" + "=" * 80)
    print("Example 6: KL Divergence Calculation")
    print("=" * 80)

    import torch
    import torch.nn.functional as F

    # Create two inference modules (policy and reference)
    policy = InferenceModule(
        model_name="gpt2",
        backend="huggingface",
        device="cpu"
    )

    # In practice, reference would be a frozen copy of the initial policy
    reference = policy  # For demo purposes

    prompts = ["What is AI?", "Explain machine learning"]
    responses = [" Artificial Intelligence", " A subset of AI"]

    # Get logits from both models
    policy_logits = policy.get_logits(prompts, responses)
    ref_logits = reference.get_logits(prompts, responses)

    # Calculate KL divergence
    policy_logprobs = F.log_softmax(policy_logits, dim=-1)
    ref_probs = F.softmax(ref_logits, dim=-1)

    kl_div = F.kl_div(
        policy_logprobs,
        ref_probs,
        reduction='batchmean',
        log_target=False
    )

    print(f"\nKL Divergence between policy and reference: {kl_div.item():.6f}")
    print("\nNote: In RL training, you would:")
    print("  1. Keep reference model frozen")
    print("  2. Update policy based on rewards + KL penalty")
    print("  3. Periodically update reference to current policy")


# ============================================================================
# Example 7: Multi-Backend Comparison
# ============================================================================

def example_backend_comparison():
    """Compare HuggingFace and vLLM backends (if vLLM is available)."""
    print("\n" + "=" * 80)
    print("Example 7: Backend Comparison")
    print("=" * 80)

    import time

    prompt = "Explain reinforcement learning:"

    # HuggingFace backend
    print("\nTesting HuggingFace backend...")
    hf_inference = InferenceModule(
        "gpt2",
        backend="huggingface",
        device="cpu"
    )

    start = time.time()
    hf_result = hf_inference.generate(
        prompt,
        max_new_tokens=50,
        temperature=0.8
    )
    hf_time = time.time() - start

    print(f"HuggingFace time: {hf_time:.4f}s")
    print(f"Result: {hf_result.texts[0][:100]}...")

    # vLLM backend (if available)
    try:
        print("\nTesting vLLM backend...")
        vllm_inference = InferenceModule(
            "gpt2",
            backend="vllm",
            gpu_memory_utilization=0.5
        )

        start = time.time()
        vllm_result = vllm_inference.generate(
            prompt,
            max_new_tokens=50,
            temperature=0.8
        )
        vllm_time = time.time() - start

        print(f"vLLM time: {vllm_time:.4f}s")
        print(f"Result: {vllm_result.texts[0][:100]}...")
        print(f"\nSpeedup: {hf_time/vllm_time:.2f}x")

    except Exception as e:
        print(f"\nvLLM not available: {e}")
        print("Install with: pip install vllm")


# ============================================================================
# Main
# ============================================================================

def main():
    """Run all examples."""
    examples = [
        ("Basic Generation", example_basic_generation),
        ("Logits and Logprobs", example_logits_and_logprobs),
        ("Tool Calling", example_tool_calling),
        ("Caching", example_caching),
        ("Factory Function", example_factory_function),
        ("KL Divergence", example_kl_divergence),
        ("Backend Comparison", example_backend_comparison),
    ]

    print("\n" + "=" * 80)
    print("InferenceModule Examples")
    print("=" * 80)
    print("\nSelect examples to run:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    print(f"  {len(examples)+1}. Run all examples")
    print("  0. Exit")

    try:
        choice = input("\nEnter choice (or comma-separated list): ").strip()

        if choice == "0":
            return

        if choice == str(len(examples) + 1) or choice == "":
            # Run all
            for name, func in examples:
                try:
                    func()
                except Exception as e:
                    print(f"\nError in {name}: {e}")
                    import traceback
                    traceback.print_exc()
        else:
            # Run selected
            choices = [int(c.strip()) for c in choice.split(",")]
            for c in choices:
                if 1 <= c <= len(examples):
                    try:
                        examples[c-1][1]()
                    except Exception as e:
                        print(f"\nError in {examples[c-1][0]}: {e}")
                        import traceback
                        traceback.print_exc()

    except KeyboardInterrupt:
        print("\n\nExiting...")
    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    # For automated testing, run all examples
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        print("Running all examples in non-interactive mode...\n")
        try:
            example_basic_generation()
        except Exception as e:
            print(f"Note: Some examples may fail without proper dependencies")
            print(f"Error: {e}")
    else:
        main()
