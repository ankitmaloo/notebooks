"""
InferenceModule: Production-ready inference component for RL training.

Supports multiple backends (vLLM, HuggingFace Transformers) with unified interface
for generation, logit computation, and tool-calling capabilities.

Author: RL Primitives
License: MIT
"""

import json
import re
import hashlib
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
from functools import lru_cache
import warnings

import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer
)

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    warnings.warn("vLLM not available. Install with: pip install vllm")


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class GenerationResult:
    """Result from generation with metadata."""
    texts: List[str]
    token_ids: Optional[List[List[int]]] = None
    logprobs: Optional[List[torch.Tensor]] = None
    logits: Optional[List[torch.Tensor]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ToolCall:
    """Parsed tool call from model output."""
    name: str
    arguments: Dict[str, Any]
    raw_text: str
    start_pos: int
    end_pos: int


# ============================================================================
# Backend Abstraction
# ============================================================================

class InferenceBackend(ABC):
    """Abstract base class for inference backends."""

    @abstractmethod
    def generate(
        self,
        prompts: Union[str, List[str]],
        **kwargs
    ) -> GenerationResult:
        """Generate text from prompts."""
        pass

    @abstractmethod
    def get_logits(
        self,
        prompts: Union[str, List[str]],
        responses: Union[str, List[str]]
    ) -> torch.Tensor:
        """Get logits for prompt-response pairs."""
        pass

    @abstractmethod
    def get_logprobs(
        self,
        prompts: Union[str, List[str]],
        responses: Union[str, List[str]]
    ) -> torch.Tensor:
        """Get log probabilities for prompt-response pairs."""
        pass


class HuggingFaceBackend(InferenceBackend):
    """HuggingFace Transformers backend for inference."""

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        dtype: str = "auto",
        trust_remote_code: bool = False,
        **model_kwargs
    ):
        """
        Initialize HuggingFace backend.

        Args:
            model_name: HuggingFace model identifier
            device: Device placement ("auto", "cuda", "cpu")
            dtype: Model dtype ("auto", "float16", "bfloat16", "float32")
            trust_remote_code: Whether to trust remote code
            **model_kwargs: Additional arguments for model loading
        """
        self.model_name = model_name
        self.device = device

        # Setup dtype
        dtype_map = {
            "auto": "auto",
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        self.dtype = dtype_map.get(dtype, "auto")

        # Load model and tokenizer
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code
        )

        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            device_map=device if device != "auto" else "auto",
            trust_remote_code=trust_remote_code,
            **model_kwargs
        )

        self.model.eval()

        # Determine actual device
        if device == "auto":
            self.actual_device = next(self.model.parameters()).device
        else:
            self.actual_device = torch.device(device)

        print(f"Model loaded on device: {self.actual_device}")

    def generate(
        self,
        prompts: Union[str, List[str]],
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        do_sample: bool = True,
        num_return_sequences: int = 1,
        return_logits: bool = False,
        return_logprobs: bool = False,
        **kwargs
    ) -> GenerationResult:
        """
        Generate text from prompts.

        Args:
            prompts: Single prompt or list of prompts
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            do_sample: Whether to use sampling
            num_return_sequences: Number of sequences to generate per prompt
            return_logits: Whether to return logits
            return_logprobs: Whether to return log probabilities
            **kwargs: Additional generation parameters

        Returns:
            GenerationResult with generated texts and optional logits/logprobs
        """
        # Ensure prompts is a list
        if isinstance(prompts, str):
            prompts = [prompts]

        # Tokenize inputs
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=kwargs.get("max_length", 2048)
        ).to(self.actual_device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k if top_k > 0 else None,
                do_sample=do_sample,
                num_return_sequences=num_return_sequences,
                return_dict_in_generate=True,
                output_scores=return_logits or return_logprobs,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )

        # Decode generated sequences
        generated_sequences = outputs.sequences

        # Remove prompt tokens from generated sequences
        prompt_lengths = inputs.input_ids.shape[1]
        generated_tokens = generated_sequences[:, prompt_lengths:]

        # Decode to text
        texts = self.tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True
        )

        # Prepare result
        result = GenerationResult(
            texts=texts,
            token_ids=generated_tokens.cpu().tolist()
        )

        # Add logits if requested
        if return_logits and hasattr(outputs, 'scores'):
            # scores is a tuple of tensors (one per generation step)
            # Stack them: [num_steps, batch_size, vocab_size]
            logits_stack = torch.stack(outputs.scores, dim=0)
            # Transpose to [batch_size, num_steps, vocab_size]
            logits = logits_stack.transpose(0, 1)
            result.logits = logits

        # Add log probabilities if requested
        if return_logprobs and hasattr(outputs, 'scores'):
            transition_scores = self.model.compute_transition_scores(
                outputs.sequences,
                outputs.scores,
                normalize_logits=True
            )
            # Get only the generated part (exclude prompt)
            result.logprobs = transition_scores[:, prompt_lengths:]

        return result

    def get_logits(
        self,
        prompts: Union[str, List[str]],
        responses: Union[str, List[str]]
    ) -> torch.Tensor:
        """
        Get logits for prompt-response pairs.

        Args:
            prompts: Single prompt or list of prompts
            responses: Single response or list of responses

        Returns:
            Tensor of shape [batch_size, seq_len, vocab_size]
        """
        # Ensure lists
        if isinstance(prompts, str):
            prompts = [prompts]
        if isinstance(responses, str):
            responses = [responses]

        assert len(prompts) == len(responses), \
            f"Prompts and responses must have same length: {len(prompts)} vs {len(responses)}"

        # Combine prompts and responses
        full_texts = [p + r for p, r in zip(prompts, responses)]

        # Tokenize
        inputs = self.tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(self.actual_device)

        # Get logits via forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        return logits

    def get_logprobs(
        self,
        prompts: Union[str, List[str]],
        responses: Union[str, List[str]]
    ) -> torch.Tensor:
        """
        Get log probabilities for prompt-response pairs.

        Args:
            prompts: Single prompt or list of prompts
            responses: Single response or list of responses

        Returns:
            Tensor of shape [batch_size, seq_len] with log probabilities
        """
        # Get logits
        logits = self.get_logits(prompts, responses)

        # Ensure lists
        if isinstance(prompts, str):
            prompts = [prompts]
        if isinstance(responses, str):
            responses = [responses]

        # Get response tokens for indexing
        full_texts = [p + r for p, r in zip(prompts, responses)]
        inputs = self.tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(self.actual_device)

        # Compute log probabilities
        # Shift logits and labels for next token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = inputs.input_ids[:, 1:].contiguous()

        # Get log probs
        log_probs = F.log_softmax(shift_logits, dim=-1)

        # Gather log probs for actual tokens
        token_log_probs = torch.gather(
            log_probs,
            dim=2,
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        return token_log_probs


class VLLMBackend(InferenceBackend):
    """vLLM backend for high-performance inference."""

    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        gpu_memory_utilization: float = 0.9,
        **llm_kwargs
    ):
        """
        Initialize vLLM backend.

        Args:
            model_name: HuggingFace model identifier
            tensor_parallel_size: Number of GPUs for tensor parallelism
            dtype: Model dtype ("auto", "float16", "bfloat16", "float32")
            gpu_memory_utilization: GPU memory utilization ratio
            **llm_kwargs: Additional arguments for vLLM LLM
        """
        if not VLLM_AVAILABLE:
            raise ImportError(
                "vLLM is not installed. Install with: pip install vllm"
            )

        self.model_name = model_name

        print(f"Loading model with vLLM: {model_name}")
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            **llm_kwargs
        )

        # Load tokenizer separately for utility functions
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("vLLM model loaded successfully")

    def generate(
        self,
        prompts: Union[str, List[str]],
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        num_return_sequences: int = 1,
        return_logits: bool = False,
        return_logprobs: bool = False,
        **kwargs
    ) -> GenerationResult:
        """
        Generate text from prompts using vLLM.

        Args:
            prompts: Single prompt or list of prompts
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            num_return_sequences: Number of sequences per prompt
            return_logits: Whether to return logits (Note: limited in vLLM)
            return_logprobs: Whether to return log probabilities
            **kwargs: Additional sampling parameters

        Returns:
            GenerationResult with generated texts
        """
        # Ensure prompts is a list
        if isinstance(prompts, str):
            prompts = [prompts]

        # Create sampling params
        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k if top_k > 0 else -1,
            n=num_return_sequences,
            logprobs=1 if return_logprobs else None,
            **kwargs
        )

        # Generate
        outputs = self.llm.generate(prompts, sampling_params)

        # Extract results
        texts = []
        all_logprobs = [] if return_logprobs else None

        for output in outputs:
            for completion in output.outputs:
                texts.append(completion.text)

                if return_logprobs and completion.logprobs:
                    # Convert vLLM logprobs to tensor format
                    logprobs_list = [
                        list(token_logprobs.values())[0]
                        if token_logprobs else 0.0
                        for token_logprobs in completion.logprobs
                    ]
                    all_logprobs.append(torch.tensor(logprobs_list))

        result = GenerationResult(
            texts=texts,
            logprobs=all_logprobs if return_logprobs else None
        )

        if return_logits:
            warnings.warn(
                "vLLM does not directly support returning logits. "
                "Use HuggingFace backend if logits are required."
            )

        return result

    def get_logits(
        self,
        prompts: Union[str, List[str]],
        responses: Union[str, List[str]]
    ) -> torch.Tensor:
        """
        Get logits for prompt-response pairs.

        Note: vLLM doesn't directly expose logits. This is a workaround
        that may not be as efficient as HuggingFace backend.

        Args:
            prompts: Single prompt or list of prompts
            responses: Single response or list of responses

        Returns:
            Tensor of logits (Note: requires loading separate model)
        """
        raise NotImplementedError(
            "vLLM backend does not support direct logit extraction. "
            "For logit computation, use HuggingFaceBackend or implement "
            "a custom solution using the underlying model."
        )

    def get_logprobs(
        self,
        prompts: Union[str, List[str]],
        responses: Union[str, List[str]]
    ) -> torch.Tensor:
        """
        Get log probabilities for prompt-response pairs.

        This computes logprobs by conditioning on the full prompt+response.

        Args:
            prompts: Single prompt or list of prompts
            responses: Single response or list of responses

        Returns:
            Tensor of log probabilities
        """
        # Ensure lists
        if isinstance(prompts, str):
            prompts = [prompts]
        if isinstance(responses, str):
            responses = [responses]

        # For vLLM, we can get logprobs by generating with the response as constraint
        # This is a workaround - not ideal but functional
        warnings.warn(
            "vLLM logprobs computation is approximate. "
            "For exact logprobs, consider using HuggingFaceBackend."
        )

        # Combine and tokenize to get response lengths
        full_texts = [p + r for p, r in zip(prompts, responses)]

        # Use vLLM to get logprobs for the full sequence
        sampling_params = SamplingParams(
            max_tokens=1,  # We just want logprobs, not generation
            logprobs=1,
            prompt_logprobs=1
        )

        outputs = self.llm.generate(full_texts, sampling_params)

        # Extract prompt logprobs (these include the response portion)
        all_logprobs = []
        for output in outputs:
            if output.prompt_logprobs:
                logprobs_list = [
                    list(token_logprobs.values())[0] if token_logprobs else 0.0
                    for token_logprobs in output.prompt_logprobs
                ]
                all_logprobs.append(torch.tensor(logprobs_list))

        # Stack into tensor
        if all_logprobs:
            # Pad to same length
            max_len = max(lp.shape[0] for lp in all_logprobs)
            padded = []
            for lp in all_logprobs:
                if lp.shape[0] < max_len:
                    padding = torch.zeros(max_len - lp.shape[0])
                    lp = torch.cat([lp, padding])
                padded.append(lp)
            return torch.stack(padded)

        return torch.tensor([])


# ============================================================================
# Tool Call Parsing
# ============================================================================

class ToolCallParser:
    """Parser for extracting tool calls from model outputs."""

    @staticmethod
    def parse_json_tool_calls(text: str) -> List[ToolCall]:
        """
        Parse tool calls in JSON format.

        Expected format:
        ```json
        {
            "tool": "tool_name",
            "arguments": {"arg1": "value1", "arg2": "value2"}
        }
        ```

        Args:
            text: Generated text potentially containing tool calls

        Returns:
            List of parsed ToolCall objects
        """
        tool_calls = []

        # Find JSON blocks
        json_pattern = r'```json\s*(.*?)\s*```'
        matches = re.finditer(json_pattern, text, re.DOTALL)

        for match in matches:
            try:
                json_str = match.group(1)
                data = json.loads(json_str)

                # Support both single and multiple tool calls
                if isinstance(data, dict):
                    data = [data]

                for item in data:
                    if 'tool' in item or 'name' in item:
                        tool_call = ToolCall(
                            name=item.get('tool', item.get('name')),
                            arguments=item.get('arguments', item.get('args', {})),
                            raw_text=match.group(0),
                            start_pos=match.start(),
                            end_pos=match.end()
                        )
                        tool_calls.append(tool_call)
            except json.JSONDecodeError:
                continue

        return tool_calls

    @staticmethod
    def parse_xml_tool_calls(text: str) -> List[ToolCall]:
        """
        Parse tool calls in XML format (Anthropic-style).

        Expected format:
        <function_calls>
        <invoke>
        <tool_name>tool_name</tool_name>
        <parameters>
        <param1>value1</param1>
        <param2>value2</param2>
        </parameters>
        </invoke>
        </function_calls>

        Args:
            text: Generated text potentially containing tool calls

        Returns:
            List of parsed ToolCall objects
        """
        tool_calls = []

        # Simple XML parsing for tool calls
        invoke_pattern = r'<invoke>(.*?)</invoke>'
        matches = re.finditer(invoke_pattern, text, re.DOTALL)

        for match in matches:
            invoke_content = match.group(1)

            # Extract tool name
            name_match = re.search(r'<tool_name>(.*?)</tool_name>', invoke_content)
            if not name_match:
                continue

            tool_name = name_match.group(1).strip()

            # Extract parameters
            params_match = re.search(
                r'<parameters>(.*?)</parameters>',
                invoke_content,
                re.DOTALL
            )

            arguments = {}
            if params_match:
                params_content = params_match.group(1)
                # Find all parameter tags
                param_matches = re.finditer(
                    r'<(\w+)>(.*?)</\1>',
                    params_content,
                    re.DOTALL
                )
                for pm in param_matches:
                    param_name = pm.group(1)
                    param_value = pm.group(2).strip()
                    arguments[param_name] = param_value

            tool_call = ToolCall(
                name=tool_name,
                arguments=arguments,
                raw_text=match.group(0),
                start_pos=match.start(),
                end_pos=match.end()
            )
            tool_calls.append(tool_call)

        return tool_calls

    @staticmethod
    def parse_function_style_calls(text: str) -> List[ToolCall]:
        """
        Parse tool calls in function call format.

        Expected format:
        tool_name(arg1="value1", arg2="value2")

        Args:
            text: Generated text potentially containing tool calls

        Returns:
            List of parsed ToolCall objects
        """
        tool_calls = []

        # Pattern for function-style calls
        pattern = r'(\w+)\((.*?)\)'
        matches = re.finditer(pattern, text)

        for match in matches:
            tool_name = match.group(1)
            args_str = match.group(2)

            # Parse arguments (simple key=value parsing)
            arguments = {}
            if args_str.strip():
                # Split by comma (naive, doesn't handle nested structures)
                arg_pairs = args_str.split(',')
                for pair in arg_pairs:
                    if '=' in pair:
                        key, value = pair.split('=', 1)
                        key = key.strip()
                        value = value.strip().strip('"\'')
                        arguments[key] = value

            tool_call = ToolCall(
                name=tool_name,
                arguments=arguments,
                raw_text=match.group(0),
                start_pos=match.start(),
                end_pos=match.end()
            )
            tool_calls.append(tool_call)

        return tool_calls


# ============================================================================
# Main Inference Module
# ============================================================================

class InferenceModule:
    """
    Main inference module with support for multiple backends.

    Handles all generation tasks including standard text generation,
    logit/logprob computation, and tool-calling capabilities.
    """

    def __init__(
        self,
        model_name: str,
        backend: str = "huggingface",
        cache_size: int = 1000,
        tool_parser: str = "json",
        **backend_kwargs
    ):
        """
        Initialize InferenceModule.

        Args:
            model_name: HuggingFace model identifier
            backend: Backend to use ("huggingface" or "vllm")
            cache_size: Size of generation cache (0 to disable)
            tool_parser: Tool call parsing format ("json", "xml", "function")
            **backend_kwargs: Additional arguments for backend initialization
        """
        self.model_name = model_name
        self.backend_name = backend.lower()
        self.cache_size = cache_size

        # Initialize backend
        if self.backend_name == "huggingface":
            self.backend = HuggingFaceBackend(model_name, **backend_kwargs)
        elif self.backend_name == "vllm":
            self.backend = VLLMBackend(model_name, **backend_kwargs)
        else:
            raise ValueError(
                f"Unknown backend: {backend}. Choose 'huggingface' or 'vllm'"
            )

        # Setup generation cache
        self.generation_cache: Dict[str, GenerationResult] = {}

        # Setup tool parser
        self.tool_parser_type = tool_parser
        self.tool_parser = ToolCallParser()

        print(f"InferenceModule initialized with {self.backend_name} backend")

    def _get_cache_key(self, prompts: Union[str, List[str]], **kwargs) -> str:
        """Generate cache key for prompts and generation params."""
        # Convert prompts to string
        if isinstance(prompts, list):
            prompt_str = "|||".join(prompts)
        else:
            prompt_str = prompts

        # Add kwargs to key
        kwargs_str = json.dumps(kwargs, sort_keys=True)
        full_key = f"{prompt_str}:::{kwargs_str}"

        # Hash for efficiency
        return hashlib.md5(full_key.encode()).hexdigest()

    def generate(
        self,
        prompts: Union[str, List[str]],
        use_cache: bool = True,
        **kwargs
    ) -> GenerationResult:
        """
        Core generation method.

        Args:
            prompts: Single prompt or list of prompts
            use_cache: Whether to use generation cache
            **kwargs: Generation parameters (temperature, max_new_tokens, etc.)

        Returns:
            GenerationResult with generated texts and optional metadata

        Example:
            >>> inference = InferenceModule("gpt2")
            >>> result = inference.generate("Hello, world!", max_new_tokens=50)
            >>> print(result.texts[0])
        """
        # Check cache
        if use_cache and self.cache_size > 0:
            cache_key = self._get_cache_key(prompts, **kwargs)
            if cache_key in self.generation_cache:
                return self.generation_cache[cache_key]

        # Generate
        result = self.backend.generate(prompts, **kwargs)

        # Update cache
        if use_cache and self.cache_size > 0:
            cache_key = self._get_cache_key(prompts, **kwargs)
            self.generation_cache[cache_key] = result

            # Evict oldest if cache is full
            if len(self.generation_cache) > self.cache_size:
                oldest_key = next(iter(self.generation_cache))
                del self.generation_cache[oldest_key]

        return result

    def get_logits(
        self,
        prompts: Union[str, List[str]],
        responses: Union[str, List[str]]
    ) -> torch.Tensor:
        """
        Get logits for prompt-response pairs (for KL calculation).

        Args:
            prompts: Single prompt or list of prompts
            responses: Single response or list of responses

        Returns:
            Tensor of shape [batch_size, seq_len, vocab_size]

        Example:
            >>> prompts = ["What is 2+2?", "What is 3+3?"]
            >>> responses = [" 4", " 6"]
            >>> logits = inference.get_logits(prompts, responses)
            >>> print(logits.shape)  # [2, seq_len, vocab_size]
        """
        return self.backend.get_logits(prompts, responses)

    def get_logprobs(
        self,
        prompts: Union[str, List[str]],
        responses: Union[str, List[str]]
    ) -> torch.Tensor:
        """
        Get log probabilities for prompt-response pairs (for policy gradient).

        Args:
            prompts: Single prompt or list of prompts
            responses: Single response or list of responses

        Returns:
            Tensor of shape [batch_size, seq_len] with log probabilities

        Example:
            >>> prompts = ["Complete: The sky is"]
            >>> responses = [" blue"]
            >>> logprobs = inference.get_logprobs(prompts, responses)
            >>> print(logprobs.mean().item())  # Average log probability
        """
        return self.backend.get_logprobs(prompts, responses)

    def batch_generate_with_tools(
        self,
        prompts: Union[str, List[str]],
        available_tools: Optional[List[Dict[str, Any]]] = None,
        max_new_tokens: int = 512,
        **kwargs
    ) -> Tuple[GenerationResult, List[List[ToolCall]]]:
        """
        Generate with tool calling support.

        Args:
            prompts: Single prompt or list of prompts
            available_tools: List of tool definitions (optional)
            max_new_tokens: Maximum tokens to generate
            **kwargs: Additional generation parameters

        Returns:
            Tuple of (GenerationResult, List of ToolCalls per prompt)

        Example:
            >>> tools = [{"name": "search", "description": "Search the web"}]
            >>> result, tool_calls = inference.batch_generate_with_tools(
            ...     "Search for Python tutorials",
            ...     available_tools=tools
            ... )
            >>> print(f"Generated: {result.texts[0]}")
            >>> print(f"Tool calls: {tool_calls[0]}")
        """
        # Optionally augment prompts with tool information
        if available_tools:
            augmented_prompts = self._augment_prompts_with_tools(
                prompts,
                available_tools
            )
        else:
            augmented_prompts = prompts

        # Generate responses
        result = self.generate(
            augmented_prompts,
            max_new_tokens=max_new_tokens,
            use_cache=False,  # Don't cache tool calls
            **kwargs
        )

        # Parse tool calls from responses
        all_tool_calls = []
        for text in result.texts:
            tool_calls = self.parse_tool_calls(text)
            all_tool_calls.append(tool_calls)

        return result, all_tool_calls

    def parse_tool_calls(self, text: str) -> List[ToolCall]:
        """
        Parse tool calls from generated text.

        Args:
            text: Generated text potentially containing tool calls

        Returns:
            List of parsed ToolCall objects
        """
        if self.tool_parser_type == "json":
            return self.tool_parser.parse_json_tool_calls(text)
        elif self.tool_parser_type == "xml":
            return self.tool_parser.parse_xml_tool_calls(text)
        elif self.tool_parser_type == "function":
            return self.tool_parser.parse_function_style_calls(text)
        else:
            # Try all parsers
            calls = self.tool_parser.parse_json_tool_calls(text)
            if not calls:
                calls = self.tool_parser.parse_xml_tool_calls(text)
            if not calls:
                calls = self.tool_parser.parse_function_style_calls(text)
            return calls

    def _augment_prompts_with_tools(
        self,
        prompts: Union[str, List[str]],
        tools: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Augment prompts with tool information.

        Args:
            prompts: Original prompts
            tools: Available tools with descriptions

        Returns:
            Augmented prompts with tool information
        """
        # Ensure prompts is a list
        if isinstance(prompts, str):
            prompts = [prompts]

        # Format tools
        tools_description = "\n\nAvailable tools:\n"
        for tool in tools:
            name = tool.get("name", "unknown")
            desc = tool.get("description", "No description")
            params = tool.get("parameters", {})

            tools_description += f"\n- {name}: {desc}\n"
            if params:
                tools_description += f"  Parameters: {json.dumps(params)}\n"

        tools_description += "\nTo use a tool, respond with JSON:\n"
        tools_description += '```json\n{"tool": "tool_name", "arguments": {...}}\n```\n'

        # Augment each prompt
        augmented = [prompt + tools_description for prompt in prompts]

        return augmented

    def clear_cache(self):
        """Clear the generation cache."""
        self.generation_cache.clear()
        print("Generation cache cleared")

    def __repr__(self) -> str:
        return (
            f"InferenceModule(model={self.model_name}, "
            f"backend={self.backend_name}, "
            f"cache_size={self.cache_size})"
        )


# ============================================================================
# Utility Functions
# ============================================================================

def create_inference_module(
    model_name: str,
    backend: str = "auto",
    **kwargs
) -> InferenceModule:
    """
    Factory function to create InferenceModule with automatic backend selection.

    Args:
        model_name: HuggingFace model identifier
        backend: Backend to use ("auto", "huggingface", "vllm")
        **kwargs: Additional arguments for InferenceModule

    Returns:
        Configured InferenceModule instance

    Example:
        >>> # Automatically select best backend
        >>> inference = create_inference_module("meta-llama/Llama-2-7b-hf")
        >>>
        >>> # Force specific backend
        >>> inference = create_inference_module(
        ...     "gpt2",
        ...     backend="huggingface",
        ...     dtype="float16"
        ... )
    """
    if backend == "auto":
        # Prefer vLLM if available, fallback to HuggingFace
        if VLLM_AVAILABLE:
            backend = "vllm"
            print("Auto-selected vLLM backend")
        else:
            backend = "huggingface"
            print("Auto-selected HuggingFace backend (vLLM not available)")

    return InferenceModule(model_name, backend=backend, **kwargs)


if __name__ == "__main__":
    # Example usage
    print("=" * 80)
    print("InferenceModule Example Usage")
    print("=" * 80)

    # Example 1: Basic generation with HuggingFace backend
    print("\n1. Basic Generation (HuggingFace)")
    print("-" * 40)

    try:
        inference = InferenceModule(
            "gpt2",
            backend="huggingface",
            device="cpu"  # Use CPU for example
        )

        result = inference.generate(
            "The future of AI is",
            max_new_tokens=30,
            temperature=0.8
        )

        print(f"Generated: {result.texts[0]}")
    except Exception as e:
        print(f"Error: {e}")

    # Example 2: Batch generation
    print("\n2. Batch Generation")
    print("-" * 40)

    try:
        prompts = [
            "Explain quantum computing:",
            "What is machine learning?"
        ]

        result = inference.generate(
            prompts,
            max_new_tokens=50,
            do_sample=True
        )

        for i, text in enumerate(result.texts):
            print(f"Prompt {i+1}: {text[:100]}...")
    except Exception as e:
        print(f"Error: {e}")

    # Example 3: Tool calling
    print("\n3. Tool Calling")
    print("-" * 40)

    try:
        tools = [
            {
                "name": "search",
                "description": "Search the web",
                "parameters": {"query": "string"}
            }
        ]

        prompt = "I need to search for the latest AI news."
        result, tool_calls = inference.batch_generate_with_tools(
            prompt,
            available_tools=tools,
            max_new_tokens=100
        )

        print(f"Generated: {result.texts[0][:200]}...")
        print(f"Tool calls found: {len(tool_calls[0])}")
        for tc in tool_calls[0]:
            print(f"  - {tc.name}: {tc.arguments}")
    except Exception as e:
        print(f"Error: {e}")

    print("\n" + "=" * 80)
    print("Examples complete!")
    print("=" * 80)
