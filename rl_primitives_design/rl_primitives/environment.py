"""
Environment Module for RL Primitives

This module provides the core environment abstractions for RL training of LLMs.
Environments handle pure interaction logic - state transitions, prompting, and termination.
Reward computation is handled separately by RewardComputer.

Design Philosophy:
- Environments define "how to interact" with the model
- Environments do NOT compute rewards (that's RewardComputer's job)
- State transitions are explicit and trackable
- Prompting logic is customizable per environment
- Terminal conditions are environment-specific

Author: RL Primitives Team
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from copy import deepcopy
import json


@dataclass
class State:
    """
    Base state class for trajectory tracking.

    This dataclass holds all information about a trajectory's current state.
    It's designed to be:
    - Immutable (use copy() to create new states)
    - Serializable (can convert to/from dict)
    - Extensible (subclass for environment-specific state)

    Attributes:
        prompt: Initial prompt or current prompt for this step
        response: Last generated response from the model
        step_count: Number of steps taken in this trajectory
        done: Whether this trajectory is complete
        history: List of all (prompt, response) pairs in this trajectory
        metadata: Environment-specific metadata (tools, budget, etc.)
        trajectory_id: Unique identifier for this trajectory (useful for tracking)
    """
    prompt: str
    response: str = ""
    step_count: int = 0
    done: bool = False
    history: List[Dict[str, str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    trajectory_id: Optional[str] = None

    def copy(self) -> "State":
        """Create a deep copy of this state."""
        return deepcopy(self)

    def add_turn(self, prompt: str, response: str) -> None:
        """Add a turn to the history."""
        self.history.append({"prompt": prompt, "response": response})

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization."""
        return {
            "prompt": self.prompt,
            "response": self.response,
            "step_count": self.step_count,
            "done": self.done,
            "history": self.history,
            "metadata": self.metadata,
            "trajectory_id": self.trajectory_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "State":
        """Create state from dictionary."""
        return cls(**data)

    def __repr__(self) -> str:
        return (
            f"State(step={self.step_count}, done={self.done}, "
            f"prompt_len={len(self.prompt)}, response_len={len(self.response)})"
        )


@dataclass
class SVRLState(State):
    """
    Extended state for Self-Verification RL environments.

    Adds tracking for verification budget, tool calls, and verification history.

    Additional Attributes:
        initial_budget: Starting budget for verification
        current_budget: Remaining budget
        verification_history: List of all verification attempts
        tools_available: List of available verification tools
        verification_count: Number of verifications performed
    """
    initial_budget: float = 10.0
    current_budget: float = 10.0
    verification_history: List[Dict[str, Any]] = field(default_factory=list)
    tools_available: List[str] = field(default_factory=list)
    verification_count: int = 0

    def can_afford_verification(self, cost: float) -> bool:
        """Check if we have budget for a verification."""
        return self.current_budget >= cost

    def spend_budget(self, cost: float) -> None:
        """Spend budget on a verification."""
        self.current_budget -= cost
        self.verification_count += 1

    def add_verification(self, tool: str, result: Any, cost: float) -> None:
        """Record a verification attempt."""
        self.verification_history.append({
            "step": self.step_count,
            "tool": tool,
            "result": result,
            "cost": cost,
            "budget_remaining": self.current_budget
        })


class BaseEnvironment(ABC):
    """
    Abstract base class for RL environments.

    Environments define the interaction logic between prompts and model responses.
    They handle:
    - State initialization (reset)
    - State transitions (step)
    - Termination conditions (is_terminal)
    - Prompt construction (build_prompt)
    - State updates from responses (update_state)

    Environments do NOT compute rewards - that's handled by RewardComputer.

    Usage:
        class MyEnv(BaseEnvironment):
            def reset(self) -> State:
                return State(prompt="Solve this problem: ...")

            def step(self, state: State) -> State:
                # Generate response, update state
                pass

            def is_terminal(self, state: State) -> bool:
                return state.step_count >= 10
    """

    def __init__(self, inference_module=None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize environment.

        Args:
            inference_module: Module for generating responses (can be None for testing)
            config: Environment-specific configuration
        """
        self.inference = inference_module
        self.config = config or {}

    @abstractmethod
    def reset(self) -> State:
        """
        Initialize a new trajectory.

        Returns:
            Initial state for a new trajectory
        """
        pass

    @abstractmethod
    def step(self, state: State) -> State:
        """
        Execute one step of interaction.

        This should:
        1. Build prompt from current state
        2. Generate response (via inference module or custom logic)
        3. Update state with new information
        4. Increment step count

        Args:
            state: Current state

        Returns:
            New state after this step
        """
        pass

    @abstractmethod
    def is_terminal(self, state: State) -> bool:
        """
        Check if trajectory should terminate.

        Common termination conditions:
        - Max steps reached
        - Budget exhausted
        - Special token in response ("DONE", "COMPLETE", etc.)
        - Task-specific completion criteria

        Args:
            state: Current state

        Returns:
            True if trajectory is complete, False otherwise
        """
        pass

    @abstractmethod
    def build_prompt(self, state: State) -> str:
        """
        Construct prompt for current state.

        This defines how the environment communicates with the model.
        Can include:
        - Original task
        - Conversation history
        - Available tools/actions
        - Remaining budget
        - Instructions

        Args:
            state: Current state

        Returns:
            Prompt string for model generation
        """
        pass

    @abstractmethod
    def update_state(self, state: State, response: str) -> State:
        """
        Update state based on model response.

        This is where environment-specific logic happens:
        - Parse special commands from response
        - Execute tool calls
        - Update budget
        - Check for completion signals

        Args:
            state: Current state
            response: Model's generated response

        Returns:
            Updated state
        """
        pass

    def batch_step(self, states: List[State]) -> List[State]:
        """
        Step multiple states at once (for efficient batched inference).

        Default implementation steps each state individually.
        Override for custom batching logic.

        Args:
            states: List of current states

        Returns:
            List of new states
        """
        return [self.step(state) for state in states]


class SimpleTextEnvironment(BaseEnvironment):
    """
    Simple single-turn text generation environment.

    This environment:
    - Starts with a prompt
    - Generates one response
    - Terminates after one step

    Useful for:
    - Basic supervised fine-tuning style tasks
    - Single-turn QA
    - Simple instruction following
    - Testing and debugging

    Example:
        env = SimpleTextEnvironment(
            prompts=["What is 2+2?", "Explain photosynthesis"],
            max_steps=1
        )

        state = env.reset()
        state = env.step(state)
        # state now contains the model's response
    """

    def __init__(
        self,
        prompts: Optional[List[str]] = None,
        inference_module=None,
        max_steps: int = 1,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize simple text environment.

        Args:
            prompts: List of prompts to sample from (if None, must be provided in reset)
            inference_module: Module for generating responses
            max_steps: Maximum steps before termination (default: 1)
            config: Additional configuration
        """
        super().__init__(inference_module, config)
        self.prompts = prompts or []
        self.max_steps = max_steps
        self.current_prompt_idx = 0

    def reset(self, prompt: Optional[str] = None) -> State:
        """
        Initialize with a new prompt.

        Args:
            prompt: Specific prompt to use (if None, cycles through self.prompts)

        Returns:
            Initial state with the prompt
        """
        if prompt is None:
            if not self.prompts:
                raise ValueError("No prompts available. Provide prompts in __init__ or reset()")
            prompt = self.prompts[self.current_prompt_idx % len(self.prompts)]
            self.current_prompt_idx += 1

        return State(
            prompt=prompt,
            step_count=0,
            done=False,
            metadata={"max_steps": self.max_steps}
        )

    def step(self, state: State) -> State:
        """
        Generate response for the prompt.

        Args:
            state: Current state

        Returns:
            State with generated response
        """
        new_state = state.copy()

        # Build prompt
        prompt = self.build_prompt(new_state)

        # Generate response (if inference module available)
        if self.inference is not None:
            response = self.inference.generate([prompt])[0]
        else:
            # Placeholder for testing without inference
            response = f"[Generated response for: {prompt[:50]}...]"

        # Update state
        new_state = self.update_state(new_state, response)
        new_state.step_count += 1

        # Check if terminal
        if self.is_terminal(new_state):
            new_state.done = True

        return new_state

    def is_terminal(self, state: State) -> bool:
        """
        Terminate after max_steps or if response contains completion signal.

        Args:
            state: Current state

        Returns:
            True if should terminate
        """
        if state.step_count >= self.max_steps:
            return True

        # Check for completion signals in response
        completion_signals = self.config.get("completion_signals", ["[DONE]", "[COMPLETE]"])
        if any(signal in state.response for signal in completion_signals):
            return True

        return False

    def build_prompt(self, state: State) -> str:
        """
        Simple prompt construction - just return the original prompt.

        Override this for more complex prompting (few-shot, etc.)

        Args:
            state: Current state

        Returns:
            Prompt string
        """
        return state.prompt

    def update_state(self, state: State, response: str) -> State:
        """
        Update state with the response.

        Args:
            state: Current state
            response: Generated response

        Returns:
            Updated state
        """
        new_state = state.copy()
        new_state.response = response
        new_state.add_turn(state.prompt, response)
        return new_state


class SVRLEnvironment(BaseEnvironment):
    """
    Self-Verification RL Environment with budget and tool calls.

    This environment allows the model to:
    - Use verification tools (at a cost)
    - Manage a budget for verifications
    - Decide when to verify vs. when to trust its response
    - Learn to balance accuracy vs. efficiency

    The model can:
    1. Generate a response
    2. Optionally call verification tools
    3. Update its response based on verification
    4. Decide when to submit final answer

    Useful for:
    - Teaching models to use tools judiciously
    - Learning when verification is worth the cost
    - Multi-step reasoning with verification
    - Self-correction and iterative refinement

    Example:
        tools = {
            "calculator": {"cost": 1.0, "fn": calculator_fn},
            "web_search": {"cost": 2.0, "fn": search_fn}
        }

        env = SVRLEnvironment(
            task_prompts=["Calculate 1234 * 5678"],
            verification_tools=tools,
            initial_budget=10.0
        )

        state = env.reset()
        while not env.is_terminal(state):
            state = env.step(state)
    """

    def __init__(
        self,
        task_prompts: Optional[List[str]] = None,
        verification_tools: Optional[Dict[str, Dict[str, Any]]] = None,
        initial_budget: float = 10.0,
        inference_module=None,
        max_steps: int = 50,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize SVRL environment.

        Args:
            task_prompts: List of tasks to sample from
            verification_tools: Dict mapping tool names to {cost, fn} dicts
            initial_budget: Starting budget for verifications
            inference_module: Module for generating responses
            max_steps: Maximum steps before forced termination
            config: Additional configuration
        """
        super().__init__(inference_module, config)
        self.task_prompts = task_prompts or []
        self.verification_tools = verification_tools or self._default_tools()
        self.initial_budget = initial_budget
        self.max_steps = max_steps
        self.current_task_idx = 0

    def _default_tools(self) -> Dict[str, Dict[str, Any]]:
        """
        Default verification tools (placeholders).

        In production, replace with actual tool implementations.
        """
        return {
            "verify_calculation": {
                "cost": 1.0,
                "description": "Verify a mathematical calculation",
                "fn": lambda x: {"verified": True, "confidence": 0.95}
            },
            "check_facts": {
                "cost": 2.0,
                "description": "Check factual accuracy of a statement",
                "fn": lambda x: {"verified": True, "sources": []}
            },
            "run_code": {
                "cost": 1.5,
                "description": "Execute code and return output",
                "fn": lambda x: {"output": "Success", "error": None}
            }
        }

    def reset(self, task_prompt: Optional[str] = None) -> SVRLState:
        """
        Initialize a new SVRL task.

        Args:
            task_prompt: Specific task (if None, cycles through task_prompts)

        Returns:
            Initial SVRL state
        """
        if task_prompt is None:
            if not self.task_prompts:
                raise ValueError("No task prompts available")
            task_prompt = self.task_prompts[self.current_task_idx % len(self.task_prompts)]
            self.current_task_idx += 1

        return SVRLState(
            prompt=task_prompt,
            step_count=0,
            done=False,
            initial_budget=self.initial_budget,
            current_budget=self.initial_budget,
            tools_available=list(self.verification_tools.keys()),
            metadata={
                "max_steps": self.max_steps,
                "task": task_prompt
            }
        )

    def step(self, state: SVRLState) -> SVRLState:
        """
        Execute one SVRL step.

        Process:
        1. Build prompt with available tools and budget
        2. Generate response
        3. Parse for tool calls
        4. Execute tool calls (if affordable)
        5. Update state

        Args:
            state: Current SVRL state

        Returns:
            Updated state
        """
        new_state = state.copy()

        # Build prompt
        prompt = self.build_prompt(new_state)

        # Generate response
        if self.inference is not None:
            response = self.inference.generate([prompt])[0]
        else:
            # Placeholder for testing
            response = f"[Response with potential tool calls]\nFINAL ANSWER: {state.prompt[:20]}"

        # Update state with response
        new_state = self.update_state(new_state, response)
        new_state.step_count += 1

        # Check terminal
        if self.is_terminal(new_state):
            new_state.done = True

        return new_state

    def is_terminal(self, state: SVRLState) -> bool:
        """
        Check termination conditions.

        Terminates if:
        - Budget exhausted
        - Max steps reached
        - Response contains completion signal

        Args:
            state: Current state

        Returns:
            True if should terminate
        """
        # Budget exhausted
        if state.current_budget <= 0:
            return True

        # Max steps
        if state.step_count >= self.max_steps:
            return True

        # Completion signals
        completion_signals = ["FINAL ANSWER:", "[DONE]", "[COMPLETE]"]
        if any(signal in state.response for signal in completion_signals):
            return True

        return False

    def build_prompt(self, state: SVRLState) -> str:
        """
        Build prompt with task, tools, budget, and history.

        Format:
        ```
        Task: {original task}

        Available Tools:
        - tool1 (cost: X): description
        - tool2 (cost: Y): description

        Current Budget: {budget}
        Verification History: {history}

        Instructions:
        You can use tools by writing: USE_TOOL[tool_name](args)
        When ready to submit, write: FINAL ANSWER: {your answer}

        Your response:
        ```

        Args:
            state: Current state

        Returns:
            Formatted prompt
        """
        # Build tools section
        tools_section = "Available Tools:\n"
        for tool_name, tool_info in self.verification_tools.items():
            cost = tool_info['cost']
            desc = tool_info.get('description', 'No description')
            tools_section += f"- {tool_name} (cost: {cost}): {desc}\n"

        # Build verification history
        history_section = ""
        if state.verification_history:
            history_section = "\nVerification History:\n"
            for i, verif in enumerate(state.verification_history[-3:], 1):  # Last 3
                history_section += f"{i}. {verif['tool']}: {verif['result']} (cost: {verif['cost']})\n"

        # Build full prompt
        prompt = f"""Task: {state.metadata['task']}

{tools_section}
Current Budget: {state.current_budget:.1f} / {state.initial_budget:.1f}
Verifications Used: {state.verification_count}
{history_section}

Instructions:
You can use tools by writing: USE_TOOL[tool_name](arguments)
When ready to submit your final answer, write: FINAL ANSWER: [your answer]

Think carefully about whether verification is worth the cost.

Your response:
"""
        return prompt

    def update_state(self, state: SVRLState, response: str) -> SVRLState:
        """
        Update state based on response.

        Parses response for:
        - Tool calls (USE_TOOL[name](args))
        - Final answer (FINAL ANSWER: ...)

        Executes tool calls if budget allows.

        Args:
            state: Current state
            response: Generated response

        Returns:
            Updated state
        """
        new_state = state.copy()
        new_state.response = response

        # Parse and execute tool calls
        tool_calls = self._parse_tool_calls(response)

        for tool_call in tool_calls:
            tool_name = tool_call['tool']
            tool_args = tool_call['args']

            if tool_name not in self.verification_tools:
                continue

            tool_info = self.verification_tools[tool_name]
            cost = tool_info['cost']

            # Check if can afford
            if new_state.can_afford_verification(cost):
                # Execute tool
                tool_fn = tool_info['fn']
                result = tool_fn(tool_args)

                # Update state
                new_state.spend_budget(cost)
                new_state.add_verification(tool_name, result, cost)

        # Add to history
        new_state.add_turn(state.prompt, response)

        return new_state

    def _parse_tool_calls(self, response: str) -> List[Dict[str, Any]]:
        """
        Parse tool calls from response.

        Expected format: USE_TOOL[tool_name](arguments)

        Args:
            response: Generated response

        Returns:
            List of tool calls with {tool, args}
        """
        import re

        tool_calls = []
        pattern = r'USE_TOOL\[(\w+)\]\((.*?)\)'

        matches = re.finditer(pattern, response)
        for match in matches:
            tool_name = match.group(1)
            args = match.group(2)
            tool_calls.append({
                'tool': tool_name,
                'args': args
            })

        return tool_calls

    def execute_tool(self, tool_name: str, args: str) -> Any:
        """
        Execute a verification tool.

        Args:
            tool_name: Name of tool to execute
            args: Arguments for the tool

        Returns:
            Tool execution result
        """
        if tool_name not in self.verification_tools:
            return {"error": f"Tool {tool_name} not found"}

        tool_fn = self.verification_tools[tool_name]['fn']
        try:
            result = tool_fn(args)
            return result
        except Exception as e:
            return {"error": str(e)}


# ============================================================================
# Example: Multi-turn Conversation Environment
# ============================================================================

class ConversationEnvironment(BaseEnvironment):
    """
    Multi-turn conversation environment.

    Useful for:
    - Dialogue tasks
    - Multi-turn reasoning
    - Interactive problem solving
    - Customer service simulations

    Example:
        env = ConversationEnvironment(
            initial_prompts=["Help me plan a vacation"],
            max_turns=10
        )
    """

    def __init__(
        self,
        initial_prompts: Optional[List[str]] = None,
        inference_module=None,
        max_turns: int = 10,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(inference_module, config)
        self.initial_prompts = initial_prompts or []
        self.max_turns = max_turns
        self.current_prompt_idx = 0

    def reset(self, initial_prompt: Optional[str] = None) -> State:
        if initial_prompt is None:
            if not self.initial_prompts:
                raise ValueError("No initial prompts available")
            initial_prompt = self.initial_prompts[self.current_prompt_idx % len(self.initial_prompts)]
            self.current_prompt_idx += 1

        return State(
            prompt=initial_prompt,
            step_count=0,
            done=False,
            history=[],
            metadata={"max_turns": self.max_turns}
        )

    def step(self, state: State) -> State:
        new_state = state.copy()

        # Build conversation prompt
        prompt = self.build_prompt(new_state)

        # Generate response
        if self.inference is not None:
            response = self.inference.generate([prompt])[0]
        else:
            response = f"[Turn {state.step_count + 1} response]"

        # Update state
        new_state = self.update_state(new_state, response)
        new_state.step_count += 1

        if self.is_terminal(new_state):
            new_state.done = True

        return new_state

    def is_terminal(self, state: State) -> bool:
        if state.step_count >= self.max_turns:
            return True

        # Check for conversation end signals
        end_signals = ["goodbye", "thank you, that's all", "[END]"]
        if any(signal.lower() in state.response.lower() for signal in end_signals):
            return True

        return False

    def build_prompt(self, state: State) -> str:
        """Build conversation history into prompt."""
        if not state.history:
            return f"User: {state.prompt}\nAssistant:"

        # Build full conversation
        conversation = ""
        for turn in state.history:
            conversation += f"User: {turn['prompt']}\nAssistant: {turn['response']}\n"

        conversation += f"User: {state.prompt}\nAssistant:"
        return conversation

    def update_state(self, state: State, response: str) -> State:
        new_state = state.copy()
        new_state.response = response
        new_state.add_turn(state.prompt, response)

        # Generate next user turn (in production, this would come from dataset)
        # For now, placeholder
        next_user_prompt = "[User's next question...]"
        new_state.prompt = next_user_prompt

        return new_state


# ============================================================================
# Utility Functions
# ============================================================================

def create_environment(env_type: str, **kwargs) -> BaseEnvironment:
    """
    Factory function for creating environments.

    Args:
        env_type: Type of environment ("simple", "svrl", "conversation")
        **kwargs: Arguments to pass to environment constructor

    Returns:
        Environment instance

    Example:
        env = create_environment(
            "svrl",
            task_prompts=["Calculate 123 * 456"],
            initial_budget=10.0
        )
    """
    env_map = {
        "simple": SimpleTextEnvironment,
        "svrl": SVRLEnvironment,
        "conversation": ConversationEnvironment,
    }

    if env_type not in env_map:
        raise ValueError(f"Unknown environment type: {env_type}. Choose from {list(env_map.keys())}")

    return env_map[env_type](**kwargs)


def demo_environment():
    """
    Demo script showing how to use environments.

    Run this to see environments in action.
    """
    print("=" * 80)
    print("Environment Demo")
    print("=" * 80)

    # Demo 1: Simple Text Environment
    print("\n1. Simple Text Environment")
    print("-" * 40)
    simple_env = SimpleTextEnvironment(
        prompts=["What is 2+2?", "Explain machine learning"],
        max_steps=1
    )

    state = simple_env.reset()
    print(f"Initial state: {state}")
    print(f"Prompt: {state.prompt}")

    state = simple_env.step(state)
    print(f"After step: {state}")
    print(f"Response: {state.response}")
    print(f"Terminal: {simple_env.is_terminal(state)}")

    # Demo 2: SVRL Environment
    print("\n2. SVRL Environment")
    print("-" * 40)
    svrl_env = SVRLEnvironment(
        task_prompts=["Calculate 1234 * 5678"],
        initial_budget=10.0,
        max_steps=5
    )

    state = svrl_env.reset()
    print(f"Initial budget: {state.current_budget}")
    print(f"Available tools: {state.tools_available}")

    for i in range(3):
        state = svrl_env.step(state)
        print(f"\nStep {i+1}:")
        print(f"  Budget: {state.current_budget:.1f}")
        print(f"  Verifications: {state.verification_count}")
        print(f"  Terminal: {svrl_env.is_terminal(state)}")

        if state.done:
            break

    # Demo 3: Factory
    print("\n3. Factory Function")
    print("-" * 40)
    env = create_environment(
        "simple",
        prompts=["Test prompt"],
        max_steps=1
    )
    print(f"Created environment: {type(env).__name__}")

    print("\n" + "=" * 80)
    print("Demo complete!")
    print("=" * 80)


if __name__ == "__main__":
    # Run demo if executed directly
    demo_environment()
