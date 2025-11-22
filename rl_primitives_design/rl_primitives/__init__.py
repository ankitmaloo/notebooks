"""
RL Primitives - Modular components for RL training of LLMs

This package provides reusable, production-ready components for reinforcement learning:
- InferenceModule: Multi-backend inference with generation, logits, and tool calling
- BackpropModule: Gradient computation and model updates
- Environment: Interaction logic and state transitions
- RolloutManager: Parallel trajectory collection
- RewardComputer: Reward computation strategies (absolute, relative, Pareto)
- Algorithms: Complete training loops (PPO, GRPO, REINFORCE)

Based on the architecture in v2.md
"""

# Core modules
from .backprop import BackpropModule, BackpropConfig
from .inference import (
    InferenceModule,
    InferenceBackend,
    HuggingFaceBackend,
    VLLMBackend,
    GenerationResult,
    ToolCall,
    ToolCallParser,
    create_inference_module,
)
from .environment import (
    BaseEnvironment,
    State,
    SVRLState,
    SimpleTextEnvironment,
    SVRLEnvironment,
    ConversationEnvironment,
    create_environment,
)
from .rollout_manager import (
    RolloutManager,
    RolloutConfig,
    RolloutStats,
    BatchStats,
    create_rollout_manager,
)
from .reward_computer import (
    RewardComputer,
    SVRLRewardComputer,
    ParetoRewardComputer,
    create_reward_computer,
)
from .algorithms import (
    RLAlgorithm,
    PPOAlgorithm,
    GRPOAlgorithm,
    REINFORCEAlgorithm,
    AlgorithmConfig,
    TrainingMetrics,
    create_algorithm,
    load_algorithm_from_checkpoint,
)

__all__ = [
    # Backprop
    'BackpropModule',
    'BackpropConfig',
    # Inference
    'InferenceModule',
    'InferenceBackend',
    'HuggingFaceBackend',
    'VLLMBackend',
    'GenerationResult',
    'ToolCall',
    'ToolCallParser',
    'create_inference_module',
    # Environment
    'BaseEnvironment',
    'State',
    'SVRLState',
    'SimpleTextEnvironment',
    'SVRLEnvironment',
    'ConversationEnvironment',
    'create_environment',
    # Rollout Manager
    'RolloutManager',
    'RolloutConfig',
    'RolloutStats',
    'BatchStats',
    'create_rollout_manager',
    # Reward Computer
    'RewardComputer',
    'SVRLRewardComputer',
    'ParetoRewardComputer',
    'create_reward_computer',
    # Algorithms
    'RLAlgorithm',
    'PPOAlgorithm',
    'GRPOAlgorithm',
    'REINFORCEAlgorithm',
    'AlgorithmConfig',
    'TrainingMetrics',
    'create_algorithm',
    'load_algorithm_from_checkpoint',
]

__version__ = '0.1.0'
