# Subagent Task Decomposition for Dry Debugging

When performing dry debugging, use specialized subagents with clear priority order. Focus on catching wrong choices first, then correctness issues.

## Priority Order

**Tier 1: Critical (Must Fix)**
- Subagent 10: Instruction Compliance Auditor
- Subagent 11: Model/Framework Choice Validator (NEW)
- Subagent 1: Syntax & Import Checker

**Tier 2: Correctness Issues**
- Subagent 2: Training Loop Analyzer
- Subagent 3: Tensor Operations Auditor
- Subagent 4: Device Management Inspector

**Tier 3: Optimization & Best Practices**
- Subagent 5: Memory Management Validator
- Subagent 6: Logging & Monitoring Verifier
- Subagent 7: Model I/O Specialist
- Subagent 8: Dependency & Library Auditor
- Subagent 9: Data Pipeline Examiner

## Subagent 10: Instruction Compliance Auditor (HIGHEST PRIORITY)
**Responsibility**: Compare code against ALL user instructions
**Checks**:
- Model name matches specification
- Hyperparameters match (LR, batch size, epochs, etc.)
- Framework choice matches (PyTorch vs vLLM vs SGLang)
- Output paths/naming match
- Approach matches (LoRA vs full fine-tune, etc.)
- All user requirements implemented
- ALL assumptions explicitly listed

**Output**: Detailed comparison of requested vs implemented

**Critical**: This is the MOST important check. Wrong model/config wastes entire training run.

## Subagent 11: Model/Framework Choice Validator (NEW - CRITICAL)
**Responsibility**: Validate model and framework choices are appropriate
**Checks**:
- Is model appropriate for stated task?
  - Instruction following → Need instruct/chat variant
  - Code generation → Need code-specialized model
  - Base model when chat needed?
- Should they use vLLM instead of transformers for inference?
- Should they use SGLang for structured output?
- Is model size appropriate for stated hardware?
- Using community model when official version exists?

**Output**: Flag inappropriate model/framework choices with better alternatives

**Examples**:
- User wants chat → Flag if using base model instead of chat variant
- User doing inference → Suggest vLLM if using vanilla transformers
- User needs JSON output → Suggest SGLang if using manual parsing

## Subagent 1: Syntax & Import Checker
**Responsibility**: Identify syntax errors and import issues
**Checks**:
- Valid Python syntax
- All imports present for used libraries
- Proper import statements (no circular imports)
- Correct import paths

**Output**: List of syntax errors and missing imports with line numbers

## Subagent 2: Training Loop Analyzer  
**Responsibility**: Verify training loop correctness
**Checks**:
- Proper order: `zero_grad()` → forward → `backward()` → `step()`
- Gradient accumulation logic (if present)
- Learning rate scheduler placement
- Model in correct mode (`train()` vs `eval()`)

**Output**: Training loop issues with severity ratings

## Subagent 12: Overengineering Detector (NEW - IMPORTANT)
**Responsibility**: Catch unnecessary complexity
**Checks**:
- Using DDP/FSDP for single GPU?
- Custom training loop when Trainer API would work?
- Implementing features that already exist in frameworks?
- Overly complex data loading for simple tasks?
- Using advanced features without need (mixed precision on small models, etc.)
- More code than necessary?

**Output**: Flag overengineered solutions with simpler alternatives

**Examples**:
- 200-line custom training loop → "Use HF Trainer instead"
- DDP setup for 1 GPU → "Remove distributed code"
- Custom gradient accumulation → "Trainer has this built-in"

## Subagent 3: Tensor Operations Auditor
**Responsibility**: Check tensor operations and shapes
**Checks**:
- Tensor shape compatibility in operations
- Proper use of `view()`, `reshape()`, `unsqueeze()`, `squeeze()`
- Correct loss function input shapes
- Batch dimension handling

**Output**: Shape mismatch errors and tensor operation issues

## Subagent 4: Device Management Inspector
**Responsibility**: Verify device (CPU/GPU) handling
**Checks**:
- Model and data on same device
- Proper `.to(device)` calls
- CUDA availability checks
- Multi-GPU setup correctness (if applicable)

**Output**: Device mismatch errors and CUDA issues

## Subagent 5: Memory Management Validator
**Responsibility**: Check for memory leaks and efficiency
**Checks**:
- `torch.no_grad()` in validation loops
- Proper `.detach()` usage when accumulating losses
- Gradient checkpointing (if needed for large models)
- Memory-efficient data loading

**Output**: Memory leak warnings and efficiency suggestions

## Subagent 6: Logging & Monitoring Verifier
**Responsibility**: Ensure proper experiment tracking
**Checks**:
- wandb/tensorboard initialization
- Correct logging frequency and metrics
- All required config saved
- Proper experiment naming

**Output**: Missing logging setup and configuration issues

## Subagent 7: Model I/O Specialist
**Responsibility**: Verify model loading and saving
**Checks**:
- Correct model name matches user instructions
- Proper checkpoint saving (includes optimizer, scheduler state)
- HuggingFace `save_pretrained()` usage
- Model loading from checkpoints

**Output**: Model saving/loading issues and naming mismatches

## Subagent 8: Dependency & Library Auditor
**Responsibility**: Check library versions and compatibility
**Checks**:
- All required packages installed (`!pip install` cells)
- Library versions specified and up-to-date
- Known deprecated patterns
- Version conflicts

**Output**: Outdated libraries, deprecated patterns, missing installations

## Subagent 9: Data Pipeline Examiner
**Responsibility**: Validate data loading and preprocessing
**Checks**:
- Correct DataLoader configuration
- Tokenization properly configured
- Data augmentation logic
- Proper train/val/test split handling
- Distributed sampler usage (if DDP)

**Output**: Data loading errors and configuration issues

## Subagent 10: Instruction Compliance Auditor
**Responsibility**: Verify code matches user requirements
**Checks**:
- Model architecture matches specification
- Hyperparameters match requested values
- Output directory and naming conventions followed
- Any specific requirements from user instructions
- Assumptions made are explicitly stated

**Output**: Deviations from instructions with justifications or missing confirmations

## Execution Flow

1. **Gather Instructions**: Collect ALL user instructions (chat, docs, comments)
2. **Priority Analysis**: Run Tier 1 subagents first (Instruction Compliance, Model Choice, Syntax)
3. **Correctness Check**: Run Tier 2 subagents (Training Loop, Tensors, Devices)
4. **Optimization Check**: Run Tier 3 subagents (Memory, Logging, Dependencies, etc.)
5. **Issue Aggregation**: Collect findings organized by severity
6. **Report Generation**: Critical errors first, then warnings/suggestions

## Critical Principle

**Stop immediately if Tier 1 issues found.** No point checking training loop correctness if they're using the wrong model.

Report structure:
1. Critical errors (wrong model, wrong configs, syntax errors)
2. Instruction compliance report  
3. Overengineering check
4. Warnings & suggestions (conversational)

## Communication Protocol

Each subagent returns issues in this format:
```python
{
    'agent_id': 'subagent_name',
    'severity': 'CRITICAL' | 'WARNING' | 'SUGGESTION',
    'category': 'Training Loop' | 'Tensor Ops' | 'Device' | etc.,
    'line_number': int or None,
    'message': 'Description of issue',
    'suggestion': 'How to fix it',
    'code_snippet': 'Relevant code excerpt (if available)'
}
```

## Example Usage

For a given notebook/script:
1. Instantiate all 10 subagents
2. Each subagent analyzes relevant aspects
3. Aggregate results
4. Present comprehensive report organized by:
   - Critical errors (must fix)
   - Warnings (should fix)
   - Suggestions (nice to have)
