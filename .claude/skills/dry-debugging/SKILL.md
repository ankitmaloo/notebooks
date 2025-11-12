---
name: dry-debugging
description: "Static analysis for PyTorch, HuggingFace, vLLM, and SGLang training/inference code. Use when: (1) User asks to check/debug/review training or inference code, (2) User wants validation before running, (3) Code involves PyTorch, transformers, vLLM, or SGLang. Catches critical errors (wrong model choice, incorrect configs, overengineering) and compares against source instructions to flag deviations and assumptions."
---

# Dry Debugging for ML Training Code

Static analysis for PyTorch/HuggingFace/vLLM/SGLang code. Catches critical errors before execution and compares code against user instructions.

## Core Principle

Models commonly use wrong models, wrong configs, or overengineer solutions. This skill catches these issues by comparing code against source instructions and flagging deviations.

## When to Use This Skill

Use dry debugging when:
- User provides code and asks to check, debug, or review it
- User wants validation before running
- Code involves PyTorch, HuggingFace, vLLM, or SGLang
- User mentions "dry run" or "check for errors"

## Focus Areas

The skill prioritizes catching:
1. **Wrong model choice** - Using inappropriate model for the task
2. **Wrong configs** - Incorrect hyperparameters, batch sizes, learning rates
3. **Overengineering** - Unnecessary complexity, redundant code
4. **Instruction deviations** - Code doesn't match what user asked for
5. **Critical syntax/logic errors** - Will cause immediate failure

## Analysis Process

### Step 1: Extract Source Instructions

**Critical first step**: Gather ALL user instructions from:
- Chat messages throughout the conversation
- Uploaded documentation files
- Inline comments in code
- Requirements specifications

Build a complete picture of what the user actually asked for. Look for:
- Model name/architecture specified
- Hyperparameters mentioned (LR, batch size, epochs, etc.)
- Framework preferences (vLLM vs vanilla transformers)
- Output requirements (save paths, naming)
- Constraints mentioned (memory limits, speed requirements)

### Step 2: Multi-Agent Analysis

Deploy 10 specialized subagents. Read `references/subagent_tasks.md` for details.

Focus order (by severity):
1. **Critical Errors** - Must fix before running
   - Wrong model choice for task
   - Incorrect core configs
   - Syntax errors
   - Missing imports
   - Will-fail-immediately issues

2. **Warnings** - Should fix (conversational)
   - Overengineered solutions
   - Suboptimal patterns
   - Memory inefficiencies
   - Missing best practices

3. **Suggestions** - Nice to have (conversational)
   - Code improvements
   - Optimizations
   - Alternative approaches

### Step 3: Instruction Compliance (Most Important)

Compare code line-by-line against source instructions:

**Red flags to catch**:
- Using `model = AutoModel.from_pretrained('gpt2')` when user asked for Llama
- Batch size = 32 when user specified 16
- Using complex multi-GPU setup when user has single GPU
- Implementing full fine-tuning when user wanted LoRA
- Using vanilla transformers when vLLM would be better for inference
- Overengineering with unnecessary distributed training code

**Questions to answer**:
- Does model match specification? 
- Do hyperparameters match?
- Is the approach right for the task?
- Is it overengineered?
- Are there simpler alternatives?

**For deviations**:
- WITH justification → Note it, explain why
- WITHOUT justification → Flag as ERROR requiring user confirmation
- Assumptions made → List explicitly for user approval

### Step 4: Automated Checks

Use scripts for baseline detection:

```bash
python scripts/analyze_training_code.py <file.py>  # Syntax, imports, basic patterns
python scripts/check_dependencies.py <file.py>     # Versions, installations
```

But remember: Scripts catch syntax/imports. **Your job is catching wrong choices and overengineering.**

### Step 5: Reference Common Patterns

Consult `references/common_errors.md` for:
- Typical wrong model choices
- Common config mistakes
- Overengineering patterns
- Framework-specific issues

### Step 6: Framework-Specific Checks

**PyTorch/HuggingFace**:
- Model choice appropriate for task?
- Using Trainer API when simpler approach would work?
- Unnecessary distributed training code?

**vLLM/SGLang** (for inference):
- Should they be using vLLM instead of vanilla transformers for inference?
- SGLang for complex prompting/structured output?
- Overengineering with custom sampling when framework has it built-in?

## Output Format

### Part 1: Critical Errors (Must Fix)

List only issues that:
- Will cause immediate failure (syntax, imports, type errors)
- Use wrong model for the task
- Have incorrect critical configs
- Deviate from user instructions without justification

Format:
```
🔴 CRITICAL ERROR #1: Wrong Model Choice
Line: N/A (Architecture decision)
User asked for: "Llama-7B for instruction following"
Code uses: GPT-2
Why this is wrong: GPT-2 is not instruction-tuned, much smaller (124M vs 7B params)
Fix: Use 'meta-llama/Llama-2-7b-chat-hf' or similar instruction-tuned model

🔴 CRITICAL ERROR #2: Batch Size Mismatch
Line: 45
User specified: batch_size = 8
Code has: batch_size = 32
Fix: Change to per_device_train_batch_size=8
```

**Purpose**: These must be fixed. Stop the user from running with wrong choices.

### Part 2: Instruction Compliance Report

Compare against source instructions:

```
✅ MATCHES INSTRUCTIONS:
- Model architecture: Llama-7B ✓
- Learning rate: 2e-4 ✓
- Using LoRA adapters ✓

❌ DEVIATIONS FROM INSTRUCTIONS:
- Batch size: Asked 8, got 32
- Output dir: Asked './checkpoints', got './model'
- Using AdamW instead of asked Adam (This is actually better - recommend keeping)

⚠️ ASSUMPTIONS MADE (Not in instructions):
- Assumed LoRA rank r=16 (user didn't specify)
- Assumed warmup steps = 100 (user didn't mention)
- Using gradient accumulation = 4 (user didn't ask for this)

👉 User should confirm these assumptions are acceptable.
```

**Purpose**: Show exactly where code matches/differs from what was asked.

### Part 3: Overengineering Check

```
🔧 OVERENGINEERING DETECTED:

Issue: Using DDP (DistributedDataParallel) for single-GPU training
Evidence: Code has multi-GPU setup but user mentioned "on my A100"
Impact: Unnecessary complexity, harder to debug
Recommendation: Remove DDP code, use simple .to(device)

Issue: Custom training loop when Trainer would work
Evidence: 200 lines of manual epoch/batch loops
Impact: Reinventing the wheel, more bugs
Recommendation: Use HF Trainer API - it handles all of this
```

**Purpose**: Catch unnecessary complexity that makes code harder to maintain.

### Part 4: Warnings & Suggestions (Conversational)

Only after critical errors are addressed, mention:
- Missing best practices
- Optimization opportunities  
- Style improvements

Keep this section brief and conversational.

```
A few suggestions:
- Consider adding wandb logging for experiment tracking
- You might want gradient checkpointing to save memory
- Could add evaluation loop to track validation loss
```

### Final Summary

```
SEVERITY BREAKDOWN:
- Critical Errors: X (MUST FIX)
- Deviations: Y (REVIEW WITH USER)
- Warnings: Z (conversational)

ACTION REQUIRED:
1. Fix critical errors #1, #2, #3
2. Confirm assumptions with user
3. Consider removing overengineered parts
```

## Example Interaction

```
User: "I need a model for customer service chat. Check this code."
[Code uses GPT-2 with custom training loop]

Claude: Let me dry debug this code. First, gathering your instructions...

CRITICAL ERRORS FOUND:

🔴 CRITICAL #1: Wrong Model Choice
User wants: Customer service chat
Code uses: GPT-2 (base model, 124M params)
Problem: GPT-2 is not instruction-tuned and too small for quality chat
Fix: Use 'meta-llama/Llama-2-7b-chat-hf' or 'mistralai/Mistral-7B-Instruct-v0.2'

🔴 CRITICAL #2: Overengineered Training
Code has: 200+ lines of custom training loop
Problem: Reinventing the wheel, more bugs, harder to maintain
Fix: Use HuggingFace Trainer API - handles all of this automatically

INSTRUCTION COMPLIANCE:
❌ Model type: You asked for chat model, got base GPT-2
✅ Learning rate: 2e-4 as requested
⚠️ Assumption: Code assumes 16GB GPU (you didn't specify hardware)

ACTION REQUIRED:
1. Change to instruction-tuned model (Llama-2-chat or Mistral-Instruct)
2. Replace custom loop with Trainer API
3. Confirm: Do you have 16GB+ GPU?

[Warnings and suggestions would follow only after these are addressed]
```

## Key Reminders

### Priority Order
1. **Instruction compliance** - Does code match what user asked for?
2. **Model/framework choice** - Right tool for the job?
3. **Overengineering** - Can it be simpler?
4. **Critical errors** - Will it run?
5. **Warnings** - Best practices (conversational)

### Scope
- **Frameworks**: PyTorch, HuggingFace Transformers, vLLM, SGLang only
- **Focus**: Wrong choices, wrong configs, overengineering
- **Not focus**: Generic Python style, formatting, documentation

### User Instructions
Source instructions come from:
- Current and previous chat messages
- Uploaded documentation
- Code comments
- Requirements files

Gather ALL of these before analyzing.

### Severity Levels
- **Critical**: Must fix (wrong model, wrong configs, syntax errors, instruction mismatches)
- **Warning**: Should consider (conversational, not blocking)
- **Suggestion**: Nice to have (conversational)

User expects to fix critical errors immediately. Warnings/suggestions are for conversation.

## Final Checklist

Before completing dry debug:
- [ ] Gathered ALL user instructions from conversation and docs
- [ ] Checked model choice is appropriate for task
- [ ] Verified configs match user specifications  
- [ ] Flagged overengineering (DDP on 1 GPU, custom loops, etc.)
- [ ] Listed all assumptions explicitly
- [ ] Ran syntax/import checks
- [ ] Put critical errors first in report
- [ ] Made warnings/suggestions conversational, not prescriptive
