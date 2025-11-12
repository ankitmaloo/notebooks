# Common Training Code Errors

## 0. Wrong Model/Framework Choices (MOST CRITICAL)

### Using wrong model for task
```python
# ❌ WRONG - User asked for instruction-following
model = AutoModelForCausalLM.from_pretrained('gpt2')

# ✅ CORRECT
model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-chat-hf')
```

### Using base model instead of chat/instruct variant
```python
# ❌ WRONG - For chat applications
model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf')

# ✅ CORRECT
model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-chat-hf')
```

### Using transformers for inference when vLLM is better
```python
# ❌ SUBOPTIMAL - For production inference
from transformers import pipeline
generator = pipeline('text-generation', model='llama-7b')
outputs = generator(prompts)

# ✅ BETTER - Much faster inference
from vllm import LLM
llm = LLM(model='meta-llama/Llama-2-7b-hf')
outputs = llm.generate(prompts)
```

### Not using SGLang for structured output
```python
# ❌ SUBOPTIMAL - Complex regex parsing
output = model.generate(prompt)
result = complex_regex_parsing(output)

# ✅ BETTER - SGLang handles structure
import sglang as sgl
@sgl.function
def extract_structured(s):
    # SGLang ensures proper JSON/structure
```

## 0.1 Overengineering Patterns

### Using DDP for single GPU
```python
# ❌ OVERENGINEERED - User has 1 GPU
import torch.distributed as dist
dist.init_process_group(...)
model = DDP(model)

# ✅ SIMPLE
model = model.to('cuda')
```

### Custom training loop when Trainer works
```python
# ❌ OVERENGINEERED - 200+ lines of manual loops
for epoch in range(num_epochs):
    for batch in dataloader:
        # Manual gradient accumulation
        # Manual mixed precision
        # Manual checkpointing
        # Manual logging
        ...

# ✅ SIMPLE - Trainer handles all this
from transformers import Trainer, TrainingArguments
trainer = Trainer(model=model, args=training_args, ...)
trainer.train()
```

### Implementing features that already exist
```python
# ❌ OVERENGINEERED - Custom gradient accumulation
accumulated_loss = 0
for i, batch in enumerate(dataloader):
    loss = model(batch).loss / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# ✅ SIMPLE - Trainer has this built-in
training_args = TrainingArguments(
    gradient_accumulation_steps=4,
    ...
)
```

## 1. Training Loop Errors

### Missing gradient zeroing
```python
# ❌ WRONG
for batch in dataloader:
    loss = model(batch)
    loss.backward()
    optimizer.step()

# ✅ CORRECT
for batch in dataloader:
    optimizer.zero_grad()  # Must zero gradients!
    loss = model(batch)
    loss.backward()
    optimizer.step()
```

### Wrong operation order
Standard order: `zero_grad()` → forward pass → `backward()` → `step()`

### Missing model.train() / model.eval()
```python
# ❌ WRONG - model stays in eval mode
model.eval()
for batch in train_loader:
    loss = model(batch)
    loss.backward()  # No gradient updates!

# ✅ CORRECT
model.train()  # Switch to training mode
for batch in train_loader:
    loss = model(batch)
    loss.backward()
```

### Gradient accumulation errors
```python
# ❌ WRONG - step() called too frequently
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    optimizer.step()  # Wrong! Accumulation broken

# ✅ CORRECT
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

## 2. Tensor Shape Errors

### Batch dimension mismatch
```python
# ❌ WRONG
logits = model(input_ids)  # Shape: [batch, seq_len, vocab]
loss = criterion(logits, labels)  # Shape mismatch!

# ✅ CORRECT
logits = model(input_ids)  # Shape: [batch, seq_len, vocab]
loss = criterion(logits.view(-1, vocab_size), labels.view(-1))
```

### Missing unsqueeze/squeeze
```python
# ❌ WRONG - single sample without batch dim
output = model(input)  # Input: [seq_len, features]

# ✅ CORRECT
output = model(input.unsqueeze(0))  # Input: [1, seq_len, features]
```

## 3. Device Mismatch Errors

### Model and data on different devices
```python
# ❌ WRONG
model = model.to('cuda')
for batch in dataloader:
    output = model(batch)  # batch still on CPU!

# ✅ CORRECT
model = model.to('cuda')
for batch in dataloader:
    batch = {k: v.to('cuda') for k, v in batch.items()}
    output = model(batch)
```

### Missing device check
```python
# ❌ WRONG - assumes CUDA is available
model = model.to('cuda')

# ✅ CORRECT
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
```

## 4. Data Loading Errors

### Wrong shuffle in distributed training
```python
# ❌ WRONG - breaks DDP
train_loader = DataLoader(dataset, shuffle=True)

# ✅ CORRECT
from torch.utils.data.distributed import DistributedSampler
sampler = DistributedSampler(dataset)
train_loader = DataLoader(dataset, sampler=sampler)
```

### Missing pin_memory for GPU training
```python
# ❌ SUBOPTIMAL
train_loader = DataLoader(dataset, batch_size=32)

# ✅ BETTER
train_loader = DataLoader(dataset, batch_size=32, 
                         pin_memory=True, num_workers=4)
```

## 5. Memory Management Errors

### Not detaching in validation
```python
# ❌ WRONG - keeps full computation graph
model.eval()
for batch in val_loader:
    val_loss += model(batch).item()  # Memory leak!

# ✅ CORRECT
model.eval()
with torch.no_grad():  # Disable gradient computation
    for batch in val_loader:
        val_loss += model(batch).item()
```

### Accumulating tensors without detach
```python
# ❌ WRONG - keeps computation graph
losses = []
for batch in dataloader:
    loss = model(batch)
    losses.append(loss)  # Memory leak!

# ✅ CORRECT
losses = []
for batch in dataloader:
    loss = model(batch)
    losses.append(loss.detach().cpu().item())
```

## 6. Learning Rate Scheduler Errors

### Wrong scheduler step timing
```python
# ❌ WRONG - scheduler steps before optimizer
for epoch in range(epochs):
    scheduler.step()  # Too early!
    for batch in dataloader:
        optimizer.step()

# ✅ CORRECT - depends on scheduler type
for epoch in range(epochs):
    for batch in dataloader:
        optimizer.step()
    scheduler.step()  # After epoch for StepLR, CosineAnnealingLR
    
# Or for OneCycleLR:
for epoch in range(epochs):
    for batch in dataloader:
        optimizer.step()
        scheduler.step()  # After each batch
```

## 7. Logging & Monitoring Errors

### Missing wandb initialization
```python
# ❌ WRONG
import wandb
for epoch in range(epochs):
    wandb.log({'loss': loss})  # Will fail!

# ✅ CORRECT
import wandb
wandb.init(project='my-project', name='experiment-1', config={
    'learning_rate': lr,
    'batch_size': batch_size
})
for epoch in range(epochs):
    wandb.log({'loss': loss})
```

### Logging on wrong step
```python
# ❌ WRONG - logs before training step
wandb.log({'loss': loss.item()})
optimizer.step()

# ✅ CORRECT - logs after step
optimizer.step()
wandb.log({'loss': loss.item(), 'step': global_step})
```

## 8. Model Saving Errors

### Not saving optimizer state
```python
# ❌ INCOMPLETE
torch.save(model.state_dict(), 'model.pt')

# ✅ COMPLETE
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, 'checkpoint.pt')
```

### Wrong model name
```python
# ❌ WRONG - doesn't match user request
model.save_pretrained('./my-custom-name')  # User asked for 'llama-finetuned'

# ✅ CORRECT - follows instructions
model.save_pretrained('./llama-finetuned')  # Matches user specification
```

## 9. Distributed Training Errors

### Missing DDP initialization
```python
# ❌ WRONG - using DDP patterns without setup
model = nn.parallel.DistributedDataParallel(model)  # Will fail!

# ✅ CORRECT
import torch.distributed as dist
dist.init_process_group(backend='nccl')
local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)
model = model.to(local_rank)
model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
```

### Not using rank-specific seeds
```python
# ❌ WRONG - all processes have same randomness
torch.manual_seed(42)

# ✅ CORRECT
rank = dist.get_rank()
torch.manual_seed(42 + rank)
```

## 10. HuggingFace-Specific Errors

### Wrong tokenizer padding
```python
# ❌ WRONG - padding token not set
tokenizer = AutoTokenizer.from_pretrained('gpt2')
inputs = tokenizer(texts, padding=True)  # Will use wrong padding!

# ✅ CORRECT
tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
inputs = tokenizer(texts, padding=True, return_tensors='pt')
```

### Missing attention_mask
```python
# ❌ WRONG - no attention mask
outputs = model(input_ids)

# ✅ CORRECT
outputs = model(input_ids=input_ids, attention_mask=attention_mask)
```

### Not freezing base model layers
```python
# ❌ WRONG - fine-tuning entire model when only needed adapter
model = AutoModel.from_pretrained('llama-7b')
# Training all 7B parameters!

# ✅ CORRECT - using LoRA
from peft import get_peft_model, LoraConfig
config = LoraConfig(r=8, lora_alpha=32, target_modules=['q_proj', 'v_proj'])
model = get_peft_model(model, config)
# Only training ~few M parameters
```

## 11. Configuration Errors

### Hardcoded hyperparameters
```python
# ❌ WRONG - values scattered in code
lr = 1e-4
batch_size = 16
epochs = 10

# ✅ CORRECT - centralized config
config = {
    'learning_rate': 1e-4,
    'batch_size': 16,
    'epochs': 10,
    'model_name': 'llama-7b',
    'output_dir': './checkpoints'
}
```

### Missing reproducibility setup
```python
# ❌ WRONG - no seed setting
model = train()

# ✅ CORRECT
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)
```
