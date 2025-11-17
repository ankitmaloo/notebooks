# Setting Up Environments in Notebooks

Two approaches for running your RL training environments in notebooks.

## Option 1: Direct Notebook Setup (Recommended for Modal/Containers)

**Use Case:** Running a notebook on Modal, in Docker, or any containerized environment.

**Why it works:** Containers are already isolated, so we use `uv pip install --system` instead of creating a venv.

### Quick Start

**In your notebook's first cell:**

```python
# For verl-vllm
%run images/verl-vllm/setup_notebook.py
```

or

```python
# For verl-sglang
%run images/verl-sglang/setup_notebook.py
```

**What it does:**
1. Installs uv
2. Installs all packages with `--system` flag
3. Runs sanity checks
4. Prints GPU info

**After setup, you can immediately:**
```python
import torch
from vllm import LLM
from transformers import AutoModelForCausalLM

# Your training code here...
```

---

## Option 2: Modal Notebooks with Custom Images

**Use Case:** Using Modal's hosted notebooks with pre-built images.

### Step 1: Create Modal App with Image

Create `modal_training_app.py`:

```python
import modal

app = modal.App("rl-training")

# Build custom image
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "git-lfs", "curl", "build-essential")
    .run_commands(
        # Install uv
        "curl -LsSf https://astral.sh/uv/install.sh | sh",
        "export PATH=$HOME/.local/bin:$PATH",
    )
    .copy_local_dir("images", "/root/images")
    .run_commands(
        # Setup using notebook script with --system
        "cd /root && python images/verl-vllm/setup_notebook.py",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# Example function using the image
@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=3600,
)
def train():
    """Training function with custom environment."""
    import torch
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    # Your training code
    pass
```

### Step 2: Deploy

```bash
modal deploy modal_training_app.py
```

### Step 3: Use in Modal Notebook

In a Modal Notebook:

```python
import modal

app = modal.App.lookup("rl-training")

# Run code in your custom environment
with app.run():
    result = train.remote()
```

---

## Option 3: Modal's Built-in `%uv` Magic

**Modal Notebooks** have native `%uv` support:

```python
# In a Modal Notebook cell
%uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
%uv pip install vllm transformers wandb

# Then use immediately
import torch
from vllm import LLM
```

**Pros:** Simple, no setup script needed
**Cons:** Need to install each time notebook restarts

---

## Comparison

| Approach | Setup Time | Persistence | Best For |
|----------|------------|-------------|----------|
| Option 1: `setup_notebook.py` | 5-10 min | Session only | Quick experiments, Docker containers |
| Option 2: Custom Modal Image | Build once | Permanent | Production, repeated use |
| Option 3: `%uv` magic | ~1 min | Session only | Quick tests, small deps |

---

## Example: Complete Modal Notebook Workflow

**Notebook Cell 1 (Setup):**
```python
%run images/verl-vllm/setup_notebook.py
```

**Notebook Cell 2 (Download Model):**
```python
from huggingface_hub import snapshot_download
import os

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

model_path = snapshot_download(
    "meta-llama/Llama-2-7b-hf",
    token=os.getenv("HF_TOKEN")  # Set in Modal Secrets
)
print(f"Model: {model_path}")
```

**Notebook Cell 3 (Training):**
```python
from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Your PPO/GRPO training loop
# ...
```

---

## Why `--system` Works in Notebooks/Containers

From uv docs:

> The `--system` flag is appropriate in continuous integration and containerized environments.

**Key insight:** Modal/Docker notebooks run in isolated containers, so installing to "system" Python is safe and equivalent to a venv.

**What happens:**
```bash
# Traditional approach (creates nested venv in container - unnecessary)
uv venv .venv
source .venv/bin/activate
uv pip install torch

# Better approach (install directly in container's Python)
uv pip install --system torch
```

---

## Troubleshooting

### "uv not found"

```python
import subprocess
subprocess.run(["curl", "-LsSf", "https://astral.sh/uv/install.sh", "|", "sh"], shell=True)
```

### Packages not found after install

```python
import sys
import os
os.environ["PATH"] = f"{os.path.expanduser('~/.local/bin')}:{os.environ['PATH']}"
```

### GPU not detected

Check Modal function has GPU:
```python
@app.function(gpu="A100")  # Must specify GPU
def my_func():
    pass
```

---

## Best Practice: Pre-built Image for Speed

For repeated use, build once:

```dockerfile
# Dockerfile
FROM python:3.12-slim

RUN apt-get update && apt-get install -y git git-lfs curl build-essential
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

COPY images /root/images
RUN python /root/images/verl-vllm/setup_notebook.py

# Now your notebooks start instantly
```

Then use in Modal:
```python
image = modal.Image.from_dockerfile("Dockerfile")
```
