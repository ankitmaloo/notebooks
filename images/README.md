# GPU Environment Images

Pre-configured environments for fast RL training setup. No conda, only uv.

## Quick Start

```bash
# On your GPU machine (Modal, etc.)
source images/verl-vllm/setup.sh
# Wait for setup + sanity check
# If all green, you're ready to train!
```

## Available Environments

| Environment | Backend | PyTorch | Use Case |
|-------------|---------|---------|----------|
| `verl-vllm` | vLLM | 2.9 | Fast inference, production-ready |
| `verl-sglang` | SGLang | 2.8 | Flexible serving, latest features |

## Structure

```
images/
├── base/              # Common setup (uv, git-lfs, hf-cli)
├── verl-vllm/         # verl + vLLM environment
├── verl-sglang/       # verl + SGLang environment
└── utils/             # Helper scripts
```

## Usage Patterns

### Pattern 1: Quick Setup on GPU Instance

```bash
# SSH into GPU machine or Modal
cd your-project
source images/verl-vllm/setup.sh

# Environment ready, start training
python train.py
```

### Pattern 2: Reactivate Existing Environment

```bash
# After initial setup, just activate
source ~/.venvs/verl-vllm/bin/activate
python sanity_check.py  # Optional verification
```

---

## Notebook Connectivity

### Option A: Local Notebook → Remote GPU Server

Your local Jupyter notebook connects to GPU server's kernel.

**1. On GPU Server (Modal/Remote):**

```bash
# Setup environment first
source images/verl-vllm/setup.sh

# Start Jupyter server (no browser)
jupyter notebook --no-browser --port=8888 --ip=0.0.0.0

# Note the token from output:
# http://0.0.0.0:8888/?token=YOUR_TOKEN_HERE
```

**2. SSH Tunnel from Local:**

```bash
# In new terminal
ssh -N -L 8888:localhost:8888 user@gpu-server
```

**3. Open in Local Browser:**

```
http://localhost:8888/?token=YOUR_TOKEN_HERE
```

**4. Select Kernel:**

In Jupyter, select kernel: `Python (verl-vllm)` or `Python (verl-sglang)`

---

### Option B: Notebook Running Directly on Server

Run notebook entirely on the GPU server (e.g., Modal).

**Modal Example:**

```python
import modal

app = modal.App("rl-training")

# Create image with your setup
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "git-lfs", "curl")
    .run_commands(
        "curl -LsSf https://astral.sh/uv/install.sh | sh",
        "export PATH=$HOME/.local/bin:$PATH",
    )
    .copy_local_dir("images", "/root/images")
    .run_commands(
        "chmod +x /root/images/verl-vllm/setup.sh",
        "cd /root && source images/verl-vllm/setup.sh",
    )
)

@app.function(
    image=image,
    gpu="A100",
    timeout=3600,
)
def run_notebook():
    import subprocess
    # Run your notebook
    subprocess.run([
        "jupyter", "nbconvert", "--execute",
        "--to", "notebook", "--inplace",
        "your_notebook.ipynb"
    ])
```

**VSCode Remote SSH:**

1. Install "Remote - SSH" extension
2. Connect to GPU server
3. Open folder with your project
4. Open `.ipynb` file
5. Select kernel: `Python (verl-vllm)`

**JupyterLab on Server:**

```bash
source ~/.venvs/verl-vllm/bin/activate
jupyter lab --no-browser --port=8888 --ip=0.0.0.0

# Then tunnel from local machine
```

---

## Environment Details

### verl-vllm

- Python 3.12
- PyTorch 2.9 + CUDA 12.4
- vLLM (latest)
- FlashInfer (latest for torch 2.9)
- Ray (with serve, tune, dashboard)
- Full verl dependencies
- Jupyter kernel registered

**Best for:** Production inference, stable performance

### verl-sglang

- Python 3.12
- PyTorch 2.8 + CUDA 12.4
- SGLang (latest)
- FlashInfer (for torch 2.8)
- Ray (with serve, tune, dashboard)
- Full verl dependencies
- Jupyter kernel registered

**Best for:** Experimental features, flexible serving

---

## Troubleshooting

### Sanity Check Fails

**GPU not detected:**
```bash
nvidia-smi  # Check driver
# If missing, install NVIDIA drivers
```

**Import errors:**
```bash
# Re-run setup to reinstall
source images/verl-vllm/setup.sh
```

**Ray components missing:**
```bash
source ~/.venvs/verl-vllm/bin/activate
uv pip install "ray[default,serve,tune]" aiohttp py-spy gpustat
```

### CUDA Version Mismatch

```bash
# Check your CUDA version
nvcc --version
nvidia-smi  # Driver CUDA version

# Environments expect CUDA 12.4
# If different, modify setup.sh to match your CUDA
```

### Kernel Not Showing in Jupyter

```bash
source ~/.venvs/verl-vllm/bin/activate
python -m ipykernel install --user --name "verl-vllm"

# List kernels
jupyter kernelspec list
```

---

## Workflow Example

```bash
# 1. Setup (one-time)
source images/verl-vllm/setup.sh

# 2. Download model
huggingface-cli download meta-llama/Llama-2-7b-hf

# 3. Your training script
python train_ppo.py \
  --model meta-llama/Llama-2-7b-hf \
  --dataset your-hf-dataset \
  --reward_fn custom_reward.py

# 4. Monitor with wandb (auto-configured)
```

All the boring stuff is handled. Focus on your model, dataset, and reward function.
