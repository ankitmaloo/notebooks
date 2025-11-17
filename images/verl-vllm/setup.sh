#!/bin/bash
# verl + vLLM environment setup
# Python 3.12, PyTorch 2.9, vLLM latest, CUDA 12.4
# Usage: source setup.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="verl-vllm"
ENV_PATH="$HOME/.venvs/$ENV_NAME"

echo "=== Setting up $ENV_NAME environment ==="

# Source base setup
source "$SCRIPT_DIR/../base/setup_base.sh"

# Create virtual environment with Python 3.12
echo "Creating virtual environment at $ENV_PATH..."
uv venv "$ENV_PATH" --python 3.12

# Activate environment
source "$ENV_PATH/bin/activate"

echo "Python: $(python --version)"
echo "Environment: $ENV_PATH"

# Install PyTorch 2.9 with CUDA 12.4
echo ""
echo "Installing PyTorch 2.9 with CUDA 12.4..."
uv pip install torch==2.9.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install vLLM (latest compatible with torch 2.9)
echo ""
echo "Installing vLLM..."
uv pip install vllm

# Install FlashInfer latest
echo ""
echo "Installing FlashInfer..."
uv pip install flashinfer-python -i https://flashinfer.ai/whl/cu124/torch2.9/

# Install Ray with all extras (dashboard, default, serve)
echo ""
echo "Installing Ray with all components..."
uv pip install "ray[default,serve,tune]"
# Ray extras that don't auto-install
uv pip install aiohttp aiohttp-cors py-spy gpustat prometheus_client opencensus

# Install verl core dependencies
echo ""
echo "Installing verl and RL training dependencies..."
uv pip install \
    accelerate \
    codetiming \
    datasets \
    dill \
    hydra-core \
    liger-kernel \
    "numpy<2.0.0" \
    pandas \
    peft \
    "pyarrow>=19.0.0" \
    pybind11 \
    pylatexenc \
    "tensordict>=0.8.0,<=0.10.0,!=0.9.0" \
    torchdata \
    transformers \
    wandb \
    "packaging>=20.0" \
    uvicorn \
    fastapi \
    latex2sympy2_extended \
    math_verify \
    tensorboard \
    ipykernel \
    jupyter \
    ipywidgets \
    tqdm

# Install verl itself
echo ""
echo "Installing verl..."
uv pip install verl

# Register Jupyter kernel
echo ""
echo "Registering Jupyter kernel..."
python -m ipykernel install --user --name "$ENV_NAME" --display-name "Python ($ENV_NAME)"

echo ""
echo "=== Running sanity check ==="
python "$SCRIPT_DIR/sanity_check.py"

echo ""
echo "=== Setup complete! ==="
echo "Environment: $ENV_PATH"
echo "To activate: source $ENV_PATH/bin/activate"
echo "Jupyter kernel: $ENV_NAME"
echo ""
