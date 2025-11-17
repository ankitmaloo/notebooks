#!/bin/bash
# Base GPU environment setup
# Installs: uv, git-lfs, common GPU tools
# Usage: source setup_base.sh

set -e

echo "=== Setting up base GPU environment ==="

# Check if running on GPU machine
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "WARNING: nvidia-smi not found. Ensure you're on a GPU machine."
fi

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "uv version: $(uv --version)"

# Install git-lfs if not present
if ! command -v git-lfs &> /dev/null; then
    echo "Installing git-lfs..."
    if command -v apt-get &> /dev/null; then
        sudo apt-get update && sudo apt-get install -y git-lfs
    elif command -v yum &> /dev/null; then
        sudo yum install -y git-lfs
    else
        echo "WARNING: Could not auto-install git-lfs. Install manually."
    fi
fi

# Initialize git-lfs
git lfs install --skip-repo 2>/dev/null || true

# Install HuggingFace CLI
if ! command -v huggingface-cli &> /dev/null; then
    echo "Installing huggingface-cli..."
    uv pip install --system huggingface_hub[cli] hf_transfer
fi

# Enable fast HF downloads
export HF_HUB_ENABLE_HF_TRANSFER=1

echo ""
echo "=== Base setup complete ==="
echo "- uv: $(uv --version)"
echo "- git-lfs: $(git lfs version 2>/dev/null || echo 'not installed')"
echo "- huggingface-cli: $(huggingface-cli --version 2>/dev/null || echo 'not installed')"
echo ""
