"""
Dockerfile Generator for RL Training Environments
Creates customized Docker images for different training setups
"""
from pathlib import Path
from typing import List, Optional, Dict
import argparse


# ============================================================================
# Dockerfile Templates
# ============================================================================

BASEIMAGE_CUDA118 = "nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04"
BASEIMAGE_CUDA121 = "nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04"
BASEIMAGE_CUDA124 = "nvidia/cuda:12.4.0-cudnn9-devel-ubuntu22.04"


def generate_base_dockerfile(
    cuda_version: str = "12.1",
    python_version: str = "3.10",
    torch_version: str = "2.4.0",
    install_verl: bool = True,
    install_flash_attn: bool = True,
    additional_packages: Optional[List[str]] = None
) -> str:
    """Generate a base Dockerfile for RL training"""
    
    # Select base image
    base_images = {
        "11.8": BASEIMAGE_CUDA118,
        "12.1": BASEIMAGE_CUDA121,
        "12.4": BASEIMAGE_CUDA124,
    }
    base_image = base_images.get(cuda_version, BASEIMAGE_CUDA121)
    
    # Determine CUDA index URL for PyTorch
    cuda_short = cuda_version.replace(".", "")
    torch_index_url = f"https://download.pytorch.org/whl/cu{cuda_short}"
    
    dockerfile = f"""# Auto-generated Dockerfile for RL Training
FROM {base_image}

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    git \\
    wget \\
    curl \\
    vim \\
    build-essential \\
    cmake \\
    ninja-build \\
    libopenmpi-dev \\
    && rm -rf /var/lib/apt/lists/*

# Install Python {python_version}
RUN apt-get update && apt-get install -y \\
    python{python_version} \\
    python{python_version}-dev \\
    python{python_version}-venv \\
    python3-pip \\
    && rm -rf /var/lib/apt/lists/*

# Set Python {python_version} as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python{python_version} 1 && \\
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python{python_version} 1

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Install PyTorch {torch_version}
RUN pip install torch=={torch_version} torchvision torchaudio --index-url {torch_index_url}

# Install transformers and core dependencies
RUN pip install \\
    transformers>=4.36.0 \\
    accelerate>=0.25.0 \\
    datasets>=2.16.0 \\
    tokenizers>=0.15.0 \\
    huggingface-hub>=0.20.0 \\
    safetensors>=0.4.1

# Install training utilities
RUN pip install \\
    wandb \\
    tensorboard \\
    peft \\
    bitsandbytes \\
    scipy \\
    scikit-learn \\
    pandas \\
    numpy \\
    tqdm

"""

    # Add Flash Attention
    if install_flash_attn:
        dockerfile += """# Install Flash Attention 2
RUN pip install flash-attn --no-build-isolation

"""

    # Add verl
    if install_verl:
        dockerfile += """# Install verl and dependencies
RUN pip install vllm==0.5.4 ray==2.10
RUN git clone https://github.com/volcengine/verl.git /workspace/verl && \\
    cd /workspace/verl && \\
    pip install -e .

"""

    # Add additional packages
    if additional_packages:
        packages_str = " \\\n    ".join(additional_packages)
        dockerfile += f"""# Install additional packages
RUN pip install \\
    {packages_str}

"""

    # Add workspace setup
    dockerfile += """# Set up workspace
WORKDIR /workspace
RUN mkdir -p /workspace/data /workspace/checkpoints /workspace/logs

# Copy training scripts (if present)
# COPY scripts/ /workspace/scripts/
# COPY configs/ /workspace/configs/

# Expose ports for tensorboard and wandb
EXPOSE 6006 8080

# Default command
CMD ["/bin/bash"]
"""

    return dockerfile


def generate_verl_dockerfile(
    cuda_version: str = "12.1",
    python_version: str = "3.10",
    use_fsdp: bool = True,
    use_megatron: bool = False
) -> str:
    """Generate a specialized Dockerfile for verl training"""
    
    base_images = {
        "11.8": BASEIMAGE_CUDA118,
        "12.1": BASEIMAGE_CUDA121,
        "12.4": BASEIMAGE_CUDA124,
    }
    base_image = base_images.get(cuda_version, BASEIMAGE_CUDA121)
    
    dockerfile = f"""# Auto-generated Dockerfile for verl Training
FROM {base_image}

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda

# System dependencies
RUN apt-get update && apt-get install -y \\
    git wget curl vim build-essential cmake ninja-build libopenmpi-dev \\
    python{python_version} python{python_version}-dev python{python_version}-venv python3-pip \\
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python{python_version} 1

# Python dependencies
RUN python -m pip install --upgrade pip setuptools wheel

# PyTorch (let vllm handle the installation)
RUN pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# Core dependencies
RUN pip install vllm==0.5.4 ray==2.10 flash-attn --no-build-isolation

"""

    if use_fsdp:
        dockerfile += """# FSDP Backend (Recommended for prototyping)
RUN git clone https://github.com/volcengine/verl.git /workspace/verl && \\
    cd /workspace/verl && \\
    pip install -e .

"""
    
    if use_megatron:
        dockerfile += """# Megatron-LM Backend (For production scaling)
# Install Apex
RUN pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation \\
    --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" \\
    git+https://github.com/NVIDIA/apex

# Install Transformer Engine
RUN pip install git+https://github.com/NVIDIA/TransformerEngine.git@v1.7

# Install Megatron-LM
RUN cd /workspace && \\
    git clone -b core_v0.4.0 https://github.com/NVIDIA/Megatron-LM.git && \\
    cd Megatron-LM && \\
    cp /workspace/verl/patches/megatron_v4.patch . && \\
    git apply megatron_v4.patch && \\
    pip install -e .

ENV PYTHONPATH=$PYTHONPATH:/workspace/Megatron-LM

"""

    dockerfile += """# Additional ML utilities
RUN pip install transformers accelerate datasets wandb tensorboard

WORKDIR /workspace
RUN mkdir -p /workspace/data /workspace/checkpoints /workspace/logs

EXPOSE 6006 8080
CMD ["/bin/bash"]
"""

    return dockerfile


def generate_nanochat_style_dockerfile(
    cuda_version: str = "12.1",
    python_version: str = "3.10"
) -> str:
    """Generate a Dockerfile inspired by nanochat's minimal setup"""
    
    base_images = {
        "11.8": BASEIMAGE_CUDA118,
        "12.1": BASEIMAGE_CUDA121,
        "12.4": BASEIMAGE_CUDA124,
    }
    base_image = base_images.get(cuda_version, BASEIMAGE_CUDA121)
    
    dockerfile = f"""# Minimal Dockerfile for nanochat-style Training
FROM {base_image}

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Minimal system setup
RUN apt-get update && apt-get install -y \\
    git curl python{python_version} python{python_version}-venv python3-pip \\
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

WORKDIR /workspace

# Install core dependencies with uv
RUN uv pip install --system \\
    torch==2.4.0 \\
    transformers \\
    accelerate \\
    wandb \\
    numpy \\
    tqdm

# Simple and minimal - ready for custom training code
CMD ["/bin/bash"]
"""

    return dockerfile


# ============================================================================
# Main Generator Function
# ============================================================================

def generate_dockerfile(
    output_path: str = "Dockerfile",
    style: str = "base",
    cuda_version: str = "12.1",
    python_version: str = "3.10",
    **kwargs
) -> str:
    """
    Generate a Dockerfile for RL training
    
    Args:
        output_path: Path to save Dockerfile
        style: "base", "verl", or "nanochat"
        cuda_version: CUDA version ("11.8", "12.1", "12.4")
        python_version: Python version (e.g., "3.10")
        **kwargs: Additional arguments for specific styles
    
    Returns:
        Generated Dockerfile content
    """
    generators = {
        "base": generate_base_dockerfile,
        "verl": generate_verl_dockerfile,
        "nanochat": generate_nanochat_style_dockerfile,
    }
    
    if style not in generators:
        raise ValueError(f"Unknown style: {style}. Choose from {list(generators.keys())}")
    
    generator = generators[style]
    dockerfile_content = generator(cuda_version=cuda_version, python_version=python_version, **kwargs)
    
    # Write to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(dockerfile_content)
    
    print(f"✅ Generated {style} Dockerfile at: {output_path}")
    print(f"\nTo build: docker build -t rl-training:{style} -f {output_path} .")
    print(f"To run: docker run --gpus all -it --rm -v $(pwd):/workspace rl-training:{style}")
    
    return dockerfile_content


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate Dockerfile for RL training")
    parser.add_argument("--style", type=str, default="base", choices=["base", "verl", "nanochat"],
                        help="Dockerfile style")
    parser.add_argument("--output", type=str, default="Dockerfile", help="Output path")
    parser.add_argument("--cuda", type=str, default="12.1", choices=["11.8", "12.1", "12.4"],
                        help="CUDA version")
    parser.add_argument("--python", type=str, default="3.10", help="Python version")
    parser.add_argument("--verl", action="store_true", help="Install verl (base style only)")
    parser.add_argument("--flash-attn", action="store_true", default=True, help="Install flash attention")
    parser.add_argument("--megatron", action="store_true", help="Use Megatron-LM backend (verl style only)")
    
    args = parser.parse_args()
    
    kwargs = {}
    if args.style == "base":
        kwargs = {
            "install_verl": args.verl,
            "install_flash_attn": args.flash_attn,
        }
    elif args.style == "verl":
        kwargs = {
            "use_megatron": args.megatron,
        }
    
    generate_dockerfile(
        output_path=args.output,
        style=args.style,
        cuda_version=args.cuda,
        python_version=args.python,
        **kwargs
    )


if __name__ == "__main__":
    main()
