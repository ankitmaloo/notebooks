"""
Setup script for running in Jupyter notebooks (Modal, containers, etc.)
Run this in a notebook cell to set up the verl-sglang environment.

Usage in notebook:
    %run setup_notebook.py
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: str, description: str = ""):
    """Run shell command and display output."""
    if description:
        print(f"\n{'='*60}")
        print(f"{description}")
        print('='*60)

    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True
    )

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)

    if result.returncode != 0:
        print(f"ERROR: Command failed with code {result.returncode}")
        return False
    return True


def main():
    print("Setting up verl-sglang environment in notebook...")

    # Check if we're in a notebook
    try:
        get_ipython()
        print("✓ Running in Jupyter environment")
    except NameError:
        print("WARNING: Not in Jupyter. Run this with: python setup_notebook.py")

    # Install uv if needed
    if not Path.home().joinpath(".local/bin/uv").exists():
        run_command(
            "curl -LsSf https://astral.sh/uv/install.sh | sh",
            "Installing uv"
        )

    # Add uv to PATH for current session
    uv_path = str(Path.home() / ".local/bin")
    if uv_path not in sys.path:
        import os
        os.environ["PATH"] = f"{uv_path}:{os.environ['PATH']}"

    print(f"\n✓ uv available at: {uv_path}/uv")

    # Install packages using --system flag (for containers/notebooks)
    packages = [
        # PyTorch with CUDA 12.4
        "--index-url https://download.pytorch.org/whl/cu124 torch==2.8.0 torchvision torchaudio",
        # SGLang and FlashInfer
        "'sglang[all]'",
        "-i https://flashinfer.ai/whl/cu124/torch2.8/ flashinfer-python",
        # Ray with extras
        "'ray[default,serve,tune]' aiohttp aiohttp-cors py-spy gpustat prometheus_client opencensus",
        # verl and RL dependencies
        """accelerate codetiming datasets dill hydra-core liger-kernel
           'numpy<2.0.0' pandas peft 'pyarrow>=19.0.0' pybind11 pylatexenc
           'tensordict>=0.8.0,<=0.10.0,!=0.9.0' torchdata transformers wandb
           'packaging>=20.0' uvicorn fastapi latex2sympy2_extended math_verify
           tensorboard ipykernel jupyter ipywidgets tqdm verl""",
    ]

    for pkg_group in packages:
        success = run_command(
            f"uv pip install --system {pkg_group}",
            f"Installing: {pkg_group.split()[0]}..."
        )
        if not success:
            print(f"Failed to install: {pkg_group}")
            return

    # Enable HF fast downloads
    import os
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    print("\n" + "="*60)
    print("Running sanity check...")
    print("="*60)

    # Import check
    try:
        import torch
        import transformers
        import sglang
        import ray
        import verl

        print(f"✓ PyTorch: {torch.__version__}")
        print(f"✓ Transformers: {transformers.__version__}")
        print(f"✓ SGLang: {sglang.__version__}")
        print(f"✓ Ray: {ray.__version__}")
        print(f"✓ verl: {verl.__version__}")

        if torch.cuda.is_available():
            print(f"\n✓ GPU: {torch.cuda.get_device_name(0)}")
            print(f"✓ CUDA: {torch.version.cuda}")
            print(f"✓ Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("\n⚠ WARNING: CUDA not available")

        print("\n" + "="*60)
        print("✓ ALL CHECKS PASSED - Ready to train!")
        print("="*60)

    except ImportError as e:
        print(f"\n✗ Import failed: {e}")
        print("Some packages may not have installed correctly.")


if __name__ == "__main__":
    main()
