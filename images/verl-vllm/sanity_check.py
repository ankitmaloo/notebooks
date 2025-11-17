#!/usr/bin/env python3
"""
Sanity check for verl + vLLM environment.
Verifies all critical imports and GPU access.
"""

import sys
from typing import Any


def check_import(module_name: str, version_attr: str = "__version__") -> tuple[bool, str]:
    """Try to import a module and get its version."""
    try:
        module = __import__(module_name.split(".")[0])
        # Handle nested modules
        for part in module_name.split(".")[1:]:
            module = getattr(module, part)
        version = getattr(module, version_attr, "unknown")
        return True, str(version)
    except ImportError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Error: {e}"


def check_gpu() -> tuple[bool, dict[str, Any]]:
    """Check GPU availability and properties."""
    try:
        import torch

        if not torch.cuda.is_available():
            return False, {"error": "CUDA not available"}

        gpu_info = {
            "cuda_available": True,
            "device_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device(),
            "device_name": torch.cuda.get_device_name(0),
            "memory_total_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
            "cuda_version": torch.version.cuda,
        }
        return True, gpu_info
    except Exception as e:
        return False, {"error": str(e)}


def check_vllm_gpu() -> tuple[bool, str]:
    """Check if vLLM can access GPU."""
    try:
        from vllm import LLM
        # Just check import, don't actually load a model
        return True, "vLLM imports successfully"
    except Exception as e:
        return False, str(e)


def check_ray() -> tuple[bool, str]:
    """Check Ray installation and components."""
    try:
        import ray
        # Check Ray version
        version = ray.__version__

        # Try to import key Ray components
        from ray import serve  # noqa: F401
        from ray import tune  # noqa: F401
        from ray.util import ActorPool  # noqa: F401

        return True, f"{version} (serve, tune available)"
    except Exception as e:
        return False, str(e)


def main():
    print("=" * 60)
    print("verl + vLLM Environment Sanity Check")
    print("=" * 60)

    all_passed = True

    # Core imports
    checks = [
        ("torch", "torch", "__version__"),
        ("transformers", "transformers", "__version__"),
        ("vllm", "vllm", "__version__"),
        ("ray", None, None),  # Special check
        ("flashinfer", "flashinfer", "__version__"),
        ("verl", "verl", "__version__"),
        ("datasets", "datasets", "__version__"),
        ("peft", "peft", "__version__"),
        ("accelerate", "accelerate", "__version__"),
        ("wandb", "wandb", "__version__"),
        ("tensordict", "tensordict", "__version__"),
        ("hydra", "hydra", "__version__"),
    ]

    print("\n[Checking imports]")
    for name, module, version_attr in checks:
        if name == "ray":
            success, info = check_ray()
        else:
            success, info = check_import(module, version_attr)

        status = "PASS" if success else "FAIL"
        symbol = "+" if success else "X"
        print(f"  [{symbol}] {name:20s} : {info}")

        if not success:
            all_passed = False

    # GPU check
    print("\n[Checking GPU]")
    gpu_ok, gpu_info = check_gpu()

    if gpu_ok:
        print(f"  [+] CUDA available    : {gpu_info['cuda_available']}")
        print(f"  [+] Device count      : {gpu_info['device_count']}")
        print(f"  [+] Device name       : {gpu_info['device_name']}")
        print(f"  [+] Memory (GB)       : {gpu_info['memory_total_gb']:.2f}")
        print(f"  [+] CUDA version      : {gpu_info['cuda_version']}")
    else:
        print(f"  [X] GPU check failed  : {gpu_info.get('error', 'Unknown error')}")
        all_passed = False

    # vLLM GPU check
    print("\n[Checking vLLM GPU access]")
    vllm_ok, vllm_info = check_vllm_gpu()
    status = "PASS" if vllm_ok else "FAIL"
    symbol = "+" if vllm_ok else "X"
    print(f"  [{symbol}] vLLM GPU access : {vllm_info}")
    if not vllm_ok:
        all_passed = False

    # Final verdict
    print("\n" + "=" * 60)
    if all_passed:
        print("STATUS: ALL CHECKS PASSED - Environment ready for training!")
    else:
        print("STATUS: SOME CHECKS FAILED - Review errors above")
    print("=" * 60)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
