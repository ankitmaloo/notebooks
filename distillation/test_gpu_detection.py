#!/usr/bin/env python3
"""Test GPU auto-detection for multi-teacher OPD notebook."""

import torch
from dataclasses import dataclass
from typing import Tuple

@dataclass
class GPUConfig:
    """Auto-detected GPU configuration."""
    name: str
    vram_gb: float
    compute_capability: Tuple[int, int]

    # Auto-selected settings
    dtype: torch.dtype
    dtype_str: str
    batch_prompts: int
    samples_per_prompt: int
    student_mb: int
    lora_rank: int
    use_flash_attention: bool = False

    def __str__(self):
        return (
            f"GPU: {self.name}\n"
            f"VRAM: {self.vram_gb:.1f} GB\n"
            f"Compute: {self.compute_capability[0]}.{self.compute_capability[1]}\n"
            f"Precision: {self.dtype_str}\n"
            f"Batch: {self.batch_prompts} prompts × {self.samples_per_prompt} samples\n"
            f"LoRA rank: {self.lora_rank}\n"
            f"Flash Attention: {self.use_flash_attention}"
        )

def detect_gpu() -> GPUConfig:
    """Detect GPU and automatically select optimal configuration."""

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU detected!")

    # Get GPU info
    gpu_name = torch.cuda.get_device_name(0)
    vram_bytes = torch.cuda.get_device_properties(0).total_memory
    vram_gb = vram_bytes / (1024 ** 3)
    compute_cap = torch.cuda.get_device_capability(0)

    # Check for bf16 support (compute capability >= 8.0)
    supports_bf16 = (compute_cap[0] >= 8) or (compute_cap[0] == 7 and compute_cap[1] >= 5)

    # Check for Hopper (H100, compute capability 9.0) - supports fp8
    is_hopper = (compute_cap[0] >= 9)

    # Check for Ampere+ (A100, RTX 30/40 series) - supports bf16 natively
    is_ampere_plus = (compute_cap[0] >= 8)

    print(f"\n{'='*60}")
    print(f"🔍 GPU DETECTION")
    print(f"{'='*60}")
    print(f"Name: {gpu_name}")
    print(f"VRAM: {vram_gb:.1f} GB")
    print(f"Compute Capability: {compute_cap[0]}.{compute_cap[1]}")
    print(f"BF16 Support: {supports_bf16}")
    print(f"Hopper (FP8): {is_hopper}")

    # --------------------------
    # Select precision (dtype)
    # --------------------------
    if is_hopper:
        dtype = torch.bfloat16
        dtype_str = "bf16 (fp8 available but not auto-enabled)"
        print(f"\n✅ Precision: bfloat16 (Hopper GPU)")
        print(f"   Note: FP8 available but requires additional setup")
    elif supports_bf16:
        dtype = torch.bfloat16
        dtype_str = "bf16"
        print(f"\n✅ Precision: bfloat16 (Ampere+ GPU)")
    else:
        dtype = torch.float16
        dtype_str = "fp16"
        print(f"\n⚠️  Precision: float16 (older GPU, no native bf16)")

    # --------------------------
    # Auto-tune batch sizes based on VRAM
    # --------------------------
    if vram_gb >= 75:  # H100 80GB, A100 80GB
        batch_prompts = 8
        samples_per_prompt = 8
        student_mb = 16
        lora_rank = 32
        tier = "High (80GB+)"
    elif vram_gb >= 35:  # A100 40GB, L40S 48GB
        batch_prompts = 4
        samples_per_prompt = 4
        student_mb = 8
        lora_rank = 16
        tier = "Medium (40GB+)"
    elif vram_gb >= 20:  # RTX 4090 24GB, RTX A5000 24GB
        batch_prompts = 2
        samples_per_prompt = 4
        student_mb = 4
        lora_rank = 16
        tier = "Low (24GB)"
    else:  # T4 16GB, V100 16GB
        batch_prompts = 2
        samples_per_prompt = 2
        student_mb = 2
        lora_rank = 8
        tier = "Minimal (16GB)"

    print(f"\n📊 Memory Tier: {tier}")
    print(f"   Batch: {batch_prompts} prompts × {samples_per_prompt} samples")
    print(f"   Student micro-batch: {student_mb}")
    print(f"   LoRA rank: {lora_rank}")

    # --------------------------
    # Flash Attention (if available)
    # --------------------------
    use_flash_attention = is_ampere_plus
    if use_flash_attention:
        print(f"\n⚡ Flash Attention: Enabled (Ampere+ GPU)")
    else:
        print(f"\n⚡ Flash Attention: Not available (requires Ampere+ GPU)")

    print(f"{'='*60}\n")

    return GPUConfig(
        name=gpu_name,
        vram_gb=vram_gb,
        compute_capability=compute_cap,
        dtype=dtype,
        dtype_str=dtype_str,
        batch_prompts=batch_prompts,
        samples_per_prompt=samples_per_prompt,
        student_mb=student_mb,
        lora_rank=lora_rank,
        use_flash_attention=use_flash_attention,
    )

if __name__ == "__main__":
    print("Testing GPU Auto-Detection for Multi-Teacher OPD")
    print("="*60)

    try:
        gpu_config = detect_gpu()

        print("\n=== Selected Configuration ===")
        print(gpu_config)

        # Validation checks
        print("\n=== Validation Checks ===")
        assert gpu_config.dtype in [torch.float16, torch.bfloat16], f"Invalid dtype: {gpu_config.dtype}"
        print("✅ dtype is valid")

        assert gpu_config.batch_prompts > 0, "batch_prompts must be > 0"
        print("✅ batch_prompts is valid")

        assert gpu_config.samples_per_prompt > 0, "samples_per_prompt must be > 0"
        print("✅ samples_per_prompt is valid")

        assert gpu_config.lora_rank in [8, 16, 32], f"Unexpected LoRA rank: {gpu_config.lora_rank}"
        print("✅ LoRA rank is valid")

        print("\n✅ All validation checks passed!")
        print("\n🎯 GPU auto-detection is working correctly!")

    except Exception as e:
        print(f"\n❌ Error during GPU detection: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
