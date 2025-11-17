#!/usr/bin/env python3
"""
Quick model download utility with progress tracking.
Uses hf_transfer for fast downloads.
"""

import argparse
import os
import sys


def download_model(model_id: str, revision: str = "main", token: str = None):
    """Download a model from HuggingFace Hub."""

    # Enable fast downloads
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub not installed")
        print("Run: uv pip install huggingface_hub[cli] hf_transfer")
        sys.exit(1)

    print(f"Downloading: {model_id}")
    print(f"Revision: {revision}")

    try:
        path = snapshot_download(
            repo_id=model_id,
            revision=revision,
            token=token,
            resume_download=True,
        )
        print(f"\nDownload complete!")
        print(f"Model path: {path}")
        return path
    except Exception as e:
        print(f"\nERROR: {e}")
        if "401" in str(e) or "403" in str(e):
            print("\nAccess denied. For gated models:")
            print("1. Accept model license on HuggingFace")
            print("2. Set HF_TOKEN or pass --token")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Download HuggingFace models")
    parser.add_argument("model_id", help="Model ID (e.g., meta-llama/Llama-2-7b-hf)")
    parser.add_argument("--revision", default="main", help="Model revision/branch")
    parser.add_argument("--token", default=None, help="HuggingFace token (or set HF_TOKEN)")

    args = parser.parse_args()

    # Use env token if not provided
    token = args.token or os.environ.get("HF_TOKEN")

    download_model(args.model_id, args.revision, token)


if __name__ == "__main__":
    main()
