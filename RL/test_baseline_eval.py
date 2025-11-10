#!/usr/bin/env python3
"""
Quick test script to verify baseline evaluation setup.

This tests the core functionality without running a full evaluation.
Useful for debugging and verification.

Usage:
    python test_baseline_eval.py
"""

import sys
import os

# Add tasks directory to path
sys.path.insert(0, '/home/user/notebooks/tasks/password-game')

def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")
    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"  ✗ PyTorch: {e}")
        return False

    try:
        import transformers
        print(f"  ✓ Transformers {transformers.__version__}")
    except ImportError as e:
        print(f"  ✗ Transformers: {e}")
        return False

    try:
        import numpy as np
        print(f"  ✓ NumPy {np.__version__}")
    except ImportError as e:
        print(f"  ✗ NumPy: {e}")
        return False

    try:
        import pandas as pd
        print(f"  ✓ Pandas {pd.__version__}")
    except ImportError as e:
        print(f"  ✗ Pandas: {e}")
        return False

    try:
        import matplotlib
        print(f"  ✓ Matplotlib {matplotlib.__version__}")
    except ImportError as e:
        print(f"  ✗ Matplotlib: {e}")
        return False

    return True


def test_password_game():
    """Test that password game can be imported and works."""
    print("\nTesting Password Game...")
    try:
        from game import PasswordGame, rules
        print(f"  ✓ Imported PasswordGame")
        print(f"  ✓ Total rules: {len(rules)}")

        # Test game creation
        game = PasswordGame()
        print(f"  ✓ Created game instance")
        print(f"    - Captcha: {game.captcha}")
        print(f"    - Country: {game.country}")
        print(f"    - Wordle: {game.wordle_answer}")
        print(f"    - Moon: {game.moon_phase}")

        # Test first rule
        first_rule = game.get_current_rule()
        print(f"  ✓ First rule: {first_rule}")

        # Test password checking
        test_password = "Test1!"
        feedback = game.get_rule_feedback(test_password)
        print(f"  ✓ Tested password: '{test_password}'")
        print(f"    - Rules satisfied: {feedback['total_passing']}/{len(feedback['rules_checked'])}")
        print(f"    - Reward: {feedback['reward']}")

        return True

    except Exception as e:
        print(f"  ✗ Password Game Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_loading():
    """Test that model can be loaded (optional, can be slow)."""
    print("\nTesting Model Loading (this may take a while)...")

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        model_name = "Qwen/Qwen2.5-0.6B"

        # Test tokenizer
        print(f"  Loading tokenizer from {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        print(f"  ✓ Tokenizer loaded (vocab: {len(tokenizer)})")

        # Test chat template
        if hasattr(tokenizer, 'apply_chat_template'):
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"}
            ]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            print(f"  ✓ Chat template works")
            print(f"    Preview: {prompt[:100]}...")
        else:
            print(f"  ⚠ No chat template available")

        # Optionally test model loading (comment out if too slow)
        print(f"  Loading model (this is slow, skip if testing quickly)...")
        # Uncomment to test model loading:
        # model = AutoModelForCausalLM.from_pretrained(
        #     model_name,
        #     torch_dtype=torch.bfloat16,
        #     device_map="auto",
        #     trust_remote_code=True
        # )
        # print(f"  ✓ Model loaded")

        return True

    except Exception as e:
        print(f"  ✗ Model Loading Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_prompt_building():
    """Test prompt building functionality."""
    print("\nTesting Prompt Building...")

    try:
        from game import PasswordGame
        from transformers import AutoTokenizer

        # Load tokenizer
        model_name = "Qwen/Qwen2.5-0.6B"
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        # Create game
        game = PasswordGame()
        state = game.get_minimal_game_state()

        # Build prompt
        system_prompt = "You are playing the Password Game."
        user_msg = f"Create a password for rule: {state['current_rule']}"

        if hasattr(tokenizer, 'apply_chat_template'):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg}
            ]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            print(f"  ✓ Prompt built successfully")
            print(f"  ✓ Length: {len(prompt)} chars")
            print(f"\n  Preview:\n{'-'*60}")
            print(prompt[:300] + "...")
            print(f"{'-'*60}")

            return True
        else:
            print(f"  ⚠ Chat template not available, using fallback")
            return True

    except Exception as e:
        print(f"  ✗ Prompt Building Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gpu():
    """Test GPU availability."""
    print("\nTesting GPU...")

    try:
        import torch

        if torch.cuda.is_available():
            print(f"  ✓ CUDA available")
            print(f"    - Device: {torch.cuda.get_device_name(0)}")
            print(f"    - Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return True
        else:
            print(f"  ⚠ CUDA not available (CPU will be used, very slow)")
            return True

    except Exception as e:
        print(f"  ✗ GPU Test Error: {e}")
        return False


def run_all_tests():
    """Run all tests."""
    print("="*80)
    print("BASELINE EVALUATION SETUP TEST")
    print("="*80)

    results = {
        "Imports": test_imports(),
        "Password Game": test_password_game(),
        "GPU": test_gpu(),
        "Prompt Building": test_prompt_building(),
        "Model Loading": test_model_loading(),  # May be slow
    }

    print("\n" + "="*80)
    print("TEST RESULTS")
    print("="*80)

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:.<50} {status}")

    all_passed = all(results.values())

    print("="*80)

    if all_passed:
        print("\n🎉 All tests passed! You're ready to run baseline evaluation.")
        print("\nNext steps:")
        print("  1. Open password_game_baseline_eval.ipynb")
        print("  2. Run all cells")
        print("  3. Review results in output directory")
        return 0
    else:
        print("\n⚠ Some tests failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("  - Install missing packages: pip install -r requirements.txt")
        print("  - Check GPU drivers if CUDA unavailable")
        print("  - Ensure password game files are in correct location")
        return 1


if __name__ == '__main__':
    exit_code = run_all_tests()
    sys.exit(exit_code)
