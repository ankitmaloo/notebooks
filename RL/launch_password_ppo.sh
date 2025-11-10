#!/bin/bash

# Launch script for Password Game PPO Training
# Usage: ./launch_password_ppo.sh [config_name]

set -e

# Default config
CONFIG=${1:-"default"}

echo "========================================"
echo "Password Game PPO Training"
echo "========================================"
echo "Config: $CONFIG"
echo ""

# Activate environment if needed
# source /path/to/venv/bin/activate

# Set API keys (if not in environment)
# export WANDB_API_KEY="your_key_here"
# export HF_TOKEN="your_token_here"
# export OPENAI_API_KEY="your_key_here"

# Choose configuration
case $CONFIG in
    "default")
        echo "Running default configuration..."
        python verl_password_game_ppo.py \
            --model Qwen/Qwen2.5-0.6B \
            --epochs 5 \
            --episodes-per-epoch 100 \
            --batch-size 4 \
            --lr 5e-7 \
            --wandb-project verl-password-game \
            --seed 42
        ;;

    "quick-test")
        echo "Running quick test (3 epochs, fewer episodes)..."
        python verl_password_game_ppo.py \
            --model Qwen/Qwen2.5-0.6B \
            --epochs 3 \
            --episodes-per-epoch 50 \
            --batch-size 2 \
            --lr 5e-7 \
            --wandb-project verl-password-game-test \
            --seed 42
        ;;

    "high-lr")
        echo "Running with higher learning rate..."
        python verl_password_game_ppo.py \
            --model Qwen/Qwen2.5-0.6B \
            --epochs 5 \
            --episodes-per-epoch 100 \
            --batch-size 4 \
            --lr 1e-6 \
            --wandb-project verl-password-game-highlr \
            --seed 42
        ;;

    "no-thinking")
        echo "Running without thinking mode..."
        python verl_password_game_ppo.py \
            --model Qwen/Qwen2.5-0.6B \
            --epochs 5 \
            --episodes-per-epoch 100 \
            --batch-size 4 \
            --lr 5e-7 \
            --no-thinking \
            --wandb-project verl-password-game-nothink \
            --seed 42
        ;;

    "long-run")
        echo "Running extended training (10 epochs)..."
        python verl_password_game_ppo.py \
            --model Qwen/Qwen2.5-0.6B \
            --epochs 10 \
            --episodes-per-epoch 150 \
            --batch-size 4 \
            --lr 5e-7 \
            --wandb-project verl-password-game-long \
            --seed 42
        ;;

    *)
        echo "Unknown configuration: $CONFIG"
        echo ""
        echo "Available configurations:"
        echo "  default      - Standard 5-epoch training"
        echo "  quick-test   - Quick 3-epoch test run"
        echo "  high-lr      - Higher learning rate experiment"
        echo "  no-thinking  - Disable thinking mode"
        echo "  long-run     - Extended 10-epoch training"
        echo ""
        echo "Usage: ./launch_password_ppo.sh [config_name]"
        exit 1
        ;;
esac

echo ""
echo "========================================"
echo "Training complete!"
echo "========================================"
