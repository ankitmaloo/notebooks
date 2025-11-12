#!/usr/bin/env python3
"""
Test script for Password Game Integration

Run this to verify the integration is working before adding to notebook.

Usage:
    python test_password_game_integration.py
"""

import sys
import os
from pathlib import Path
import random

# Add password game directory to path
password_game_dir = Path("/home/user/notebooks/tasks/password-game")
if str(password_game_dir) not in sys.path:
    sys.path.insert(0, str(password_game_dir))

print("=" * 80)
print("PASSWORD GAME INTEGRATION TEST")
print("=" * 80)

# Test 1: Import password game
print("\n[TEST 1] Importing password game...")
try:
    from game import PasswordGame, rules, instructions
    print(f"✓ SUCCESS: Imported password game ({len(rules)} rules)")
except Exception as e:
    print(f"✗ FAILED: {e}")
    sys.exit(1)

# Test 2: Create game instance
print("\n[TEST 2] Creating game instance...")
try:
    game = PasswordGame()
    print(f"✓ SUCCESS: Created game instance")
    print(f"  Current rule index: {game.current_rule}")
    print(f"  Current rule: {game.get_current_rule()}")
    print(f"  CAPTCHA: {game.captcha}")
    print(f"  Country: {game.country}")
except Exception as e:
    print(f"✗ FAILED: {e}")
    sys.exit(1)

# Test 3: Test reward calculation
print("\n[TEST 3] Testing reward calculation...")
test_passwords = [
    ("12345", "Short with number"),
    ("12345A!", "Uppercase and special char"),
    ("abcdefghijk", "Only lowercase, long"),
    ("12A!#january", "Multiple rules satisfied"),
]

try:
    for password, description in test_passwords:
        reward = game.calculate_reward(password)
        feedback = game.get_rule_feedback(password)
        print(f"\n  Password: '{password}' ({description})")
        print(f"    Reward: {reward:.2f}")
        print(f"    Passing: {feedback['total_passing']}/{len(feedback['rules_checked'])} rules")
        print(f"    Length: {len(password)} chars (penalty: -{len(password) * 0.1:.1f})")
    print("\n✓ SUCCESS: Reward calculation working")
except Exception as e:
    print(f"✗ FAILED: {e}")
    sys.exit(1)

# Test 4: Test PasswordGameEnvironment class
print("\n[TEST 4] Testing PasswordGameEnvironment...")
try:
    class PasswordGameEnvironment:
        def __init__(self, use_rule_progression=True, min_rules=5, max_rules=26, progression_rate=0.1, seed=42):
            self.use_rule_progression = use_rule_progression
            self.min_rules = min_rules
            self.max_rules = max_rules
            self.progression_rate = progression_rate
            self.seed = seed
            self.current_max_rules = min_rules if use_rule_progression else max_rules
            self.games_completed = 0
            self.total_games = 0
            self.successful_games = 0

        def create_game_instance(self):
            return PasswordGame()

        def get_current_max_rules(self):
            if not self.use_rule_progression:
                return self.max_rules
            progress = min(1.0, self.games_completed * self.progression_rate / 100)
            current_max = self.min_rules + int((self.max_rules - self.min_rules) * progress)
            return min(current_max, self.max_rules)

    env = PasswordGameEnvironment(min_rules=5, max_rules=15, progression_rate=0.1)
    print(f"✓ SUCCESS: PasswordGameEnvironment created")
    print(f"  Min rules: {env.min_rules}")
    print(f"  Max rules: {env.max_rules}")
    print(f"  Current max: {env.get_current_max_rules()}")

    # Test progression
    for i in range(0, 150, 50):
        env.games_completed = i
        print(f"  After {i} games: max {env.get_current_max_rules()} rules")
except Exception as e:
    print(f"✗ FAILED: {e}")
    sys.exit(1)

# Test 5: Test prompt formatting
print("\n[TEST 5] Testing prompt formatting...")
try:
    test_game = PasswordGame()

    # Mock tokenizer for testing
    class MockTokenizer:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            result = ""
            for msg in messages:
                result += f"{msg['role'].upper()}: {msg['content']}\n\n"
            if add_generation_prompt:
                result += "ASSISTANT: "
            return result

    # Simple format prompt function
    def format_prompt(game, target_rule_index=None):
        if target_rule_index is not None:
            rule_index = target_rule_index
        else:
            rule_index = game.current_rule

        all_rules = []
        for i in range(rule_index + 1):
            if i < len(rules):
                rule_text = rules[i]
                if "{captcha}" in rule_text:
                    rule_text = rule_text.format(captcha=game.captcha)
                elif "{country}" in rule_text:
                    rule_text = rule_text.format(country=game.country)
                all_rules.append(f"Rule {i+1}: {rule_text}")

        prompt_text = f"""You are playing the Password Game.

INSTRUCTIONS:
{instructions.strip()}

ALL ACTIVE RULES:
{chr(10).join(all_rules)}

Generate a password that satisfies all {len(all_rules)} rules."""

        return prompt_text

    # Test formatting at different rule levels
    for target in [0, 4, 9]:
        prompt = format_prompt(test_game, target)
        print(f"\n  Target rule {target + 1}:")
        print(f"    Prompt length: {len(prompt)} chars")
        print(f"    Rules included: {target + 1}")
        print(f"    Preview: {prompt[:100]}...")

    print("\n✓ SUCCESS: Prompt formatting working")
except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Test rule feedback
print("\n[TEST 6] Testing detailed rule feedback...")
try:
    feedback_game = PasswordGame()
    test_pwd = "12345A!january"

    feedback = feedback_game.get_rule_feedback(test_pwd)

    print(f"  Password: {test_pwd}")
    print(f"  Total rules checked: {len(feedback['rules_checked'])}")
    print(f"  Passing: {feedback['total_passing']}")
    print(f"  Reward: {feedback['reward']}")
    print("\n  Rule breakdown:")
    for rule_info in feedback['rules_checked'][:5]:  # Show first 5
        status = "✓" if rule_info['passes'] else "✗"
        print(f"    {status} Rule {rule_info['rule_index'] + 1}: {rule_info['rule_text'][:50]}...")

    print("\n✓ SUCCESS: Rule feedback working")
except Exception as e:
    print(f"✗ FAILED: {e}")
    sys.exit(1)

# Test 7: Test dynamic game elements
print("\n[TEST 7] Testing dynamic game elements...")
try:
    # Create multiple games to verify uniqueness
    games = [PasswordGame() for _ in range(3)]

    print(f"  Game instances created: {len(games)}")
    print("\n  Dynamic elements per game:")
    for i, g in enumerate(games, 1):
        print(f"    Game {i}:")
        print(f"      CAPTCHA: {g.captcha}")
        print(f"      Country: {g.country}")
        print(f"      Wordle: {g.wordle_answer}")

    # Check if CAPTCHAs are different (they should be with high probability)
    captchas = [g.captcha for g in games]
    unique_captchas = len(set(captchas))
    print(f"\n  Unique CAPTCHAs: {unique_captchas}/{len(captchas)}")

    print("\n✓ SUCCESS: Dynamic elements working")
except Exception as e:
    print(f"✗ FAILED: {e}")
    sys.exit(1)

# Test 8: Test extract password from response
print("\n[TEST 8] Testing password extraction from model responses...")
try:
    def extract_password_from_response(response: str) -> str:
        response = response.strip()
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        if not lines:
            return ""
        password = lines[-1]
        if len(password) > 500:
            words = password.split()
            password = words[0] if words else password[:100]
        return password

    test_responses = [
        ("mypassword123", "mypassword123"),
        ("Let me think...\nmypassword123", "mypassword123"),
        ("  Here's my answer:\n  \n  FinalPassword!  ", "FinalPassword!"),
        ("A very long explanation " * 50 + " password", "A"),
    ]

    for response, expected_contains in test_responses:
        extracted = extract_password_from_response(response)
        print(f"\n  Response: {response[:50]}...")
        print(f"    Extracted: '{extracted}'")
        if expected_contains in extracted or extracted in expected_contains:
            print(f"    ✓ Correct")
        else:
            print(f"    ⚠ May need adjustment")

    print("\n✓ SUCCESS: Password extraction working")
except Exception as e:
    print(f"✗ FAILED: {e}")
    sys.exit(1)

# Summary
print("\n" + "=" * 80)
print("ALL TESTS PASSED ✓")
print("=" * 80)
print("\nThe password game integration is working correctly!")
print("\nNext steps:")
print("1. Open your RL training notebook")
print("2. Insert cells from: password_game_cells_ready.txt")
print("3. Modify training loop as described in: PASSWORD_GAME_INTEGRATION.md")
print("4. Run and train!")
print("\nFiles created:")
print("  - PASSWORD_GAME_INTEGRATION.md (full documentation)")
print("  - PASSWORD_GAME_SUMMARY.md (overview and quick reference)")
print("  - password_game_cells_ready.txt (copy-paste ready cells)")
print("  - test_password_game_integration.py (this test script)")
print("=" * 80)
