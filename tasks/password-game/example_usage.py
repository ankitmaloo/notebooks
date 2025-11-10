"""
Example Usage of Password Game Environment Wrapper

This script demonstrates various use cases of the PasswordGameEnv wrapper,
including integration patterns for RL training.

Prerequisites:
- FastAPI server running on localhost:8000
- Run: uvicorn main:app --reload

Author: Claude Code
Date: 2025-11-10
"""

import time
from password_game_env import (
    PasswordGameEnv,
    BatchPasswordGameEnv,
    calculate_rule_progress_reward,
    create_observation_dict
)


def example_1_basic_usage():
    """Example 1: Basic single-environment usage"""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Basic Usage")
    print("=" * 70)

    env = PasswordGameEnv(base_url="http://localhost:8000")

    # Start a new game
    state = env.start_game()
    if not state:
        print("Failed to start game. Is the API server running?")
        return

    print(f"\nGame started successfully!")
    print(f"Token: {state.token[:16]}...")
    print(f"Current rule (index {state.current_rule_index}): {state.current_rule}")

    # Try a few passwords
    test_passwords = [
        "abc",           # Too short
        "Hello",         # No number
        "Hello1",        # Need uppercase and special char
        "Hello1!",       # This should pass rule 0-3
    ]

    for password in test_passwords:
        print(f"\nTesting password: '{password}'")
        feedback = env.get_feedback(password)

        if feedback:
            print(f"  Length: {feedback.length}")
            print(f"  Passing rules: {feedback.total_passing}/{len(feedback.rules_checked)}")
            print(f"  Reward: {feedback.reward}")

            failing = feedback.first_failing_rule()
            if failing:
                print(f"  First failing: Rule {failing['rule_index']}: {failing['rule_text']}")
            else:
                print(f"  All rules passed!")

    # Clean up
    env.close()
    print("\nEnvironment closed.")


def example_2_game_progression():
    """Example 2: Playing through multiple rules"""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Game Progression")
    print("=" * 70)

    with PasswordGameEnv(base_url="http://localhost:8000") as env:
        state = env.start_game()
        if not state:
            print("Failed to start game.")
            return

        # Track progress
        rule_index = 0
        max_rules = 5  # Limit for demo

        while env.is_active() and rule_index < max_rules:
            print(f"\n--- Rule {state.current_rule_index} ---")
            print(f"Rule: {state.current_rule}")

            # For demo, we'll submit progressively better passwords
            if rule_index == 0:
                password = "Hello"
            elif rule_index == 1:
                password = "Hello1"
            elif rule_index == 2:
                password = "Hello1A"
            elif rule_index == 3:
                password = "Hello1A!"
            else:
                password = "Hello1A!9999"  # Digits sum to 25+

            print(f"Submitting: '{password}'")

            # Get feedback first
            feedback = env.get_feedback(password)
            if feedback:
                print(f"  Passing: {feedback.total_passing}/{len(feedback.rules_checked)}")

            # Submit password
            result = env.submit_password(password)

            if result.success:
                if result.game_ended:
                    print(f"\nGame ended! Final reward: {result.reward}")
                    break
                else:
                    state = result.new_state
                    rule_index += 1
            else:
                print(f"Submission failed: {result.error}")
                break

        print(f"\nCompleted {rule_index} rules")


def example_3_batch_operations():
    """Example 3: Batch operations for parallel training"""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Batch Operations")
    print("=" * 70)

    num_envs = 4
    batch_env = BatchPasswordGameEnv(
        num_envs=num_envs,
        base_url="http://localhost:8000"
    )

    # Reset all environments
    print(f"\nStarting {num_envs} parallel environments...")
    states = batch_env.reset()

    # Check states
    active_count = sum(1 for s in states if s and s.game_active)
    print(f"Active environments: {active_count}/{num_envs}")

    # Print current rules
    print("\nCurrent rules for each environment:")
    for i, state in enumerate(states):
        if state:
            print(f"  Env {i}: Rule {state.current_rule_index}: {state.current_rule}")

    # Test batch feedback
    test_passwords = ["Test1!", "Hello2@", "World3#", "Python4$"]
    print(f"\nGetting feedback for batch passwords...")

    feedbacks = batch_env.get_feedback_batch(test_passwords)

    for i, (pwd, fb) in enumerate(zip(test_passwords, feedbacks)):
        if fb:
            print(f"  Env {i}: '{pwd}' -> {fb.total_passing} passing, reward={fb.reward}")

    # Submit batch
    print(f"\nSubmitting passwords...")
    results = batch_env.submit_batch(test_passwords)

    for i, result in enumerate(results):
        if result.success:
            status = "ended" if result.game_ended else "continuing"
            print(f"  Env {i}: {status}")

    # Clean up
    batch_env.close()
    print("\nBatch environment closed.")


def example_4_error_handling():
    """Example 4: Error handling and edge cases"""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Error Handling")
    print("=" * 70)

    # Test with invalid URL
    print("\nTest 1: Invalid API URL")
    env = PasswordGameEnv(
        base_url="http://localhost:9999",  # Wrong port
        timeout=2.0,
        max_retries=1
    )

    state = env.start_game()
    if state:
        print("  Unexpectedly succeeded")
    else:
        print("  Correctly handled connection error")

    # Test with valid URL
    print("\nTest 2: Valid API but invalid token")
    env2 = PasswordGameEnv(base_url="http://localhost:8000")
    env2.token = "invalid-token-12345"

    feedback = env2.get_feedback("test")
    if feedback:
        print("  Unexpectedly succeeded")
    else:
        print("  Correctly handled invalid token")

    # Test context manager
    print("\nTest 3: Context manager cleanup")
    try:
        with PasswordGameEnv(base_url="http://localhost:8000") as env3:
            state = env3.start_game()
            if state:
                print(f"  Started game with token: {state.token[:16]}...")
            # Context manager will auto-close
        print("  Context manager cleanup successful")
    except Exception as e:
        print(f"  Error: {e}")

    env.close()
    env2.close()


def example_5_rl_integration():
    """Example 5: Integration pattern for RL training"""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: RL Training Integration Pattern")
    print("=" * 70)

    class SimpleRLAgent:
        """Dummy RL agent for demonstration"""

        def select_action(self, observation):
            """Select action based on observation"""
            # In real RL: use policy network
            # Here: simple heuristic based on rule index
            rule_index = observation.get("current_rule_index", 0)

            if rule_index == 0:
                return "Hello"
            elif rule_index == 1:
                return "Hello1"
            elif rule_index == 2:
                return "Hello1A"
            elif rule_index == 3:
                return "Hello1A!"
            else:
                return "Hello1A!9999"

        def update(self, observation, action, reward, next_observation, done):
            """Update agent (dummy)"""
            print(f"    [Agent] Updated with reward: {reward:.2f}")

    # Initialize
    env = PasswordGameEnv(base_url="http://localhost:8000")
    agent = SimpleRLAgent()

    print("\nRunning RL training loop simulation...")

    # Training loop
    num_episodes = 2
    max_steps_per_episode = 3

    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1}/{num_episodes} ---")

        # Reset environment
        state = env.reset()
        if not state:
            print("Failed to reset environment")
            break

        episode_reward = 0
        step = 0

        while env.is_active() and step < max_steps_per_episode:
            # Create observation
            observation = create_observation_dict(state)

            # Agent selects action
            password = agent.select_action(observation)
            print(f"  Step {step + 1}: Submitting '{password}'")

            # Get feedback before submitting (for shaping reward)
            feedback = env.get_feedback(password)

            # Execute action
            result = env.submit_password(password)

            if result.success:
                # Calculate reward
                if result.game_ended:
                    reward = result.reward or 0.0
                    done = True
                    next_observation = None
                else:
                    # Shaped reward based on progress
                    progress_reward = calculate_rule_progress_reward(
                        state, result.new_state, False
                    )
                    reward = progress_reward
                    done = False
                    next_observation = create_observation_dict(result.new_state, feedback)

                episode_reward += reward

                # Update agent
                agent.update(observation, password, reward, next_observation, done)

                if done:
                    print(f"  Episode ended. Total reward: {episode_reward:.2f}")
                    break

                # Update state
                state = result.new_state
                step += 1
            else:
                print(f"  Submission failed: {result.error}")
                break

    env.close()
    print("\nRL training simulation completed.")


def example_6_performance_test():
    """Example 6: Performance and throughput test"""
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Performance Test")
    print("=" * 70)

    num_requests = 10

    # Test single environment
    print(f"\nTest 1: Single environment ({num_requests} feedback requests)")
    env = PasswordGameEnv(base_url="http://localhost:8000")
    state = env.start_game()

    if state:
        start_time = time.time()
        for i in range(num_requests):
            env.get_feedback(f"Test{i}!")
        elapsed = time.time() - start_time

        print(f"  Completed in {elapsed:.2f}s")
        print(f"  Throughput: {num_requests / elapsed:.2f} requests/sec")

    env.close()

    # Test batch environment
    num_envs = 4
    print(f"\nTest 2: Batch environment ({num_envs} parallel envs)")

    batch_env = BatchPasswordGameEnv(num_envs=num_envs, base_url="http://localhost:8000")
    batch_env.reset()

    passwords = [f"Test{i}!" for i in range(num_envs)]

    start_time = time.time()
    for i in range(num_requests // num_envs):
        batch_env.get_feedback_batch(passwords)
    elapsed = time.time() - start_time

    total_requests = (num_requests // num_envs) * num_envs
    print(f"  Completed {total_requests} requests in {elapsed:.2f}s")
    print(f"  Throughput: {total_requests / elapsed:.2f} requests/sec")

    batch_env.close()


def main():
    """Run all examples"""
    print("\n" + "=" * 70)
    print("PASSWORD GAME ENVIRONMENT - USAGE EXAMPLES")
    print("=" * 70)
    print("\nMake sure the FastAPI server is running:")
    print("  $ cd /home/user/notebooks/tasks/password-game")
    print("  $ uvicorn main:app --reload")
    print()

    try:
        # Run examples
        example_1_basic_usage()
        example_2_game_progression()
        example_3_batch_operations()
        example_4_error_handling()
        example_5_rl_integration()
        example_6_performance_test()

        print("\n" + "=" * 70)
        print("ALL EXAMPLES COMPLETED!")
        print("=" * 70)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\n\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
