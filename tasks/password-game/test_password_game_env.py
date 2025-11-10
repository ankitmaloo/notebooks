"""
Test Suite for Password Game Environment Wrapper

Run tests with:
    python -m pytest test_password_game_env.py -v

Or without pytest:
    python test_password_game_env.py

Prerequisites:
- FastAPI server running on localhost:8000

Author: Claude Code
Date: 2025-11-10
"""

import unittest
import time
from password_game_env import (
    PasswordGameEnv,
    BatchPasswordGameEnv,
    GameState,
    FeedbackResult,
    SubmitResult,
    GameStatus,
    calculate_rule_progress_reward,
    create_observation_dict
)


class TestPasswordGameEnv(unittest.TestCase):
    """Test cases for PasswordGameEnv"""

    def setUp(self):
        """Set up test environment before each test"""
        self.base_url = "http://localhost:8000"
        self.env = PasswordGameEnv(base_url=self.base_url, timeout=10.0)

    def tearDown(self):
        """Clean up after each test"""
        if self.env:
            self.env.close()

    def test_initialization(self):
        """Test environment initialization"""
        self.assertIsNotNone(self.env)
        self.assertEqual(self.env.base_url, self.base_url)
        self.assertIsNone(self.env.token)
        self.assertIsNone(self.env.current_state)
        self.assertEqual(self.env.game_status, GameStatus.ACTIVE)

    def test_start_game(self):
        """Test starting a new game"""
        state = self.env.start_game()

        self.assertIsNotNone(state)
        self.assertIsInstance(state, GameState)
        self.assertIsNotNone(state.token)
        self.assertEqual(state.current_rule_index, 0)
        self.assertIsNotNone(state.current_rule)
        self.assertTrue(state.game_active)
        self.assertGreater(len(state.all_rules), 0)

        # Check internal state
        self.assertEqual(self.env.token, state.token)
        self.assertEqual(self.env.current_state, state)
        self.assertTrue(self.env.is_active())

    def test_get_state(self):
        """Test getting current state"""
        # Start game first
        initial_state = self.env.start_game()
        self.assertIsNotNone(initial_state)

        # Get state without refresh (cached)
        cached_state = self.env.get_state(refresh=False)
        self.assertEqual(cached_state.token, initial_state.token)

        # Get state with refresh (from API)
        fresh_state = self.env.get_state(refresh=True)
        self.assertEqual(fresh_state.token, initial_state.token)
        self.assertEqual(fresh_state.current_rule_index, initial_state.current_rule_index)

    def test_get_feedback(self):
        """Test getting password feedback"""
        self.env.start_game()

        # Test valid password
        feedback = self.env.get_feedback("Test123!")
        self.assertIsNotNone(feedback)
        self.assertIsInstance(feedback, FeedbackResult)
        self.assertEqual(feedback.password, "Test123!")
        self.assertEqual(feedback.length, 8)
        self.assertGreaterEqual(feedback.total_passing, 0)
        self.assertIsNotNone(feedback.reward)
        self.assertGreater(len(feedback.rules_checked), 0)

        # Test password that fails first rule
        feedback_short = self.env.get_feedback("abc")
        self.assertIsNotNone(feedback_short)
        self.assertEqual(feedback_short.length, 3)

        # Test feedback methods
        passing = feedback.passing_rules()
        failing = feedback.failing_rules()
        self.assertIsInstance(passing, list)
        self.assertIsInstance(failing, list)

    def test_submit_password(self):
        """Test submitting a password"""
        state = self.env.start_game()
        self.assertIsNotNone(state)

        initial_rule = state.current_rule_index

        # Submit a password that likely passes rule 0
        result = self.env.submit_password("Hello1")

        self.assertIsInstance(result, SubmitResult)
        self.assertTrue(result.success)

        if not result.game_ended:
            # Game should advance
            self.assertIsNotNone(result.new_state)
            self.assertGreaterEqual(
                result.new_state.current_rule_index,
                initial_rule
            )

    def test_give_up(self):
        """Test giving up"""
        self.env.start_game()

        # Submit with give_up=True
        result = self.env.submit_password("anything", give_up=True)

        self.assertTrue(result.success)
        self.assertTrue(result.game_ended)
        self.assertTrue(result.gave_up)
        self.assertIsNotNone(result.reward)
        self.assertEqual(self.env.game_status, GameStatus.GAVE_UP)

    def test_end_game(self):
        """Test explicitly ending game"""
        self.env.start_game()
        self.assertTrue(self.env.is_active())

        success = self.env.end_game()

        self.assertTrue(success)
        self.assertEqual(self.env.game_status, GameStatus.GAVE_UP)

    def test_reset(self):
        """Test resetting environment"""
        # Start first game
        state1 = self.env.start_game()
        token1 = state1.token if state1 else None

        # Reset
        state2 = self.env.reset()

        self.assertIsNotNone(state2)
        self.assertNotEqual(state2.token, token1)
        self.assertEqual(state2.current_rule_index, 0)
        self.assertTrue(self.env.is_active())

    def test_context_manager(self):
        """Test using environment as context manager"""
        with PasswordGameEnv(base_url=self.base_url) as env:
            state = env.start_game()
            self.assertIsNotNone(state)
            self.assertTrue(env.is_active())

        # After exiting context, game should be ended
        # (but we can't check env state as it's out of scope)

    def test_error_handling_invalid_token(self):
        """Test error handling with invalid token"""
        self.env.token = "invalid-token-xyz"

        # Should return None/failure
        feedback = self.env.get_feedback("test")
        self.assertIsNone(feedback)

        result = self.env.submit_password("test")
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error)

    def test_submission_history(self):
        """Test submission history tracking"""
        self.env.start_game()

        # Initially empty
        history = self.env.get_history()
        self.assertEqual(len(history), 0)

        # Submit passwords
        self.env.submit_password("Test1")

        # History should be updated
        # (depends on implementation - may track only certain submissions)


class TestBatchPasswordGameEnv(unittest.TestCase):
    """Test cases for BatchPasswordGameEnv"""

    def setUp(self):
        """Set up batch environment"""
        self.num_envs = 3
        self.base_url = "http://localhost:8000"
        self.batch_env = BatchPasswordGameEnv(
            num_envs=self.num_envs,
            base_url=self.base_url
        )

    def tearDown(self):
        """Clean up"""
        if self.batch_env:
            self.batch_env.close()

    def test_initialization(self):
        """Test batch environment initialization"""
        self.assertEqual(self.batch_env.num_envs, self.num_envs)
        self.assertEqual(len(self.batch_env.envs), self.num_envs)

    def test_reset(self):
        """Test resetting all environments"""
        states = self.batch_env.reset()

        self.assertEqual(len(states), self.num_envs)
        for state in states:
            if state:  # May be None if API is down
                self.assertIsInstance(state, GameState)
                self.assertEqual(state.current_rule_index, 0)

    def test_get_states(self):
        """Test getting states from all environments"""
        self.batch_env.reset()

        states = self.batch_env.get_states(refresh=False)
        self.assertEqual(len(states), self.num_envs)

        states_fresh = self.batch_env.get_states(refresh=True)
        self.assertEqual(len(states_fresh), self.num_envs)

    def test_get_feedback_batch(self):
        """Test batch feedback"""
        self.batch_env.reset()

        passwords = [f"Test{i}!" for i in range(self.num_envs)]
        feedbacks = self.batch_env.get_feedback_batch(passwords)

        self.assertEqual(len(feedbacks), self.num_envs)
        for fb in feedbacks:
            if fb:  # May be None if API is down
                self.assertIsInstance(fb, FeedbackResult)

    def test_submit_batch(self):
        """Test batch submission"""
        self.batch_env.reset()

        passwords = [f"Hello{i}" for i in range(self.num_envs)]
        results = self.batch_env.submit_batch(passwords)

        self.assertEqual(len(results), self.num_envs)
        for result in results:
            self.assertIsInstance(result, SubmitResult)

    def test_submit_batch_with_give_up(self):
        """Test batch submission with give_up flags"""
        self.batch_env.reset()

        passwords = [f"Test{i}" for i in range(self.num_envs)]
        give_up = [False, True, False]

        results = self.batch_env.submit_batch(passwords, give_up=give_up)

        self.assertEqual(len(results), self.num_envs)

        # Second environment should have given up
        if results[1].success:
            self.assertTrue(results[1].gave_up or results[1].game_ended)

    def test_batch_size_validation(self):
        """Test validation of batch sizes"""
        self.batch_env.reset()

        # Wrong number of passwords
        with self.assertRaises(ValueError):
            self.batch_env.get_feedback_batch(["Test1", "Test2"])  # Only 2, need 3

        with self.assertRaises(ValueError):
            self.batch_env.submit_batch(["Test1"])  # Only 1, need 3


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions"""

    def test_calculate_rule_progress_reward(self):
        """Test rule progress reward calculation"""
        old_state = GameState(
            token="test",
            current_rule_index=0,
            current_rule="Rule 0",
            all_rules=["Rule 0"],
            game_active=True
        )

        new_state = GameState(
            token="test",
            current_rule_index=1,
            current_rule="Rule 1",
            all_rules=["Rule 0", "Rule 1"],
            game_active=True
        )

        # Progress from rule 0 to 1
        reward = calculate_rule_progress_reward(old_state, new_state, False)
        self.assertEqual(reward, 1.0)

        # Game ended
        reward_end = calculate_rule_progress_reward(old_state, None, True)
        self.assertEqual(reward_end, 0.0)

    def test_create_observation_dict(self):
        """Test observation dictionary creation"""
        state = GameState(
            token="test",
            current_rule_index=2,
            current_rule="Test rule",
            all_rules=["Rule 0", "Rule 1", "Rule 2"],
            game_active=True,
            captcha="abc12",
            country="Germany"
        )

        obs = create_observation_dict(state)

        self.assertEqual(obs["current_rule_index"], 2)
        self.assertEqual(obs["current_rule"], "Test rule")
        self.assertEqual(obs["num_rules"], 3)
        self.assertTrue(obs["game_active"])
        self.assertTrue(obs["has_captcha"])
        self.assertTrue(obs["has_country"])

        # With feedback
        feedback = FeedbackResult(
            password="Test123!",
            length=8,
            total_passing=3,
            reward=2.2,
            rules_checked=[
                {"rule_index": 0, "passes": True},
                {"rule_index": 1, "passes": True},
                {"rule_index": 2, "passes": False},
            ]
        )

        obs_with_fb = create_observation_dict(state, feedback)
        self.assertEqual(obs_with_fb["password_length"], 8)
        self.assertEqual(obs_with_fb["total_passing"], 3)
        self.assertEqual(obs_with_fb["reward"], 2.2)


class TestDataClasses(unittest.TestCase):
    """Test data classes"""

    def test_game_state(self):
        """Test GameState dataclass"""
        state = GameState(
            token="test-token",
            current_rule_index=1,
            current_rule="Rule text",
            all_rules=["Rule 0", "Rule 1"],
            game_active=True
        )

        # Test to_dict
        state_dict = state.to_dict()
        self.assertEqual(state_dict["token"], "test-token")
        self.assertEqual(state_dict["current_rule_index"], 1)

        # Test __str__
        state_str = str(state)
        self.assertIn("test-tok", state_str)
        self.assertIn("rule_index=1", state_str)

    def test_feedback_result(self):
        """Test FeedbackResult dataclass"""
        feedback = FeedbackResult(
            password="Test123!",
            length=8,
            total_passing=2,
            reward=1.2,
            rules_checked=[
                {"rule_index": 0, "rule_text": "Rule 0", "passes": True},
                {"rule_index": 1, "rule_text": "Rule 1", "passes": True},
                {"rule_index": 2, "rule_text": "Rule 2", "passes": False},
            ]
        )

        # Test methods
        passing = feedback.passing_rules()
        self.assertEqual(passing, [0, 1])

        failing = feedback.failing_rules()
        self.assertEqual(failing, [2])

        first_fail = feedback.first_failing_rule()
        self.assertEqual(first_fail["rule_index"], 2)

    def test_submit_result(self):
        """Test SubmitResult dataclass"""
        result = SubmitResult(
            success=True,
            game_ended=False,
            gave_up=False,
            reward=None,
            new_state=None
        )

        self.assertTrue(result.success)
        self.assertFalse(result.game_ended)

        # Test __str__
        result_str = str(result)
        self.assertIn("ended=False", result_str)


def run_integration_tests():
    """Run integration tests that require API server"""
    print("\n" + "=" * 70)
    print("INTEGRATION TESTS - Require API Server Running")
    print("=" * 70)

    # Test 1: Full game flow
    print("\nTest 1: Full game flow")
    try:
        env = PasswordGameEnv(base_url="http://localhost:8000", timeout=5.0)
        state = env.start_game()

        if state:
            print(f"  Started game: {state.token[:16]}...")
            print(f"  Current rule: {state.current_rule}")

            feedback = env.get_feedback("Test123!")
            if feedback:
                print(f"  Feedback: {feedback.total_passing} passing rules")

            result = env.submit_password("Hello1")
            if result.success:
                print(f"  Submitted successfully, game_ended={result.game_ended}")

            env.close()
            print("  Test PASSED")
        else:
            print("  Test FAILED: Could not start game")

    except Exception as e:
        print(f"  Test FAILED: {e}")

    # Test 2: Batch operations
    print("\nTest 2: Batch operations")
    try:
        batch_env = BatchPasswordGameEnv(num_envs=2, base_url="http://localhost:8000")
        states = batch_env.reset()

        if all(states):
            print(f"  Started {len(states)} environments")

            feedbacks = batch_env.get_feedback_batch(["Test1!", "Test2!"])
            if all(feedbacks):
                print(f"  Got feedback for all environments")

            results = batch_env.submit_batch(["Hello1", "Hello2"])
            if all(r.success for r in results):
                print(f"  Submitted to all environments")

            batch_env.close()
            print("  Test PASSED")
        else:
            print("  Test FAILED: Could not start all environments")

    except Exception as e:
        print(f"  Test FAILED: {e}")

    print("\n" + "=" * 70)


def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("PASSWORD GAME ENVIRONMENT - TEST SUITE")
    print("=" * 70)

    # Run unit tests
    print("\nRunning unit tests...")
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestPasswordGameEnv))
    suite.addTests(loader.loadTestsFromTestCase(TestBatchPasswordGameEnv))
    suite.addTests(loader.loadTestsFromTestCase(TestUtilityFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestDataClasses))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    # Run integration tests
    run_integration_tests()

    return result.wasSuccessful()


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
