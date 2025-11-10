"""
Password Game Environment Wrapper

A comprehensive Python class for interfacing with the Password Game API.
Designed for RL training with robust error handling and batch operations support.

Author: Claude Code
Date: 2025-11-10
"""

import requests
import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GameStatus(Enum):
    """Enumeration for game status"""
    ACTIVE = "active"
    COMPLETED = "completed"
    GAVE_UP = "gave_up"
    ERROR = "error"
    TIMEOUT = "timeout"


@dataclass
class GameState:
    """Data class representing the current game state"""
    token: str
    current_rule_index: int
    current_rule: Optional[str]
    all_rules: List[str]
    game_active: bool
    captcha: Optional[str] = None
    country: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)

    def __str__(self) -> str:
        """String representation"""
        return (f"GameState(token={self.token[:8]}..., "
                f"rule_index={self.current_rule_index}, "
                f"active={self.game_active})")


@dataclass
class FeedbackResult:
    """Data class for password feedback"""
    password: str
    length: int
    total_passing: int
    reward: float
    rules_checked: List[Dict]

    def passing_rules(self) -> List[int]:
        """Get indices of passing rules"""
        return [r["rule_index"] for r in self.rules_checked if r["passes"]]

    def failing_rules(self) -> List[int]:
        """Get indices of failing rules"""
        return [r["rule_index"] for r in self.rules_checked if not r["passes"]]

    def first_failing_rule(self) -> Optional[Dict]:
        """Get first failing rule"""
        for rule in self.rules_checked:
            if not rule["passes"]:
                return rule
        return None


@dataclass
class SubmitResult:
    """Data class for submission results"""
    success: bool
    game_ended: bool
    gave_up: bool
    reward: Optional[float]
    new_state: Optional[GameState]
    error: Optional[str] = None

    def __str__(self) -> str:
        if self.error:
            return f"SubmitResult(error={self.error})"
        return f"SubmitResult(ended={self.game_ended}, reward={self.reward})"


class PasswordGameEnv:
    """
    Password Game Environment Wrapper

    A comprehensive wrapper for the Password Game API that handles:
    - Game session management
    - API communication with retries and timeouts
    - State tracking and caching
    - Batch operations for RL training
    - Comprehensive error handling

    Usage:
        env = PasswordGameEnv(base_url="http://localhost:8000")
        state = env.start_game()
        feedback = env.get_feedback("MyPassword123!")
        result = env.submit_password("MyPassword123!")
        env.reset()
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        verify_ssl: bool = True
    ):
        """
        Initialize the Password Game Environment

        Args:
            base_url: Base URL of the FastAPI server
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts for failed requests
            retry_delay: Delay between retries in seconds
            verify_ssl: Whether to verify SSL certificates
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.verify_ssl = verify_ssl

        # State tracking
        self.token: Optional[str] = None
        self.current_state: Optional[GameState] = None
        self.game_status: GameStatus = GameStatus.ACTIVE
        self.submission_history: List[Tuple[str, float]] = []

        # Session for connection pooling
        self.session = requests.Session()

        logger.info(f"Initialized PasswordGameEnv with base_url: {base_url}")

    def _make_request(
        self,
        method: str,
        endpoint: str,
        json_data: Optional[Dict] = None,
        retries: Optional[int] = None
    ) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """
        Make HTTP request with retry logic

        Args:
            method: HTTP method (GET, POST)
            endpoint: API endpoint
            json_data: JSON data for POST requests
            retries: Number of retries (uses self.max_retries if None)

        Returns:
            Tuple of (success, response_data, error_message)
        """
        if retries is None:
            retries = self.max_retries

        url = f"{self.base_url}{endpoint}"
        last_error = None

        for attempt in range(retries + 1):
            try:
                if method.upper() == "GET":
                    response = self.session.get(
                        url,
                        timeout=self.timeout,
                        verify=self.verify_ssl
                    )
                elif method.upper() == "POST":
                    response = self.session.post(
                        url,
                        json=json_data,
                        timeout=self.timeout,
                        verify=self.verify_ssl
                    )
                else:
                    return False, None, f"Unsupported HTTP method: {method}"

                # Check for HTTP errors
                response.raise_for_status()

                # Parse JSON response
                data = response.json()
                return True, data, None

            except requests.exceptions.Timeout as e:
                last_error = f"Request timeout: {str(e)}"
                logger.warning(f"Attempt {attempt + 1}/{retries + 1} failed: {last_error}")

            except requests.exceptions.ConnectionError as e:
                last_error = f"Connection error: {str(e)}"
                logger.warning(f"Attempt {attempt + 1}/{retries + 1} failed: {last_error}")

            except requests.exceptions.HTTPError as e:
                last_error = f"HTTP error: {e.response.status_code} - {e.response.text}"
                logger.error(f"HTTP error on attempt {attempt + 1}: {last_error}")
                # Don't retry on 404 or 4xx client errors
                if e.response.status_code == 404 or 400 <= e.response.status_code < 500:
                    return False, None, last_error

            except Exception as e:
                last_error = f"Unexpected error: {str(e)}"
                logger.error(f"Unexpected error on attempt {attempt + 1}: {last_error}")

            # Wait before retry (except on last attempt)
            if attempt < retries:
                time.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff

        # All retries exhausted
        logger.error(f"All {retries + 1} attempts failed for {endpoint}")
        return False, None, last_error

    def start_game(self) -> Optional[GameState]:
        """
        Start a new game session

        Returns:
            GameState object if successful, None otherwise
        """
        logger.info("Starting new game...")

        success, data, error = self._make_request("POST", "/start")

        if not success:
            logger.error(f"Failed to start game: {error}")
            self.game_status = GameStatus.ERROR
            return None

        # Extract game state
        self.token = data.get("token")
        self.current_state = GameState(
            token=self.token,
            current_rule_index=data.get("current_rule_index", 0),
            current_rule=data.get("current_rule"),
            all_rules=data.get("all_rules", [data.get("current_rule")]),
            game_active=data.get("game_active", True),
            captcha=data.get("captcha"),
            country=data.get("country")
        )
        self.game_status = GameStatus.ACTIVE
        self.submission_history = []

        logger.info(f"Game started successfully: {self.current_state}")
        return self.current_state

    def get_state(self, refresh: bool = False) -> Optional[GameState]:
        """
        Get current game state

        Args:
            refresh: If True, fetch fresh state from API. Otherwise return cached state.

        Returns:
            GameState object if successful, None otherwise
        """
        if not refresh and self.current_state is not None:
            return self.current_state

        if self.token is None:
            logger.warning("No active game session. Call start_game() first.")
            return None

        success, data, error = self._make_request("GET", f"/state/{self.token}")

        if not success:
            logger.error(f"Failed to get state: {error}")
            return None

        # Update cached state
        self.current_state = GameState(
            token=self.token,
            current_rule_index=data.get("current_rule_index", 0),
            current_rule=data.get("current_rule"),
            all_rules=data.get("all_rules", []),
            game_active=data.get("game_active", True),
            captcha=data.get("captcha"),
            country=data.get("country")
        )

        return self.current_state

    def get_feedback(self, password: str) -> Optional[FeedbackResult]:
        """
        Get detailed feedback for a password without advancing the game

        Args:
            password: Password string to test

        Returns:
            FeedbackResult object if successful, None otherwise
        """
        if self.token is None:
            logger.warning("No active game session. Call start_game() first.")
            return None

        payload = {"password": password, "give_up": False}
        success, data, error = self._make_request(
            "POST",
            f"/feedback/{self.token}",
            json_data=payload
        )

        if not success:
            logger.error(f"Failed to get feedback: {error}")
            return None

        feedback = FeedbackResult(
            password=data.get("password", password),
            length=data.get("length", len(password)),
            total_passing=data.get("total_passing", 0),
            reward=data.get("reward", 0.0),
            rules_checked=data.get("rules_checked", [])
        )

        return feedback

    def submit_password(
        self,
        password: str,
        give_up: bool = False
    ) -> SubmitResult:
        """
        Submit a password and advance to the next rule

        Args:
            password: Password string to submit
            give_up: If True, end the game

        Returns:
            SubmitResult object with outcome
        """
        if self.token is None:
            return SubmitResult(
                success=False,
                game_ended=True,
                gave_up=False,
                reward=None,
                new_state=None,
                error="No active game session"
            )

        payload = {"password": password, "give_up": give_up}
        success, data, error = self._make_request(
            "POST",
            f"/submit/{self.token}",
            json_data=payload
        )

        if not success:
            logger.error(f"Failed to submit password: {error}")
            return SubmitResult(
                success=False,
                game_ended=False,
                gave_up=False,
                reward=None,
                new_state=None,
                error=error
            )

        # Check if game ended
        game_ended = data.get("game_ended", False)
        gave_up_result = data.get("gave_up", False)
        reward = data.get("reward")

        if game_ended:
            self.game_status = GameStatus.GAVE_UP if gave_up_result else GameStatus.COMPLETED
            logger.info(f"Game ended. Status: {self.game_status}, Reward: {reward}")

            return SubmitResult(
                success=True,
                game_ended=True,
                gave_up=gave_up_result,
                reward=reward,
                new_state=None
            )

        # Game continues - update state
        self.current_state = GameState(
            token=self.token,
            current_rule_index=data.get("current_rule_index", 0),
            current_rule=data.get("current_rule"),
            all_rules=data.get("all_rules", []),
            game_active=data.get("game_active", True),
            captcha=data.get("captcha"),
            country=data.get("country")
        )

        # Track submission history
        if reward is not None:
            self.submission_history.append((password, reward))

        return SubmitResult(
            success=True,
            game_ended=False,
            gave_up=False,
            reward=reward,
            new_state=self.current_state
        )

    def end_game(self) -> bool:
        """
        Explicitly end the current game

        Returns:
            True if successful, False otherwise
        """
        if self.token is None:
            logger.warning("No active game session to end.")
            return False

        success, data, error = self._make_request("POST", f"/end/{self.token}")

        if success:
            self.game_status = GameStatus.GAVE_UP
            logger.info("Game ended successfully")
            return True
        else:
            logger.error(f"Failed to end game: {error}")
            return False

    def reset(self) -> Optional[GameState]:
        """
        Reset environment by starting a new game

        Returns:
            New GameState if successful, None otherwise
        """
        logger.info("Resetting environment...")

        # Try to end current game if exists
        if self.token is not None:
            self.end_game()

        # Clear state
        self.token = None
        self.current_state = None
        self.game_status = GameStatus.ACTIVE
        self.submission_history = []

        # Start new game
        return self.start_game()

    def is_active(self) -> bool:
        """Check if game is currently active"""
        return (
            self.current_state is not None and
            self.current_state.game_active and
            self.game_status == GameStatus.ACTIVE
        )

    def get_history(self) -> List[Tuple[str, float]]:
        """Get submission history (password, reward) tuples"""
        return self.submission_history.copy()

    def close(self):
        """Clean up resources"""
        if self.token is not None:
            self.end_game()
        self.session.close()
        logger.info("Environment closed")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()

    def __del__(self):
        """Destructor"""
        try:
            self.close()
        except:
            pass


class BatchPasswordGameEnv:
    """
    Batch wrapper for running multiple Password Game environments in parallel

    Useful for RL training with parallel experience collection.

    Usage:
        batch_env = BatchPasswordGameEnv(num_envs=4, base_url="http://localhost:8000")
        states = batch_env.reset()
        feedbacks = batch_env.get_feedback_batch(["pass1", "pass2", "pass3", "pass4"])
        results = batch_env.submit_batch(["pass1", "pass2", "pass3", "pass4"])
    """

    def __init__(
        self,
        num_envs: int,
        base_url: str = "http://localhost:8000",
        max_workers: Optional[int] = None,
        **env_kwargs
    ):
        """
        Initialize batch environment

        Args:
            num_envs: Number of parallel environments
            base_url: Base URL of the API server
            max_workers: Max thread pool workers (defaults to num_envs)
            **env_kwargs: Additional kwargs passed to PasswordGameEnv
        """
        self.num_envs = num_envs
        self.base_url = base_url
        self.max_workers = max_workers or num_envs

        # Create environments
        self.envs = [
            PasswordGameEnv(base_url=base_url, **env_kwargs)
            for _ in range(num_envs)
        ]

        logger.info(f"Initialized BatchPasswordGameEnv with {num_envs} environments")

    def reset(self) -> List[Optional[GameState]]:
        """
        Reset all environments

        Returns:
            List of GameState objects
        """
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(env.reset) for env in self.envs]
            states = [future.result() for future in futures]

        return states

    def get_states(self, refresh: bool = False) -> List[Optional[GameState]]:
        """
        Get current states from all environments

        Args:
            refresh: Whether to refresh from API

        Returns:
            List of GameState objects
        """
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(env.get_state, refresh)
                for env in self.envs
            ]
            states = [future.result() for future in futures]

        return states

    def get_feedback_batch(
        self,
        passwords: List[str]
    ) -> List[Optional[FeedbackResult]]:
        """
        Get feedback for passwords in batch

        Args:
            passwords: List of password strings (length must equal num_envs)

        Returns:
            List of FeedbackResult objects
        """
        if len(passwords) != self.num_envs:
            raise ValueError(
                f"Expected {self.num_envs} passwords, got {len(passwords)}"
            )

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(env.get_feedback, password)
                for env, password in zip(self.envs, passwords)
            ]
            feedbacks = [future.result() for future in futures]

        return feedbacks

    def submit_batch(
        self,
        passwords: List[str],
        give_up: Optional[List[bool]] = None
    ) -> List[SubmitResult]:
        """
        Submit passwords to all environments in batch

        Args:
            passwords: List of password strings (length must equal num_envs)
            give_up: Optional list of give_up flags

        Returns:
            List of SubmitResult objects
        """
        if len(passwords) != self.num_envs:
            raise ValueError(
                f"Expected {self.num_envs} passwords, got {len(passwords)}"
            )

        if give_up is None:
            give_up = [False] * self.num_envs
        elif len(give_up) != self.num_envs:
            raise ValueError(
                f"Expected {self.num_envs} give_up flags, got {len(give_up)}"
            )

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(env.submit_password, password, giveup)
                for env, password, giveup in zip(self.envs, passwords, give_up)
            ]
            results = [future.result() for future in futures]

        return results

    def close(self):
        """Close all environments"""
        for env in self.envs:
            env.close()
        logger.info("Batch environment closed")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


# Utility functions for RL training

def calculate_rule_progress_reward(
    old_state: GameState,
    new_state: Optional[GameState],
    game_ended: bool
) -> float:
    """
    Calculate shaped reward based on rule progression

    Args:
        old_state: State before action
        new_state: State after action (None if game ended)
        game_ended: Whether game ended

    Returns:
        Shaped reward value
    """
    if game_ended:
        # Terminal state - no additional shaping
        return 0.0

    if new_state is None:
        # Error state
        return -1.0

    # Reward for progressing to next rule
    rule_progress = new_state.current_rule_index - old_state.current_rule_index
    return float(rule_progress)


def create_observation_dict(state: GameState, feedback: Optional[FeedbackResult] = None) -> Dict:
    """
    Create standardized observation dictionary for RL agent

    Args:
        state: Current game state
        feedback: Optional feedback result

    Returns:
        Dictionary observation
    """
    obs = {
        "current_rule_index": state.current_rule_index,
        "current_rule": state.current_rule or "",
        "all_rules": state.all_rules,
        "num_rules": len(state.all_rules),
        "game_active": state.game_active,
        "has_captcha": state.captcha is not None,
        "has_country": state.country is not None,
    }

    if feedback is not None:
        obs.update({
            "password_length": feedback.length,
            "total_passing": feedback.total_passing,
            "reward": feedback.reward,
            "num_failing": len(feedback.failing_rules()),
        })

    return obs


if __name__ == "__main__":
    # Example usage
    print("Password Game Environment Wrapper")
    print("=" * 50)

    # Single environment example
    print("\n1. Single Environment Example:")
    with PasswordGameEnv(base_url="http://localhost:8000") as env:
        # Start game
        state = env.start_game()
        if state:
            print(f"Game started: {state}")
            print(f"Current rule: {state.current_rule}")

            # Test a password
            test_password = "Test123!"
            feedback = env.get_feedback(test_password)
            if feedback:
                print(f"\nFeedback for '{test_password}':")
                print(f"  Passing: {feedback.total_passing}/{len(feedback.rules_checked)} rules")
                print(f"  Reward: {feedback.reward}")

                if feedback.first_failing_rule():
                    print(f"  First failing: {feedback.first_failing_rule()['rule_text']}")

    # Batch environment example
    print("\n2. Batch Environment Example:")
    with BatchPasswordGameEnv(num_envs=3, base_url="http://localhost:8000") as batch_env:
        # Reset all environments
        states = batch_env.reset()
        print(f"Started {len(states)} games")

        # Get feedback for multiple passwords
        test_passwords = ["abc12", "Test1", "Hello!"]
        feedbacks = batch_env.get_feedback_batch(test_passwords)

        for i, (pwd, fb) in enumerate(zip(test_passwords, feedbacks)):
            if fb:
                print(f"  Env {i}: '{pwd}' -> {fb.total_passing} passing rules")

    print("\n" + "=" * 50)
    print("Examples completed successfully!")
