from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from game import PasswordGame
import uuid
from typing import Dict

app = FastAPI()
game_sessions: Dict[str, PasswordGame] = {}

class PasswordSubmission(BaseModel):
    password: str
    give_up: bool = False

def get_game_session(token: str) -> PasswordGame:
    """Get game session by token, raise 404 if not found."""
    if token not in game_sessions:
        raise HTTPException(status_code=404, detail="Game session not found")
    return game_sessions[token]

@app.post("/start")
def start_game():
    token = str(uuid.uuid4())
    game_instance = PasswordGame()
    game_sessions[token] = game_instance

    api_guide = {
        "endpoints": {
            "submit": f"POST /submit/{token} - Submit password and advance to next rule",
            "state": f"GET /state/{token} - Get current game state without advancing",
            "feedback": f"POST /feedback/{token} - Test password against all current rules",
            "end": f"POST /end/{token} - End the current game"
        },
        "usage": "Submit passwords that satisfy the current rule AND all previous rules"
    }

    return {
        "token": token,
        "current_rule_index": game_instance.current_rule,
        "current_rule": game_instance.get_current_rule(),
        "game_active": game_instance.game_active,
        "instructions": game_instance.get_instructions(),
        "api_guide": api_guide
    }

@app.get("/state/{token}")
def get_game_state(token: str):
    game_instance = get_game_session(token)
    return game_instance.get_minimal_game_state()

@app.post("/submit/{token}")
def submit_password(token: str, submission: PasswordSubmission):
    game_instance = get_game_session(token)

    if not game_instance.game_active:
        return {"error": "Game is not active"}

    # Handle give up
    if submission.give_up:
        game_instance.end_game()
        reward = game_instance.calculate_reward(submission.password)
        feedback = game_instance.get_rule_feedback(submission.password)
        return {
            "game_ended": True,
            "gave_up": True,
            "reward": reward,
            "final_password": submission.password,
            "rule_feedback": feedback
        }

    # Advance to next rule
    game_instance.advance_rule()

    # Check if game ended naturally
    if not game_instance.game_active:
        reward = game_instance.calculate_reward(submission.password)
        feedback = game_instance.get_rule_feedback(submission.password)
        return {
            "game_ended": True,
            "gave_up": False,
            "reward": reward,
            "final_password": submission.password,
            "rule_feedback": feedback
        }

    return game_instance.get_minimal_game_state()

@app.post("/feedback/{token}")
def get_password_feedback(token: str, submission: PasswordSubmission):
    """Get detailed rule feedback for a password without advancing the game."""
    game_instance = get_game_session(token)
    return game_instance.get_rule_feedback(submission.password)

@app.post("/end/{token}")
def end_game(token: str):
    game_instance = get_game_session(token)
    game_instance.end_game()
    return {"message": "Game ended"}