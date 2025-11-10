import os
import json
from openai import OpenAI
from keys import *

client = OpenAI(api_key=OPENAI_API_KEY)

SCHEMA = {
    "type": "json_schema",
    "strict":True,
    "name": "AnswerAndDate",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "The concise answer or summary."
                },
                "date": {
                    "type": "string",
                    "format": "date",
                    "description": "Today's date in YYYY-MM-DD."
                }
            },
            "required": ["answer", "date"]
        }
}

def ask_with_search(prompt: str,  model: str = "gpt-4.1", temperature: float = 0.1) -> dict:
    """
    Generic function that calls OpenAI Responses API with web search enabled.

    Args:
        prompt: The search query/prompt
        schema: The JSON schema for response format
        model: OpenAI model to use
        temperature: Temperature for response generation

    Returns:
        dict: Parsed JSON response based on provided schema
    """
    response = client.responses.create(
        model=model,
        input=prompt,
        text={
            "format": SCHEMA
        },
        tools=[
            {
                "type": "web_search",
                "user_location": {
                    "type": "approximate"
                },
                "search_context_size": "medium"
            }
        ],
        temperature=temperature,
        top_p=1,
        store=True,
        include=["web_search_call.action.sources"]
    )
    return json.loads(response.output_text)

def utils_get_wordle() -> dict:
    """
    Gets today's Wordle answer using web search.
    Returns: { "answer": str, "date": "YYYY-MM-DD" }
    """
    prompt = "What is today's wordle answer? In your answer only include the word, no other text"
    return ask_with_search(prompt,temperature=1)

def utils_get_emoji() -> dict:
    """
    Gets today's actual moon phase emoji using web search.
    Returns: { "emoji": str, "date": "YYYY-MM-DD" }
    """
    prompt = "What is today's current moon phase? Return only the appropriate emoji: 🌑 🌒 🌓 🌔 🌕 🌖 🌗 🌘"
    return ask_with_search(prompt, temperature=0.1)
