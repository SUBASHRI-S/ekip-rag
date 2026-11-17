import os
import time

from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, OpenAI

load_dotenv()

_client = None


def get_client():
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=os.environ["LLM_BASE_URL"],
            api_key=os.environ["LLM_API_KEY"],
        )
    return _client


def generate(
    prompt: str,
    system: str | None = None,
    model: str | None = None,
    attempts: int = 4,
) -> str:
    """Ask the LLM a question. Retries when the server is busy."""
    model = model or os.environ["LLM_MODEL"]
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    for attempt in range(1, attempts + 1):
        try:
            response = get_client().chat.completions.create(model=model, messages=messages)
            return response.choices[0].message.content
        except (APIStatusError, APIConnectionError) as error:
            code = getattr(error, "status_code", None)  # None means a network problem
            temporary = code is None or code in (429, 500, 502, 503, 504)
            if not temporary or attempt == attempts:
                raise
            wait = 2 ** attempt
            print(f"LLM is busy or unreachable (error {code}). Retrying in {wait} seconds...")
            time.sleep(wait)

            