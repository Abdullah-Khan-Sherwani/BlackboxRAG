"""
Centralized LLM client — single point of access for all LLM calls.

Uses DeepSeek V3.1 via NVIDIA's OpenAI-compatible API.
Handles initialization, retries, and error handling.
"""
import os
from typing import Any

from dotenv import load_dotenv
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

try:
    from openai import OpenAI, RateLimitError
except ImportError:  # Allow Ollama-only runs without OpenAI package installed.
    OpenAI = None

    class RateLimitError(Exception):
        pass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(BASE_DIR, ".env"))

MODEL = "deepseek-ai/deepseek-v3.1"
BASE_URL = "https://integrate.api.nvidia.com/v1"

_client: Any | None = None


def _get_client() -> Any:
    """Lazy-initialize the OpenAI client (module-level singleton)."""
    global _client
    if _client is None:
        if OpenAI is None:
            raise RuntimeError(
                "OpenAI package is not installed. Install 'openai' only if using DeepSeek/NVIDIA API. "
                "For local usage, select Ollama in the UI."
            )
        api_key = os.environ.get("NVIDIA_API_KEY")
        if not api_key:
            raise RuntimeError(
                "NVIDIA_API_KEY not found in environment. Check your .env file."
            )
        _client = OpenAI(base_url=BASE_URL, api_key=api_key)
    return _client


@retry(
    retry=retry_if_exception_type(RateLimitError),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    stop=stop_after_attempt(3),
    reraise=True,
)
def call_llm(prompt: str, system: str | None = None) -> str:
    """Send a prompt to the LLM and return the response text.

    Args:
        prompt: The user message / main prompt content.
        system: Optional system message for role-setting.

    Returns:
        The model's response as a string.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = _get_client().chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()
