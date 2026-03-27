import json
import logging
import time
from typing import Dict, Any, Optional
import requests
from requests.exceptions import RequestException

logger = logging.getLogger(__name__)

OLLAMA_API_URL = "http://localhost:11434/api"
OLLAMA_CHAT_URL = f"{OLLAMA_API_URL}/chat"


def is_ollama_running() -> bool:
    """Check if the Ollama local service is running and responsive."""
    try:
        response = requests.get(OLLAMA_API_URL[:-4]) # get base localhost:11434
        return response.status_code == 200
    except RequestException:
        return False

def check_model_available(model_name: str) -> bool:
    """Check if a specific model is downloaded and available in Ollama."""
    try:
        response = requests.get(f"{OLLAMA_API_URL}/tags")
        if response.status_code == 200:
            models = response.json().get("models", [])
            return any(m.get("name") == model_name for m in models)
        return False
    except RequestException:
        return False

def call_ollama(
    prompt: str,
    system_prompt: Optional[str] = None,
    model: str = "qwen2.5:32b",
    temperature: float = 0.1,
    max_tokens: int = 2048,
    json_mode: bool = False,
    retries: int = 3,
    timeout: int = 120
) -> str:
    """
    Call the local Ollama LLM for inference.
    
    Args:
        prompt: The user prompt.
        system_prompt: System instructions to guide the model.
        model: Name of the Ollama model.
        temperature: Sampling temperature.
        max_tokens: Max tokens to generate.
        json_mode: If True, forces Ollama to output valid JSON format.
        retries: Number of retry attempts on failure.
        timeout: Wait time in seconds for the model to respond (heavy models can be slow).
        
    Returns:
        The generated text from the model.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens
        }
    }
    
    if json_mode:
        payload["format"] = "json"

    for attempt in range(retries):
        try:
            response = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=timeout)
            response.raise_for_status()
            
            result = response.json()
            return result["message"]["content"]
            
        except requests.exceptions.Timeout:
            logger.warning(f"Ollama timeout on attempt {attempt + 1}/{retries} - Model might need more time or context is too large.")
            if attempt == retries - 1:
                raise Exception(f"Ollama call timed out after {retries} attempts.")
            time.sleep(2 ** attempt)
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"Ollama request failed on attempt {attempt + 1}/{retries}: {str(e)}")
            if attempt == retries - 1:
                raise Exception(f"Failed to communicate with Ollama: {str(e)}")
            time.sleep(2 ** attempt)

    return ""
