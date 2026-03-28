"""
Centralized LLM client — single point of access for all LLM calls.

Uses DeepSeek V3.1 via NVIDIA's OpenAI-compatible API.
Handles initialization, retries, and error handling.
"""
import os

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from openai import OpenAI, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(BASE_DIR, ".env"))

MODEL = "deepseek-ai/deepseek-v3.1"
BASE_URL = "https://integrate.api.nvidia.com/v1"
HF_EVAL_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

_client: OpenAI | None = None
_hf_client: InferenceClient | None = None


def _get_client() -> OpenAI:
    """Lazy-initialize the OpenAI client (module-level singleton)."""
    global _client
    if _client is None:
        api_key = os.environ.get("NVIDIA_API_KEY")
        if not api_key:
            raise RuntimeError(
                "NVIDIA_API_KEY not found in environment. Check your .env file."
            )
        _client = OpenAI(base_url=BASE_URL, api_key=api_key)
    return _client


def _get_hf_client() -> InferenceClient:
    """Lazy-initialize the Hugging Face Inference client."""
    global _hf_client
    if _hf_client is None:
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        _hf_client = InferenceClient(token=hf_token)
    return _hf_client


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


def call_eval_llm(prompt: str, system: str | None = None) -> str:
    """Call evaluation judge model.

    Backend is controlled by EVAL_LLM_PROVIDER env var:
    - "nvidia" (default): NVIDIA OpenAI-compatible endpoint
    - "hf": Hugging Face Inference API
    """
    provider = os.environ.get("EVAL_LLM_PROVIDER", "nvidia").strip().lower()

    if provider == "hf":
        model = os.environ.get("EVAL_HF_MODEL", HF_EVAL_MODEL)
        combined_prompt = prompt if not system else f"System:\n{system}\n\nUser:\n{prompt}"
        out = _get_hf_client().text_generation(
            combined_prompt,
            model=model,
            max_new_tokens=700,
            temperature=0.0,
            do_sample=False,
        )
        return out.strip()

    # Default to NVIDIA path for eval too.
    return call_llm(prompt, system=system)
