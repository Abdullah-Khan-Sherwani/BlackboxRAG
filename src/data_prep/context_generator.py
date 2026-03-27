import json
import logging
from pathlib import Path
from tqdm import tqdm
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.llm.ollama_client import call_ollama, is_ollama_running

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CACHE_FILE = project_root / "data" / "processed" / "context_cache.json"

CONTEXT_SYSTEM_PROMPT = """
You are an expert AI extraction system helping to optimize documents for search retrieval.
Your task is to concisely situate the given chunk within the context of the larger document.
"""

def generate_context_prompt(whole_document: str, chunk_content: str) -> str:
    """
    Format strictly follows Anthropic's recommended Contextual Retrieval prompt.
    """
    return f"""<document>
{whole_document}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk_content}
</chunk>

Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""


def contextualize_chunks(chunks: list[dict], whole_document: str, model: str = "qwen2.5:32b") -> list[dict]:
    """
    Takes a list of chunk dictionaries and generates a context string for each one.
    The output chunk will have an augmented 'contextualized_text' field which prepends the context.
    
    Expected chunk format: {'chunk_id': '...', 'text': '...'}
    """
    if not is_ollama_running():
        raise RuntimeError("Ollama is not running. Context generation requires the local LLM.")
        
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    # Load cache
    context_cache = {}
    if CACHE_FILE.exists():
        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            context_cache = json.load(f)
            
    contextualized_chunks = []
    
    for chunk in tqdm(chunks, desc="Generating contexts for chunks"):
        chunk_id = chunk.get('chunk_id')
        chunk_text = chunk.get('text', '')
        
        # Check cache first
        if chunk_id in context_cache:
            context = context_cache[chunk_id]
        else:
            # Generate new context
            prompt = generate_context_prompt(whole_document, chunk_text)
            try:
                # Temperature 0 for deterministic, factual outputs
                context = call_ollama(
                    prompt=prompt,
                    system_prompt=CONTEXT_SYSTEM_PROMPT,
                    model=model,
                    temperature=0.0,
                    max_tokens=150
                )
                
                # Save to cache strictly as text
                context = context.strip()
                context_cache[chunk_id] = context
                
                # Write back cache periodically or immediately
                with open(CACHE_FILE, 'w', encoding='utf-8') as f:
                    json.dump(context_cache, f, indent=4)
                    
            except Exception as e:
                logger.error(f"Failed to generate context for chunk {chunk_id}: {str(e)}")
                context = ""
                
        # Build contextualized text
        augmented_text = f"{context}\n\n{chunk_text}" if context else chunk_text
        
        # Create a new chunk dict preserving original structure
        new_chunk = chunk.copy() if isinstance(chunk, dict) else dict(chunk)
        new_chunk['context'] = context
        new_chunk['contextualized_text'] = augmented_text
        contextualized_chunks.append(new_chunk)
        
    return contextualized_chunks
