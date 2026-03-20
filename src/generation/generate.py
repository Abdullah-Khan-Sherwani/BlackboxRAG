"""
RAG Answer Generation using Google Gemini.
Takes retrieved chunks + user query → calls Gemini → produces an answer.
"""
import os
import sys

from dotenv import load_dotenv
import google.generativeai as genai

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.retrieval.query import load_model, init_pinecone, retrieve

GEMINI_MODEL = "gemini-2.0-flash"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SAMPLE_QUERIES = [
    "What are common causes of engine failure during takeoff?",
    "Cessna accidents in instrument meteorological conditions",
    "How does pilot experience affect landing accident outcomes?",
]


def init_gemini():
    """Load environment variables and configure the Gemini API."""
    load_dotenv(os.path.join(BASE_DIR, ".env"))
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not found in environment. Check your .env file.")
    genai.configure(api_key=api_key)


def build_prompt(query, retrieved_chunks):
    """Build a prompt with system instruction, context chunks, and the user query."""
    system = (
        "You are an NTSB aviation safety expert. Answer based only on the provided context. "
        "If the context is insufficient, say so."
    )

    context_blocks = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        meta = chunk.metadata
        header = (
            f"[Context {i}] NTSB No: {meta.get('ntsb_no', 'N/A')} | "
            f"Date: {meta.get('event_date', 'N/A')} | "
            f"Aircraft: {meta.get('make', 'N/A')} {meta.get('model', 'N/A')}"
        )
        text = meta.get("text", "")
        context_blocks.append(f"{header}\n{text}")

    context_str = "\n\n".join(context_blocks)

    prompt = f"{system}\n\n--- Context ---\n{context_str}\n\n--- Question ---\n{query}"
    return prompt


def generate_answer(query, retrieved_chunks):
    """Generate an answer using Gemini given the query and retrieved chunks."""
    prompt = build_prompt(query, retrieved_chunks)
    model = genai.GenerativeModel(GEMINI_MODEL)
    response = model.generate_content(prompt)
    return response.text


def rag_pipeline(query, strategy, top_k=5, model=None, index=None):
    """End-to-end RAG: retrieve chunks then generate an answer.

    Returns a dict with query, strategy, answer, sources, and num_chunks.
    """
    matches = retrieve(query, strategy, top_k=top_k, model=model, index=index)
    answer = generate_answer(query, matches)
    source_ids = [m.id for m in matches]
    return {
        "query": query,
        "strategy": strategy,
        "answer": answer,
        "sources": source_ids,
        "num_chunks": len(matches),
    }


def main():
    # Load resources once
    jina_model = load_model()
    index = init_pinecone()
    init_gemini()

    strategies = ["fixed", "recursive", "semantic"]

    for query in SAMPLE_QUERIES:
        for strategy in strategies:
            print(f"\n{'='*80}")
            print(f"Query: {query}")
            print(f"Strategy: {strategy}")
            print("-" * 80)

            result = rag_pipeline(query, strategy, top_k=5, model=jina_model, index=index)

            print(f"\nAnswer:\n{result['answer']}")
            print(f"\nSources ({result['num_chunks']} chunks): {result['sources']}")


if __name__ == "__main__":
    main()
