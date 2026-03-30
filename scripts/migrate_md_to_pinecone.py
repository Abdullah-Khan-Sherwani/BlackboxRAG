"""
Migrate markdown-extracted reports into Pinecone with metadata-augmented embeddings.

Pipeline:
1. Load .md reports from dataset-pipeline/data/extracted/extracted
2. Chunk each report (section-aware or recursive)
3. Build contextualized document text with provenance headers
4. Generate document embeddings locally
5. Optionally reset (delete + recreate) the Pinecone index
6. Upsert vectors in batches

Notes:
- Document embeddings are generated locally in this script.
- Query embeddings are generated in-app at retrieval time (src/retrieval/query.py).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from transformers import AutoModel

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.llm.ollama_client import call_ollama  # noqa: E402

MD_DIR = BASE_DIR / "dataset-pipeline" / "data" / "extracted" / "extracted"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

DEFAULT_INDEX_NAME = "ntsb-rag"
DEFAULT_MODEL_NAME = "jinaai/jina-embeddings-v5-text-nano"
DEFAULT_SUMMARIZER_MODEL = "qwen2.5:32b"
DEFAULT_BATCH_SIZE = 100

ENTITY_DENSE_SUMMARY_PROMPT = """You are an expert technical editor. Your task is to create a high-density, entity-preserving summary of the provided text chunk.

Rules:
1. MANDATORY: Preserve all unique identifiers (e.g., Flight IDs, Case Numbers, Part IDs).
2. PRECISION: Retain all numerical values, units, and technical parameters exactly as written.
3. ENTITY INTEGRITY: Keep all proper names (people, organizations, locations) and specific roles intact.
4. NO GENERALIZATION: Do not use vague terms like "various," "technical issues," or "several." If the text says "three engine valves," you must write "three engine valves."
5. DYNAMIC LENGTH: Provide enough detail to cover all unique technical points in the chunk. Do not cut the summary short if the content is dense.
6. OBJECTIVITY: Do not interpret, infer, or provide commentary. Only summarize explicit facts.

Chunk:
{chunk_text}
"""


def split_text_with_overlap(text: str, chunk_size: int = 1500, overlap: int = 200) -> list[str]:
    """Split text into overlapping character windows with simple boundary cleanup."""
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []

    chunks: list[str] = []
    start = 0
    n = len(cleaned)

    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            boundary = cleaned.rfind(" ", start, end)
            if boundary > start + int(chunk_size * 0.7):
                end = boundary

        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= n:
            break
        start = max(0, end - overlap)

    return chunks


def infer_role(section_title: str, text: str) -> str:
    """Infer a coarse role label used in chunk-level provenance formatting."""
    label = f"{section_title} {text}".lower()
    if "captain" in label:
        return "Captain"
    if "first officer" in label or "copilot" in label or "co-pilot" in label:
        return "First Officer"
    if "pilot" in label:
        return "Pilot"
    if "engine" in label or "maintenance" in label:
        return "Maintenance/Engineering"
    if "atc" in label or "air traffic" in label or "controller" in label:
        return "ATC"
    if section_title:
        return section_title
    return "Unknown"


def _density_summary_fallback(chunk_text: str) -> str:
    """Fast local fallback summary preserving high-information lines."""
    cleaned = re.sub(r"\s+", " ", chunk_text).strip()
    if not cleaned:
        return "No context available"

    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    selected = []
    numeric_or_id = re.compile(r"\d|[A-Z]{2,}\d+|\b[A-Z]{2,}[\-/]\d+")

    for sent in sentences:
        if numeric_or_id.search(sent):
            selected.append(sent.strip())

    if not selected:
        selected = [s.strip() for s in sentences[:2] if s.strip()]

    joined = " ".join(selected)

    # Dynamic length by information density.
    density_hits = len(numeric_or_id.findall(cleaned))
    max_len = 260 if density_hits < 6 else 420 if density_hits < 14 else 620
    return joined[:max_len].strip()


def summarize_entity_dense(
    chunk_text: str,
    summarizer_model: str,
    use_ollama_summary: bool,
    summary_timeout: int,
    summary_cache: dict[str, str],
) -> str:
    """Generate entity-dense summary with optional local Ollama model."""
    key = hashlib.md5(chunk_text.encode("utf-8")).hexdigest()
    if key in summary_cache:
        return summary_cache[key]

    if not use_ollama_summary:
        out = _density_summary_fallback(chunk_text)
        summary_cache[key] = out
        return out

    prompt = ENTITY_DENSE_SUMMARY_PROMPT.format(chunk_text=chunk_text)
    try:
        out = call_ollama(
            prompt=prompt,
            system_prompt="You are a strict technical summarizer.",
            model=summarizer_model,
            temperature=0.0,
            max_tokens=600,
            timeout=summary_timeout,
        ).strip()
        if not out:
            out = _density_summary_fallback(chunk_text)
    except Exception:
        out = _density_summary_fallback(chunk_text)

    summary_cache[key] = out
    return out


def chunk_markdown_section_aware_local(md_file_path: Path) -> list[dict]:
    with md_file_path.open("r", encoding="utf-8") as f:
        content = f.read()

    sections = re.split(r"\n(##\s+.*?)\n", content)
    parsed_sections = []

    if sections and sections[0].strip():
        parsed_sections.append({"title": "Introduction/Header", "content": sections[0].strip()})

    for i in range(1, len(sections), 2):
        header = sections[i].replace("##", "").strip()
        text = sections[i + 1].strip() if i + 1 < len(sections) else ""
        if text:
            parsed_sections.append({"title": header, "content": text})

    report_id = md_file_path.stem
    chunks: list[dict] = []

    for sec_idx, section in enumerate(parsed_sections):
        if not section["content"] or re.match(r"^[\W_]+$", section["content"]):
            continue

        for chunk_idx, chunk_text in enumerate(split_text_with_overlap(section["content"])):
            chunks.append(
                {
                    "chunk_id": f"{report_id}_sec{sec_idx:02d}_{chunk_idx:03d}",
                    "report_id": report_id,
                    "section_title": section["title"],
                    "role": infer_role(section["title"], chunk_text),
                    "source_filename": md_file_path.name,
                    "entity_id": report_id,
                    "text": f"Section: {section['title']}\n{chunk_text}",
                }
            )

    return chunks


def chunk_markdown_recursive_local(md_file_path: Path) -> list[dict]:
    with md_file_path.open("r", encoding="utf-8") as f:
        content = f.read()

    report_id = md_file_path.stem
    chunks: list[dict] = []
    for chunk_idx, chunk_text in enumerate(split_text_with_overlap(content)):
        chunks.append(
            {
                "chunk_id": f"{report_id}_rec_{chunk_idx:03d}",
                "report_id": report_id,
                "section_title": "",
                "role": infer_role("", chunk_text),
                "source_filename": md_file_path.name,
                "entity_id": report_id,
                "text": chunk_text,
            }
        )
    return chunks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chunk markdown files, embed, and upsert to Pinecone."
    )
    parser.add_argument(
        "--strategy",
        choices=["section", "recursive"],
        default="section",
        help="Chunking strategy for markdown source files.",
    )
    parser.add_argument(
        "--index-name",
        default=DEFAULT_INDEX_NAME,
        help="Pinecone index name.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Embedding model used for both retrieval documents and queries.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Pinecone upsert batch size.",
    )
    parser.add_argument(
        "--reset-index",
        action="store_true",
        help="Delete existing index and recreate it before upload.",
    )
    parser.add_argument(
        "--use-ollama-summary",
        action="store_true",
        help=(
            "Use local Ollama summarization (qwen2.5:32b by default) for entity-dense context summaries. "
            "If disabled, a deterministic local density summary is used."
        ),
    )
    parser.add_argument(
        "--summarizer-model",
        default=DEFAULT_SUMMARIZER_MODEL,
        help="Local Ollama model used for entity-dense summaries.",
    )
    parser.add_argument(
        "--summary-timeout",
        type=int,
        default=120,
        help="Per-chunk timeout (seconds) for Ollama summarization.",
    )
    parser.add_argument(
        "--ollama-summary-max-chunks",
        type=int,
        default=0,
        help=(
            "Maximum number of chunks to summarize with Ollama before switching to deterministic fallback. "
            "0 means no limit."
        ),
    )
    return parser.parse_args()


def add_provenance_context(
    chunks: list[dict],
    use_ollama_summary: bool,
    summarizer_model: str,
    summary_timeout: int,
    ollama_summary_max_chunks: int,
) -> list[dict]:
    """Add context summaries and bake provenance headers into embedding text."""
    summary_cache: dict[str, str] = {}

    ollama_budget = max(0, ollama_summary_max_chunks)

    for idx, chunk in enumerate(chunks, start=1):
        should_use_ollama = use_ollama_summary and (ollama_budget == 0 or idx <= ollama_budget)
        summary = summarize_entity_dense(
            chunk_text=chunk.get("text", ""),
            summarizer_model=summarizer_model,
            use_ollama_summary=should_use_ollama,
            summary_timeout=summary_timeout,
            summary_cache=summary_cache,
        )
        chunk["context_summary"] = summary
        source = chunk.get("source_filename", "unknown.md")
        entity_id = chunk.get("entity_id", chunk.get("report_id", ""))

        chunk["contextualized_text"] = (
            f"[Source: {source}] [Entity_ID: {entity_id}] [Context: {summary}]\n"
            f"{chunk.get('text', '')}"
        )

        if idx % 1000 == 0 or idx == len(chunks):
            print(f"  Added provenance headers for {idx}/{len(chunks)} chunks")

    return chunks


def load_markdown_chunks(strategy: str) -> list[dict]:
    md_files = sorted(MD_DIR.glob("*.md"))
    if not md_files:
        raise RuntimeError(f"No markdown files found in: {MD_DIR}")

    print(f"Found {len(md_files)} markdown files in {MD_DIR}")
    chunks: list[dict] = []
    for idx, md_file in enumerate(md_files, start=1):
        if strategy == "section":
            file_chunks = chunk_markdown_section_aware_local(md_file)
        else:
            file_chunks = chunk_markdown_recursive_local(md_file)

        chunks.extend(file_chunks)
        if idx % 25 == 0 or idx == len(md_files):
            print(f"  Chunked {idx}/{len(md_files)} files -> {len(chunks)} chunks")

    return chunks


def embed_chunks(chunks: list[dict], model_name: str) -> np.ndarray:
    print(f"Loading embedding model: {model_name}")
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

    texts = [c.get("contextualized_text", c.get("text", "")) for c in chunks]
    print(f"Encoding {len(texts)} chunks...")

    batch_size = 64
    all_vectors: list[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            vec = model.encode(texts=batch, task="retrieval", prompt_name="passage")
        except ValueError:
            vec = model.encode(texts=batch, task="retrieval")
        all_vectors.append(np.asarray(vec, dtype=np.float32))
        print(f"  Encoded {min(i + batch_size, len(texts))}/{len(texts)}")

    embeddings = np.concatenate(all_vectors, axis=0)
    print(f"Embeddings shape: {embeddings.shape}")
    return embeddings


def save_local_artifacts(chunks: list[dict], embeddings: np.ndarray, strategy: str) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    chunks_path = PROCESSED_DIR / f"chunks_md_{strategy}.json"
    emb_path = PROCESSED_DIR / f"embeddings_md_{strategy}.npz"

    with chunks_path.open("w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    np.savez_compressed(
        emb_path,
        chunk_ids=np.array([c["chunk_id"] for c in chunks]),
        embeddings=embeddings,
    )

    print(f"Saved chunks to: {chunks_path}")
    print(f"Saved embeddings to: {emb_path}")


def init_index(index_name: str, dimension: int, reset_index: bool):
    load_dotenv(BASE_DIR / ".env")
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY is not set in environment or .env")

    pc = Pinecone(api_key=api_key)
    existing = {idx.name for idx in pc.list_indexes()}

    if reset_index and index_name in existing:
        print(f"Deleting existing index '{index_name}' (remove old data)...")
        pc.delete_index(index_name)
        existing.remove(index_name)

    if index_name not in existing:
        print(f"Creating index '{index_name}' with dimension={dimension}...")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    return pc.Index(index_name)


def upsert(index, chunks: list[dict], embeddings: np.ndarray, strategy: str, batch_size: int) -> None:
    vectors = []
    for chunk, emb in zip(chunks, embeddings):
        vectors.append(
            {
                "id": chunk["chunk_id"],
                "values": emb.tolist(),
                "metadata": {
                    "report_id": chunk.get("report_id", ""),
                    "entity_id": chunk.get("entity_id", chunk.get("report_id", "")),
                    "source_filename": chunk.get("source_filename", ""),
                    "section_title": chunk.get("section_title", ""),
                    "role": chunk.get("role", "Unknown"),
                    "context_summary": chunk.get("context_summary", ""),
                    "strategy": strategy,
                    "source": "dataset-pipeline/data/extracted/extracted",
                },
            }
        )

    total = len(vectors)
    print(f"Upserting {total} vectors in batches of {batch_size}...")
    for i in range(0, total, batch_size):
        batch = vectors[i : i + batch_size]
        index.upsert(vectors=batch)
        print(f"  Upserted {min(i + batch_size, total)}/{total}")


def main() -> None:
    args = parse_args()

    chunks = load_markdown_chunks(args.strategy)
    chunks = add_provenance_context(
        chunks=chunks,
        use_ollama_summary=args.use_ollama_summary,
        summarizer_model=args.summarizer_model,
        summary_timeout=args.summary_timeout,
        ollama_summary_max_chunks=args.ollama_summary_max_chunks,
    )
    embeddings = embed_chunks(chunks, args.model_name)
    save_local_artifacts(chunks, embeddings, args.strategy)

    index = init_index(
        index_name=args.index_name,
        dimension=int(embeddings.shape[1]),
        reset_index=args.reset_index,
    )
    upsert(
        index=index,
        chunks=chunks,
        embeddings=embeddings,
        strategy=args.strategy,
        batch_size=args.batch_size,
    )

    print("\nIndex stats:")
    print(index.describe_index_stats())


if __name__ == "__main__":
    main()
