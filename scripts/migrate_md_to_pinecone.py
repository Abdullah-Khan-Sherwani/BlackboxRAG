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
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from transformers import AutoModel

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.llm.ollama_client import call_ollama  # noqa: E402
from src.data_prep.chunking import (  # noqa: E402
    chunk_markdown_baseline_fixed,
    chunk_markdown_baseline_recursive,
    chunk_markdown_baseline_semantic,
    chunk_markdown_md_recursive,
    chunk_markdown_parent_child,
)

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

METADATA_EXTRACTION_PROMPT = """You are extracting structured aviation-accident context from one text chunk.

Return strict JSON with keys:
- context_summary: concise factual summary, preserving exact entities and values
- entities: array of people/organizations explicitly named in the text
- aircraft_components: array of physical aircraft components explicitly mentioned
- numerics: comma-separated string of all exact times, measurements, quantities, and coded identifiers

Rules:
1. Only include information present in the text.
2. Do not infer or normalize values.
3. Keep numerics exact (e.g., '98 feet', '891 hours', '0142:26', '6.8 nm').
4. Output JSON only.

Chunk:
{chunk_text}
"""


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


def _metadata_fallback(chunk_text: str) -> dict[str, str]:
    """Deterministic fallback extraction for entities/components/numerics."""
    summary = _density_summary_fallback(chunk_text)

    entities = re.findall(
        r"\b(?:National Transportation Safety Board|Federal Aviation Administration|Korean Air|Guam|NTSB|FAA|KCAB|CERAP|ATC|CVR|FDR)\b",
        chunk_text,
        flags=re.IGNORECASE,
    )
    components = re.findall(
        r"\b(?:engine|engines|flaps|landing gear|glideslope|localizer|autopilot|wipers|VOR|DME|GPWS|runway|altimeter|radar)\b",
        chunk_text,
        flags=re.IGNORECASE,
    )
    numerics = re.findall(
        r"\b\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?\b|\b\d+(?:\.\d+)?\s?(?:feet|ft|hours|nm|knots|kg|miles|percent|°|msl|agl|tokens?)\b|\b[A-Z]{2,}\d{2,}\b|\b\d+(?:\.\d+)?\b",
        chunk_text,
        flags=re.IGNORECASE,
    )

    uniq_entities = list(dict.fromkeys([e.strip() for e in entities if e.strip()]))[:20]
    uniq_components = list(dict.fromkeys([c.strip().lower() for c in components if c.strip()]))[:20]
    uniq_numerics = list(dict.fromkeys([n.strip() for n in numerics if n.strip()]))[:50]

    return {
        "context_summary": summary or "No context available",
        "entities": ", ".join(uniq_entities) if uniq_entities else "None",
        "aircraft_components": ", ".join(uniq_components) if uniq_components else "None",
        "numerics": ", ".join(uniq_numerics) if uniq_numerics else "None",
    }


def _extract_context_metadata(
    chunk_text: str,
    summarizer_model: str,
    use_ollama_summary: bool,
    summary_timeout: int,
    metadata_cache: dict[str, dict[str, str]],
) -> dict[str, str]:
    """Extract dense context metadata for each chunk."""
    key = hashlib.md5(chunk_text.encode("utf-8")).hexdigest()
    if key in metadata_cache:
        return metadata_cache[key]

    if not use_ollama_summary:
        out = _metadata_fallback(chunk_text)
        metadata_cache[key] = out
        return out

    prompt = METADATA_EXTRACTION_PROMPT.format(chunk_text=chunk_text)
    try:
        raw = call_ollama(
            prompt=prompt,
            system_prompt="You extract strict JSON metadata.",
            model=summarizer_model,
            temperature=0.0,
            max_tokens=700,
            timeout=summary_timeout,
        ).strip()
        parsed = json.loads(raw)
        out = {
            "context_summary": str(parsed.get("context_summary", "")).strip() or _density_summary_fallback(chunk_text),
            "entities": ", ".join(parsed.get("entities", [])) if isinstance(parsed.get("entities"), list) else str(parsed.get("entities", "None")),
            "aircraft_components": ", ".join(parsed.get("aircraft_components", [])) if isinstance(parsed.get("aircraft_components"), list) else str(parsed.get("aircraft_components", "None")),
            "numerics": str(parsed.get("numerics", "None")).strip() or "None",
        }
    except Exception:
        out = _metadata_fallback(chunk_text)

    metadata_cache[key] = out
    return out


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chunk markdown files, embed, and upsert to Pinecone."
    )
    parser.add_argument(
        "--mode",
        choices=["advanced", "baseline"],
        default="advanced",
        help="Ingestion mode. baseline disables chunk-level extraction and stores only report-level metadata.",
    )
    parser.add_argument(
        "--strategy",
        choices=[
            "md_recursive",
            "parent_child",
            "fixed",
            "recursive",
            "semantic",
            "section",
            "parent",
        ],
        default="md_recursive",
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
        "--embed-batch-size",
        type=int,
        default=64,
        help="Embedding encode batch size.",
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
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Limit markdown processing to the first N files (0 means all files).",
    )
    parser.add_argument(
        "--from-prebuilt",
        action="store_true",
        help=(
            "Load chunks from pre-built JSON in data/processed/ instead of re-chunking. "
            "Skips chunking and provenance generation — uses contextualized_text already in the JSON."
        ),
    )
    return parser.parse_args()


def load_prebuilt_chunks(strategy: str, mode: str) -> list[dict]:
    """Load pre-built chunk JSON from data/processed/ without re-chunking."""
    if mode == "baseline":
        filename = PROCESSED_DIR / f"chunks_baseline_{strategy}.json"
    else:
        filename = PROCESSED_DIR / f"chunks_md_{strategy}.json"

    if not filename.exists():
        raise FileNotFoundError(
            f"Pre-built chunk file not found: {filename}\n"
            "Unzip chunks_md_recursive.zip into data/processed/ first."
        )

    print(f"Loading pre-built chunks from: {filename}")
    with filename.open("r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks.")
    return chunks


def canonical_strategy(strategy: str, mode: str) -> str:
    """Map legacy aliases while keeping baseline strategies distinct."""
    if strategy in {"section", "md_recursive"}:
        return "md_recursive"
    if strategy in {"parent", "parent_child"}:
        return "parent_child"

    if mode == "advanced" and strategy in {"fixed", "recursive", "semantic"}:
        raise ValueError(
            f"Advanced mode supports only md_recursive/parent_child (got: {strategy}). "
            "Use --mode baseline for fixed/recursive/semantic."
        )
    if mode == "baseline" and strategy in {"md_recursive", "parent_child"}:
        raise ValueError(
            f"Baseline mode supports only fixed/recursive/semantic (got: {strategy})."
        )

    return strategy


def add_provenance_context(
    chunks: list[dict],
    use_ollama_summary: bool,
    summarizer_model: str,
    summary_timeout: int,
    ollama_summary_max_chunks: int,
) -> list[dict]:
    """Add context summaries and bake provenance headers into embedding text."""
    summary_cache: dict[str, dict[str, str]] = {}
    total = len(chunks)
    start_ts = time.time()
    print(
        f"Starting context summarization for {total} chunks "
        f"(ollama enabled: {use_ollama_summary}, ollama budget: {ollama_summary_max_chunks or 'unlimited'})",
        flush=True,
    )

    ollama_budget = max(0, ollama_summary_max_chunks)

    for idx, chunk in enumerate(chunks, start=1):
        should_use_ollama = use_ollama_summary and (ollama_budget == 0 or idx <= ollama_budget)
        extracted = _extract_context_metadata(
            chunk_text=chunk.get("text", ""),
            summarizer_model=summarizer_model,
            use_ollama_summary=should_use_ollama,
            summary_timeout=summary_timeout,
            metadata_cache=summary_cache,
        )
        summary = extracted.get("context_summary", "No context available")
        chunk["entities"] = extracted.get("entities", "None")
        chunk["aircraft_components"] = extracted.get("aircraft_components", "None")
        chunk["numerics"] = extracted.get("numerics", "None")
        chunk["context_summary"] = summary
        report_id = chunk.get("report_id", "unknown")
        ntsb_no = chunk.get("ntsb_no", report_id)
        event_date = chunk.get("event_date", "unknown")
        make = chunk.get("make", "unknown")
        model = chunk.get("model", "unknown")
        section_title = chunk.get("section_title", "Unknown Section")
        role = chunk.get("role", "Unknown")

        chunk["contextualized_text"] = (
            f"[REPORT: {ntsb_no} | DATE: {event_date} | AIRCRAFT: {make} {model}]\n"
            f"[SECTION: {section_title} | ROLE: {role}]\n"
            f"[CHUNK SUMMARY: {summary}]\n"
            f"[ENTITIES: {chunk.get('entities', 'None')} | COMPONENTS: {chunk.get('aircraft_components', 'None')} | NUMERICS: {chunk.get('numerics', 'None')}]\n"
            "--- RAW TEXT ---\n"
            f"{chunk.get('text', '')}"
        )

        if idx % 25 == 0 or idx == total:
            elapsed = max(time.time() - start_ts, 1e-6)
            rate = idx / elapsed
            remaining = max(total - idx, 0)
            eta_sec = int(remaining / max(rate, 1e-6))
            eta_h = eta_sec // 3600
            eta_m = (eta_sec % 3600) // 60
            source_mode = "ollama" if should_use_ollama else "fallback"
            print(
                f"  Summary progress: {idx}/{total} chunks summary done "
                f"| mode: {source_mode} | elapsed: {elapsed/60:.1f}m "
                f"| rate: {rate:.2f}/s | ETA: {eta_h:02d}:{eta_m:02d}",
                flush=True,
            )

    return chunks


def _normalize_chunk_metadata(chunk: dict, md_file: Path, mode: str) -> dict:
    """Ensure required provenance keys exist regardless of chunking strategy."""
    report_id = chunk.get("report_id") or chunk.get("ntsb_no") or md_file.stem
    section_title = chunk.get("section_title", "")
    text = chunk.get("text", "")

    if not chunk.get("source_filename"):
        chunk["source_filename"] = md_file.name
    if not chunk.get("entity_id"):
        chunk["entity_id"] = report_id
    if not chunk.get("report_id"):
        chunk["report_id"] = report_id
    if not chunk.get("ntsb_no"):
        chunk["ntsb_no"] = report_id
    if not chunk.get("event_date"):
        chunk["event_date"] = "unknown"
    if not chunk.get("make"):
        chunk["make"] = "unknown"
    if not chunk.get("model"):
        chunk["model"] = "unknown"

    if mode == "advanced":
        if not chunk.get("role"):
            chunk["role"] = infer_role(section_title, text)
        if not chunk.get("entities"):
            chunk["entities"] = "None"
        if not chunk.get("aircraft_components"):
            chunk["aircraft_components"] = "None"
        if not chunk.get("numerics"):
            chunk["numerics"] = "None"
    else:
        # Baseline keeps only report-level metadata fields.
        chunk.pop("role", None)
        chunk.pop("context_summary", None)
        chunk.pop("entities", None)
        chunk.pop("aircraft_components", None)
        chunk.pop("numerics", None)
        chunk.pop("parent_text", None)

    return chunk


def load_markdown_chunks(strategy: str, mode: str, max_files: int = 0) -> list[dict]:
    md_files = sorted(MD_DIR.glob("*.md"))
    if not md_files:
        raise RuntimeError(f"No markdown files found in: {MD_DIR}")

    if max_files and max_files > 0:
        md_files = md_files[:max_files]

    print(f"Found {len(md_files)} markdown files in {MD_DIR}")
    chunks: list[dict] = []
    for idx, md_file in enumerate(md_files, start=1):
        print(f"  [{idx}/{len(md_files)}] Chunking file: {md_file.name}", flush=True)
        if mode == "advanced":
            if strategy in {"md_recursive", "section"}:
                file_chunks = chunk_markdown_md_recursive(str(md_file))
            elif strategy in {"parent_child", "parent"}:
                file_chunks = chunk_markdown_parent_child(str(md_file))
            else:
                raise ValueError(f"Unsupported advanced strategy: {strategy}")
        else:
            if strategy == "fixed":
                file_chunks = chunk_markdown_baseline_fixed(str(md_file))
            elif strategy == "recursive":
                file_chunks = chunk_markdown_baseline_recursive(str(md_file))
            elif strategy == "semantic":
                file_chunks = chunk_markdown_baseline_semantic(str(md_file))
            else:
                raise ValueError(f"Unsupported baseline strategy: {strategy}")

        file_chunks = [_normalize_chunk_metadata(c, md_file, mode=mode) for c in file_chunks]

        chunks.extend(file_chunks)
        print(
            f"      -> {len(file_chunks)} chunks from {md_file.name} | cumulative: {len(chunks)}",
            flush=True,
        )

        if idx % 25 == 0 or idx == len(md_files):
            print(f"  Chunked {idx}/{len(md_files)} files -> {len(chunks)} chunks", flush=True)

    return chunks


def embed_chunks(chunks: list[dict], model_name: str, embed_batch_size: int) -> np.ndarray:
    print(f"Loading embedding model: {model_name}")
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

    texts = [c.get("contextualized_text", c.get("text", "")) for c in chunks]
    total = len(texts)
    print(f"Encoding {total} chunks...")

    batch_size = max(1, int(embed_batch_size))
    all_vectors: list[np.ndarray] = []
    start_ts = time.time()

    for i in range(0, total, batch_size):
        batch = texts[i : i + batch_size]
        try:
            vec = model.encode(texts=batch, task="retrieval", prompt_name="passage")
        except ValueError:
            vec = model.encode(texts=batch, task="retrieval")
        all_vectors.append(np.asarray(vec, dtype=np.float32))

        done = min(i + batch_size, total)
        elapsed = max(time.time() - start_ts, 1e-6)
        rate = done / elapsed
        eta_sec = int((total - done) / max(rate, 1e-6))
        eta_h, eta_m, eta_s = eta_sec // 3600, (eta_sec % 3600) // 60, eta_sec % 60
        print(
            f"  Encoded {done}/{total} | elapsed: {elapsed/60:.1f}m "
            f"| rate: {rate:.1f}/s | ETA: {eta_h:02d}:{eta_m:02d}:{eta_s:02d}",
            flush=True,
        )

    embeddings = np.concatenate(all_vectors, axis=0)
    total_elapsed = time.time() - start_ts
    print(f"Embeddings shape: {embeddings.shape} | total time: {total_elapsed/60:.1f}m")
    return embeddings


def save_local_artifacts(chunks: list[dict], embeddings: np.ndarray, strategy: str, mode: str) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if mode == "baseline":
        chunks_path = PROCESSED_DIR / f"chunks_baseline_{strategy}.json"
        emb_path = PROCESSED_DIR / f"embeddings_baseline_{strategy}.npz"
    else:
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


def upsert(index, chunks: list[dict], embeddings: np.ndarray, strategy: str, mode: str, batch_size: int) -> None:
    vectors = []
    for chunk, emb in zip(chunks, embeddings):
        base_meta = {
            "report_id": chunk.get("report_id", ""),
            "ntsb_no": chunk.get("ntsb_no", chunk.get("report_id", "")),
            "event_date": chunk.get("event_date", "unknown"),
            "make": chunk.get("make", "unknown"),
            "model": chunk.get("model", "unknown"),
            "strategy": strategy,
            "mode": mode,
            "source": "dataset-pipeline/data/extracted/extracted",
        }

        if mode == "advanced":
            base_meta.update(
                {
                    "entity_id": chunk.get("entity_id", chunk.get("report_id", "")),
                    "source_filename": chunk.get("source_filename", ""),
                    "section_title": chunk.get("section_title", ""),
                    "role": chunk.get("role", "Unknown"),
                    "context_summary": chunk.get("context_summary", ""),
                    "entities": chunk.get("entities", "None"),
                    "aircraft_components": chunk.get("aircraft_components", "None"),
                    "numerics": chunk.get("numerics", "None"),
                    "parent_id": chunk.get("parent_id", ""),
                }
            )

        vectors.append(
            {
                "id": chunk["chunk_id"],
                "values": emb.tolist(),
                "metadata": base_meta,
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
    strategy = canonical_strategy(args.strategy, mode=args.mode)

    if args.from_prebuilt:
        chunks = load_prebuilt_chunks(strategy, mode=args.mode)
        # contextualized_text already baked in — ensure fallback for any missing ones
        for chunk in chunks:
            if not chunk.get("contextualized_text"):
                chunk["contextualized_text"] = chunk.get("text", "")
    else:
        chunks = load_markdown_chunks(strategy, mode=args.mode, max_files=args.max_files)
        if args.mode == "advanced":
            chunks = add_provenance_context(
                chunks=chunks,
                use_ollama_summary=args.use_ollama_summary,
                summarizer_model=args.summarizer_model,
                summary_timeout=args.summary_timeout,
                ollama_summary_max_chunks=args.ollama_summary_max_chunks,
            )
        else:
            for chunk in chunks:
                chunk["contextualized_text"] = chunk.get("text", "")

    embeddings = embed_chunks(chunks, args.model_name, args.embed_batch_size)
    save_local_artifacts(chunks, embeddings, strategy, mode=args.mode)

    index = init_index(
        index_name=args.index_name,
        dimension=int(embeddings.shape[1]),
        reset_index=args.reset_index,
    )
    upsert(
        index=index,
        chunks=chunks,
        embeddings=embeddings,
        strategy=strategy,
        mode=args.mode,
        batch_size=args.batch_size,
    )

    print("\nIndex stats:")
    print(index.describe_index_stats())


if __name__ == "__main__":
    main()
