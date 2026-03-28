"""
Clean full report text and chunk documents using multiple strategies.

A. Fixed-size character splitting
B. Recursive character splitting
C. Semantic chunking
D. Parent-child chunking
"""
import json
import os
import re

import pandas as pd
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SAMPLE_PATH = os.path.join(BASE_DIR, "data", "processed", "sampled_reports.csv")
OUT_FIXED_PATH = os.path.join(BASE_DIR, "data", "processed", "chunks_fixed.json")
OUT_REC_PATH = os.path.join(BASE_DIR, "data", "processed", "chunks_recursive.json")
OUT_SEM_PATH = os.path.join(BASE_DIR, "data", "processed", "chunks_semantic.json")
OUT_PARENT_PATH = os.path.join(BASE_DIR, "data", "processed", "chunks_parent.json")


def clean_report(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"Page \d+\s*of\s*\d+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_metadata(row, chunk_idx, strategy):
    return {
        "chunk_id": f"{row['NtsbNo']}_{strategy}_{chunk_idx:03d}",
        "ntsb_no": str(row["NtsbNo"]),
        "event_date": str(row["EventDate"]),
        "state": str(row.get("State", "")),
        "make": str(row.get("Make", "")),
        "model": str(row.get("Model", "")),
        "phase_of_flight": str(row.get("BroadPhaseofFlight", "")),
        "weather": str(row.get("WeatherCondition", "")),
    }


def chunk_fixed(df):
    """Strategy A: Baseline fixed-size character splitting."""
    splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=200, separator="")

    chunks = []
    for _, row in df.iterrows():
        text = clean_report(row["rep_text"])
        header = (
            f"Accident {row['NtsbNo']} ({row.get('Make', '')} {row.get('Model', '')}, "
            f"{row.get('EventDate', '')[:10]}): "
        )

        for i, chunk_text in enumerate(splitter.split_text(text)):
            chunk_data = build_metadata(row, i, "fixed")
            chunk_data["text"] = header + chunk_text
            chunks.append(chunk_data)
    return chunks


def chunk_recursive(df):
    """Strategy B: Baseline recursive character splitting."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " "],
    )

    chunks = []
    for _, row in df.iterrows():
        text = clean_report(row["rep_text"])
        header = (
            f"Accident {row['NtsbNo']} ({row.get('Make', '')} {row.get('Model', '')}, "
            f"{row.get('EventDate', '')[:10]}): "
        )

        for i, chunk_text in enumerate(splitter.split_text(text)):
            chunk_data = build_metadata(row, i, "rec")
            chunk_data["text"] = header + chunk_text
            chunks.append(chunk_data)
    return chunks


def chunk_semantic(df):
    """Strategy C: Semantic chunking using embedding breakpoints."""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    semantic_chunker = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")

    chunks = []
    for idx, row in df.iterrows():
        text = clean_report(row["rep_text"])
        header = (
            f"Accident {row['NtsbNo']} ({row.get('Make', '')} {row.get('Model', '')}, "
            f"{row.get('EventDate', '')[:10]}): "
        )

        if len(text) < 100:
            doc_chunks = [text]
        else:
            try:
                doc_chunks = semantic_chunker.split_text(text)
            except Exception as e:
                print(
                    f"Warning: Semantic split failed for {row['NtsbNo']}, "
                    f"falling back to recursive. Error: {e}"
                )
                splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
                doc_chunks = splitter.split_text(text)

        for i, chunk_text in enumerate(doc_chunks):
            chunk_data = build_metadata(row, i, "sem")
            chunk_data["text"] = header + chunk_text
            chunks.append(chunk_data)

        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(df)} reports semantically...")

    return chunks


def chunk_markdown_section_aware(md_file_path: str):
    """Section-aware chunking for markdown reports."""
    with open(md_file_path, "r", encoding="utf-8") as f:
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

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " "],
    )

    chunks = []
    report_id = os.path.basename(md_file_path).replace(".md", "")

    for sec_idx, section in enumerate(parsed_sections):
        if not section["content"] or re.match(r"^[\W_]+$", section["content"]):
            continue

        for chunk_idx, chunk_text in enumerate(splitter.split_text(section["content"])):
            chunks.append(
                {
                    "chunk_id": f"{report_id}_sec{sec_idx:02d}_{chunk_idx:03d}",
                    "report_id": report_id,
                    "section_title": section["title"],
                    "text": f"Section: {section['title']}\n{chunk_text}",
                }
            )

    return chunks


def chunk_markdown_recursive(md_file_path: str):
    """Recursive markdown chunking without section boundaries."""
    with open(md_file_path, "r", encoding="utf-8") as f:
        content = f.read()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " "],
    )

    chunks = []
    report_id = os.path.basename(md_file_path).replace(".md", "")
    for chunk_idx, chunk_text in enumerate(splitter.split_text(content)):
        chunks.append(
            {
                "chunk_id": f"{report_id}_rec_{chunk_idx:03d}",
                "report_id": report_id,
                "text": chunk_text,
            }
        )

    return chunks


def chunk_parent(df):
    """Strategy D: Parent-child chunking for richer generation context."""
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1800,
        chunk_overlap=250,
        separators=["\n\n", "\n", ". ", " "],
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
        separators=["\n\n", "\n", ". ", " "],
    )

    chunks = []
    for _, row in df.iterrows():
        text = clean_report(row["rep_text"])
        header = (
            f"Accident {row['NtsbNo']} ({row.get('Make', '')} {row.get('Model', '')}, "
            f"{row.get('EventDate', '')[:10]}): "
        )

        parent_chunks = parent_splitter.split_text(text)
        for p_idx, parent_text in enumerate(parent_chunks):
            parent_id = f"{row['NtsbNo']}_parent_{p_idx:03d}"
            child_chunks = child_splitter.split_text(parent_text)

            for c_idx, child_text in enumerate(child_chunks):
                chunk_data = build_metadata(row, c_idx, f"parent{p_idx:03d}")
                chunk_data["text"] = header + child_text
                chunk_data["parent_id"] = parent_id
                chunk_data["parent_text"] = header + parent_text
                chunks.append(chunk_data)

    return chunks


def main():
    print(f"Loading data from {SAMPLE_PATH}")
    df = pd.read_csv(SAMPLE_PATH, sep=";", encoding="utf-8")
    print(f"Loaded {len(df)} reports.")

    print("\nRunning Strategy A: Fixed-Size Chunking...")
    chunks_fixed = chunk_fixed(df)
    with open(OUT_FIXED_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks_fixed, f, indent=2)
    print(f"  -> Generated {len(chunks_fixed)} fixed chunks")

    print("\nRunning Strategy B: Recursive Character Chunking...")
    chunks_rec = chunk_recursive(df)
    with open(OUT_REC_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks_rec, f, indent=2)
    print(f"  -> Generated {len(chunks_rec)} recursive chunks")

    print("\nRunning Strategy C: Semantic Chunking...")
    chunks_sem = chunk_semantic(df)
    with open(OUT_SEM_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks_sem, f, indent=2)
    print(f"  -> Generated {len(chunks_sem)} semantic chunks")

    print("\nRunning Strategy D: Parent-Child Chunking...")
    chunks_parent = chunk_parent(df)
    with open(OUT_PARENT_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks_parent, f, indent=2)
    print(f"  -> Generated {len(chunks_parent)} parent-child chunks")

    print("\nDone! Ready for embeddings.")


if __name__ == "__main__":
    main()
