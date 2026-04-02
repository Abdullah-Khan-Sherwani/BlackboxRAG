# BlackboxRAG Project Quick Understanding

This file is a fast onboarding map for new contributors (human or model).
It explains what each part of the repo does and where to find it.

## 1) What this project is

BlackboxRAG is a Retrieval-Augmented Generation (RAG) system for NTSB aviation accident reports.

High-level flow:
1. Build/select corpus
2. Chunk documents (multiple strategies)
3. Generate embeddings
4. Upsert vectors to Pinecone (`ntsb-rag`)
5. Retrieve (semantic + BM25 hybrid)
6. Rerank + answer generation
7. Evaluate faithfulness/relevancy and run ablations

Primary runtime UI:
- `streamlit run src/ui/app.py`

## 2) Where everything is (top-level map)

- `README.md`
  - High-level project description and stack.

- `QUICK_START.md`
  - Practical setup/run workflow, expected local artifacts, and strategy notes.

- `requirements.txt`
  - Main environment dependencies for app/retrieval/evaluation.

- `src/`
  - Main application code (retrieval, generation, eval, UI, prep).

- `scripts/`
  - Utility scripts for corpus prep and pipeline runs.

- `dataset-pipeline/`
  - Separate pipeline for downloading/extracting NTSB PDFs to markdown.

- `data/`
  - Raw dataset and processed artifacts (chunks, embeddings, eval outputs).

- `docs/`
  - Guides, reports, assignment/checklist docs.

- Root utility/experimental files:
  - `query.py` and `query_prod.py`: CLI query tools.
  - `retrieval_enhanced.py`, `demo_enhanced_rag.py`, `accuracy_check_tool.py`, `compare_chunks.py`: analysis/experimental helpers.

## 3) Main app code (`src/`)

### `src/ui/`
- `app.py`
  - Streamlit front-end.
  - Lets user pick strategy/mode/provider and run end-to-end RAG.
  - Loads retrieval models/indexes with Streamlit caching.

### `src/retrieval/`
- `query.py`
  - Semantic retrieval from Pinecone.
  - Resolves available local chunk artifacts per strategy.
  - Loads local chunk text by `chunk_id` and enriches Pinecone matches.

- `hybrid.py`
  - Hybrid retrieval stack:
    - BM25 retrieval
    - semantic retrieval
    - fusion (RRF)
    - cross-encoder reranking
    - optional query-expansion helpers

- `report_mapper.py`
  - Detects report IDs from query text and builds Pinecone metadata filters.
  - Ensures strategy-aware filtering (`strategy` metadata).

- `upsert.py`
  - Generic upsert helper for chunk+embedding artifacts into Pinecone.

### `src/generation/`
- `generate.py`
  - Prompt construction and answer generation.
  - Supports multiple LLM providers (local Ollama and API-backed models via centralized client).

### `src/evaluation/`
- `evaluate.py`
  - Computes faithfulness/relevancy metrics.
  - Runs multi-strategy eval loops.
  - Includes manual 10-question compare mode with reference answers.

- `ablation.py`
  - Ablation routines across retrieval/chunking settings.

### `src/data_prep/`
- `chunking.py`
  - Chunking functions for baseline and markdown-focused strategies.
  - Includes section-aware and recursive-style methods and metadata attachment helpers.

- `embeddings.py`
  - Embedding generation for chunk files and artifact save (`.npz`).

- `context_generator.py`
  - Optional context augmentation for chunks using local Ollama model.

### `src/llm/`
- `client.py`
  - Centralized LLM client.
  - Handles NVIDIA API-backed calls and optional HF eval backend.

- `ollama_client.py`
  - Local Ollama connectivity and calls used across pipeline/runtime.

## 4) Data extraction pipeline (`dataset-pipeline/`)

This is a distinct subsystem that prepares markdown from official NTSB PDFs.

- `dataset-pipeline/config/settings.py`
  - Paths, download settings, URL patterns, filtering thresholds.

- `dataset-pipeline/core/models.py`
  - Dataclasses/enums for report records and extraction results.

- `dataset-pipeline/core/tracker.py`
  - SQLite tracker (`tracker.db`) for download/extraction status and resumability.

- `dataset-pipeline/extraction/docling_extractor.py`
  - PDF -> structured markdown using Docling.

- `dataset-pipeline/processing/validator.py`
  - Quality checks for extracted markdown (word count, required sections).

- `dataset-pipeline/scripts/download_aars.py`
  - Enumerates and downloads AAR/AIR PDFs.

- `dataset-pipeline/data/`
  - Pipeline-local raw/pdf/extracted/output folders and logs.

## 5) Data and artifacts (`data/`)

### `data/raw/`
- Canonical source CSV dataset.

### `data/processed/`
Common artifacts used by retrieval/eval:
- Chunk JSON files (strategy-specific), e.g.:
  - `chunks_fixed.json`
  - `chunks_recursive.json`
  - `chunks_semantic.json`
  - plus markdown strategy artifacts (including nested folder variants)
- Embedding files (`*.npz`)
- Metadata and report outputs, e.g.:
  - `sampled_reports.csv`
  - `report_metadata.json`
  - `RAG_ANALYSIS_SUMMARY.csv`

### Strategy naming in current code
Current retrieval code supports these strategy labels:
- `section`
- `md_recursive`
- `parent_child`
- `fixed`
- `recursive`
- `semantic`

Retrieval code includes compatibility logic for legacy/alternate file names in `data/processed/` and `data/processed/chunks_md_recursive/`.

## 6) Operational scripts (`scripts/`)

- `scripts/build_corpus.py`
  - Builds the curated sampled corpus from the raw CSV.

- `scripts/extract_metadata.py`
  - Extracts structured metadata from markdown reports (regex + LLM-assisted fields).

- `scripts/run_pipeline.py`
  - End-to-end script for chunking and optional context generation.
  - Upsert stage is present as a placeholder in this script; dedicated upsert scripts/modules are used in practice.

## 7) Runtime pathways

### A) Typical user-facing pathway
1. Start UI: `streamlit run src/ui/app.py`
2. UI loads Jina model, Pinecone index, reranker, and BM25 index for selected strategy.
3. Retrieval returns chunks.
4. Generator produces cited answer.
5. Eval metrics (faithfulness/relevancy) can be computed in UI flow.

### B) Evaluation pathway
1. Run `src/evaluation/evaluate.py`.
2. For each query/strategy/mode:
   - retrieve contexts
   - generate answer
   - score faithfulness + relevancy
3. Persist CSV outputs in `data/processed/`.

### C) Data preparation pathway
1. Build/select source corpus (`scripts/build_corpus.py`).
2. Produce chunks (`src/data_prep/chunking.py` or markdown pipeline outputs).
3. Generate embeddings (`src/data_prep/embeddings.py`).
4. Upsert to Pinecone (`src/retrieval/upsert.py` and related scripts).

## 8) Environment and external dependencies

Expected env vars in `.env`:
- `PINECONE_API_KEY`
- `NVIDIA_API_KEY` (if using NVIDIA-hosted models)
- Optional eval provider/model vars for HF-based judge mode.

Core external systems:
- Pinecone index: `ntsb-rag`
- Embeddings: Jina v5 model in retrieval path
- Local LLM option: Ollama (`qwen2.5:32b` often used)

## 9) What to read first as a newcomer

If you only have 15 minutes, read in this order:
1. `README.md`
2. `QUICK_START.md`
3. `src/ui/app.py`
4. `src/retrieval/query.py`
5. `src/retrieval/hybrid.py`
6. `src/generation/generate.py`
7. `src/evaluation/evaluate.py`
8. `dataset-pipeline/config/settings.py`
9. `dataset-pipeline/extraction/docling_extractor.py`

## 10) Known repo reality notes

- There are both "core production" modules under `src/` and additional root-level experimental/legacy tools.
- Artifact naming/layout has evolved; retrieval now includes fallback resolution across old/new names and folders.
- Not every script is fully symmetric with the latest strategy set, so prefer `src/ui/app.py` + retrieval modules as canonical runtime path.
