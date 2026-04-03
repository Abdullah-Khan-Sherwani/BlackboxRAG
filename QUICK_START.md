# QUICK START — BlackboxRAG (NTSB Aviation Accident RAG System)

## Overview

Retrieval-Augmented Generation over 101 NTSB aviation accident reports.  
Query → Hybrid Retrieval (Pinecone + BM25 + RRF) → Cross-encoder Rerank → LLM Answer.

**The only entry point that matters:**
```bash
streamlit run src/ui/app.py
```

---

## Stack (current)

| Component      | Choice                                |
|----------------|---------------------------------------|
| Vector DB      | Pinecone (`ntsb-rag`, cosine, 768-dim)|
| Embedding      | Jina v5 (`jina-embeddings-v5-text-nano`, 768-dim) |
| Lexical Search | BM25 (`rank_bm25`)                    |
| Reranker       | `cross-encoder/qnli-distilroberta-base`|
| LLM (default)  | Ollama local (`qwen2.5:32b`)          |
| LLM (API)      | DeepSeek V3.1 or GPT-4o via NVIDIA API|
| UI             | Streamlit                             |

---

## Collaborator Workflow — How We Share Files

Large chunk files cannot be committed to Git directly (they are gitignored).  
We share them via **`data/processed/chunks_md_recursive.zip`**, which IS committed.

**Workflow:**
1. Pull latest from GitHub
2. Extract the zip: `cd data/processed && python -c "import zipfile; zipfile.ZipFile('chunks_md_recursive.zip').extractall('.')"`
3. Copy your `.env` file into the project root (see API Keys below)
4. Run: `streamlit run src/ui/app.py`

**When you add new chunk files (e.g. baseline strategies), rebuild the zip and push:**
```bash
cd data/processed
python -c "
import zipfile
files = [
    'chunks_md_section.json',
    'chunks_md_section_enriched.json',
    'chunks_md_md_recursive.json',
    'chunks_md_parent_child.json',
    'chunks_md_recursive.json',
    'chunks_baseline_recursive.json',   # add when ready
    'chunks_baseline_semantic.json',    # add when ready
]
import os
with zipfile.ZipFile('chunks_md_recursive.zip', 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for f in files:
        if os.path.exists(f):
            z.write(f)
            print('Added:', f)
"
```
Then `git add data/processed/chunks_md_recursive.zip && git commit && git push`.

---

## API Keys

Create a file at `.env` in the project root (this file is gitignored — never commit it):

```
PINECONE_API_KEY=<your key>
NVIDIA_API_KEY=<your key>        # for DeepSeek/GPT via NVIDIA API
```

There is also a copy at `data/processed/env` (plaintext, gitignored) used as a fallback
by `scripts/upsert_section_chunks.py`. Keep both in sync.

---

## Required Local Files

### Files tracked in Git (always present after pull)
```
src/                          all source code
scripts/                      operational scripts
data/processed/
    chunks_md_recursive.zip   THE zip — extract this first
    embeddings_md_section.npz pre-built section embeddings (gitignored via *.npz)
requirements.txt
.env.example
```

### Files produced by extracting the zip
```
data/processed/
    chunks_md_section.json          section strategy chunks  (23,381 chunks)
    chunks_md_section_enriched.json section chunks + report metadata (23,381 chunks)
    chunks_md_md_recursive.json     md_recursive strategy chunks (89,675 chunks)
    chunks_md_parent_child.json     parent_child strategy chunks  (26,520 chunks)
    chunks_md_recursive.json        legacy alias — same as md_recursive
```

### Files NOT in the zip (collaborator adds separately)
```
data/processed/
    chunks_baseline_recursive.json  baseline recursive — friend upserting this
    chunks_baseline_semantic.json   baseline semantic  — friend upserting this
```
Once those files exist locally, they will automatically appear in the UI strategy dropdown
(controlled by `available_strategies()` in `src/retrieval/query.py`).

---

## Pinecone Index State (`ntsb-rag`)

**Total vectors: 139,576** (as of Apr 2, 2026)

| Strategy      | Vectors  | Chunk ID pattern         | Status        |
|---------------|----------|--------------------------|---------------|
| `md_recursive`| ~88,000  | `AAR0001_mdrec_000_000`  | live          |
| `parent_child`| ~28,000  | `AAR0001_pchild_000_000` | live          |
| `section`     | ~23,381  | `AAR0001_sec74_002`      | live          |
| `recursive`   | —        | `AAR0001_base_rec_000`   | pending (friend)|
| `semantic`    | —        | `AAR0001_base_sem_000`   | pending (friend)|

---

## Critical Section Naming Change

There is a naming distinction that caused bugs — be aware of it:

| Strategy name | Local file                        | Chunk function                        | Chunking method                              |
|---------------|-----------------------------------|---------------------------------------|----------------------------------------------|
| `section`     | `chunks_md_section.json`          | original `chunk_markdown_section_aware()` (lines 130–212 of `chunking.py`) | Regex splits on `##` headers |
| `md_recursive`| `chunks_md_md_recursive.json`     | `chunk_markdown_md_recursive()` (line 486) | LangChain `MarkdownHeaderTextSplitter` + `RecursiveCharacterTextSplitter` |

**The trap:** `chunking.py` lines 581–586 define backward-compatible aliases:
```python
def chunk_markdown_section_aware(md_file_path):   # line 581 — OVERRIDES the original
    return chunk_markdown_md_recursive(md_file_path)

def chunk_markdown_recursive(md_file_path):        # line 585
    return chunk_markdown_md_recursive(md_file_path)
```
Both aliases now produce `md_recursive` output. The original regex-based section chunking
at line 130 is **dead code** (overridden by the alias at 581).

`chunks_md_section.json` was generated before the alias was written — it contains the
true regex-based section chunks. It is a distinct dataset from `chunks_md_md_recursive.json`.

**In the UI:** "section" and "md_recursive" are genuinely different chunking strategies
with separate Pinecone vectors.

---

## Bugs Found and Fixed (Apr 2, 2026)

### Bug 1 — Text lookup always returned empty strings
**File:** `src/retrieval/query.py`, `retrieve()` function  
**Problem:** Text was hardcoded to look up from `chunks_md_section.json` first, then one
alternate strategy. Since Pinecone returns `_mdrec_` and `_pchild_` IDs, not `_sec_` IDs,
`section_dict.get(match.id)` always returned `{}` → `match.metadata["text"] = ""`.  
**Fix:** Each Pinecone match now routes to the correct local chunk store via
`match.metadata.get("strategy")`, lazy-loaded per strategy. Correct text is returned for
all three strategies including parent-context expansion for `parent_child`.

### Bug 2 — Strategy selector in UI was cosmetic (no effect on Pinecone results)
**File:** `src/retrieval/report_mapper.py`, `get_pinecone_filter()`  
**Problem:** Strategy filter was removed in a prior commit with the comment "index may only
contain one strategy". With `md_recursive`, `parent_child`, and `section` now all present
in Pinecone, returning `{}` as the filter caused every strategy to return the same mixed
results.  
**Fix:** Filter always includes `{"strategy": {"$eq": strategy}}`. For specific-report
queries, it adds `{"ntsb_no": ...}` via `$and`.

### Bug 3 — "parent" and "parent_child" both appeared in UI dropdown
**Files:** `src/retrieval/query.py` (`ALL_STRATEGIES`), `src/retrieval/hybrid.py` (`STRATEGIES`)  
**Problem:** Legacy alias `"parent"` mapped to the same file as `"parent_child"`, creating
a duplicate entry in the strategy dropdown.  
**Fix:** Removed `"parent"` from both lists. `_canonical_strategy("parent")` still maps to
`"parent_child"` for any legacy callers.

### Bug 4 — `upsert_section_chunks.py` would regenerate embeddings from scratch
**File:** `scripts/upsert_section_chunks.py`  
**Problem:** The old script re-embedded all 23,381 chunks using the Jina model instead of
loading the pre-built `embeddings_md_section.npz`. It also referenced a wrong filename
(`embeddings_section.npz`) and had sparse metadata.  
**Fix:** Rewrote the script to load `embeddings_md_section.npz` directly and apply full
report-level metadata from `chunks_md_section_enriched.json`.

### Bug 5 — Section chunks had no report metadata in Pinecone
**Files:** `scripts/enrich_section_chunks.py` (new), `scripts/upsert_section_chunks.py`  
**Problem:** `chunks_md_section.json` only had 4 fields: `chunk_id`, `report_id`,
`section_title`, `text`. Pinecone vectors for section would be missing `ntsb_no`,
`event_date`, `make`, `model`, `state`, etc.  
**Fix:** `enrich_section_chunks.py` joins `chunks_md_section.json` with
`chunks_md_md_recursive.json` on `report_id` to copy all safe report-level fields (101/101
reports matched). The enriched output is `chunks_md_section_enriched.json` — this is what
gets upserted.

---

## Setup from Scratch (New Machine)

```bash
# 1. Clone
git clone https://github.com/Abdullah-Khan-Sherwani/BlackboxRAG.git
cd BlackboxRAG

# 2. Create virtualenv and install
python -m venv venv
venv/Scripts/activate          # Windows
# source venv/bin/activate     # Mac/Linux
pip install -r requirements.txt

# 3. Extract chunk files from zip
cd data/processed
python -c "import zipfile; zipfile.ZipFile('chunks_md_recursive.zip').extractall('.')"
cd ../..

# 4. Add your .env file (get keys from a teammate)
# .env contents:
#   PINECONE_API_KEY=...
#   NVIDIA_API_KEY=...

# 5. Run
streamlit run src/ui/app.py
```

---

## Adding a New Chunking Strategy (for friend adding baseline strategies)

1. Generate your chunk JSON and save to `data/processed/chunks_baseline_recursive.json`
   (or `chunks_baseline_semantic.json`)
2. Generate embeddings, save as `data/processed/embeddings_baseline_recursive.npz`
   with keys `chunk_ids` (array) and `embeddings` (float32 array, shape N×768)
3. Upsert to Pinecone — each vector's metadata must include `"strategy": "recursive"`
   (or `"semantic"`) so the filter works
4. Add the json file to `chunks_md_recursive.zip` (see zip rebuild command above) and push
5. The strategy will automatically appear in the UI and evaluation once the local file exists

The Pinecone filter relies on the `strategy` metadata field. Chunk IDs should follow the
pattern `REPORTID_base_rec_NNN` / `REPORTID_base_sem_NNN`.

---

## Running Evaluation and Ablation

```bash
# Evaluation (all available strategies, semantic + hybrid)
venv/Scripts/python.exe src/evaluation/evaluate.py

# Fast run (5 queries, top_k=3, BM25 fallback if Pinecone down)
venv/Scripts/python.exe src/evaluation/evaluate.py --fast --allow-bm25-fallback

# Manual QA comparison (10 specific reference questions)
venv/Scripts/python.exe src/evaluation/evaluate.py --manual-qa-only

# Ablation study (all strategies x all modes)
venv/Scripts/python.exe src/evaluation/ablation.py

# Fresh ablation run
venv/Scripts/python.exe src/evaluation/ablation.py --fresh
```

Results saved to:
- `data/evaluation_results.csv`
- `data/ablation_detailed.csv`
- `data/ablation_summary.csv`

---

## Re-upserting Section Chunks (only needed if Pinecone index is reset)

```bash
# Step 1 — enrich section chunks with report metadata (~10 seconds, no network)
venv/Scripts/python.exe scripts/enrich_section_chunks.py

# Step 2 — upsert to Pinecone (~5-10 minutes)
venv/Scripts/python.exe scripts/upsert_section_chunks.py
```

---

## Key Source Files

```
src/
  retrieval/
    query.py          Core retrieval: Pinecone query, text lookup routing, available_strategies()
    hybrid.py         BM25, RRF fusion, reranking, query expansion, neighbor enrichment
    report_mapper.py  Pinecone filter builder: strategy filter + optional NTSB-number filter
    upsert.py         Legacy upsert for old CSV-based strategies (not currently used)
  data_prep/
    chunking.py       All chunking strategies — see NAMING CHANGE section above
  evaluation/
    evaluate.py       Faithfulness + Relevancy metrics, run_evaluation(), MANUAL_COMPARE_QA
    ablation.py       Ablation study runner
  ui/
    app.py            Streamlit UI
  generation/
    generate.py       LLM answer generation
  llm/
    client.py         NVIDIA API client (DeepSeek / GPT)
    ollama_client.py  Local Ollama client

scripts/
  enrich_section_chunks.py   Joins section chunks with md_recursive metadata
  upsert_section_chunks.py   Upserts enriched section chunks to Pinecone
  build_corpus.py            Selects 101 reports from raw dataset
```
