# Aviation Safety RAG: Implementation Guide

**Last Updated:** 2026-03-20
**Dataset:** `final_reports_2016-23_cons_2024-12-24.csv` (Garcia et al., Zenodo, 2025)
**Deadline:** April 5, 2026

---

## Problem Statement

Aviation safety professionals cannot reason *across* NTSB accident reports — existing tools (NTSB CAROL, crashinvestigations.ai, etc.) treat each report in isolation. No interactive system supports reasoning across a corpus to surface patterns, synthesize causal chains, or track safety recommendations at scale.

We build a RAG system over 75-100 NTSB final investigation reports (2016-2023, Garcia et al., Zenodo, 2025) that generates answers grounded in and cited to specific NtsbNo report IDs, evaluated via a systematic ablation study.

### Query Classes

The system supports a **spectrum of queries** through a single retrieval pipeline — the same hybrid search runs for all queries. What differs is the LLM prompt template and how many reports the retriever naturally pulls back:

1. **Single-accident Q/A** — "What caused ERA24LA084?" or "Describe the Cessna crash near Dallas in 2022." The retriever focuses on one report. The user does NOT need to know the NtsbNo — semantic search matches by description, location, date, aircraft, etc.

2. **Cross-document comparison & synthesis** — "Compare engine failure patterns in Cessna vs Piper" or "What recurring factors contribute to loss-of-control in IMC?" The retriever pulls chunks from multiple reports; the LLM synthesizes patterns and cites each source.

These are not hard categories — they blend naturally. A user can start with a single-accident query and follow up with "Were there similar accidents?" and the system will retrieve related reports and compare. The pipeline does not restrict itself to one report unless the query naturally narrows to one.

> **Stretch goal (if time permits):** Safety recommendation tracking queries linking findings to regulatory actions.

---

## Mandated Technical Stack

These are assignment requirements, not suggestions.

| Component | Required | Notes |
|-----------|----------|-------|
| **Vector DB** | Pinecone (Free Starter) | Cloud-based. Do NOT store indexes locally. |
| **Query Embedding** | `all-MiniLM-L6-v2` | Runs on-app in HF Space. 384-dim vectors. |
| **Pre-computed Embeddings** | Generate locally | Upsert to Pinecone ahead of time. |
| **LLM Generation** | HF Inference API | Mistral-7B, Llama-3-8B, or TinyAya. |
| **Retrieval** | Hybrid + Reranking | BM25 + Semantic + RRF or Cross-Encoder. Mandatory. |
| **Hosting** | HF Spaces | Streamlit or Gradio. Must be publicly accessible. |
| **Evaluation** | LLM-as-a-Judge | Faithfulness + Relevancy on 10-20 test queries. |
| **Ablation** | Required | Compare chunking strategies + retrieval modes. |

### Swappable Components (future flexibility)

These can be changed without breaking the architecture:

| Component | Current Choice | Alternatives |
|-----------|---------------|-------------|
| Embedding model (local) | `all-MiniLM-L6-v2` (384d) | `all-mpnet-base-v2` (768d), `bge-small-en-v1.5` |
| LLM | Mistral-7B via HF API | Llama-3-8B, TinyAya, any HF Inference model |
| Reranker | Cross-encoder (`qnli-distilroberta-base`) | NVIDIA NIM reranker, Cohere rerank |
| BM25 implementation | `rank_bm25` Python library | Elasticsearch, custom TF-IDF |
| Evaluation framework | `deepeval` | Custom LLM-as-a-Judge, RAGAS |
| UI framework | Streamlit | Gradio |
| Corpus size | 75-100 reports | Up to 7,462 available |

---

## Dataset Summary

- **Source:** 7,462 unique NTSB final reports, semicolon-delimited CSV
- **We use:** 75-100 sampled reports (assignment requirement: 50-100 docs, 500+ chunks)
- **Key columns:** `rep_text` (full report, ~10.5K chars avg), `ProbableCause`, `Findings`, `NtsbNo`, `EventDate`, `State`, `Make`, `Model`, `BroadPhaseofFlight`, `WeatherCondition`
- **Quality:** 100% non-null `rep_text`, 99.7%+ metadata completeness, clean UTF-8

---

## Architecture Overview

```
User Query
    |
    v
[all-MiniLM-L6-v2] ---> Dense Embedding
    |                         |
    |                    [Pinecone] ---> Top-K dense results
    |                         |
    +---> [BM25 Index] ----> Top-K lexical results
                              |
                         [RRF Fusion] ---> Merged ranked list
                              |
                         [Reranker] ---> Final top-K chunks
                              |
                    [HF Inference API] ---> Generated answer + citations
                              |
                         [Streamlit UI] ---> Display answer, chunks, scores
```

---

## Phase 1: Data Preparation (Day 1)

### 1.1 Load and Sample

```python
import pandas as pd

df = pd.read_csv('final_reports_2016-23_cons_2024-12-24.csv', sep=';', encoding='utf-8')
df = df.drop_duplicates(subset=['rep_text'], keep='first')

# Sample 75-100 reports with diversity across years, states, aircraft
# Strategy: stratified sample to cover different accident types
sample = df.groupby(df['EventDate'].str[:4]).apply(
    lambda x: x.sample(min(len(x), 12), random_state=42)
).reset_index(drop=True)

# Verify we hit 75-100 range; adjust per-year count if needed
print(f"Sampled {len(sample)} reports")
```

### 1.2 Clean Text

```python
import re

def clean_report(text):
    text = re.sub(r'Page \d+ of \d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

sample['rep_text_clean'] = sample['rep_text'].apply(clean_report)
```

### 1.3 Chunking (Two Strategies for Ablation)

**Strategy A: Recursive Character Splitting**
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter_recursive = RecursiveCharacterTextSplitter(
    chunk_size=1500,        # adjustable
    chunk_overlap=200,      # adjustable
    separators=["\n\n", "\n", ". ", " "]
)
```

**Strategy B: Section-Aware / Semantic Chunking**
```python
# Split on NTSB report section headers, then sub-split large sections
SECTION_MARKERS = [
    "History of Flight", "Pilot Information", "Aircraft Information",
    "Meteorological Information", "Wreckage and Impact Information",
    "Medical and Pathological Information", "Probable Cause",
    "Findings", "Recommendations"
]
# Implementation: split at markers, then apply recursive splitter to oversized sections
```

### 1.4 Build Chunk Records

```python
chunks = []
for _, row in sample.iterrows():
    doc_chunks = splitter.split_text(row['rep_text_clean'])
    for i, chunk_text in enumerate(doc_chunks):
        chunks.append({
            'chunk_id': f"{row['NtsbNo']}_{i:03d}",
            'text': chunk_text,
            'ntsb_no': row['NtsbNo'],
            'event_date': row['EventDate'],
            'state': row.get('State', ''),
            'make': row.get('Make', ''),
            'model': row.get('Model', ''),
            'phase_of_flight': row.get('BroadPhaseofFlight', ''),
            'weather': row.get('WeatherCondition', ''),
            'probable_cause': row.get('ProbableCause', ''),
        })

print(f"Generated {len(chunks)} chunks from {len(sample)} reports")
# Expect 500-2000 chunks depending on strategy
```

---

## Phase 2: Embedding + Pinecone Upsert (Day 1-2)

### 2.1 Generate Embeddings Locally

```python
from sentence_transformers import SentenceTransformer

embed_model = SentenceTransformer('all-MiniLM-L6-v2')  # 384 dimensions
embeddings = embed_model.encode(
    [c['text'] for c in chunks],
    batch_size=64,
    show_progress_bar=True
)
```

### 2.2 Upsert to Pinecone

```python
from pinecone import Pinecone

pc = Pinecone(api_key="YOUR_PINECONE_API_KEY")

# Create index (once)
# pc.create_index(
#     name="ntsb-rag",
#     dimension=384,
#     metric="cosine",
#     spec=ServerlessSpec(cloud="aws", region="us-east-1")
# )

index = pc.Index("ntsb-rag")

# Upsert in batches
BATCH_SIZE = 100
for i in range(0, len(chunks), BATCH_SIZE):
    batch = chunks[i:i+BATCH_SIZE]
    vectors = [
        {
            'id': b['chunk_id'],
            'values': embeddings[i+j].tolist(),
            'metadata': {
                'text': b['text'],
                'ntsb_no': b['ntsb_no'],
                'event_date': b['event_date'],
                'state': b['state'],
                'make': b['make'],
                'model': b['model'],
                'phase_of_flight': b['phase_of_flight'],
                'weather': b['weather'],
            }
        }
        for j, b in enumerate(batch)
    ]
    index.upsert(vectors=vectors)
```

---

## Phase 3: Hybrid Retrieval Pipeline (Day 2-3)

### 3.1 BM25 Index (runs on-app)

```python
from rank_bm25 import BM25Okapi
import pickle

# Build BM25 index from chunk texts (do this during data prep, save as pickle)
corpus_tokens = [c['text'].lower().split() for c in chunks]
bm25 = BM25Okapi(corpus_tokens)

# Save for deployment
with open('bm25_index.pkl', 'wb') as f:
    pickle.dump({'bm25': bm25, 'chunks': chunks}, f)
```

### 3.2 Hybrid Search with RRF

```python
def reciprocal_rank_fusion(dense_ids, bm25_ids, k=60):
    """Fuse two ranked lists using Reciprocal Rank Fusion."""
    scores = {}
    for rank, doc_id in enumerate(dense_ids):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    for rank, doc_id in enumerate(bm25_ids):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

def hybrid_retrieve(query, top_k=10, final_k=5):
    # Dense retrieval via Pinecone
    query_embedding = embed_model.encode(query).tolist()
    dense_results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    dense_ids = [r['id'] for r in dense_results['matches']]

    # BM25 retrieval
    query_tokens = query.lower().split()
    bm25_scores = bm25.get_scores(query_tokens)
    bm25_top = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k]
    bm25_ids = [chunks[i]['chunk_id'] for i in bm25_top]

    # Fuse
    fused_ids = reciprocal_rank_fusion(dense_ids, bm25_ids)[:final_k]
    return fused_ids
```

### 3.3 Reranking

```python
from sentence_transformers import CrossEncoder

# Option A: Cross-encoder reranker (runs on-app or pre-computed)
reranker = CrossEncoder('cross-encoder/qnli-distilroberta-base')

def rerank(query, chunk_ids, top_k=5):
    pairs = [(query, get_chunk_text(cid)) for cid in chunk_ids]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(chunk_ids, scores), key=lambda x: x[1], reverse=True)
    return [cid for cid, _ in ranked[:top_k]]

# Option B: NVIDIA NIM reranker via API (alternative)
# import requests
# response = requests.post(NIM_RERANK_URL, json={...})
```

---

## Phase 4: LLM Generation (Day 3)

### 4.1 HF Inference API

```python
from huggingface_hub import InferenceClient

client = InferenceClient(model="mistralai/Mistral-7B-Instruct-v0.3", token="YOUR_HF_TOKEN")
# Alternatives: "meta-llama/Llama-3-8B-Instruct", "CohereForAI/aya-23-8B"

def generate_answer(query, context_chunks):
    context = "\n\n".join([
        f"[Report {c['ntsb_no']}]: {c['text']}"
        for c in context_chunks
    ])

    prompt = f"""You are an aviation safety analyst. Answer the question using ONLY the provided NTSB reports.
Cite specific report IDs (e.g., [ERA24LA084]) for every claim.
If the reports don't contain enough information, say so.

REPORTS:
{context}

QUESTION: {query}

ANSWER:"""

    response = client.text_generation(prompt, max_new_tokens=512, temperature=0.3)
    return response
```

### 4.2 Cross-Document Synthesis Prompt (Causal Chain Queries)

```python
SYNTHESIS_PROMPT = """You are an aviation safety analyst performing cross-document synthesis.
Given multiple NTSB accident reports, identify recurring causal factors, patterns, and systemic issues.

For each pattern you identify:
1. Name the pattern
2. List the specific reports where it appears (cite NtsbNo)
3. Describe how the factor contributed to each accident
4. Note any common conditions (weather, phase of flight, aircraft type)

REPORTS:
{context}

QUESTION: {query}

CROSS-DOCUMENT ANALYSIS:"""
```

---

## Phase 5: Evaluation with deepeval (Day 4)

### 5.1 Setup

```bash
pip install deepeval
# Judge LLM options:
# Option A: OpenAI (best quality, costs money)
export OPENAI_API_KEY="sk-..."
# Option B: Ollama (free, local)
deepeval set-ollama --model=llama3
# Option C: HF model (free, needs wrapper - see deepeval docs)
```

### 5.2 Faithfulness (Claim Extraction + Verification)

Per the assignment: extract claims from generated answer, verify each against retrieved context.

```python
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric

def build_test_cases(test_queries):
    """Run each query through the RAG pipeline and build test cases."""
    cases = []
    for q in test_queries:
        # Run your full pipeline
        chunk_ids = hybrid_retrieve(q)
        chunk_ids = rerank(q, chunk_ids)
        context_chunks = [get_chunk_by_id(cid) for cid in chunk_ids]
        answer = generate_answer(q, context_chunks)

        cases.append(LLMTestCase(
            input=q,
            actual_output=answer,
            retrieval_context=[c['text'] for c in context_chunks]
        ))
    return cases

# 10-20 test queries required
test_queries = [
    "What caused the Cessna 172 crash near Dallas in 2022?",
    "What recurring factors contribute to loss-of-control accidents in IMC conditions?",
    "How often is pilot inexperience cited as a contributing factor in fatal accidents?",
    "What are common causes of runway excursion accidents?",
    "Compare engine failure patterns between single-engine and multi-engine aircraft.",
    # ... add 5-15 more
]

cases = build_test_cases(test_queries)
```

### 5.3 Run Evaluation

```python
faithfulness = FaithfulnessMetric(threshold=0.7)
relevancy = AnswerRelevancyMetric(threshold=0.7)

results = evaluate(cases, metrics=[faithfulness, relevancy])

# Report: show extracted claims + verification for at least 3 examples (assignment requirement)
for case in cases[:3]:
    faithfulness.measure(case)
    print(f"Query: {case.input}")
    print(f"Faithfulness: {faithfulness.score}")
    print(f"Claims: {faithfulness.claims}")
    print(f"Verdicts: {faithfulness.verdicts}")
    print()
```

### 5.4 Relevancy (Alternate Query Generation)

Per assignment: generate 3 questions from the answer, compute cosine similarity with original query.

```python
# deepeval's AnswerRelevancyMetric does this internally
# For manual implementation / report detail:
def compute_relevancy(query, answer, embed_model):
    # Generate 3 questions from the answer using LLM
    generated_qs = llm_generate_questions(answer, n=3)

    # Embed original query and generated questions
    q_emb = embed_model.encode(query)
    gen_embs = embed_model.encode(generated_qs)

    # Cosine similarity
    from sklearn.metrics.pairwise import cosine_similarity
    sims = cosine_similarity([q_emb], gen_embs)[0]
    return float(sims.mean())
```

---

## Phase 6: Ablation Study (Day 4-5)

### Required Comparisons

| Config | Chunking | Retrieval | Reranking |
|--------|----------|-----------|-----------|
| A | Recursive (1500 chars) | Semantic only (Pinecone) | None |
| B | Recursive (1500 chars) | Hybrid (BM25 + Pinecone + RRF) | Cross-encoder |
| C | Section-aware | Semantic only (Pinecone) | None |
| D | Section-aware | Hybrid (BM25 + Pinecone + RRF) | Cross-encoder |

> Add more configs if time allows (e.g., vary chunk size, top-k, with/without metadata headers).

### Run Ablation

```python
configs = {
    'A': {'chunking': 'recursive', 'retrieval': 'semantic_only', 'reranking': False},
    'B': {'chunking': 'recursive', 'retrieval': 'hybrid_rrf', 'reranking': True},
    'C': {'chunking': 'section_aware', 'retrieval': 'semantic_only', 'reranking': False},
    'D': {'chunking': 'section_aware', 'retrieval': 'hybrid_rrf', 'reranking': True},
}

results = {}
for name, config in configs.items():
    # Rebuild chunks with this chunking strategy
    # Run pipeline with this retrieval mode
    # Evaluate with deepeval
    cases = build_test_cases_with_config(test_queries, config)
    eval_results = evaluate(cases, metrics=[faithfulness, relevancy])
    results[name] = {
        'faithfulness': avg_score(eval_results, 'faithfulness'),
        'relevancy': avg_score(eval_results, 'relevancy'),
    }

# Format as table for report
```

### Expected Report Table

| Config | Chunking | Retrieval | Faithfulness | Relevancy |
|--------|----------|-----------|-------------|-----------|
| A | Recursive | Semantic only | ? | ? |
| B | Recursive | Hybrid + Rerank | ? | ? |
| C | Section-aware | Semantic only | ? | ? |
| D | Section-aware | Hybrid + Rerank | ? | ? |

---

## Phase 7: Streamlit UI on HF Spaces (Day 5)

### 7.1 App Structure

```python
# app.py
import streamlit as st

st.title("Aviation Safety RAG")
st.caption("Cross-document reasoning over NTSB accident reports (2016-2023)")

query = st.text_input("Ask a question about aviation accidents:")

if query:
    with st.spinner("Retrieving and generating..."):
        # 1. Retrieve
        chunk_ids = hybrid_retrieve(query)
        chunk_ids = rerank(query, chunk_ids)
        context_chunks = [get_chunk_by_id(cid) for cid in chunk_ids]

        # 2. Generate
        answer = generate_answer(query, context_chunks)

        # 3. Evaluate (optional: compute live or pre-computed)
        faith_score, relev_score = compute_scores(query, answer, context_chunks)

    # Display: Generated Answer
    st.subheader("Answer")
    st.write(answer)

    # Display: Retrieved Context
    st.subheader("Retrieved Chunks")
    for c in context_chunks:
        with st.expander(f"[{c['ntsb_no']}] - {c['state']}, {c['event_date']}"):
            st.write(c['text'])

    # Display: Scores
    col1, col2 = st.columns(2)
    col1.metric("Faithfulness", f"{faith_score:.2f}")
    col2.metric("Relevancy", f"{relev_score:.2f}")
```

### 7.2 HF Space Requirements

```
# requirements.txt
streamlit
sentence-transformers
pinecone-client
rank-bm25
huggingface-hub
deepeval  # if computing scores live
```

### 7.3 Deployment

```bash
# Create HF Space (Streamlit SDK), upload:
# - app.py
# - requirements.txt
# - bm25_index.pkl (pre-built BM25 + chunk data)
# - Any config files
# Set secrets: PINECONE_API_KEY, HF_TOKEN
```

---

## Timeline

| Day | Focus | Deliverable |
|-----|-------|------------|
| 1 | Data prep, chunking (2 strategies), embed, Pinecone upsert | Chunks in Pinecone |
| 2 | BM25 index, hybrid retrieval + RRF fusion | Working retriever |
| 3 | Reranking + LLM generation via HF API | End-to-end pipeline |
| 4 | deepeval evaluation, ablation study runs | Scores + ablation table |
| 5 | Streamlit UI, deploy to HF Spaces | Live app |
| 6 | Report writing, polish ablation tables, example queries | Report draft |
| 7 | Buffer / fix issues / final submission | Submitted |

---

## Key References

- **Dataset:** Garcia et al., Zenodo, 2025 — NTSB final reports 2016-2023
- **RAG Techniques:** github.com/NirDiamant/RAG_Techniques (fusion_retrieval, reranking notebooks)
- **Evaluation:** github.com/confident-ai/deepeval (FaithfulnessMetric, AnswerRelevancyMetric)
- **Pinecone:** Free Starter tier, 384-dim index with cosine metric
- **HF Inference API:** Free tier with rate limits

---

## Decision Log

Track key decisions and their rationale here as the project evolves.

| Date | Decision | Rationale | Reversible? |
|------|----------|-----------|-------------|
| 2026-03-20 | Sample 75-100 reports, not all 7,462 | Assignment requires 50-100 docs; smaller corpus = faster iteration | Yes |
| 2026-03-20 | Drop safety-rec-tracking query class | HasSafetyRec only ~40% populated; 1 week timeline | Yes |
| 2026-03-20 | Use deepeval over custom eval | Matches assignment's faithfulness/relevancy requirements exactly | Yes |
| | | | |
