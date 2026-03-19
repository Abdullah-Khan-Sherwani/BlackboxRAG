# BlackboxRAG

Cross-document reasoning over NTSB aviation accident reports.

**Live Demo:** [HuggingFace Space](https://huggingface.co/spaces/YOUR_USERNAME/blackboxrag)

## What it does

Ask questions about aviation accidents — BlackboxRAG retrieves from 75-100 NTSB final investigation reports (2016-2023) and generates cited answers. It handles single-accident lookups, cross-document comparison, and causal pattern synthesis.

## How it works

Query → Hybrid Retrieval (Pinecone + BM25 + RRF) → Reranking → LLM Generation (HF Inference API) → Cited Answer

## Stack

| Component | Choice |
|-----------|--------|
| Vector DB | Pinecone |
| Embedding | all-MiniLM-L6-v2 |
| Lexical Search | BM25 (rank_bm25) |
| Reranker | Cross-encoder |
| LLM | Mistral-7B via HF Inference API |
| UI | Streamlit on HF Spaces |
| Evaluation | deepeval (Faithfulness + Relevancy) |

## Dataset

Garcia et al., Zenodo, 2025 — NTSB final reports 2016-2023.

## Team

- [Member 1](https://github.com/PLACEHOLDER)
- [Member 2](https://github.com/PLACEHOLDER)
- [Member 3](https://github.com/PLACEHOLDER)
