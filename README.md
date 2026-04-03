# BlackboxRAG

Retrieval-Augmented Generation over 101 NTSB aviation accident reports (2016–2023).

## What it does

Ask questions about aviation accidents — BlackboxRAG retrieves from NTSB final investigation
reports and generates cited answers. Handles single-accident deep dives and cross-document
pattern queries.

## Stack

| Component      | Choice                                          |
|----------------|-------------------------------------------------|
| Vector DB      | Pinecone (`ntsb-rag`, cosine, 768-dim)          |
| Embedding      | Jina v5 (`jina-embeddings-v5-text-nano`)        |
| Lexical Search | BM25 (`rank_bm25`)                              |
| Reranker       | `cross-encoder/qnli-distilroberta-base`          |
| LLM            | Ollama local / DeepSeek V3 / GPT-4o (NVIDIA)   |
| UI             | Streamlit                                       |

## Run

```bash
streamlit run src/ui/app.py
```

See [QUICK_START.md](QUICK_START.md) for setup, collaborator workflow, and known issues.
