# RAG Source Code Layout

Use this structure for implementation code:

- src/data_prep/
  - Data loading, deduplication, cleaning, chunking, embedding generation, Pinecone upsert.
- src/retrieval/
  - BM25 index, dense retrieval, hybrid fusion (RRF), reranking.
- src/generation/
  - Prompt templates and Hugging Face Inference API answer generation.
- src/evaluation/
  - Faithfulness and relevancy scoring, test query set runners, ablation scripts.
- src/ui/
  - Streamlit UI components and interaction logic.
- scripts/
  - One-off runnable scripts (build indexes, run ablations, export reports).
- configs/
  - Environment-specific settings, model names, retrieval defaults.

Deployment note:
- Hugging Face Spaces typically expects app.py in repository root.
- Keep core logic in src/, and use a thin root-level app.py entrypoint that imports src/ui.

Data/document locations already organized:
- data/raw/ for original datasets
- data/processed/ for sampled/cleaned outputs and derived artifacts
- docs/ for assignment/report/guides/checklists
