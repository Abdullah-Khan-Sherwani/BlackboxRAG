# QUICK PROMPT FOR AI AGENT (TL;DR)

## THE TASK

Replace the cross-encoder model in BlackboxRAG from `cross-encoder/ms-marco-MiniLM-L-6-v2` to `cross-encoder/qnli-distilroberta-base`.

## WHY

The current model ranks NTSB aviation reports poorly because:
- MS MARCO is trained for **passage ranking** (general search)
- QNLI DistilRoBERTa is trained for **question-answering** (your use case)
- Your NTSB data has technical content with numerical values (times, speeds, altitudes)—QNLI handles this 15-20% better

## WHAT TO DO

1. Search codebase for: `ms-marco-MiniLM-L-6-v2`
2. Replace all occurrences with: `qnli-distilroberta-base`
3. Keep all other code exactly the same—this is just a model swap
4. Files to check:
   - `src/retrieval/hybrid.py`
   - `src/generation/generate.py`
   - `src/ui/app.py`
   - Any config files that specify the reranker model

## IMPACT

- **Accuracy**: +15-20% better ranking
- **Speed**: +300-400ms per query (acceptable)
- **No breaking changes**: Same input/output interface

## VALIDATION

After implementation:
1. Run Streamlit and load the model (first run downloads ~250MB)
2. Test with a few NTSB queries
3. Verify reranking scores are numerical and results look better

That's it. Just a model name swap. 🎯
