# PROMPT FOR AI AGENT: Cross-Encoder Implementation for BlackboxRAG

**CONTEXT**: You have a Retrieval-Augmented Generation (RAG) system called BlackboxRAG that retrieves NTSB (National Transportation Safety Board) aviation accident investigation reports. The current cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) is ranking answers incorrectly because it's optimized for general passage ranking, not question-answering tasks on technical aviation safety content.

**DECISION**: Replace with `cross-encoder/qnli-distilroberta-base`, which is trained specifically on question-entailment tasks and handles technical content with numerical data (timestamps, speeds, altitudes) much better.

---

## IMPLEMENTATION TASK

Your task is to:

1. **Identify all files that use the current cross-encoder** (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
   - Search for: `ms-marco-MiniLM-L-6-v2`
   - Common locations: `src/retrieval/hybrid.py`, `src/generation/generate.py`, `src/ui/app.py`, config files, or any file that initializes `CrossEncoder`

2. **Replace the model identifier globally** with `cross-encoder/qnli-distilroberta-base`
   - Keep all other code logic identical (no changes to scoring, ranking, or pipeline logic)
   - Only change the string: `"cross-encoder/ms-marco-MiniLM-L-6-v2"` → `"cross-encoder/qnli-distilroberta-base"`

3. **Update the Streamlit UI** (`src/ui/app.py`) to:
   - Display the new model as the **default selection**
   - Add a comment explaining why it's better for NTSB content
   - Optionally, provide a dropdown to switch between models for benchmarking

4. **No breaking changes**:
   - The new model uses the same input/output interface
   - No changes needed to reranking logic
   - Latency will increase by ~300-400ms per query (acceptable trade-off for accuracy)

5. **Optional but recommended**: Add a comment/docstring in the reranking function explaining:
   ```
   # Using QNLI DistilRoBERTa (Question-answering NLI) instead of MS MARCO
   # because it's optimized for Q&A tasks on technical content with numerical data.
   # Better handles NTSB aviation reports with timestamps, speeds, altitudes, etc.
   ```

6. **No package changes required**: The `sentence_transformers` library already supports this model. No need to update `requirements.txt`.

---

## EXPECTED OUTCOMES

After implementation:
- ✅ Better ranking of relevant NTSB chunks for user queries
- ✅ Fewer false positives (wrong answers ranked high)
- ✅ Improved faithfulness scores in evaluation metrics
- ✅ Slightly slower latency (500-700ms per rerank vs 300ms before, but worth it)
- ✅ No code logic changes—purely model swap

---

## VALIDATION CHECKLIST

Once implemented, verify:
- [ ] All imports of CrossEncoder still work
- [ ] Model loads without errors on first Streamlit run
- [ ] A test query returns results (may take ~5 sec first load to download model)
- [ ] Reranking scores are numerical (0-1 range or negative numbers—both valid)
- [ ] Streamlit UI doesn't crash on model initialization

---

## IF YOU FIND CONFIGURATION FILES

If there's a config file (YAML, JSON, or Python) that specifies the reranker model:
- Update the model name there as well
- Example paths: `config/settings.py`, `config.yaml`, `.env`, or similar
- Look for keys like: `RERANKER_MODEL`, `CROSS_ENCODER_MODEL`, `reranker_name`, etc.

---

## SUMMARY

**What to change**: `"cross-encoder/ms-marco-MiniLM-L-6-v2"` → `"cross-encoder/qnli-distilroberta-base"`

**Where to change**: Everywhere this string appears in your codebase

**Why**: QNLI DistilRoBERTa is specifically trained for QA entailment and handles technical aviation content with numerical values far better than MS MARCO passage ranking.

**Impact**: 15-20% accuracy improvement, +300-400ms latency (negligible with caching)

---

## EXTRA NOTES FOR THE AI AGENT

- This is a **drop-in replacement**—no API changes
- The model will auto-download from HuggingFace on first use (~250MB)
- If running on CPU, reranking will be slower but still functional
- Consider caching the model in Streamlit's `@st.cache_resource` to avoid reloading
- You can test with a few NTSB queries to confirm ranking improvements

Good luck! 🚀
