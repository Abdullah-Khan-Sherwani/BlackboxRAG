# Quick Reference: Enhanced NTSB RAG System

## 4 Main Tools (Pick One)

### 🏃 **Fastest: query_prod.py**
```bash
python query_prod.py "Your question about NTSB accidents"
```
- ✅ Priority-first ranking (section + keywords + BM25)
- ✅ NTSB citations included
- ✅ No slow reranker
- ⏱️ ~10-15 seconds
- 📊 Best for: Single queries, production

### ✅ **Accurate: accuracy_check_tool.py**
```bash
python accuracy_check_tool.py "Your question"
# Follow interactive prompts
# Get JSON report with accuracy %
```
- ✅ Golden chunk retrieval
- ✅ Your answer validation
- ✅ Numerical accuracy check
- ✅ Citation verification
- 📊 Best for: Critical accuracy verification

### 🌐 **Interactive: Streamlit UI**
```bash
streamlit run src/ui/app.py
```
- ✅ Web interface
- ✅ Multiple retrieval modes
- ✅ Faithfulness scoring
- ✅ Real-time adjustment
- ⏱️ ~15-20 seconds per query
- 📊 Best for: Exploration, learning

### 📚 **Demo: demo_enhanced_rag.py**
```bash
python demo_enhanced_rag.py
```
- ✅ Shows complete pipeline
- ✅ Demonstrates all features
- ✅ Validation examples
- 📊 Best for: Understanding the system

---

## What Each Tool Does

### query_prod.py
```
Input: Question
Process: 
  1. Detect question type → section priorities
  2. Load 32K chunks, score by priority
  3. Select top 15 enriched with neighbors
  4. Generate answer with citations
Output: Answer + confidence indicators
```

### accuracy_check_tool.py
```
Input: Question
Process:
  1. Retrieve golden chunks (60+60 hybrid fusion)
  2. Get your LLM's response
  3. Extract all numbers & citations
  4. Check if numbers exist in source
  5. Verify NTSB citations
Output: JSON report with accuracy %
```

### Streamlit UI (src/ui/app.py)
```
Input: Question + Settings
Process:
  1. Choose retrieval mode
  2. Adjust top_k chunks
  3. Optional: Enable multi-query
  4. Retrieve with chosen method
  5. Generate answer
  6. Calculate scores
Output: Answer + evaluation metrics
```

### demo_enhanced_rag.py
```
Shows entire pipeline:
  1. Query expansion (2 variants)
  2. Hybrid retrieval (120 candidates)
  3. RRF fusion (136 unique)
  4. Answer generation
  5. Numerical validation
  6. Citation verification
```

---

## Key Numbers

### Retrieval
- Candidate pool: 120 (60 semantic + 60 BM25)
- After fusion: ~130 unique documents
- Final selection: 8-15 chunks
- Context per chunk: ±2 neighbors

### Performance
- query_prod.py: 10-15 seconds
- Streamlit: 15-20 seconds
- accuracy_check_tool.py: 20-30 seconds
- With HyDE: +2-3 seconds

### Data
- Total chunks: 32,180
- NTSB reports: 32
- Embeddings: 768-dimensional (Jina v5)
- Cross-encoder: L-12 (12 layers)

---

## Citation Format

All answers include NTSB citations:
```
[NTSB: AAR-14/01]
or
[NTSB/AAR-14/01]
or
[NTSB: DCA-16-MA-261]
```

Example answer:
```
"The Pilot Monitoring had 12,307 total flight hours 
but 0 hours as an Instructor Pilot in the Boeing 777 
[NTSB: NTSB/AAR-14/01]."
```

---

## Accuracy Validation

### What Gets Validated
✅ Numerical claims (flight hours, dates, distances)
✅ NTSB citations (format & existence)
✅ Section coverage (relevant sections mentioned?)
✅ Information completeness (2-3 key points?)

### Validation Output
```json
{
  "accuracy_score": 95.0,
  "numerical_verified": 5,
  "numerical_total": 5,
  "citations_verified": 1,
  "citations_total": 1,
  "issues": [],
  "recommendations": []
}
```

---

## Choosing the Right Tool

| Goal | Tool | Time |
|------|------|------|
| Quick answer | query_prod.py | 10-15s |
| Verify accuracy | accuracy_check_tool.py | 20-30s |
| Explore data | Streamlit | 15-20s |
| Learn system | demo_enhanced_rag.py | 30s |
| Complex question | Streamlit + multi-query | 30-40s |

---

## Common Commands

```bash
# Simple query
python query_prod.py "What caused the accident?"

# With validation
python accuracy_check_tool.py "What caused the accident?"

# Web interface
streamlit run src/ui/app.py

# See system in action
python demo_enhanced_rag.py

# Test everything
python test_enhancements.py

# Check syntax
python -m py_compile src/ui/app.py

# List available chunks
ls -lh data/processed/chunks_*.json
```

---

## Troubleshooting Quick Fixes

| Problem | Solution |
|---------|----------|
| query_prod.py slow | Try `top_k = 10` instead of 15 |
| UI not starting | Run: `pip install streamlit --upgrade` |
| Numbers not validating | Check format (12,307 vs 12307) |
| Missing sections in answer | Increase top_k or enable neighbors |
| Cross-encoder bad ranking | Use query_prod.py instead |
| Slow retrieval | Disable HyDE, use query_prod.py |

---

## Files Reference

```
Production Tools:
  query_prod.py              ← Use this most
  accuracy_check_tool.py     ← Validate accuracy
  demo_enhanced_rag.py       ← Understand system
  
Documentation:
  ENHANCEMENTS_GUIDE.md      ← Detailed guide
  IMPLEMENTATION_COMPLETE.md ← What was done
  README.md                  ← Project overview
  
Source Code (Enhanced):
  src/retrieval/hybrid.py    ← Better retrieval
  src/ui/app.py              ← Updated Streamlit
  src/generation/generate.py ← Enforces citations
  src/retrieval/query.py     ← Core retrieval
```

---

## Success Metrics

Your system is working great when:
- ✅ query_prod.py returns answers in 10-15 seconds
- ✅ All answers include [NTSB: ...] citations
- ✅ accuracy_check_tool.py reports >90% accuracy
- ✅ Numbers in answers are all verified
- ✅ Streamlit UI shows relevant retrieved chunks
- ✅ Multiple sections covered in answers

---

## Next Steps

1. **Try query_prod.py**
   ```bash
   python query_prod.py "Your first question here"
   ```

2. **Validate accuracy**
   ```bash
   python accuracy_check_tool.py "Same question"
   ```

3. **Explore Streamlit**
   ```bash
   streamlit run src/ui/app.py
   ```

4. **Read full documentation**
   - ENHANCEMENTS_GUIDE.md (detailed)
   - Code comments (implementation)

---

## Contact & Support

- Issues? Check TROUBLESHOOTING in ENHANCEMENTS_GUIDE.md
- Questions? See code comments and docstrings
- Examples? Run demo_enhanced_rag.py
- Testing? Run test_enhancements.py

---

**Happy RAG-ing! 🚀**

For detailed docs, see: ENHANCEMENTS_GUIDE.md or IMPLEMENTATION_COMPLETE.md
