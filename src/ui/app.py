"""
Streamlit UI for the NTSB RAG system.
Provides query input, strategy/mode selectors, and displays answer + context +
faithfulness/relevancy scores.

Run: streamlit run src/ui/app.py
"""
import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.retrieval.query import load_model, init_pinecone, retrieve
from src.retrieval.hybrid import build_bm25_index, load_reranker, hybrid_retrieve
from src.generation.generate import generate_answer
from src.evaluation.evaluate import compute_faithfulness, compute_relevancy

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="NTSB RAG System", page_icon="✈️", layout="wide")
st.title("✈️ NTSB Aviation Accident RAG System")
st.caption("Retrieval-Augmented Generation over NTSB accident reports")


# ── Resource caching ─────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading Jina embedding model...")
def get_jina_model():
    return load_model()


@st.cache_resource(show_spinner="Connecting to Pinecone...")
def get_pinecone_index():
    return init_pinecone()


@st.cache_resource(show_spinner="Loading cross-encoder reranker...")
def get_reranker():
    return load_reranker()


@st.cache_resource(show_spinner="Building BM25 index for {strategy}...")
def get_bm25(strategy):
    return build_bm25_index(strategy)


# ── Sidebar controls ─────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Settings")

    strategy = st.selectbox(
        "Chunking Strategy",
        ["fixed", "recursive", "semantic"],
        index=1,
        help="Choose which chunking strategy was used for the document index.",
    )

    mode = st.radio(
        "Retrieval Mode",
        ["Semantic Only", "Hybrid (BM25 + Semantic + Rerank)"],
        index=1,
    )

    top_k = st.slider("Number of chunks to retrieve", 3, 20, 5)

    run_eval = st.checkbox("Compute faithfulness & relevancy scores", value=True)

    st.divider()
    st.markdown("**Model info**")
    st.markdown("- Embeddings: Jina v5 (768-dim)")
    st.markdown("- Generator: DeepSeek V3.2 (NVIDIA)")
    st.markdown("- Reranker: ms-marco-MiniLM-L-6-v2")


# ── Main area ─────────────────────────────────────────────────────────────────

query = st.text_input(
    "Ask a question about NTSB aviation accidents:",
    placeholder="e.g., What are common causes of engine failure during takeoff?",
)

if query:
    # Load resources
    jina_model = get_jina_model()
    index = get_pinecone_index()

    is_hybrid = "Hybrid" in mode

    # Retrieve
    with st.spinner("Retrieving relevant chunks..."):
        if is_hybrid:
            reranker = get_reranker()
            bm25, chunks = get_bm25(strategy)
            matches = hybrid_retrieve(
                query, strategy, top_k=top_k,
                model=jina_model, index=index,
                bm25=bm25, chunks=chunks, reranker=reranker,
            )
        else:
            matches = retrieve(query, strategy, top_k=top_k, model=jina_model, index=index)

    # Extract context texts
    context_texts = []
    for m in matches:
        if hasattr(m, "metadata"):
            context_texts.append(m.metadata.get("text", ""))
        else:
            context_texts.append(m.get("text", ""))

    # Generate
    with st.spinner("Generating answer..."):
        answer = generate_answer(query, matches)

    # Display answer
    st.subheader("Answer")
    st.markdown(answer)

    # Evaluation scores
    if run_eval:
        col1, col2 = st.columns(2)

        with col1:
            with st.spinner("Computing faithfulness..."):
                faith_score, faith_details = compute_faithfulness(answer, context_texts)
            st.metric("Faithfulness", f"{faith_score:.1%}")

            if faith_details:
                with st.expander("Claim verification details"):
                    for fd in faith_details:
                        icon = "+" if fd.get("supported") else "-"
                        st.markdown(f"**{icon}** {fd.get('claim', '')}")
                        st.caption(fd.get("reasoning", ""))

        with col2:
            with st.spinner("Computing relevancy..."):
                rel_score, rel_alternates = compute_relevancy(query, answer, jina_model)
            st.metric("Relevancy", f"{rel_score:.1%}")

            if rel_alternates:
                with st.expander("Alternate query phrasings"):
                    for alt in rel_alternates:
                        st.markdown(f"- {alt}")

    # Retrieved context
    st.subheader("Retrieved Context")
    for i, m in enumerate(matches, 1):
        if hasattr(m, "metadata"):
            meta = m.metadata
            score = m.score
        else:
            meta = m
            score = m.get("score", 0)

        with st.expander(
            f"[{i}] NTSB {meta.get('ntsb_no', 'N/A')} — "
            f"{meta.get('make', '')} {meta.get('model', '')} | "
            f"Score: {score:.4f}"
        ):
            st.markdown(f"**Date:** {meta.get('event_date', 'N/A')}")
            st.markdown(f"**Phase:** {meta.get('phase_of_flight', 'N/A')} | "
                        f"**Weather:** {meta.get('weather', 'N/A')}")
            st.text(meta.get("text", "")[:1000])
