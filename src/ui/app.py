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

from src.retrieval.query import load_model, init_pinecone, retrieve, available_strategies
from src.retrieval.hybrid import (
    build_bm25_index, load_reranker, hybrid_retrieve,
    expand_query_variants, generate_multi_queries, generate_hyde_documents,
    bm25_retrieve, rrf_fuse_lists, rerank, enrich_with_neighbors,
)
from src.generation.generate import generate_answer
from src.evaluation.evaluate import compute_faithfulness, compute_relevancy

# -- Page config ---------------------------------------------------------------

st.set_page_config(page_title="NTSB RAG System", page_icon="✈️", layout="wide")
st.title("✈️ NTSB Aviation Accident RAG System")
st.caption("Retrieval-Augmented Generation over NTSB accident reports")


# -- Resource caching ----------------------------------------------------------

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


# -- Sidebar controls ----------------------------------------------------------

with st.sidebar:
    st.header("Settings")

    llm_provider_label = st.selectbox(
        "Generator",
        ["Ollama (Local)", "DeepSeek (NVIDIA API)", "GPT (NVIDIA API)"],
        index=0,
        help="Choose which LLM generates the final answer from retrieved chunks.",
    )
    if "Ollama" in llm_provider_label:
        llm_provider = "ollama"
    elif "GPT" in llm_provider_label:
        llm_provider = "gpt"
    else:
        llm_provider = "deepseek"
    ollama_model = st.text_input(
        "Ollama Model",
        value="qwen2.5:32b",
        help="Used only when Generator is set to Ollama (Local).",
    )

    strategies = available_strategies()
    if not strategies:
        st.error("No local chunk files were found in data/processed. Build chunks first.")
        st.stop()

    if "section" in strategies:
        default_idx = strategies.index("section")
    elif "recursive" in strategies:
        default_idx = strategies.index("recursive")
    else:
        default_idx = 0

    strategy = st.selectbox(
        "Chunking Strategy",
        strategies,
        index=default_idx,
        help="Choose which chunking strategy was used for the document index.",
    )

    mode = st.radio(
        "Retrieval Mode",
        ["Semantic Only", "Hybrid (BM25 + Semantic)", "Hybrid (with Cross-Encoder Rerank)"],
        index=1,
        help="Semantic: Use embeddings only. Hybrid: Combine BM25+Semantic. Rerank: Add cross-encoder scoring (slower, sometimes worse for domain-specific queries).",
    )

    top_k = st.slider("Number of chunks to retrieve", 3, 50, 10)

    run_eval = st.checkbox("Compute faithfulness & relevancy scores", value=True)
    use_multi_query = st.checkbox("Use Multi-Query expansion (slower, better recall for specific questions)", value=False)
    use_hyde = st.checkbox("Use HyDE expansion (slower, generates hypothetical retrieval snippets)", value=False)

    st.divider()
    st.markdown("**Model info**")
    st.markdown("- Embeddings: Jina v5 (768-dim)")
    if llm_provider == "ollama":
        st.markdown(f"- Generator: {ollama_model} (Ollama local)")
    elif llm_provider == "gpt":
        st.markdown("- Generator: GPT-4o 120B (NVIDIA)")
    else:
        st.markdown("- Generator: DeepSeek V3.1 (NVIDIA)")
    st.markdown("- Reranker: ms-marco-MiniLM-L-12-v2 (upgraded, better scoring)")


# -- Main area ----------------------------------------------------------------

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
    multi_query_variants = None
    hyde_docs = None
    if "Hybrid" in mode:
        reranker = get_reranker() if "Rerank" in mode else None
        bm25, chunks = get_bm25(strategy)

        # Step 1: Query expansion
        with st.spinner("Step 1/4 — Expanding query variants..."):
            queries = expand_query_variants(query)

        # Step 2: Multi-Query expansion (optional, slow — LLM API call)
        if use_multi_query:
            mq_label = "GPT" if llm_provider == "gpt" else "DeepSeek"
            with st.spinner(f"Step 2/4 — Generating multi-query variants ({mq_label})..."):
                multi_query_variants = generate_multi_queries(query, model=llm_provider)
                if multi_query_variants:
                    queries.extend(multi_query_variants)

        # Step 2b: HyDE expansion (optional, LLM API call)
        if use_hyde:
            with st.spinner("Step 2b/4 — Generating HyDE hypothetical snippets..."):
                hyde_docs = generate_hyde_documents(query, num_docs=2)
                if hyde_docs:
                    print("[HyDE] Generated hypothetical snippets:")
                    for i, doc in enumerate(hyde_docs, 1):
                        print(f"  [{i}] {doc}")
                    queries.extend(hyde_docs)

        # Step 3: Semantic + BM25 retrieval for each query variant (increased from 40 to 60)
        ranked_lists = []
        for i, q in enumerate(queries, 1):
            with st.spinner(f"Step 3/4 — Searching (variant {i}/{len(queries)})..."):
                ranked_lists.append(retrieve(q, strategy, top_k=60, model=jina_model, index=index))
                ranked_lists.append(bm25_retrieve(q, bm25, chunks, top_k=60))

        # Step 4: RRF fusion + optional cross-encoder rerank
        with st.spinner(f"Step 4/4 — Fusing & {'reranking' if reranker else 'selecting'} candidates..."):
            fused = rrf_fuse_lists(ranked_lists)
            if reranker:
                matches = rerank(query, fused, reranker, top_k=top_k, min_unique_reports=3)
            else:
                # Just use RRF scores, no cross-encoder reranking
                matches = sorted(fused, key=lambda x: x.get("score", 0), reverse=True)[:top_k]
            
            # Neighbor enrichment (increased window from 1 to 2 for more context)
            matches = enrich_with_neighbors(matches, chunks, window=2)
    else:
        with st.spinner("Retrieving relevant chunks..."):
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
        answer = generate_answer(
            query,
            matches,
            llm_provider=llm_provider,
            ollama_model=ollama_model,
        )

    # Multi-Query variants (if generated)
    if multi_query_variants:
        with st.expander("Multi-Query — Alternative questions used for retrieval"):
            for i, q in enumerate(multi_query_variants, 1):
                st.markdown(f"{i}. {q}")

    # HyDE docs (if generated)
    if hyde_docs:
        with st.expander("HyDE — Hypothetical snippets used for retrieval"):
            for i, doc in enumerate(hyde_docs, 1):
                st.markdown(f"{i}. {doc}")

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
    st.subheader(f"Retrieved Chunks ({len(matches)})")
    for i, m in enumerate(matches, 1):
        if hasattr(m, "metadata"):
            meta = m.metadata
            score = m.score
        else:
            meta = m
            score = m.get("score", 0)

        report_id = meta.get("ntsb_no") or meta.get("report_id", "N/A")
        section = meta.get("section_title", "")
        retrieval_source = meta.get("retrieval_strategy", "semantic")
        label = f"[{i}] {report_id}"
        if section:
            label += f" — {section}"
        label += f" | Score: {score:.4f}"

        with st.expander(label):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Report:** {report_id}")
                st.markdown(f"**Date:** {meta.get('event_date', 'N/A')}")
            with col2:
                st.markdown(f"**Section:** {section or 'N/A'}")
                st.markdown(
                    f"**Phase:** {meta.get('phase_of_flight', 'N/A')} | "
                    f"**Weather:** {meta.get('weather', 'N/A')}"
                )
                st.markdown(f"**Retrieval:** {retrieval_source}")
            st.divider()
            st.text(meta.get("text", ""))
