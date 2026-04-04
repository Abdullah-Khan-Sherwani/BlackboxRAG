"""
Streamlit UI for the NTSB RAG system — futuristic aviation chatbot.

Run: streamlit run src/ui/app.py
"""
import os
import sys
import html as html_lib
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.retrieval.query import load_model, init_pinecone, retrieve, available_strategies
from src.retrieval.hybrid import (
    build_bm25_index, load_reranker, hybrid_retrieve,
    expand_query_variants, generate_multi_queries, generate_hyde_documents,
    generate_knowledge_doc, KnowledgeResult,
    bm25_retrieve, rrf_fuse_lists, rerank, enrich_with_neighbors,
)
from src.retrieval.report_mapper import detect_report_from_query, resolve_report_number, AVAILABLE_REPORTS
from src.generation.generate import generate_answer
from src.evaluation.evaluate import compute_faithfulness, compute_relevancy

# -- Page config ---------------------------------------------------------------

st.set_page_config(page_title="BlackBox RAG", page_icon="🛩️", layout="wide")

# -- Custom CSS — futuristic dark aviation theme -------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --bg-primary: #0a0e17;
    --bg-secondary: #111827;
    --bg-card: #1a1f2e;
    --accent-cyan: #06d6a0;
    --accent-blue: #118ab2;
    --accent-orange: #f77f00;
    --accent-red: #ef476f;
    --text-primary: #e8edf5;
    --text-secondary: #8892a4;
    --border-color: #2a3040;
    --glow-cyan: 0 0 20px rgba(6, 214, 160, 0.15);
}

.stApp {
    background: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

#MainMenu, footer, header {visibility: hidden;}
.stDeployButton {display: none;}

/* ==================== ANIMATED BACKGROUND — RADAR + CLOUDS ==================== */
.sky-bg {
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    pointer-events: none;
    z-index: 0;
    overflow: hidden;
}

/* Scrolling grid lines like a HUD / radar */
.sky-bg::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background:
        linear-gradient(rgba(6, 214, 160, 0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(6, 214, 160, 0.03) 1px, transparent 1px);
    background-size: 60px 60px;
    animation: grid-scroll 20s linear infinite;
}
@keyframes grid-scroll {
    0% { transform: translate(0, 0); }
    100% { transform: translate(60px, 60px); }
}

/* Floating cloud particles */
.cloud {
    position: absolute;
    background: radial-gradient(ellipse, rgba(136,146,164,0.06) 0%, transparent 70%);
    border-radius: 50%;
    animation: cloud-drift linear infinite;
}
.cloud-1 { width: 300px; height: 80px; top: 15%; left: -300px; animation-duration: 45s; }
.cloud-2 { width: 200px; height: 60px; top: 40%; left: -200px; animation-duration: 55s; animation-delay: 10s; }
.cloud-3 { width: 350px; height: 90px; top: 70%; left: -350px; animation-duration: 60s; animation-delay: 20s; }
.cloud-4 { width: 180px; height: 50px; top: 25%; left: -180px; animation-duration: 50s; animation-delay: 30s; }
.cloud-5 { width: 260px; height: 70px; top: 55%; left: -260px; animation-duration: 40s; animation-delay: 5s; }

@keyframes cloud-drift {
    0% { transform: translateX(0); opacity: 0; }
    5% { opacity: 1; }
    95% { opacity: 1; }
    100% { transform: translateX(calc(100vw + 400px)); opacity: 0; }
}

/* Tiny moving stars / data points */
.star {
    position: absolute;
    width: 2px; height: 2px;
    background: var(--accent-cyan);
    border-radius: 50%;
    box-shadow: 0 0 4px var(--accent-cyan);
    animation: star-blink 3s ease-in-out infinite;
}
.star-1 { top: 10%; left: 20%; animation-delay: 0s; }
.star-2 { top: 30%; left: 70%; animation-delay: 1s; }
.star-3 { top: 50%; left: 40%; animation-delay: 0.5s; }
.star-4 { top: 75%; left: 85%; animation-delay: 2s; }
.star-5 { top: 20%; left: 55%; animation-delay: 1.5s; }
.star-6 { top: 60%; left: 15%; animation-delay: 0.8s; }
.star-7 { top: 85%; left: 50%; animation-delay: 2.5s; }
.star-8 { top: 45%; left: 90%; animation-delay: 0.3s; }

@keyframes star-blink {
    0%, 100% { opacity: 0.2; transform: scale(1); }
    50% { opacity: 1; transform: scale(1.8); }
}

/* ==================== HEADER ==================== */
.main-header {
    position: relative;
    z-index: 1;
    text-align: center;
    padding: 2rem 0 1.5rem 0;
    border-bottom: 1px solid var(--border-color);
    margin-bottom: 1.5rem;
}
.main-header h1 {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #06d6a0, #118ab2, #06d6a0);
    background-size: 200% 200%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    letter-spacing: 4px;
    animation: gradient-shift 4s ease-in-out infinite;
}
@keyframes gradient-shift {
    0%, 100% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
}
.main-header .subtitle {
    font-family: 'Inter', sans-serif;
    color: var(--text-secondary);
    font-size: 0.85rem;
    margin-top: 0.4rem;
    letter-spacing: 1.5px;
}
.main-header .status-dot {
    display: inline-block;
    width: 8px; height: 8px;
    background: var(--accent-cyan);
    border-radius: 50%;
    margin-right: 6px;
    box-shadow: 0 0 8px var(--accent-cyan);
    animation: pulse-dot 2s ease-in-out infinite;
}

/* Decorative plane in header */
.header-plane {
    font-size: 1.4rem;
    display: inline-block;
    animation: header-fly 6s ease-in-out infinite;
    filter: drop-shadow(0 0 8px rgba(6, 214, 160, 0.4));
}
@keyframes header-fly {
    0%, 100% { transform: translateX(-8px) translateY(2px); }
    50% { transform: translateX(8px) translateY(-2px); }
}

@keyframes pulse-dot {
    0%, 100% { opacity: 1; box-shadow: 0 0 8px var(--accent-cyan); }
    50% { opacity: 0.5; box-shadow: 0 0 16px var(--accent-cyan); }
}

/* ==================== PLANE TAKEOFF LOADER ==================== */
.plane-loader {
    position: relative;
    width: 100%;
    height: 120px;
    overflow: hidden;
    margin: 1rem 0;
    border-radius: 12px;
    background: linear-gradient(180deg, #0d1321 0%, #1a1f2e 60%, #2a3040 100%);
    border: 1px solid var(--border-color);
}

/* Runway */
.runway {
    position: absolute;
    bottom: 20px;
    left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, transparent, var(--border-color) 10%, var(--border-color) 90%, transparent);
}
.runway::before {
    content: '';
    position: absolute;
    top: -1px;
    left: 10%;
    width: 80%;
    height: 5px;
    background: repeating-linear-gradient(90deg, var(--text-secondary) 0px, var(--text-secondary) 20px, transparent 20px, transparent 40px);
    opacity: 0.3;
}

/* The plane taking off */
.plane-takeoff {
    position: absolute;
    bottom: 28px;
    left: 5%;
    font-size: 2rem;
    animation: takeoff 3s ease-in-out infinite;
    filter: drop-shadow(0 0 12px rgba(6, 214, 160, 0.5));
}
@keyframes takeoff {
    0% { transform: translateX(0) translateY(0) rotate(0deg); opacity: 1; }
    40% { transform: translateX(200px) translateY(0) rotate(0deg); }
    70% { transform: translateX(400px) translateY(-50px) rotate(-15deg); }
    100% { transform: translateX(600px) translateY(-80px) rotate(-15deg); opacity: 0.3; }
}

/* Exhaust trail */
.exhaust {
    position: absolute;
    bottom: 35px;
    left: 8%;
    width: 100px;
    height: 4px;
    background: linear-gradient(90deg, transparent, rgba(6, 214, 160, 0.3), rgba(17, 138, 178, 0.1));
    border-radius: 2px;
    animation: exhaust-trail 3s ease-in-out infinite;
}
@keyframes exhaust-trail {
    0% { width: 0; opacity: 0; }
    30% { width: 150px; opacity: 0.6; }
    70% { width: 300px; opacity: 0.3; }
    100% { width: 0; opacity: 0; }
}

.loader-text {
    position: absolute;
    bottom: 8px;
    left: 50%;
    transform: translateX(-50%);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: var(--text-secondary);
    letter-spacing: 2px;
    text-transform: uppercase;
}
.loader-text::after {
    content: '';
    animation: loader-dots 1.5s steps(3, end) infinite;
}
@keyframes loader-dots {
    0% { content: '.'; }
    33% { content: '..'; }
    66% { content: '...'; }
}

/* Altitude indicator */
.altitude-bar {
    position: absolute;
    right: 15px;
    top: 10px;
    bottom: 30px;
    width: 3px;
    background: rgba(42, 48, 64, 0.5);
    border-radius: 2px;
}
.altitude-indicator {
    position: absolute;
    right: 12px;
    width: 9px;
    height: 9px;
    background: var(--accent-cyan);
    border-radius: 50%;
    box-shadow: 0 0 6px var(--accent-cyan);
    animation: altitude-climb 3s ease-in-out infinite;
}
@keyframes altitude-climb {
    0% { bottom: 30px; }
    100% { bottom: 85px; }
}

/* Speed readout */
.speed-readout {
    position: absolute;
    left: 15px;
    top: 10px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    color: var(--accent-cyan);
    opacity: 0.7;
}
.speed-readout .val {
    animation: speed-count 3s linear infinite;
}
@keyframes speed-count {
    0% { content: '0'; }
    100% { content: '280'; }
}

/* ==================== PLANE CRASH (no results) ==================== */
.plane-crash-container {
    position: relative;
    width: 100%;
    height: 160px;
    overflow: hidden;
    margin: 1rem 0;
    border-radius: 12px;
    background: linear-gradient(180deg, #1a0a0a 0%, #2a1015 60%, #3a1a1a 100%);
    border: 1px solid rgba(239, 71, 111, 0.3);
}

.plane-crash {
    position: absolute;
    top: 15px;
    left: 10%;
    font-size: 2.2rem;
    animation: crash-dive 2.5s ease-in forwards infinite;
    filter: drop-shadow(0 0 10px rgba(239, 71, 111, 0.6));
}
@keyframes crash-dive {
    0% { transform: translateX(0) translateY(0) rotate(0deg); opacity: 1; }
    30% { transform: translateX(100px) translateY(10px) rotate(10deg); }
    60% { transform: translateX(200px) translateY(60px) rotate(35deg); }
    80% { transform: translateX(280px) translateY(100px) rotate(55deg); opacity: 1; }
    85% { transform: translateX(300px) translateY(110px) rotate(60deg); opacity: 0.8; }
    100% { transform: translateX(320px) translateY(115px) rotate(65deg); opacity: 0.2; }
}

/* Impact flash */
.impact-flash {
    position: absolute;
    bottom: 15px;
    left: 55%;
    width: 0; height: 0;
    background: radial-gradient(circle, rgba(247, 127, 0, 0.8) 0%, rgba(239, 71, 111, 0.4) 40%, transparent 70%);
    border-radius: 50%;
    animation: impact-boom 2.5s ease-out infinite;
}
@keyframes impact-boom {
    0%, 75% { width: 0; height: 0; opacity: 0; }
    80% { width: 20px; height: 20px; opacity: 0.3; }
    90% { width: 80px; height: 80px; opacity: 0.9; transform: translate(-40px, -40px); }
    100% { width: 120px; height: 120px; opacity: 0; transform: translate(-60px, -60px); }
}

/* Smoke particles */
.smoke {
    position: absolute;
    border-radius: 50%;
    background: rgba(136, 146, 164, 0.15);
    animation: smoke-rise 2.5s ease-out infinite;
}
.smoke-1 { width: 30px; height: 30px; bottom: 25px; left: 52%; animation-delay: 2s; }
.smoke-2 { width: 20px; height: 20px; bottom: 30px; left: 56%; animation-delay: 2.1s; }
.smoke-3 { width: 25px; height: 25px; bottom: 20px; left: 50%; animation-delay: 2.2s; }

@keyframes smoke-rise {
    0%, 78% { transform: translateY(0) scale(1); opacity: 0; }
    85% { opacity: 0.5; transform: translateY(-10px) scale(1.2); }
    100% { transform: translateY(-50px) scale(2); opacity: 0; }
}

.crash-text {
    position: absolute;
    bottom: 10px;
    left: 50%;
    transform: translateX(-50%);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: var(--accent-red);
    letter-spacing: 2px;
    text-transform: uppercase;
    animation: crash-blink 1s ease-in-out infinite;
}
@keyframes crash-blink {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

/* Warning stripes */
.crash-warning {
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 4px;
    background: repeating-linear-gradient(90deg, var(--accent-red) 0px, var(--accent-red) 10px, var(--accent-orange) 10px, var(--accent-orange) 20px);
    animation: warning-scroll 1s linear infinite;
}
@keyframes warning-scroll {
    0% { transform: translateX(0); }
    100% { transform: translateX(20px); }
}

/* ==================== SUCCESSFUL LANDING ==================== */
.plane-landed {
    display: inline-block;
    font-size: 1.3rem;
    animation: smooth-land 1s ease-out forwards;
    filter: drop-shadow(0 0 8px rgba(6, 214, 160, 0.4));
    margin-right: 8px;
}
@keyframes smooth-land {
    0% { transform: translateX(-30px) translateY(-20px) rotate(-10deg); opacity: 0; }
    60% { transform: translateX(5px) translateY(2px) rotate(2deg); opacity: 1; }
    100% { transform: translateX(0) translateY(0) rotate(0deg); opacity: 1; }
}

/* ==================== CHAT MESSAGES ==================== */
.user-msg {
    position: relative;
    z-index: 1;
    background: linear-gradient(135deg, #118ab2 0%, #073b4c 100%);
    color: #fff;
    padding: 1rem 1.3rem;
    border-radius: 18px 18px 4px 18px;
    margin: 0.8rem 0;
    margin-left: 15%;
    font-family: 'Inter', sans-serif;
    font-size: 0.95rem;
    box-shadow: 0 4px 15px rgba(17, 138, 178, 0.2);
}
.bot-msg {
    position: relative;
    z-index: 1;
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    color: var(--text-primary);
    padding: 1.2rem 1.5rem;
    border-radius: 18px 18px 18px 4px;
    margin: 0.8rem 0;
    margin-right: 10%;
    font-family: 'Inter', sans-serif;
    font-size: 0.93rem;
    line-height: 1.65;
    box-shadow: var(--glow-cyan);
}
.bot-msg strong, .bot-msg b { color: var(--accent-cyan); }

/* Deep Thinking badge */
.deep-think-badge {
    position: relative;
    z-index: 1;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: linear-gradient(135deg, #f77f00 0%, #ef476f 100%);
    color: #fff;
    padding: 4px 12px;
    border-radius: 20px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
    box-shadow: 0 0 15px rgba(247, 127, 0, 0.3);
}
.deep-think-badge .pulse {
    width: 6px; height: 6px;
    background: #fff;
    border-radius: 50%;
    animation: pulse-dot 1s ease-in-out infinite;
}

/* ==================== METRIC CARDS ==================== */
.metric-card {
    position: relative;
    z-index: 1;
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-card .label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 1.5px;
}
.metric-card .value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    margin-top: 0.3rem;
}
.metric-card .value.good { color: var(--accent-cyan); }
.metric-card .value.warn { color: var(--accent-orange); }
.metric-card .value.bad { color: var(--accent-red); }

/* ==================== CHUNK CARDS ==================== */
.chunk-card {
    position: relative;
    z-index: 1;
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 10px;
    padding: 1rem;
    margin: 0.5rem 0;
    font-family: 'Inter', sans-serif;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.chunk-card:hover {
    border-color: var(--accent-blue);
    box-shadow: 0 0 12px rgba(17, 138, 178, 0.15);
}
.chunk-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
}
.chunk-rank {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: var(--accent-cyan);
    background: rgba(6, 214, 160, 0.1);
    padding: 2px 8px;
    border-radius: 6px;
}
.chunk-report {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: var(--text-secondary);
}
.chunk-score {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: var(--accent-blue);
}
.chunk-text {
    font-size: 0.82rem;
    color: var(--text-secondary);
    line-height: 1.55;
    max-height: 120px;
    overflow-y: auto;
    white-space: pre-wrap;
}
.chunk-meta {
    display: flex;
    gap: 1rem;
    margin-top: 0.5rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: #5a6577;
}

/* ==================== SIDEBAR ==================== */
section[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border-color) !important;
}
section[data-testid="stSidebar"] .stMarkdown { color: var(--text-secondary); }

.deep-toggle-container {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 1rem;
    margin: 0.5rem 0 1rem 0;
}
.deep-toggle-desc {
    font-size: 0.72rem;
    color: var(--text-secondary);
    margin-top: 0.3rem;
}

/* ==================== WELCOME SCREEN ==================== */
.welcome-container {
    position: relative;
    z-index: 1;
    text-align: center;
    padding: 3rem 2rem;
    max-width: 700px;
    margin: 2rem auto;
}
.welcome-plane {
    font-size: 4rem;
    display: inline-block;
    animation: welcome-hover 3s ease-in-out infinite;
    filter: drop-shadow(0 0 20px rgba(6, 214, 160, 0.3));
}
@keyframes welcome-hover {
    0%, 100% { transform: translateY(0) rotate(-2deg); }
    50% { transform: translateY(-15px) rotate(2deg); }
}
.welcome-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.1rem;
    color: var(--text-secondary);
    margin-top: 1.5rem;
    letter-spacing: 1px;
}
.welcome-suggestions {
    display: flex;
    flex-wrap: wrap;
    gap: 0.6rem;
    justify-content: center;
    margin-top: 1.5rem;
}
.suggestion-chip {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    color: var(--text-secondary);
    padding: 0.5rem 1rem;
    border-radius: 20px;
    font-family: 'Inter', sans-serif;
    font-size: 0.78rem;
    cursor: default;
    transition: border-color 0.2s, color 0.2s;
}
.suggestion-chip:hover {
    border-color: var(--accent-cyan);
    color: var(--text-primary);
}

/* ==================== MISC ==================== */
.streamlit-expanderHeader {
    background: var(--bg-card) !important;
    border-radius: 8px !important;
    color: var(--text-secondary) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important;
}
</style>
""", unsafe_allow_html=True)

# -- Background elements -------------------------------------------------------

st.markdown("""
<div class="sky-bg">
    <div class="cloud cloud-1"></div>
    <div class="cloud cloud-2"></div>
    <div class="cloud cloud-3"></div>
    <div class="cloud cloud-4"></div>
    <div class="cloud cloud-5"></div>
    <div class="star star-1"></div>
    <div class="star star-2"></div>
    <div class="star star-3"></div>
    <div class="star star-4"></div>
    <div class="star star-5"></div>
    <div class="star star-6"></div>
    <div class="star star-7"></div>
    <div class="star star-8"></div>
</div>
""", unsafe_allow_html=True)

# -- Header --------------------------------------------------------------------

st.markdown("""
<div class="main-header">
    <h1><span class="header-plane">&#9992;</span> BLACKBOX RAG</h1>
    <div class="subtitle"><span class="status-dot"></span>AVIATION INCIDENT INTELLIGENCE SYSTEM</div>
</div>
""", unsafe_allow_html=True)


# -- Resource caching ----------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_jina_model():
    return load_model()

@st.cache_resource(show_spinner=False)
def get_pinecone_index():
    return init_pinecone()

@st.cache_resource(show_spinner=False)
def get_reranker():
    return load_reranker()

@st.cache_resource(show_spinner=False)
def get_bm25(strategy):
    return build_bm25_index(strategy)


# -- Plane loader HTML ---------------------------------------------------------

PLANE_TAKEOFF_HTML = """
<div class="plane-loader">
    <div class="speed-readout">ALT <span class="val">FL280</span> &bull; GS 250kt</div>
    <div class="altitude-bar"></div>
    <div class="altitude-indicator"></div>
    <div class="plane-takeoff">&#9992;</div>
    <div class="exhaust"></div>
    <div class="runway"></div>
    <div class="loader-text">{msg}</div>
</div>
"""

PLANE_CRASH_HTML = """
<div class="plane-crash-container">
    <div class="crash-warning"></div>
    <div class="plane-crash">&#9992;</div>
    <div class="impact-flash"></div>
    <div class="smoke smoke-1"></div>
    <div class="smoke smoke-2"></div>
    <div class="smoke smoke-3"></div>
    <div class="crash-text">&#9888; {msg}</div>
</div>
"""

PLANE_LANDED_ICON = '<span class="plane-landed">&#9992;</span>'


# -- Sidebar controls ----------------------------------------------------------

with st.sidebar:
    st.markdown("### Controls")

    # Deep Thinking Toggle
    st.markdown('<div class="deep-toggle-container">', unsafe_allow_html=True)
    deep_thinking = st.toggle("Deep Thinking", value=False, help="DeepSeek + Multi-Query + HyDE + more chunks")
    if deep_thinking:
        st.markdown(
            '<div class="deep-toggle-desc" style="color: #f77f00;">'
            'DeepSeek V3.1 &bull; Multi-Query &bull; HyDE &bull; 25 chunks</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="deep-toggle-desc">Standard mode &mdash; fast responses</div>',
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    if deep_thinking:
        llm_provider = "deepseek"
        use_multi_query = True
        use_hyde = True
        top_k = 25
        mode = "Hybrid (BM25 + Semantic)"
        st.markdown("**Generator:** DeepSeek V3.1")
        st.markdown("**Mode:** Hybrid + MQ + HyDE")
        st.markdown(f"**Chunks:** {top_k}")
    else:
        llm_provider_label = st.selectbox(
            "Generator",
            ["Ollama (Local)", "DeepSeek (NVIDIA API)", "GPT (NVIDIA API)"],
            index=0,
        )
        if "Ollama" in llm_provider_label:
            llm_provider = "ollama"
        elif "GPT" in llm_provider_label:
            llm_provider = "gpt"
        else:
            llm_provider = "deepseek"

        mode = st.radio(
            "Retrieval Mode",
            ["Semantic Only", "Hybrid (BM25 + Semantic)", "Hybrid (with Cross-Encoder Rerank)"],
            index=1,
        )
        top_k = st.slider("Chunks to retrieve", 3, 50, 10)
        use_multi_query = st.checkbox("Multi-Query expansion", value=False)
        use_hyde = st.checkbox("HyDE expansion", value=False)

    ollama_model = "qwen2.5:32b"
    use_knowledge_doc = st.checkbox("LLM Knowledge augmentation", value=False) if not deep_thinking else False

    strategies = available_strategies()
    if not strategies:
        st.error("No chunk files found.")
        st.stop()

    if "md_recursive" in strategies:
        default_idx = strategies.index("md_recursive")
    elif "section" in strategies:
        default_idx = strategies.index("section")
    else:
        default_idx = 0

    strategy = st.selectbox("Chunking Strategy", strategies, index=default_idx)

    report_options = sorted(AVAILABLE_REPORTS.keys())
    report_labels = {
        rid: f"{rid} — {AVAILABLE_REPORTS[rid].get('aircraft', '').strip() or '?'} "
             f"({AVAILABLE_REPORTS[rid].get('date', '?')})"
        for rid in report_options
    }
    selected_reports = st.multiselect(
        "Filter by report (max 5)",
        options=report_options,
        format_func=lambda rid: report_labels.get(rid, rid),
        max_selections=5,
        help="Leave empty for auto-detection.",
    )

    run_eval = st.checkbox("Compute evaluation scores", value=True)

    st.divider()
    st.markdown(
        "<div style='font-family: JetBrains Mono, monospace; font-size: 0.65rem; color: #5a6577;'>"
        "Embeddings: Jina v5 (768d)<br>"
        "Reranker: qnli-distilroberta<br>"
        "Index: 332K vectors"
        "</div>",
        unsafe_allow_html=True,
    )


# -- Chat history init ---------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# -- Display chat history ------------------------------------------------------

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-msg">{msg["content"]}</div>', unsafe_allow_html=True)
    elif msg.get("type") == "crash":
        st.markdown(PLANE_CRASH_HTML.format(msg=msg["content"]), unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="bot-msg">{PLANE_LANDED_ICON}{msg["content"]}</div>',
            unsafe_allow_html=True,
        )

# -- Welcome screen when no messages ------------------------------------------

if not st.session_state.messages:
    st.markdown("""
    <div class="welcome-container">
        <div class="welcome-plane">&#9992;</div>
        <div class="welcome-title">Ask me anything about aviation incidents</div>
        <div class="welcome-suggestions">
            <div class="suggestion-chip">What caused the TWA 800 explosion?</div>
            <div class="suggestion-chip">Korean Air 801 crash near Guam</div>
            <div class="suggestion-chip">Engine failure during takeoff</div>
            <div class="suggestion-chip">Crew resource management failures</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# -- Chat input ----------------------------------------------------------------

query = st.chat_input("Ask about aviation incidents...")

if query:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": query})
    st.markdown(f'<div class="user-msg">{query}</div>', unsafe_allow_html=True)

    # Deep Thinking badge
    if deep_thinking:
        st.markdown(
            '<div class="deep-think-badge"><span class="pulse"></span>DEEP THINKING</div>',
            unsafe_allow_html=True,
        )

    # Show plane takeoff loader
    loader_placeholder = st.empty()
    loader_placeholder.markdown(
        PLANE_TAKEOFF_HTML.format(msg="searching flight records"),
        unsafe_allow_html=True,
    )

    # Load resources
    jina_model = get_jina_model()
    index = get_pinecone_index()

    is_hybrid = "Hybrid" in mode

    # -- Retrieval pipeline ----
    multi_query_variants = None
    hyde_docs = None
    knowledge_result = None
    knowledge_doc = None
    llm_ntsb = ""
    llm_confidence = "none"
    regex_ntsb = ""
    resolved_ntsb = ""

    if is_hybrid:
        reranker = get_reranker() if "Rerank" in mode else None
        bm25, chunks = get_bm25(strategy)

        queries = expand_query_variants(query)

        # Augmentation
        aug_futures = {}
        active_augs = [
            label for label, enabled in [
                ("multi-query", use_multi_query),
                ("HyDE", use_hyde),
                ("LLM knowledge", use_knowledge_doc),
            ] if enabled
        ]

        if active_augs:
            loader_placeholder.markdown(
                PLANE_TAKEOFF_HTML.format(msg="generating query variants"),
                unsafe_allow_html=True,
            )
            with ThreadPoolExecutor(max_workers=3) as executor:
                if use_multi_query:
                    aug_futures["mq"] = executor.submit(
                        generate_multi_queries, query, llm_provider
                    )
                if use_hyde:
                    aug_futures["hyde"] = executor.submit(
                        generate_hyde_documents, query, 2, llm_provider, ollama_model
                    )
                if use_knowledge_doc:
                    aug_futures["llm_k"] = executor.submit(
                        generate_knowledge_doc, query, llm_provider, ollama_model
                    )

            if "mq" in aug_futures:
                multi_query_variants = aug_futures["mq"].result() or []
                queries.extend(multi_query_variants)
            if "hyde" in aug_futures:
                hyde_docs = aug_futures["hyde"].result() or []
                queries.extend(hyde_docs)
            if "llm_k" in aug_futures:
                knowledge_result = aug_futures["llm_k"].result()
                knowledge_doc = knowledge_result.narrative
                llm_ntsb = knowledge_result.ntsb_number
                llm_confidence = knowledge_result.confidence

        # Report detection
        regex_ntsb = detect_report_from_query(query)
        if use_knowledge_doc:
            resolved_ntsb = resolve_report_number(regex_ntsb, llm_ntsb, llm_confidence)
        else:
            resolved_ntsb = regex_ntsb

        if selected_reports:
            ntsb_override = selected_reports if len(selected_reports) > 1 else selected_reports[0]
        else:
            ntsb_override = resolved_ntsb if resolved_ntsb else None

        # Retrieval
        loader_placeholder.markdown(
            PLANE_TAKEOFF_HTML.format(msg=f"scanning {len(queries)} query variants"),
            unsafe_allow_html=True,
        )
        sem_top_k = 80 if deep_thinking else 60
        bm25_top_k = 80 if deep_thinking else 60
        ranked_lists = []
        for i, q in enumerate(queries, 1):
            ranked_lists.append(retrieve(q, strategy, top_k=sem_top_k, model=jina_model, index=index, ntsb_override=ntsb_override))
            bm25_results = bm25_retrieve(q, bm25, chunks, top_k=bm25_top_k)
            if selected_reports:
                bm25_results = [r for r in bm25_results if r.get("ntsb_no", "") in selected_reports]
            ranked_lists.append(bm25_results)

        if knowledge_doc:
            ranked_lists.append(retrieve(knowledge_doc, strategy, top_k=sem_top_k, model=jina_model, index=index, ntsb_override=ntsb_override))

        # Fusion
        loader_placeholder.markdown(
            PLANE_TAKEOFF_HTML.format(msg="fusing and ranking results"),
            unsafe_allow_html=True,
        )
        report_cap = 50 if selected_reports else 8
        fused = rrf_fuse_lists(ranked_lists, max_per_report=report_cap)
        if reranker:
            min_reports = 1 if selected_reports else 3
            matches = rerank(query, fused, reranker, top_k=top_k, min_unique_reports=min_reports)
        else:
            matches = sorted(fused, key=lambda x: x.get("score", 0), reverse=True)[:top_k]
        matches = enrich_with_neighbors(matches, chunks, window=2)
    else:
        sem_override = None
        if selected_reports:
            sem_override = selected_reports if len(selected_reports) > 1 else selected_reports[0]
        matches = retrieve(query, strategy, top_k=top_k, model=jina_model, index=index, ntsb_override=sem_override)

    # Context
    context_texts = []
    for m in matches:
        if hasattr(m, "metadata"):
            context_texts.append(m.metadata.get("text", ""))
        else:
            context_texts.append(m.get("text", ""))

    # Check if we got results
    has_results = len(matches) > 0 and any(t.strip() for t in context_texts)

    if has_results:
        # Generate answer
        loader_placeholder.markdown(
            PLANE_TAKEOFF_HTML.format(msg="analyzing evidence &bull; generating answer"),
            unsafe_allow_html=True,
        )
        answer = generate_answer(
            query, matches, llm_provider=llm_provider, ollama_model=ollama_model,
        )

        # Check if the answer itself says insufficient context
        answer_lower = answer.lower()
        is_insufficient = any(phrase in answer_lower for phrase in [
            "insufficient context", "no relevant", "cannot answer",
            "no information", "not enough context", "unable to find",
        ])
    else:
        answer = ""
        is_insufficient = True

    # Clear the loader
    loader_placeholder.empty()

    if is_insufficient and not answer.strip():
        # Full crash — no results at all
        st.markdown(
            PLANE_CRASH_HTML.format(msg="no flight data recovered &bull; try a different query"),
            unsafe_allow_html=True,
        )
        st.session_state.messages.append({
            "role": "assistant",
            "type": "crash",
            "content": "no flight data recovered &bull; try a different query",
        })
    elif is_insufficient:
        # Partial crash — got an answer but it says insufficient
        st.markdown(
            PLANE_CRASH_HTML.format(msg="partial data recovery &bull; limited evidence found"),
            unsafe_allow_html=True,
        )
        safe_answer = html_lib.escape(answer)
        st.markdown(f'<div class="bot-msg">{safe_answer}</div>', unsafe_allow_html=True)
        st.session_state.messages.append({"role": "assistant", "content": answer})
    else:
        # Successful landing
        safe_answer = html_lib.escape(answer)
        st.markdown(
            f'<div class="bot-msg">{PLANE_LANDED_ICON}{safe_answer}</div>',
            unsafe_allow_html=True,
        )
        st.session_state.messages.append({"role": "assistant", "content": answer})

    # -- Augmentation details (collapsible) --
    if multi_query_variants or hyde_docs or knowledge_doc:
        with st.expander("Retrieval augmentation details"):
            if multi_query_variants:
                st.markdown("**Multi-Query variants:**")
                for i, q in enumerate(multi_query_variants, 1):
                    st.markdown(f"  {i}. {q}")
            if hyde_docs:
                st.markdown("**HyDE hypothetical excerpts:**")
                for i, doc in enumerate(hyde_docs, 1):
                    st.markdown(f"  **{i}.** {doc[:300]}...")
            if knowledge_doc:
                st.markdown("**LLM Knowledge narrative:**")
                st.markdown(str(knowledge_doc)[:500] + "..." if len(str(knowledge_doc)) > 500 else str(knowledge_doc))

    # -- Evaluation scores --
    if run_eval and has_results:
        col1, col2 = st.columns(2)

        with col1:
            with st.spinner("Computing faithfulness..."):
                faith_score, faith_details = compute_faithfulness(answer, context_texts)
            score_class = "good" if faith_score >= 0.7 else "warn" if faith_score >= 0.4 else "bad"
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">Faithfulness</div>
                <div class="value {score_class}">{faith_score:.0%}</div>
            </div>
            """, unsafe_allow_html=True)

            if faith_details:
                with st.expander("Claim verification"):
                    for fd in faith_details:
                        icon = "+" if fd.get("supported") else "-"
                        st.markdown(f"**{icon}** {fd.get('claim', '')}")
                        st.caption(fd.get("reasoning", ""))

        with col2:
            with st.spinner("Computing relevancy..."):
                rel_score, rel_alternates = compute_relevancy(query, answer, jina_model)
            score_class = "good" if rel_score >= 0.7 else "warn" if rel_score >= 0.4 else "bad"
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">Relevancy</div>
                <div class="value {score_class}">{rel_score:.0%}</div>
            </div>
            """, unsafe_allow_html=True)

            if rel_alternates:
                with st.expander("Generated questions from answer"):
                    for alt in rel_alternates:
                        st.markdown(f"- {alt}")

    # -- Retrieved chunks --
    if matches:
        with st.expander(f"Retrieved Evidence ({len(matches)} chunks)"):
            for i, m in enumerate(matches, 1):
                if hasattr(m, "metadata"):
                    meta = m.metadata
                    score = m.score
                else:
                    meta = m
                    score = m.get("score", 0)

                report_id = meta.get("ntsb_no") or meta.get("report_id", "N/A")
                section = meta.get("section_title", "")
                text = html_lib.escape(meta.get("text", ""))
                retrieval_source = meta.get("retrieval_strategy", "semantic")

                st.markdown(f"""
                <div class="chunk-card">
                    <div class="chunk-header">
                        <span class="chunk-rank">#{i}</span>
                        <span class="chunk-report">{report_id}{(' &mdash; ' + html_lib.escape(section)) if section else ''}</span>
                        <span class="chunk-score">{score:.4f}</span>
                    </div>
                    <div class="chunk-text">{text[:400]}{'...' if len(text) > 400 else ''}</div>
                    <div class="chunk-meta">
                        <span>Date: {meta.get('event_date', '—')}</span>
                        <span>Phase: {meta.get('phase_of_flight', '—')}</span>
                        <span>Source: {retrieval_source}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
