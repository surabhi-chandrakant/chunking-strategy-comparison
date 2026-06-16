"""
Chunking Strategy Comparison — Fixed & Redesigned
"""

import os
import streamlit as st
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dataclasses import dataclass, field
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import hashlib

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

@dataclass
class Config:
    CHUNK_SIZES: list = field(default_factory=lambda: [50, 100, 200, 300, 500])
    OVERLAP_SIZES: list = field(default_factory=lambda: [0, 10, 20, 30, 50])
    DEFAULT_CHUNK_SIZE: int = 200
    DEFAULT_OVERLAP: int = 20
    SIMILARITY_THRESHOLDS: list = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5])
    DEFAULT_THRESHOLD: float = 0.3
    MAX_DOCUMENTS: int = 10

config = Config()

# ─── FIX: Updated model — mixtral-8x7b-32768 was decommissioned ───
GROQ_MODEL = "llama-3.3-70b-versatile"

class GroqService:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('GROQ_API_KEY')
        self.client = None
        self.is_available = False
        self.model = GROQ_MODEL
        self.error_message = ""

        if self.api_key and GROQ_AVAILABLE:
            try:
                self.client = Groq(api_key=self.api_key)
                test = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": "ping"}],
                    max_tokens=5
                )
                self.is_available = True
            except Exception as e:
                self.error_message = str(e)
                self.is_available = False

    def _text_to_vector(self, text: str) -> np.ndarray:
        words = text.lower().split()
        vector = np.zeros(512)
        for i, word in enumerate(words[:100]):
            h = int(hashlib.md5(f"{word}_{i}".encode()).hexdigest()[:8], 16)
            vector[h % 512] += 1.0 / (i + 1)
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector

    def generate_embedding(self, text: str) -> Optional[np.ndarray]:
        if not self.is_available or not self.client:
            return None
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Analyze text semantics concisely."},
                    {"role": "user", "content": f"Semantic summary: {text[:800]}"}
                ],
                temperature=0.1,
                max_tokens=200
            )
            return self._text_to_vector(resp.choices[0].message.content)
        except Exception as e:
            st.warning(f"Groq error: {str(e)[:100]}")
            return None


class DocumentProcessor:
    def __init__(self):
        self.documents: List[str] = []
        self.document_names: List[str] = []
        self.groq_service = None
        self.use_groq = False

    def set_groq_service(self, groq_service: GroqService, use_groq: bool = False):
        self.groq_service = groq_service
        self.use_groq = use_groq and groq_service.is_available

    def add_document(self, content: str, name: Optional[str] = None):
        if len(self.documents) >= config.MAX_DOCUMENTS:
            st.warning(f"Maximum {config.MAX_DOCUMENTS} documents allowed")
            return
        if not content.strip():
            st.warning("Empty document content")
            return
        self.documents.append(content)
        self.document_names.append(name or f"Document_{len(self.documents)}")

    def clear_documents(self):
        self.documents.clear()
        self.document_names.clear()

    def get_document_count(self) -> int:
        return len(self.documents)


class FixedSizeChunking:
    strategy_name = "Fixed Size"

    def __init__(self, chunk_size=200, overlap=20, **_):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> List[str]:
        words = text.split()
        if len(words) <= self.chunk_size:
            return [text]
        chunks, step = [], max(1, self.chunk_size - self.overlap)
        for i in range(0, len(words), step):
            part = words[i:i + self.chunk_size]
            if part:
                chunks.append(' '.join(part))
            if i + self.chunk_size >= len(words):
                break
        return chunks


class SentenceBasedChunking:
    strategy_name = "Sentence Based"

    def __init__(self, chunk_size=200, overlap=20, **_):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> List[str]:
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        if not sentences:
            return []
        if len(sentences) == 1:
            return sentences
        chunks, current, cur_len = [], [], 0
        for s in sentences:
            slen = len(s.split())
            if cur_len + slen <= self.chunk_size:
                current.append(s)
                cur_len += slen
            else:
                if current:
                    chunks.append(' '.join(current))
                current, cur_len = [s], slen
        if current:
            chunks.append(' '.join(current))
        return chunks


class SemanticChunking:
    strategy_name = "Semantic"

    def __init__(self, chunk_size=200, overlap=20, similarity_threshold=0.3, **_):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.similarity_threshold = similarity_threshold
        self.vectorizer = TfidfVectorizer()

    def chunk(self, text: str) -> List[str]:
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        if not sentences:
            return []
        if len(sentences) == 1:
            return sentences
        try:
            mat = self.vectorizer.fit_transform(sentences)
            sims = cosine_similarity(mat)
        except Exception:
            return self._fallback(sentences)

        chunks, current, cur_words = [], [sentences[0]], len(sentences[0].split())
        for i in range(1, len(sentences)):
            sim = sims[i - 1, i]
            sw = len(sentences[i].split())
            if sim > self.similarity_threshold and cur_words + sw <= self.chunk_size:
                current.append(sentences[i])
                cur_words += sw
            else:
                chunks.append(' '.join(current))
                current, cur_words = [sentences[i]], sw
        if current:
            chunks.append(' '.join(current))
        return chunks

    def _fallback(self, sentences):
        chunks, current, cur_len = [], [], 0
        for s in sentences:
            sl = len(s.split())
            if cur_len + sl <= self.chunk_size:
                current.append(s)
                cur_len += sl
            else:
                if current:
                    chunks.append(' '.join(current))
                current, cur_len = [s], sl
        if current:
            chunks.append(' '.join(current))
        return chunks


class ChunkingManager:
    STRATEGIES = {
        'Fixed Size': FixedSizeChunking,
        'Sentence Based': SentenceBasedChunking,
        'Semantic': SemanticChunking,
    }

    def __init__(self):
        self.processor = DocumentProcessor()
        self.results_cache = {}

    def process_documents(self, strategy_name: str, **kwargs) -> Dict:
        key = f"{strategy_name}_{kwargs}"
        if key in self.results_cache:
            return self.results_cache[key]
        strategy = self.STRATEGIES[strategy_name](**kwargs)
        results = {}
        for idx, doc in enumerate(self.processor.documents):
            chunks = strategy.chunk(doc)
            results[f"doc_{idx}"] = {
                'chunks': chunks,
                'count': len(chunks),
                'avg_length': np.mean([len(c.split()) for c in chunks]) if chunks else 0,
            }
        self.results_cache[key] = results
        return results

    def clear_cache(self):
        self.results_cache.clear()


class RetrievalEvaluator:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000)

    def evaluate(self, chunks: List[str], query: str) -> Dict:
        empty = {'precision': 0, 'recall': 0, 'relevance': 0, 'diversity': 0, 'top_chunks': []}
        if not chunks or not query:
            return empty
        try:
            all_texts = [query] + chunks[:50]
            mat = self.vectorizer.fit_transform(all_texts)
            sims = cosine_similarity(mat[0:1], mat[1:]).flatten()
            top_idx = np.argsort(sims)[-5:][::-1]
            top_chunks = [chunks[i] for i in top_idx if i < len(chunks)]
            return {
                'precision': float(np.mean(sims)),
                'recall': min(1.0, len(top_idx) / 5),
                'relevance': float(np.max(sims)),
                'diversity': float(1 - np.std(sims)) if len(sims) > 1 else 0,
                'top_chunks': top_chunks,
            }
        except Exception:
            return empty


SAMPLE_DOCS = [
    ("AI Introduction", "Artificial intelligence (AI) is the simulation of human intelligence in machines that are programmed to think and learn. The term can also be applied to any machine that exhibits traits associated with a human mind such as learning and problem-solving. AI technology has become increasingly important in modern society, with applications ranging from healthcare to finance. The field of AI research was founded at a workshop held on the campus of Dartmouth College in 1956."),
    ("Machine Learning", "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on the development of computer programs that can access data and use it to learn for themselves. There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning."),
    ("Deep Learning", "Deep learning is part of a broader family of machine learning methods based on artificial neural networks. Learning can be supervised, semi-supervised or unsupervised. Deep learning architectures such as deep neural networks, deep belief networks, and recurrent neural networks have been applied to fields including computer vision, speech recognition, natural language processing, and bioinformatics."),
    ("NLP", "Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language. In particular, it focuses on how to program computers to process and analyze large amounts of natural language data. NLP is used in many applications including machine translation, sentiment analysis, chatbots, and information extraction."),
    ("Computer Vision", "Computer vision is an interdisciplinary scientific field that deals with how computers can gain high-level understanding from digital images or videos. From the perspective of engineering, it seeks to understand and automate tasks that the human visual system can do. Computer vision tasks include methods for acquiring, processing, analyzing, and understanding digital images."),
]


def load_sample_documents(processor):
    processor.clear_documents()
    for name, content in SAMPLE_DOCS:
        processor.add_document(content, name)
    return len(SAMPLE_DOCS)


CSS = """
<style>
/* ── Reset & base ───────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
}

.stApp {
    background: #0f1117;
}

/* ── Sidebar ─────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #161b27 !important;
    border-right: 1px solid #242938;
}

[data-testid="stSidebar"] * {
    color: #c9d1e0 !important;
}

[data-testid="stSidebar"] .stMarkdown h3,
[data-testid="stSidebar"] .stMarkdown h4 {
    color: #7b93ff !important;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-top: 1.2rem;
    margin-bottom: 0.4rem;
}

/* ── Main content area ───────────────────────────────────── */
.main .block-container {
    background: #0f1117;
    padding: 2rem 2.5rem;
    max-width: 1200px;
}

/* ── All text in main area ───────────────────────────────── */
.stApp p, .stApp span, .stApp label,
.stApp div, .stMarkdown, .stMarkdown p {
    color: #c9d1e0 !important;
}

/* ── Page title ──────────────────────────────────────────── */
.page-title {
    font-size: 1.9rem;
    font-weight: 700;
    color: #e8eaf6 !important;
    letter-spacing: -0.03em;
    margin-bottom: 0.2rem;
}

.page-sub {
    font-size: 0.95rem;
    color: #7a85a3 !important;
    margin-bottom: 2rem;
}

/* ── Metric cards ────────────────────────────────────────── */
[data-testid="metric-container"] {
    background: #161b27;
    border: 1px solid #242938;
    border-radius: 10px;
    padding: 1rem 1.2rem !important;
}

[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #7b93ff !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
}

[data-testid="metric-container"] [data-testid="stMetricLabel"] {
    color: #7a85a3 !important;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* ── Tabs ────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: #161b27;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
    border: 1px solid #242938;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 7px;
    background: transparent;
    color: #7a85a3 !important;
    font-weight: 500;
    font-size: 0.88rem;
    padding: 0.45rem 1.1rem;
    border: none;
}

.stTabs [aria-selected="true"] {
    background: #7b93ff !important;
    color: #ffffff !important;
}

/* ── Chunk boxes ─────────────────────────────────────────── */
.chunk-card {
    background: #161b27;
    border: 1px solid #242938;
    border-left: 3px solid #7b93ff;
    border-radius: 8px;
    padding: 0.85rem 1rem;
    margin-bottom: 0.6rem;
    color: #c9d1e0 !important;
    font-size: 0.88rem;
    line-height: 1.6;
}

.chunk-label {
    font-size: 0.72rem;
    font-weight: 600;
    color: #7b93ff !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.35rem;
}

/* ── Status pills ────────────────────────────────────────── */
.pill {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 4px 0;
}
.pill-ok   { background: #1a2e22; color: #4ade80 !important; border: 1px solid #2d5a3d; }
.pill-warn { background: #2e2a14; color: #facc15 !important; border: 1px solid #5a4d1a; }
.pill-err  { background: #2e1a1a; color: #f87171 !important; border: 1px solid #5a2d2d; }
.pill-info { background: #1a222e; color: #60a5fa !important; border: 1px solid #1e3a5f; }

/* ── Expanders ───────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: #161b27;
    border: 1px solid #242938 !important;
    border-radius: 10px;
    margin-bottom: 0.5rem;
}

[data-testid="stExpander"] summary {
    color: #c9d1e0 !important;
    font-weight: 500;
}

[data-testid="stExpander"] > div {
    background: #0f1117;
}

/* ── Text inputs ─────────────────────────────────────────── */
.stTextInput input, .stTextArea textarea {
    background: #161b27 !important;
    border: 1px solid #242938 !important;
    color: #c9d1e0 !important;
    border-radius: 8px !important;
}

.stTextInput input::placeholder, .stTextArea textarea::placeholder {
    color: #4a5568 !important;
}

.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: #7b93ff !important;
    box-shadow: 0 0 0 2px rgba(123, 147, 255, 0.2) !important;
}

/* ── Buttons ─────────────────────────────────────────────── */
.stButton > button {
    background: #7b93ff !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 0.5rem 1.2rem !important;
    transition: background 0.15s ease !important;
}

.stButton > button:hover {
    background: #6578e0 !important;
}

.stButton > button[kind="secondary"] {
    background: #1e2433 !important;
    color: #c9d1e0 !important;
    border: 1px solid #2d3548 !important;
}

/* ── Select / slider ─────────────────────────────────────── */
.stSelectSlider [data-testid="stTickBar"],
.stSlider [data-testid="stTickBar"] {
    color: #4a5568 !important;
}

/* ── Radio ───────────────────────────────────────────────── */
.stRadio label {
    color: #c9d1e0 !important;
}

/* ── Checkboxes ──────────────────────────────────────────── */
.stCheckbox label {
    color: #c9d1e0 !important;
}

/* ── Dataframe ───────────────────────────────────────────── */
.stDataFrame {
    background: #161b27;
    border: 1px solid #242938;
    border-radius: 10px;
}

/* ── Alerts ──────────────────────────────────────────────── */
.stAlert {
    background: #161b27 !important;
    border-radius: 8px !important;
    color: #c9d1e0 !important;
}

/* ── Dividers ────────────────────────────────────────────── */
hr {
    border-color: #242938 !important;
}

/* ── Strategy badge ──────────────────────────────────────── */
.strategy-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #1e2433;
    border: 1px solid #7b93ff;
    color: #7b93ff !important;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.8rem;
    font-weight: 600;
}

/* ── Section headings ────────────────────────────────────── */
.section-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: #e8eaf6 !important;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ── Sidebar image ───────────────────────────────────────── */
[data-testid="stSidebar"] img {
    filter: brightness(0) invert(1) opacity(0.6);
}

/* ── Scrollbar ───────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0f1117; }
::-webkit-scrollbar-thumb { background: #2d3548; border-radius: 3px; }
</style>
"""


def main():
    st.set_page_config(
        page_title="Chunking Strategy Comparison",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CSS, unsafe_allow_html=True)

    # ── Session state ────────────────────────────────────────
    if 'manager' not in st.session_state:
        st.session_state.manager = ChunkingManager()
        st.session_state.evaluator = RetrievalEvaluator()
        st.session_state.results = {}
        st.session_state.history = []
        st.session_state.current_evaluation = None
        st.session_state.groq_service = None
        st.session_state.use_groq = False

    # ── Sidebar ──────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")

        st.markdown("#### 🔌 Groq API")
        use_groq = st.checkbox(
            "Enable Groq embeddings",
            value=st.session_state.use_groq,
            help=f"Uses {GROQ_MODEL}"
        )
        st.session_state.use_groq = use_groq

        api_key = st.text_input(
            "Groq API Key",
            type="password",
            placeholder="gsk_...",
            help="Free tier at console.groq.com"
        )

        if api_key:
            os.environ['GROQ_API_KEY'] = api_key
            gs = GroqService(api_key)
            st.session_state.groq_service = gs
            if gs.is_available:
                st.markdown(f'<span class="pill pill-ok">✓ Groq connected ({GROQ_MODEL})</span>', unsafe_allow_html=True)
            else:
                msg = gs.error_message[:90] + "…" if len(gs.error_message) > 90 else gs.error_message
                st.markdown(f'<span class="pill pill-err">✗ {msg}</span>', unsafe_allow_html=True)
        else:
            if use_groq:
                st.markdown('<span class="pill pill-warn">⚠ Enter API key to enable</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="pill pill-info">ℹ Using TF-IDF (no API needed)</span>', unsafe_allow_html=True)

        if st.session_state.groq_service:
            st.session_state.manager.processor.set_groq_service(
                st.session_state.groq_service,
                st.session_state.use_groq,
            )

        st.divider()
        st.markdown("#### 📏 Parameters")
        chunk_size = st.select_slider("Chunk size (words)", options=config.CHUNK_SIZES, value=config.DEFAULT_CHUNK_SIZE)
        overlap    = st.select_slider("Overlap (words)",    options=config.OVERLAP_SIZES, value=config.DEFAULT_OVERLAP)
        threshold  = st.select_slider("Semantic threshold", options=config.SIMILARITY_THRESHOLDS, value=config.DEFAULT_THRESHOLD,
                                      help="Only used for Semantic strategy")

        st.divider()
        st.markdown("#### 🎯 Strategy")
        strategy = st.radio("Select strategy", ["Fixed Size", "Sentence Based", "Semantic"], index=0)
        st.session_state.current_strategy = strategy
        color_map = {"Fixed Size": "#7b93ff", "Sentence Based": "#a78bfa", "Semantic": "#34d399"}
        st.markdown(
            f'<span class="strategy-badge" style="border-color:{color_map[strategy]};color:{color_map[strategy]} !important">'
            f'● {strategy}</span>',
            unsafe_allow_html=True,
        )

        st.divider()
        st.markdown("#### 📄 Documents")
        if st.button("📚 Load sample docs", use_container_width=True):
            n = load_sample_documents(st.session_state.manager.processor)
            st.session_state.results = {}
            st.session_state.history = []
            st.success(f"Loaded {n} sample documents")
            st.rerun()
        if st.button("🗑 Clear all", use_container_width=True):
            st.session_state.manager.processor.clear_documents()
            st.session_state.results = {}
            st.session_state.history = []
            st.rerun()

        st.divider()
        st.markdown("#### 📊 Status")
        doc_count = st.session_state.manager.processor.get_document_count()
        c1, c2 = st.columns(2)
        c1.metric("Docs", doc_count)
        if st.session_state.results:
            c2.metric("Chunks", sum(r['count'] for r in st.session_state.results.values()))

    # ── Header ───────────────────────────────────────────────
    st.markdown('<p class="page-title">📊 Chunking Strategy Comparison</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">Compare Fixed Size, Sentence Based, and Semantic chunking strategies</p>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📝 Documents", "🔄 Chunking", "🔍 Evaluation", "📈 Analytics"])

    # ── Tab 1: Documents ─────────────────────────────────────
    with tab1:
        st.markdown('<p class="section-title">📝 Add Document</p>', unsafe_allow_html=True)
        col1, col2 = st.columns([3, 1])
        with col1:
            doc_text = st.text_area("Content", height=140, placeholder="Paste your text here…", label_visibility="collapsed")
        with col2:
            doc_name = st.text_input("Name (optional)", placeholder="Doc name")
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            if st.button("➕ Add", use_container_width=True):
                if doc_text.strip():
                    st.session_state.manager.processor.add_document(doc_text, doc_name or None)
                    st.session_state.results = {}
                    st.success(f"Added. Total: {st.session_state.manager.processor.get_document_count()}")
                    st.rerun()
                else:
                    st.warning("Enter some text first.")

        docs = st.session_state.manager.processor.documents
        if docs:
            st.markdown('<p class="section-title">📚 Loaded Documents</p>', unsafe_allow_html=True)
            for idx, (name, content) in enumerate(zip(
                st.session_state.manager.processor.document_names, docs
            )):
                with st.expander(f"📄 {name}  ·  {len(content):,} chars"):
                    st.markdown(f'<div class="chunk-card">{content}</div>', unsafe_allow_html=True)
                    if st.button("Remove", key=f"rm_{idx}"):
                        st.session_state.manager.processor.documents.pop(idx)
                        st.session_state.manager.processor.document_names.pop(idx)
                        st.session_state.results = {}
                        st.rerun()
        else:
            st.info("No documents yet. Add one above or load the samples from the sidebar.")

    # ── Tab 2: Chunking ──────────────────────────────────────
    with tab2:
        st.markdown('<p class="section-title">🔄 Chunking Results</p>', unsafe_allow_html=True)
        if not st.session_state.manager.processor.get_document_count():
            st.info("Add documents first (Documents tab or sidebar).")
        else:
            c1, c2 = st.columns([2, 1])
            with c1:
                if st.button("▶ Process documents", use_container_width=True):
                    with st.spinner("Chunking…"):
                        kw = {'chunk_size': chunk_size, 'overlap': overlap}
                        if strategy == "Semantic":
                            kw['similarity_threshold'] = threshold
                        st.session_state.results = st.session_state.manager.process_documents(strategy, **kw)
                    st.success("Done!")
                    st.rerun()
            with c2:
                if st.session_state.use_groq and st.session_state.groq_service and st.session_state.groq_service.is_available:
                    st.markdown('<span class="pill pill-ok">✨ Groq active</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span class="pill pill-info">TF-IDF mode</span>', unsafe_allow_html=True)

            if st.session_state.results:
                res = st.session_state.results
                total = sum(r['count'] for r in res.values())
                avg   = np.mean([r['avg_length'] for r in res.values()])
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Chunks", total)
                m2.metric("Avg Length", f"{avg:.0f} words")
                m3.metric("Strategy", strategy)
                m4.metric("Documents", len(res))

                st.markdown('<p class="section-title">Chunks by Document</p>', unsafe_allow_html=True)
                for doc_idx, (key, r) in enumerate(res.items()):
                    dname = st.session_state.manager.processor.document_names[doc_idx]
                    with st.expander(f"📄 {dname}  ·  {r['count']} chunks  ·  avg {r['avg_length']:.0f} words"):
                        for i, chunk in enumerate(r['chunks'][:5]):
                            st.markdown(f'<p class="chunk-label">Chunk {i+1}</p>'
                                        f'<div class="chunk-card">{chunk}</div>', unsafe_allow_html=True)
                        if r['count'] > 5:
                            st.caption(f"… and {r['count'] - 5} more chunks not shown.")

    # ── Tab 3: Evaluation ────────────────────────────────────
    with tab3:
        st.markdown('<p class="section-title">🔍 Retrieval Evaluation</p>', unsafe_allow_html=True)
        if not st.session_state.results:
            st.info("Process documents first (Chunking tab).")
        else:
            c1, c2 = st.columns([3, 1])
            with c1:
                query = st.text_input("Query", placeholder="e.g. What is machine learning?", label_visibility="collapsed")
            with c2:
                run_eval = st.button("🔍 Evaluate", use_container_width=True)

            if run_eval:
                if query.strip():
                    all_chunks = [c for r in st.session_state.results.values() for c in r['chunks']]
                    ev = st.session_state.evaluator.evaluate(all_chunks, query)
                    st.session_state.current_evaluation = ev
                    st.session_state.history.append({
                        'timestamp': datetime.now().strftime("%H:%M:%S"),
                        'query': query[:50],
                        'strategy': strategy,
                        'results': ev,
                    })
                    st.rerun()
                else:
                    st.warning("Enter a query.")

            if st.session_state.current_evaluation:
                ev = st.session_state.current_evaluation
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Precision",  f"{ev['precision']:.3f}")
                m2.metric("Recall",     f"{ev['recall']:.3f}")
                m3.metric("Relevance",  f"{ev['relevance']:.3f}")
                m4.metric("Diversity",  f"{ev['diversity']:.3f}")

                if ev['top_chunks']:
                    st.markdown('<p class="section-title">🏆 Top Retrieved Chunks</p>', unsafe_allow_html=True)
                    for i, ch in enumerate(ev['top_chunks'], 1):
                        st.markdown(f'<p class="chunk-label">Rank {i}</p>'
                                    f'<div class="chunk-card">{ch}</div>', unsafe_allow_html=True)

                if st.session_state.history:
                    st.markdown('<p class="section-title">Query History</p>', unsafe_allow_html=True)
                    hdf = pd.DataFrame([
                        {'Time': h['timestamp'], 'Query': h['query'], 'Strategy': h['strategy'],
                         'Precision': f"{h['results']['precision']:.3f}",
                         'Recall': f"{h['results']['recall']:.3f}"}
                        for h in st.session_state.history[-6:]
                    ])
                    st.dataframe(hdf, use_container_width=True)

    # ── Tab 4: Analytics ─────────────────────────────────────
    with tab4:
        st.markdown('<p class="section-title">📈 Analytics & Comparison</p>', unsafe_allow_html=True)
        if not st.session_state.history:
            st.info("Run evaluations first (Evaluation tab).")
        else:
            df = pd.DataFrame([
                {'Strategy': h['strategy'],
                 'Precision': h['results']['precision'],
                 'Recall':    h['results']['recall'],
                 'Relevance': h['results']['relevance'],
                 'Diversity': h['results']['diversity']}
                for h in st.session_state.history
            ])

            cmap = {'Fixed Size': '#7b93ff', 'Sentence Based': '#a78bfa', 'Semantic': '#34d399'}

            st.markdown("**Strategy Performance (box plots)**")
            fig = make_subplots(rows=1, cols=4,
                                subplot_titles=['Precision', 'Recall', 'Relevance', 'Diversity'])
            for col_idx, metric in enumerate(['Precision', 'Recall', 'Relevance', 'Diversity']):
                for strat in df['Strategy'].unique():
                    data = df[df['Strategy'] == strat][metric]
                    fig.add_trace(go.Box(
                        y=data, name=strat,
                        marker_color=cmap.get(strat, '#888'),
                        legendgroup=strat,
                        showlegend=(col_idx == 0),
                    ), row=1, col=col_idx + 1)
            fig.update_layout(
                height=400, paper_bgcolor='#0f1117', plot_bgcolor='#161b27',
                font_color='#c9d1e0', legend_bgcolor='#161b27',
            )
            fig.update_yaxes(range=[0, 1], gridcolor='#242938')
            fig.update_xaxes(showgrid=False)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("**Summary Statistics**")
            summary = df.groupby('Strategy').agg(
                Precision_mean=('Precision', 'mean'),
                Precision_std=('Precision', 'std'),
                Recall_mean=('Recall', 'mean'),
                Recall_std=('Recall', 'std'),
                Relevance_mean=('Relevance', 'mean'),
            ).round(3)
            st.dataframe(summary, use_container_width=True)

            best_p = df.groupby('Strategy')['Precision'].mean().idxmax()
            best_r = df.groupby('Strategy')['Recall'].mean().idxmax()
            c1, c2 = st.columns(2)
            c1.metric("Best Precision", best_p)
            c2.metric("Best Recall", best_r)
            st.info(f"Based on your queries, **{best_p}** leads on precision.")

            if len(df) > 2:
                st.markdown("**Performance Trend**")
                df['Run'] = range(len(df))
                fig2 = px.line(df, x='Run', y=['Precision', 'Recall', 'Relevance'],
                               color_discrete_map={'Precision': '#7b93ff', 'Recall': '#34d399',
                                                   'Relevance': '#f59e0b'},
                               title="Metrics over evaluation runs")
                fig2.update_layout(paper_bgcolor='#0f1117', plot_bgcolor='#161b27',
                                   font_color='#c9d1e0', legend_bgcolor='#161b27')
                fig2.update_xaxes(gridcolor='#242938')
                fig2.update_yaxes(gridcolor='#242938', range=[0, 1])
                st.plotly_chart(fig2, use_container_width=True)


if __name__ == "__main__":
    main()