import os
import sys
import json

# Resolve project root and load .env BEFORE anything else
_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(_ROOT, ".env"), override=True)

import streamlit as st
from groq import Groq

from utils.pdf_reader import extract_pdf_text
from agents import SummarizerAgent, CriticAgent, RecommenderAgent
from band import BandRoom

st.set_page_config(
    page_title="PaperBand AI",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] { background: #0f1117; }
    [data-testid="stHeader"] { background: transparent; }
    .block-container { max-width: 960px; padding-top: 2.5rem; padding-bottom: 3rem; }

    .pb-hero { text-align: center; padding: 2.5rem 0 1.5rem; }
    .pb-hero h1 { font-size: 2.8rem; font-weight: 700; color: #e8eaf0; margin-bottom: 0.3rem; }
    .pb-hero p { color: #8b90a0; font-size: 1.05rem; margin: 0; }

    .agent-card {
        border: 1px solid #2a2d3a;
        border-radius: 10px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1.25rem;
        background: #161820;
    }
    .agent-card h3 {
        font-size: 1rem;
        font-weight: 600;
        color: #a0a8c0;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.75rem;
    }

    .score-badge { font-size: 3rem; font-weight: 800; color: #7c8cff; line-height: 1; }
    .decision-badge {
        display: inline-block;
        margin-left: 1rem;
        padding: 0.35rem 0.9rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        vertical-align: middle;
    }
    .badge-accept  { background: #1a3a2a; color: #4caf86; border: 1px solid #4caf86; }
    .badge-minor   { background: #1e2e1a; color: #90c060; border: 1px solid #90c060; }
    .badge-major   { background: #2e2a10; color: #d4a020; border: 1px solid #d4a020; }
    .badge-reject  { background: #2e1a1a; color: #d06060; border: 1px solid #d06060; }

    .pb-list { padding-left: 1.2rem; }
    .pb-list li { color: #c8ccd8; margin-bottom: 0.4rem; line-height: 1.55; }
    .pb-list li.strength   { color: #80c8a0; }
    .pb-list li.weakness   { color: #d08080; }
    .pb-list li.missing    { color: #d4a840; }
    .pb-list li.limitation { color: #a0a8d0; }
    .pb-list li.future     { color: #80b8d8; }

    .band-box {
        background: #10121a;
        border: 1px solid #2a2d3a;
        border-radius: 10px;
        padding: 1.25rem 1.5rem;
        font-family: monospace;
        font-size: 0.8rem;
        color: #8088a8;
        overflow-x: auto;
        white-space: pre-wrap;
    }

    hr.pb-divider { border: none; border-top: 1px solid #2a2d3a; margin: 2rem 0; }

    [data-testid="stFileUploader"] {
        border: 2px dashed #2a2d3a !important;
        border-radius: 10px !important;
        padding: 1rem !important;
        background: #13151e !important;
    }

    .stButton > button {
        background: #7c8cff;
        color: #fff;
        border: none;
        padding: 0.6rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1rem;
        width: 100%;
    }
    .stButton > button:hover { background: #5a6be0; color: #fff; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DECISION_BADGE = {
    "accept with minor":  "badge-minor",
    "major revisions":    "badge-major",
    "reject":             "badge-reject",
    "accept":             "badge-accept",
}


def _badge_class(decision: str) -> str:
    d = decision.lower()
    for key, cls in DECISION_BADGE.items():
        if key in d:
            return cls
    return "badge-minor"


def _safe_list(data: dict, key: str) -> list:
    val = data.get(key, [])
    return val if isinstance(val, list) else [str(val)]


def render_summary(summary: dict):
    st.markdown('<div class="agent-card">', unsafe_allow_html=True)
    st.markdown('<h3>📝 Agent 1 &nbsp;·&nbsp; Summarizer</h3>', unsafe_allow_html=True)

    title = summary.get("title", "Unknown Title")
    authors = summary.get("authors", [])
    author_str = ", ".join(authors) if isinstance(authors, list) else str(authors)

    st.markdown(f"**{title}**")
    if author_str and author_str != "Unknown":
        st.markdown(f"*{author_str}*")
    st.markdown("---")

    for label, key in [
        ("Research Problem", "research_problem"),
        ("Methodology",      "methodology"),
        ("Key Results",      "key_results"),
        ("Conclusion",       "conclusion"),
    ]:
        val = summary.get(key, "—")
        st.markdown(f"**{label}**")
        st.markdown(val)
        st.markdown("")

    st.markdown('</div>', unsafe_allow_html=True)


def render_critique(critique: dict):
    st.markdown('<div class="agent-card">', unsafe_allow_html=True)
    st.markdown('<h3>🔍 Agent 2 &nbsp;·&nbsp; Critic</h3>', unsafe_allow_html=True)

    sections = [
        ("Strengths",           "strengths",            "strength"),
        ("Weaknesses",          "weaknesses",           "weakness"),
        ("Missing Experiments", "missing_experiments",  "missing"),
        ("Limitations",         "limitations",          "limitation"),
    ]

    for label, key, css_cls in sections:
        items = _safe_list(critique, key)
        st.markdown(f"**{label}**")
        items_html = "".join(f'<li class="{css_cls}">{item}</li>' for item in items)
        st.markdown(f'<ul class="pb-list">{items_html}</ul>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


def render_recommendation(rec: dict):
    st.markdown('<div class="agent-card">', unsafe_allow_html=True)
    st.markdown('<h3>⭐ Agent 3 &nbsp;·&nbsp; Recommendation</h3>', unsafe_allow_html=True)

    score    = rec.get("score", "—")
    decision = rec.get("decision", "Unknown")
    justify  = rec.get("justification", "")
    future   = _safe_list(rec, "future_work")

    badge_cls = _badge_class(decision)
    st.markdown(
        f'<span class="score-badge">{score}</span>'
        f'<span class="decision-badge {badge_cls}">{decision}</span>',
        unsafe_allow_html=True,
    )
    st.markdown("")
    st.markdown(f"**Justification:** {justify}")
    st.markdown("**Future Work**")
    future_html = "".join(f'<li class="future">{item}</li>' for item in future)
    st.markdown(f'<ul class="pb-list">{future_html}</ul>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


def render_band_log(room: BandRoom):
    st.markdown('<hr class="pb-divider">', unsafe_allow_html=True)
    st.markdown("#### 🤝 Band &nbsp;·&nbsp; Agent Collaboration Log")
    st.markdown(
        '<div class="band-box">' + json.dumps(room.to_dict(), indent=2) + '</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Groq client — reads key directly each call, no caching issues
# ---------------------------------------------------------------------------

def get_groq_client() -> Groq:
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not found. "
            "Make sure your .env file is in the same folder as app.py "
            "and contains: GROQ_API_KEY=gsk_..."
        )
    return Groq(api_key=api_key)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(uploaded_file) -> None:
    try:
        client = get_groq_client()
    except RuntimeError as exc:
        st.error(str(exc))
        return

    room = BandRoom(name="paper-review")

    with st.spinner("Reading PDF..."):
        try:
            paper_text = extract_pdf_text(uploaded_file)
        except ValueError as exc:
            st.error(str(exc))
            return

    st.info(f"Extracted **{len(paper_text):,}** characters from PDF.")

    with st.spinner("Agent 1 · Summarizer is working..."):
        try:
            summary = SummarizerAgent(client).run(paper_text)
        except ValueError as exc:
            st.error(f"Summarizer Agent failed: {exc}")
            return

    room.send(agent="Agent1", role="Summarizer", content=summary)
    render_summary(summary)

    with st.spinner("Agent 2 · Critic is working..."):
        try:
            critique = CriticAgent(client).run(summary)
        except ValueError as exc:
            st.error(f"Critic Agent failed: {exc}")
            return

    room.send(agent="Agent2", role="Critic", content=critique)
    render_critique(critique)

    with st.spinner("Agent 3 · Recommender is working..."):
        try:
            recommendation = RecommenderAgent(client).run(summary, critique)
        except ValueError as exc:
            st.error(f"Recommender Agent failed: {exc}")
            return

    room.send(agent="Agent3", role="Recommender", content=recommendation)
    render_recommendation(recommendation)

    render_band_log(room)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="pb-hero">
        <h1>📄 PaperBand AI</h1>
        <p>Multi-agent research paper review &nbsp;·&nbsp; Summarize &nbsp;·&nbsp; Critique &nbsp;·&nbsp; Recommend</p>
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Upload a research paper (PDF)",
    type=["pdf"],
    label_visibility="collapsed",
)

if uploaded_file:
    st.markdown(f"**Uploaded:** `{uploaded_file.name}`")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Run Agent Review"):
            run_pipeline(uploaded_file)
else:
    st.markdown(
        "<p style='color:#555; text-align:center; margin-top:1rem;'>"
        "Upload a PDF above to get started.</p>",
        unsafe_allow_html=True,
    )