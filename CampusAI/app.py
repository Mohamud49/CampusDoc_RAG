"""
CampusAI — RAG University Academic Rules & Syllabus Assistant
=============================================================
Streamlit entry point. Sections:
  1. Configuration & page setup
  2. Login (admin vs user)
  3. PDF loading & chunking (admin sidebar)
  4. Vector indexing
  5. LLM & RAG pipeline
  6. Modern chat UI

Run: streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on Python path (beginner-friendly when IDE runs from subfolders)
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

import importlib

import auth
import config

# Streamlit caches imported modules; reload so login settings stay current.
importlib.reload(config)
importlib.reload(auth)
from ingestion.chunking import split_documents
from ingestion.pdf_loader import load_pdf_files, save_uploaded_pdfs
from rag.chain import ask_question, build_conversational_rag_chain, format_source_label
from rag.llm_factory import get_chat_llm
from vectorstore.chroma_store import (
    clear_vector_store,
    collection_has_documents,
    index_documents,
)

# ---------------------------------------------------------------------------
# 1. CONFIGURATION & PAGE SETUP
# ---------------------------------------------------------------------------

config.ensure_directories()
auth.ensure_default_users()

st.set_page_config(
    page_title="CampusAI | University RAG Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

EXAMPLE_QUESTIONS = [
    "What are the admission requirements of the university?",
    "What is the exam and evaluation policy?",
    "How many credits are required to graduate?",
    "What is the attendance rule?",
]


def inject_custom_css(theme: str, *, logged_in: bool) -> None:
    """
    Premium dark/light dashboard styling.
    Somali: Qurxinta dashboard-ka (mugdi / iftiin).
    """
    is_dark = theme == "Dark"
    bg = "#0f1419" if is_dark else "#f4f6fb"
    card = "#1a2332" if is_dark else "#ffffff"
    text = "#e7ecf3" if is_dark else "#1a1a2e"
    muted = "#8b9cb3" if is_dark else "#5c6b7a"
    accent = "#6366f1"
    border = "#2d3a4f" if is_dark else "#e2e8f0"
    hide_sidebar = (
        """
        [data-testid="stSidebar"] { display: none; }
        [data-testid="stSidebarCollapsedControl"] { display: none; }
        """
        if not logged_in
        else ""
    )

    st.markdown(
        f"""
        <style>
        .stApp {{ background: {bg}; color: {text}; }}
        [data-testid="stSidebar"] {{
            background: {card};
            border-right: 1px solid {border};
        }}
        {hide_sidebar}
        .campus-header {{
            font-size: 2rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            margin-bottom: 0.25rem;
        }}
        .campus-sub {{
            color: {muted};
            font-size: 0.95rem;
            margin-bottom: 1.5rem;
        }}
        .status-pill {{
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 600;
            background: {'#1e3a2f' if is_dark else '#d1fae5'};
            color: {'#6ee7b7' if is_dark else '#065f46'};
            border: 1px solid {'#34d399' if is_dark else '#6ee7b7'};
        }}
        .status-pill-warn {{
            background: {'#3d2e1a' if is_dark else '#fef3c7'};
            color: {'#fcd34d' if is_dark else '#92400e'};
            border-color: {'#f59e0b' if is_dark else '#fcd34d'};
        }}
        div[data-testid="stChatMessage"] {{
            background: {card};
            border: 1px solid {border};
            border-radius: 12px;
            padding: 0.5rem 1rem;
        }}
        .login-wrap {{
            max-width: 440px;
            margin: 6rem auto 2rem auto;
            padding: 2rem 1.75rem 1.5rem 1.75rem;
            background: {card};
            border: 1px solid {border};
            border-radius: 16px;
        }}
        .login-hint {{
            color: {muted};
            font-size: 0.85rem;
            margin-top: 1rem;
        }}
        .stAppDeployButton, [data-testid="stAppDeployButton"] {{
            display: none !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_session_state() -> None:
    """Initialize Streamlit session keys once."""
    defaults = {
        "messages": [],  # {"role": "user"|"assistant", "content": str, "sources": []}
        "chat_history": [],  # LangChain message objects for RAG
        "rag_chain": None,
        "indexed": False,
        "theme": "Dark",
        "pending_prompt": None,
        "authenticated": False,
        "username": "",
        "role": "",
        "login_error": "",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def is_admin() -> bool:
    return st.session_state.get("authenticated") and st.session_state.get("role") == "admin"


def logout() -> None:
    st.session_state.authenticated = False
    st.session_state.username = ""
    st.session_state.role = ""
    st.session_state.messages = []
    st.session_state.chat_history = []
    st.session_state.rag_chain = None
    st.session_state.pending_prompt = None
    st.session_state.login_error = ""
    st.rerun()


def render_login_page() -> None:
    """Centered login. Admin can later upload PDFs; users chat only."""
    left, center, right = st.columns([1, 1.2, 1])
    with center:
        st.markdown('<p class="campus-header">🎓 CampusAI</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="campus-sub">Sign in to continue.</p>',
            unsafe_allow_html=True,
        )
        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("Username", placeholder="admin or student")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log in", type="primary", use_container_width=True)

        if submitted:
            account = auth.authenticate(username, password)
            if account:
                st.session_state.authenticated = True
                st.session_state.username = account["username"]
                st.session_state.role = account["role"]
                st.session_state.login_error = ""
                st.rerun()
            st.session_state.login_error = "Invalid username or password."

        if st.session_state.login_error:
            st.error(st.session_state.login_error)

        st.caption(
            f"Demo accounts — **{auth.ADMIN_USERNAME}** / `{auth.ADMIN_PASSWORD}` (upload PDFs) · "
            f"**{auth.USER_USERNAME}** / `{auth.USER_PASSWORD}` (chat only)"
        )


init_session_state()
inject_custom_css(st.session_state.theme, logged_in=st.session_state.authenticated)

if not st.session_state.authenticated:
    render_login_page()
    st.stop()

# Sidebar theme + account (after login)
with st.sidebar:
    st.session_state.theme = st.selectbox(
        "Theme / Muuqaal",
        ["Dark", "Light"],
        index=0 if st.session_state.theme == "Dark" else 1,
    )
    st.caption(
        f"Signed in as **{st.session_state.username}** "
        f"({st.session_state.role})"
    )
    if st.button("Log out", use_container_width=True):
        logout()

# Re-apply CSS after theme select (sidebar exists only when logged in)
inject_custom_css(st.session_state.theme, logged_in=True)

# ---------------------------------------------------------------------------
# 3. PDF LOADING & TEXT CHUNKING (ADMIN SIDEBAR ONLY)
# ---------------------------------------------------------------------------

with st.sidebar:
    if is_admin():
        st.markdown("###  Document ingestion")
        st.caption(
            "Upload PDF handbooks or syllabi. "
            "Soo rar PDF-yada jaamacadda si CampusAI u akhriyo."
        )

        uploaded = st.file_uploader(
            "PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            help="Multiple PDFs supported / PDF badan waa la ogol yahay",
        )

        reset_index = st.checkbox(
            "Replace existing index",
            value=False,
            help="Clear Chroma and rebuild from uploaded files only",
        )

        ingest_btn = st.button(
            "Index documents / Index PDF", type="primary", use_container_width=True
        )

        if ingest_btn:
            if not uploaded:
                st.warning("Please upload at least one PDF.")
            else:
                with st.status("Processing PDFs…", expanded=True) as status:
                    st.write("Saving uploads…")
                    paths = save_uploaded_pdfs(uploaded, config.UPLOAD_DIR)
                    st.write(f"Loading {len(paths)} file(s)…")
                    raw_docs = load_pdf_files(paths)
                    st.write(f"Splitting into chunks (size={config.CHUNK_SIZE})…")
                    chunks = split_documents(raw_docs)
                    for c in chunks:
                        c.metadata.setdefault("filename", c.metadata.get("source", "unknown"))
                        c.metadata.setdefault("page_number", c.metadata.get("page", 0))
                    st.write("Embedding & storing in ChromaDB…")
                    if reset_index:
                        clear_vector_store()
                    count = index_documents(chunks, reset=False)
                    st.session_state.indexed = count > 0
                    st.session_state.rag_chain = None
                    status.update(label=f"Indexed {count} chunks", state="complete")
    else:
        st.markdown("###  Documents")
        st.caption("Only administrators can upload and index PDFs. You can ask questions about the existing knowledge base.")

    st.divider()
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

# ---------------------------------------------------------------------------
# 4. VECTOR DB STATUS & 5. RAG PIPELINE SETUP
# ---------------------------------------------------------------------------

has_vectors = collection_has_documents() or st.session_state.indexed

col_title, col_status = st.columns([3, 1])
with col_title:
    st.markdown('<p class="campus-header">🎓 CampusAI</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="campus-sub">RAG-powered university rules & syllabus assistant — MCA Minor Project</p>',
        unsafe_allow_html=True,
    )
with col_status:
    if has_vectors:
        st.markdown('<span class="status-pill">● Knowledge base ready</span>', unsafe_allow_html=True)
    elif is_admin():
        st.markdown(
            '<span class="status-pill status-pill-warn">● Upload & index PDFs</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="status-pill status-pill-warn">● Waiting for admin documents</span>',
            unsafe_allow_html=True,
        )

if has_vectors and st.session_state.rag_chain is None:
    try:
        # Lazy-init: only load LLM + chain when user can chat
        _ = get_chat_llm()
        st.session_state.rag_chain = build_conversational_rag_chain()
    except ValueError as exc:
        st.error(str(exc))
        st.info("Copy `.env.example` to `.env` and add GROQ_API_KEY or use Ollama.")

# ---------------------------------------------------------------------------
# 6. STREAMLIT MODERN CHAT UI
# ---------------------------------------------------------------------------

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            st.markdown("**Retrieved sources / Ilaha laga helay:**")
            for i, doc in enumerate(msg["sources"], start=1):
                label = format_source_label(doc)
                with st.expander(f"Source {i}: {label}"):
                    st.markdown(doc.page_content)
                    st.caption(f"Metadata: `{doc.metadata}`")

if has_vectors and not st.session_state.messages:
    st.markdown("**Try a question**")
    q_cols = st.columns(2)
    for i, example in enumerate(EXAMPLE_QUESTIONS):
        if q_cols[i % 2].button(example, use_container_width=True, key=f"example_q_{i}"):
            st.session_state.pending_prompt = example
            st.rerun()
elif not has_vectors and not is_admin():
    st.info("An administrator must upload and index PDFs before you can use the chatbot.")

prompt = st.chat_input("Ask about rules, credits, exams, syllabus…")
if st.session_state.pending_prompt:
    prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None

if prompt:
    if not has_vectors:
        if is_admin():
            st.warning("Index at least one PDF from the sidebar before chatting.")
        else:
            st.warning("An administrator must index documents before you can chat.")
    elif st.session_state.rag_chain is None:
        st.warning("Configure your LLM in `.env` to enable chat.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents & generating answer…"):
                try:
                    answer, sources = ask_question(
                        st.session_state.rag_chain,
                        prompt,
                        st.session_state.chat_history,
                    )
                    st.session_state.chat_history.append(HumanMessage(content=prompt))
                    st.session_state.chat_history.append(AIMessage(content=answer))
                except Exception as exc:
                    answer = f"Error: {exc}"
                    sources = []

            st.markdown(answer)
            if sources:
                st.markdown("**Retrieved sources / Ilaha laga helay:**")
                for i, doc in enumerate(sources, start=1):
                    label = format_source_label(doc)
                    with st.expander(f"Source {i}: {label}"):
                        st.markdown(doc.page_content)
                        st.caption(f"Metadata: `{doc.metadata}`")

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )
