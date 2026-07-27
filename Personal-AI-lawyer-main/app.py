"""
Personal AI Lawyer — free, local RAG legal research assistant.
Primary Streamlit entry. No paid LLM APIs.
"""

from __future__ import annotations

import streamlit as st

from rag_pipeline import (
    OLLAMA_MODEL,
    answer_query,
    check_ollama_available,
    retrieve_docs,
)
from vector_database import (
    ensure_seed_index_from_udhr,
    index_status,
    index_uploaded_pdf,
    load_default_vector_store,
    load_faiss,
    USER_FAISS,
)

st.set_page_config(
    page_title="Personal AI Lawyer",
    page_icon="⚖️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# --- Comforting, mobile-first visual system (not purple / not terracotta cream / not dark glow) ---
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,650&family=Source+Sans+3:wght@400;500;600&display=swap');

:root {
  --bg: #eef3f1;
  --bg-deep: #dfe9e5;
  --ink: #152826;
  --ink-soft: #3d524e;
  --brand: #1a4d45;
  --accent: #2f6f64;
  --accent-soft: #c5ddd6;
  --card: #ffffff;
  --line: rgba(26, 77, 69, 0.12);
  --warn: #6b4f1d;
  --warn-bg: #f4 indeede8c;
  --radius: 18px;
  --safe-bottom: env(safe-area-inset-bottom, 0px);
  --safe-top: env(safe-area-inset-top, 0px);
}

html, body, [class*="css"] {
  font-family: "Source Sans 3", "Segoe UI", sans-serif;
  color: var(--ink);
}

.stApp {
  background:
    radial-gradient(120% 80% at 10% -10%, #d7ebe4 0%, transparent 55%),
    radial-gradient(90% 60% at 100% 0%, #cfe0ea 0%, transparent 45%),
    linear-gradient(180deg, var(--bg) 0%, var(--bg-deep) 100%);
  padding-bottom: calc(24px + var(--safe-bottom));
}

/* Hide Streamlit chrome clutter on phones */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
div[data-testid="stToolbar"] { display: none; }

.block-container {
  padding-top: calc(1rem + var(--safe-top)) !important;
  padding-bottom: 2rem !important;
  padding-left: 1rem !important;
  padding-right: 1rem !important;
  max-width: 720px !important;
}

.brand-hero {
  text-align: center;
  padding: 1.6rem 0.75rem 1.1rem;
  animation: rise 0.7s ease-out both;
}

.brand-mark {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 56px;
  height: 56px;
  border-radius: 16px;
  background: linear-gradient(145deg, #1a4d45, #2f6f64);
  color: #f4faf8;
  font-size: 1.55rem;
  margin-bottom: 0.85rem;
  box-shadow: 0 10px 28px rgba(26, 77, 69, 0.22);
}

.brand-name {
  font-family: "Fraunces", Georgia, serif;
  font-weight: 650;
  font-size: clamp(1.85rem, 6.5vw, 2.55rem);
  line-height: 1.12;
  letter-spacing: -0.02em;
  color: var(--brand);
  margin: 0 0 0.45rem;
}

.brand-tag {
  margin: 0 auto;
  max-width: 28rem;
  font-size: 1.02rem;
  line-height: 1.45;
  color: var(--ink-soft);
}

.status-row {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  justify-content: center;
  margin: 1rem 0 0.25rem;
}

.pill {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  padding: 0.35rem 0.75rem;
  border-radius: 999px;
  font-size: 0.82rem;
  font-weight: 500;
  background: var(--card);
  border: 1px solid var(--line);
  color: var(--ink-soft);
}

.pill.ok { color: var(--accent); border-color: rgba(47, 111, 100, 0.35); background: #f2f8f6; }
.pill.warn { color: var(--warn); border-color: rgba(107, 79, 29, 0.25); background: #faf6ee; }

.panel {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: var(--radius);
  padding: 1rem 1.05rem;
  margin: 0.75rem 0;
  box-shadow: 0 8px 24px rgba(21, 40, 38, 0.04);
  animation: rise 0.55s ease-out both;
}

.panel h3 {
  font-family: "Fraunces", Georgia, serif;
  font-size: 1.15rem;
  font-weight: 650;
  color: var(--brand);
  margin: 0 0 0.35rem;
}

.panel p {
  margin: 0;
  color: var(--ink-soft);
  font-size: 0.95rem;
  line-height: 1.45;
}

.disclaimer {
  background: #faf6ee;
  border: 1px solid rgba(107, 79, 29, 0.18);
  color: var(--warn);
  border-radius: 14px;
  padding: 0.85rem 1rem;
  font-size: 0.88rem;
  line-height: 1.4;
  margin: 0.5rem 0 1rem;
}

.empty-state {
  text-align: center;
  padding: 1.5rem 1rem;
  color: var(--ink-soft);
}

.empty-state strong {
  display: block;
  font-family: "Fraunces", Georgia, serif;
  font-size: 1.2rem;
  color: var(--brand);
  margin-bottom: 0.35rem;
}

/* Chat bubbles */
div[data-testid="stChatMessage"] {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 16px;
  padding: 0.65rem 0.85rem;
  margin-bottom: 0.65rem;
}

/* Large tap targets */
.stButton > button {
  width: 100%;
  min-height: 3rem;
  border-radius: 14px !important;
  background: linear-gradient(145deg, #1a4d45, #2f6f64) !important;
  color: #f4faf8 !important;
  font-weight: 600 !important;
  font-size: 1.05rem !important;
  border: none !important;
  box-shadow: 0 8px 20px rgba(26, 77, 69, 0.2);
}

.stButton > button:hover {
  filter: brightness(1.05);
}

textarea, .stTextArea textarea {
  font-size: 16px !important; /* prevents iOS zoom */
  border-radius: 14px !important;
  min-height: 110px !important;
  border-color: var(--line) !important;
  background: #fff !important;
}

div[data-testid="stFileUploader"] section {
  border-radius: 14px !important;
  border: 1.5px dashed rgba(47, 111, 100, 0.35) !important;
  background: #f7fbfa !important;
  padding: 0.75rem !important;
}

label, .stMarkdown label {
  font-weight: 600 !important;
  color: var(--ink) !important;
}

@keyframes rise {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}

@media (max-width: 480px) {
  .brand-name { font-size: 1.9rem; }
  .block-container { padding-left: 0.75rem !important; padding-right: 0.75rem !important; }
}
</style>
"""

# Fix typo in CSS var
CUSTOM_CSS = CUSTOM_CSS.replace("--warn-bg: #f4 indeede8c;", "--warn-bg: #faf6ee;")

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def init_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "active_db" not in st.session_state:
        st.session_state.active_db = None
    if "db_source" not in st.session_state:
        st.session_state.db_source = "default"


def get_active_db():
    if st.session_state.active_db is not None and st.session_state.db_source == "upload":
        return st.session_state.active_db
    db = load_default_vector_store()
    if db is None:
        db = ensure_seed_index_from_udhr()
    return db


def render_hero():
    st.markdown(
        """
        <div class="brand-hero">
          <div class="brand-mark">⚖️</div>
          <h1 class="brand-name">Personal AI Lawyer</h1>
          <p class="brand-tag">
            Calm, private legal research on your device — grounded in your documents
            and free public law indexes. No paid cloud AI required.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status():
    ollama_ok, ollama_msg = check_ollama_available()
    status = index_status()
    pills = []
    if ollama_ok:
        pills.append(f'<span class="pill ok">Ollama · {OLLAMA_MODEL}</span>')
    else:
        pills.append('<span class="pill warn">Ollama offline · extractive mode</span>')

    if status["agent2"]:
        pills.append('<span class="pill ok">Legal corpus ready</span>')
    elif status["ready"]:
        pills.append('<span class="pill ok">Knowledge base ready</span>')
    else:
        pills.append('<span class="pill warn">Building seed index…</span>')

    st.markdown(f'<div class="status-row">{"".join(pills)}</div>', unsafe_allow_html=True)

    if not ollama_ok:
        st.caption(ollama_msg)


def main():
    init_state()
    render_hero()
    render_status()

    st.markdown(
        '<div class="disclaimer">'
        "<strong>Important:</strong> This app is for educational research only. "
        "It is <em>not</em> a lawyer and does not provide legal advice. "
        "For real matters, speak with a licensed attorney in your jurisdiction."
        "</div>",
        unsafe_allow_html=True,
    )

    with st.expander("Add your own law PDF (optional)", expanded=False):
        st.markdown(
            '<div class="panel"><h3>Your documents</h3>'
            "<p>Upload a PDF of statutes, contracts, or policies. "
            "We’ll index it locally with free embeddings and use it for answers.</p></div>",
            unsafe_allow_html=True,
        )
        uploaded = st.file_uploader(
            "Upload PDF",
            type=["pdf"],
            accept_multiple_files=False,
            label_visibility="collapsed",
            key="pdf_upload",
        )
        if uploaded is not None:
            if st.button("Index this PDF", use_container_width=True, key="index_btn"):
                with st.spinner("Reading and indexing your PDF locally…"):
                    try:
                        db = index_uploaded_pdf(uploaded)
                        st.session_state.active_db = db
                        st.session_state.db_source = "upload"
                        st.success(
                            f"Indexed **{uploaded.name}**. Questions will use this document "
                            f"(saved under `vectorstore/db_faiss_user`)."
                        )
                    except Exception as exc:
                        st.error(f"Could not index PDF: {exc}")

        if st.session_state.db_source == "upload":
            if st.button("Switch back to shared knowledge base", use_container_width=True):
                st.session_state.active_db = None
                st.session_state.db_source = "default"
                st.rerun()

    # Chat history
    if not st.session_state.messages:
        st.markdown(
            """
            <div class="panel empty-state">
              <strong>Ask anything gently</strong>
              Try: “Which articles protect freedom of assembly?” or
              “What does this document say about notice periods?”
            </div>
            """,
            unsafe_allow_html=True,
        )

    for msg in st.session_state.messages:
        role = "user" if msg["role"] == "user" else "assistant"
        with st.chat_message(role, avatar="🧑" if role == "user" else "⚖️"):
            st.markdown(msg["content"])
            if msg.get("mode"):
                label = "Local Ollama" if msg["mode"] == "ollama" else "Extractive (offline)"
                st.caption(f"Answer mode: {label}")

    # Suggested prompts (mobile-friendly)
    cols = st.columns(2)
    suggestions = [
        "Which rights protect peaceful assembly?",
        "Summarize key equality protections in the knowledge base.",
    ]
    for col, suggestion in zip(cols, suggestions):
        with col:
            if st.button(suggestion, use_container_width=True, key=f"sug_{suggestion[:12]}"):
                st.session_state._pending_q = suggestion

    user_query = st.chat_input("Ask a calm legal research question…")
    pending = st.session_state.pop("_pending_q", None)
    question = user_query or pending

    if question:
        question = question.strip()
        if not question:
            st.stop()

        st.session_state.messages.append({"role": "user", "content": question})

        with st.spinner("Searching your local knowledge base…"):
            db = get_active_db()
            if db is None and st.session_state.db_source == "upload":
                db = load_faiss(USER_FAISS)

            docs = retrieve_docs(question, faiss_db=db, k=4)
            answer, mode = answer_query(docs, question)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "mode": mode}
        )
        st.rerun()

    st.markdown(
        "<p style='text-align:center;color:#3d524e;font-size:0.8rem;margin-top:1.5rem;'>"
        "Runs fully free · MiniLM embeddings · FAISS · Ollama optional"
        "</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
