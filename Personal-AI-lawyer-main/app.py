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

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Manrope:wght@400;500;600;700&display=swap');

:root {
  --ink: #14201e;
  --ink-soft: #4a5c58;
  --ink-faint: #6f807c;
  --brand: #0f3d38;
  --accent: #1f6b5f;
  --surface: rgba(255, 255, 255, 0.78);
  --surface-solid: #ffffff;
  --line: rgba(15, 61, 56, 0.10);
  --safe-bottom: env(safe-area-inset-bottom, 0px);
  --safe-top: env(safe-area-inset-top, 0px);
}

html, body, [class*="css"] {
  font-family: "Manrope", "Segoe UI", sans-serif;
  color: var(--ink);
}

.stApp {
  background:
    radial-gradient(ellipse 90% 55% at 50% -5%, #c8e4de 0%, transparent 58%),
    radial-gradient(ellipse 70% 40% at 100% 20%, #d4e3ec 0%, transparent 50%),
    linear-gradient(180deg, #f3f7f6 0%, #e6efec 55%, #dfe9e6 100%);
  min-height: 100vh;
}

#MainMenu, footer, div[data-testid="stToolbar"] { display: none !important; }
header[data-testid="stHeader"] { background: transparent; }

.block-container {
  padding-top: calc(1.15rem + var(--safe-top)) !important;
  padding-bottom: calc(7.5rem + var(--safe-bottom)) !important;
  padding-left: 1.15rem !important;
  padding-right: 1.15rem !important;
  max-width: 640px !important;
}

/* —— Hero —— */
.hero {
  text-align: center;
  padding: 1.5rem 0.5rem 0.2rem;
  animation: rise 0.65s cubic-bezier(.22,1,.36,1) both;
}

.hero-mark {
  width: 52px;
  height: 52px;
  margin: 0 auto 1rem;
  border-radius: 16px;
  display: grid;
  place-items: center;
  font-size: 1.4rem;
  color: #f2faf8;
  background: linear-gradient(160deg, #0f3d38 0%, #1f6b5f 100%);
  box-shadow: 0 14px 32px rgba(15, 61, 56, 0.2);
  animation: floaty 5.5s ease-in-out infinite;
}

.hero h1 {
  font-family: "Instrument Serif", Georgia, serif;
  font-weight: 400;
  font-size: clamp(2.4rem, 8vw, 3.2rem);
  line-height: 1.04;
  letter-spacing: -0.03em;
  color: var(--brand);
  margin: 0 0 0.6rem;
}

.hero p {
  margin: 0 auto;
  max-width: 22rem;
  font-size: 1.02rem;
  line-height: 1.5;
  color: var(--ink-soft);
  font-weight: 500;
}

.meta-line {
  display: flex;
  align-items: center;
  justify-content: center;
  flex-wrap: wrap;
  gap: 0.4rem 0.55rem;
  margin: 1rem 0 0.15rem;
  font-size: 0.8rem;
  font-weight: 600;
  color: var(--ink-faint);
  animation: rise 0.7s 0.08s cubic-bezier(.22,1,.36,1) both;
}

.meta-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: #2a9b7a;
  box-shadow: 0 0 0 3px rgba(42, 155, 122, 0.18);
  flex-shrink: 0;
  animation: pulse-dot 2.2s ease-in-out infinite;
}

.meta-dot.warn {
  background: #c49a1a;
  box-shadow: 0 0 0 3px rgba(196, 154, 26, 0.16);
}

.meta-sep { opacity: 0.35; }

/* —— Chat —— */
div[data-testid="stChatMessage"] {
  background: var(--surface-solid);
  border: 1px solid var(--line);
  border-radius: 18px;
  padding: 0.85rem 1rem;
  margin-bottom: 0.75rem;
  box-shadow: 0 6px 18px rgba(20, 32, 30, 0.04);
  animation: rise 0.45s cubic-bezier(.22,1,.36,1) both;
}

div[data-testid="stChatMessage"] p {
  font-size: 0.98rem;
  line-height: 1.55;
}

/* —— Suggestions —— */
.suggest-label {
  margin: 1.6rem 0 0.7rem;
  text-align: center;
  font-size: 0.76rem;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--ink-faint);
  animation: rise 0.55s 0.12s ease-out both;
}

button[data-testid="baseButton-secondary"] {
  width: 100% !important;
  min-height: 3rem !important;
  margin-bottom: 0.55rem !important;
  border-radius: 999px !important;
  background: var(--surface-solid) !important;
  color: var(--brand) !important;
  border: 1px solid var(--line) !important;
  box-shadow: 0 4px 14px rgba(20, 32, 30, 0.045) !important;
  font-weight: 600 !important;
  font-size: 0.95rem !important;
  line-height: 1.3 !important;
  padding: 0.75rem 1.2rem !important;
  white-space: normal !important;
  transition: transform 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease !important;
  animation: rise 0.5s ease-out both;
}

button[data-testid="baseButton-secondary"]:hover {
  border-color: rgba(31, 107, 95, 0.4) !important;
  box-shadow: 0 10px 24px rgba(15, 61, 56, 0.1) !important;
  transform: translateY(-2px);
  background: #f7fbfa !important;
}

button[data-testid="baseButton-secondary"]:nth-of-type(1) { animation-delay: 0.1s; }
button[data-testid="baseButton-secondary"]:nth-of-type(2) { animation-delay: 0.16s; }
button[data-testid="baseButton-secondary"]:nth-of-type(3) { animation-delay: 0.22s; }

button[data-testid="baseButton-primary"] {
  border-radius: 14px !important;
  min-height: 2.85rem !important;
  background: linear-gradient(160deg, #0f3d38, #1f6b5f) !important;
  color: #f4faf8 !important;
  border: none !important;
  font-weight: 700 !important;
  box-shadow: 0 10px 22px rgba(15, 61, 56, 0.18) !important;
}

/* —— Loading —— */
.loader-card {
  margin: 1.25rem 0;
  padding: 1.35rem 1.2rem;
  border-radius: 20px;
  background: var(--surface-solid);
  border: 1px solid var(--line);
  box-shadow: 0 10px 28px rgba(20, 32, 30, 0.06);
  text-align: center;
  animation: rise 0.35s ease-out both;
}

.loader-orb {
  width: 42px;
  height: 42px;
  margin: 0 auto 0.9rem;
  border-radius: 50%;
  border: 2.5px solid rgba(31, 107, 95, 0.15);
  border-top-color: var(--accent);
  animation: spin 0.85s linear infinite;
}

.loader-card strong {
  display: block;
  font-family: "Instrument Serif", Georgia, serif;
  font-size: 1.2rem;
  color: var(--brand);
  margin-bottom: 0.25rem;
}

.loader-card span {
  color: var(--ink-soft);
  font-size: 0.9rem;
}

.loader-bars {
  display: flex;
  justify-content: center;
  gap: 5px;
  margin-top: 0.95rem;
}

.loader-bars i {
  display: block;
  width: 5px;
  height: 16px;
  border-radius: 99px;
  background: var(--accent);
  opacity: 0.35;
  animation: bar 1s ease-in-out infinite;
}

.loader-bars i:nth-child(2) { animation-delay: 0.12s; }
.loader-bars i:nth-child(3) { animation-delay: 0.24s; }
.loader-bars i:nth-child(4) { animation-delay: 0.36s; }

/* —— Chat dock —— */
[data-testid="stChatInput"] { background: transparent !important; }
[data-testid="stChatInput"] > div,
[data-testid="stBottom"] > div { background: transparent !important; }

section[data-testid="stChatInput"] textarea,
[data-testid="stChatInput"] textarea {
  background: #ffffff !important;
  border: 1px solid rgba(15, 61, 56, 0.14) !important;
  border-radius: 22px !important;
  box-shadow: 0 10px 30px rgba(20, 32, 30, 0.08) !important;
  font-size: 16px !important;
  color: var(--ink) !important;
  min-height: 52px !important;
  max-height: 140px !important;
}

div[data-testid="stBottomBlockContainer"],
div[data-testid="stBottom"] {
  background:
    linear-gradient(180deg, transparent 0%, rgba(227, 238, 234, 0.92) 28%, #dfe9e6 100%) !important;
  border-top: none !important;
  padding-bottom: calc(0.65rem + var(--safe-bottom)) !important;
}

div[data-testid="stExpander"] {
  background: transparent !important;
  border: none !important;
  margin-top: 0.35rem !important;
}

div[data-testid="stExpander"] details {
  border: 1px solid var(--line) !important;
  border-radius: 16px !important;
  background: var(--surface) !important;
  backdrop-filter: blur(8px);
}

div[data-testid="stExpander"] summary {
  font-weight: 600 !important;
  color: var(--ink-soft) !important;
  font-size: 0.9rem !important;
}

div[data-testid="stFileUploader"] section {
  border-radius: 14px !important;
  border: 1.5px dashed rgba(31, 107, 95, 0.28) !important;
  background: #f7fbfa !important;
}

div[data-testid="stSpinner"] {
  text-align: center;
}

.foot {
  text-align: center;
  margin-top: 1.75rem;
  padding: 0 0.75rem 0.5rem;
  font-size: 0.78rem;
  line-height: 1.55;
  color: var(--ink-faint);
  animation: rise 0.6s 0.2s ease-out both;
}

.foot strong { color: var(--ink-soft); font-weight: 650; }

@keyframes rise {
  from { opacity: 0; transform: translateY(12px); }
  to { opacity: 1; transform: translateY(0); }
}

@keyframes floaty {
  0%, 100% { transform: translateY(0); }
  50% { transform: translateY(-4px); }
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

@keyframes pulse-dot {
  0%, 100% { transform: scale(1); opacity: 1; }
  50% { transform: scale(1.15); opacity: 0.75; }
}

@keyframes bar {
  0%, 100% { transform: scaleY(0.55); opacity: 0.35; }
  50% { transform: scaleY(1.15); opacity: 1; }
}

@media (max-width: 480px) {
  .hero { padding-top: 1rem; }
  .hero h1 { font-size: 2.25rem; }
  .block-container {
    padding-left: 0.9rem !important;
    padding-right: 0.9rem !important;
  }
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Emoji avatars only — plain punctuation is treated as a file path by Streamlit
USER_AVATAR = "👤"
ASSISTANT_AVATAR = "⚖️"

SUGGESTIONS = [
    ("assembly", "Peaceful assembly rights"),
    ("equality", "Equality protections"),
    ("speech", "Freedom of expression"),
]

PROMPTS = {
    "assembly": "Which rights protect peaceful assembly?",
    "equality": "Summarize key equality protections in the knowledge base.",
    "speech": "What does the knowledge base say about freedom of expression?",
}


def init_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "active_db" not in st.session_state:
        st.session_state.active_db = None
    if "db_source" not in st.session_state:
        st.session_state.db_source = "default"
    if "is_thinking" not in st.session_state:
        st.session_state.is_thinking = False

    # Drop empty / broken bubbles from earlier avatar crashes
    st.session_state.messages = [
        m
        for m in st.session_state.messages
        if isinstance(m, dict) and str(m.get("content") or "").strip()
    ]


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
        <div class="hero">
          <div class="hero-mark">⚖️</div>
          <h1>Personal AI Lawyer</h1>
          <p>Private legal research on your device — calm answers grounded in your documents.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_meta():
    ollama_ok, _ = check_ollama_available()
    status = index_status()
    corpus_ok = bool(status.get("ready") or status.get("agent2"))

    mode = f"Ollama · {OLLAMA_MODEL}" if ollama_ok else "Extractive mode"
    corpus = "Corpus ready" if corpus_ok else "Building index…"
    mode_dot = "meta-dot" if ollama_ok else "meta-dot warn"
    corpus_dot = "meta-dot" if corpus_ok else "meta-dot warn"

    st.markdown(
        f"""
        <div class="meta-line">
          <span class="{mode_dot}"></span><span>{mode}</span>
          <span class="meta-sep">·</span>
          <span class="{corpus_dot}"></span><span>{corpus}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_loader(label: str = "Searching your local knowledge base"):
    st.markdown(
        f"""
        <div class="loader-card">
          <div class="loader-orb"></div>
          <strong>{label}</strong>
          <span>Reading passages from your free legal index…</span>
          <div class="loader-bars"><i></i><i></i><i></i><i></i></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_pdf_tools():
    with st.expander("Add a law PDF", expanded=False):
        st.caption("Indexed only on this device. Used for your next questions.")
        uploaded = st.file_uploader(
            "Upload PDF",
            type=["pdf"],
            accept_multiple_files=False,
            label_visibility="collapsed",
            key="pdf_upload",
        )
        if uploaded is not None:
            if st.button("Index PDF", type="primary", use_container_width=True, key="index_btn"):
                with st.spinner("Indexing locally…"):
                    try:
                        db = index_uploaded_pdf(uploaded)
                        st.session_state.active_db = db
                        st.session_state.db_source = "upload"
                        st.success(f"Ready — using **{uploaded.name}**")
                    except Exception as exc:
                        st.error(f"Could not index PDF: {exc}")

        if st.session_state.db_source == "upload":
            if st.button(
                "Use shared knowledge base instead",
                type="secondary",
                use_container_width=True,
                key="reset_db",
            ):
                st.session_state.active_db = None
                st.session_state.db_source = "default"
                st.rerun()


def render_suggestions():
    st.markdown('<p class="suggest-label">Try a question</p>', unsafe_allow_html=True)
    for key, label in SUGGESTIONS:
        if st.button(label, type="secondary", use_container_width=True, key=f"sug_{key}"):
            st.session_state._pending_q = PROMPTS[key]


def ask(question: str):
    question = question.strip()
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.is_thinking = True
    st.rerun()


def finish_thinking():
    """Run retrieval after the thinking UI has been painted."""
    question = ""
    for msg in reversed(st.session_state.messages):
        if msg.get("role") == "user":
            question = str(msg.get("content") or "")
            break
    if not question:
        st.session_state.is_thinking = False
        return

    try:
        db = get_active_db()
        if db is None and st.session_state.db_source == "upload":
            db = load_faiss(USER_FAISS)
        docs = retrieve_docs(question, faiss_db=db, k=4)
        answer, mode = answer_query(docs, question)
        if not str(answer).strip():
            answer = (
                "I couldn’t find enough grounded material in the local index. "
                "Try another question or upload a PDF."
            )
        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "mode": mode}
        )
    except Exception as exc:  # noqa: BLE001
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": (
                    "Something went wrong while searching the local knowledge base.\n\n"
                    f"`{exc}`\n\nTry again, or rebuild the index with "
                    "`python build_index.py --force --skip-remote`."
                ),
                "mode": "error",
            }
        )
    finally:
        st.session_state.is_thinking = False
        st.rerun()


def main():
    init_state()
    render_hero()
    render_meta()
    render_pdf_tools()

    has_chat = bool(st.session_state.messages)

    for msg in st.session_state.messages:
        role = "user" if msg["role"] == "user" else "assistant"
        avatar = USER_AVATAR if role == "user" else ASSISTANT_AVATAR
        with st.chat_message(role, avatar=avatar):
            st.markdown(msg["content"])
            if msg.get("mode") == "ollama":
                st.caption("Answered with local Ollama")
            elif msg.get("mode") == "extractive":
                st.caption("From your local index")
            elif msg.get("mode") == "error":
                st.caption("Error")

    if st.session_state.is_thinking:
        render_loader()
        finish_thinking()
        return

    if not has_chat:
        render_suggestions()

    pending = st.session_state.pop("_pending_q", None)
    user_query = st.chat_input("Ask about a right, statute, or your PDF…")
    question = user_query or pending
    if question:
        ask(question)

    st.markdown(
        '<p class="foot"><strong>Educational research only</strong> — not legal advice. '
        "For real matters, speak with a licensed attorney.<br>"
        "Runs free on-device · MiniLM · FAISS · Ollama optional</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
