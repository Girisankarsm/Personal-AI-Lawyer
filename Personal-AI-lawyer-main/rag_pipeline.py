"""
Free local RAG generation for Personal AI Lawyer.

Uses Ollama when available; otherwise extractive fallback from retrieved chunks.
No paid APIs.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from vector_database import get_context, retrieve_docs as _retrieve_docs

OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

CUSTOM_PROMPT = """
You are a calm educational legal research assistant.
Use ONLY the context below. If it is insufficient, say you do not know.
Do not invent citations or statutes. This is not legal advice.

Question: {question}
Context: {context}
Answer:
"""


def check_ollama_available() -> Tuple[bool, str]:
    """Return (ok, message). Never raises."""
    try:
        import urllib.request

        req = urllib.request.Request(f"{OLLAMA_BASE_URL.rstrip('/')}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=2) as resp:
            if resp.status == 200:
                return True, f"Ollama reachable · model `{OLLAMA_MODEL}`"
            return False, "Ollama responded unexpectedly. Using extractive mode."
    except Exception:  # noqa: BLE001
        return (
            False,
            f"Ollama offline at {OLLAMA_BASE_URL}. "
            f"Install from https://ollama.com then run: `ollama pull {OLLAMA_MODEL}`. "
            "Until then, answers use extractive passages from your local index.",
        )


def retrieve_docs(query: str, faiss_db=None, k: int = 4) -> List[Document]:
    """App-facing retrieval; optional in-memory FAISS override for uploaded PDFs."""
    return _retrieve_docs(query, k=k, faiss_db=faiss_db)


def _extractive_answer(documents: List[Document], query: str) -> str:
    if not documents:
        return (
            "I could not find relevant passages in the local knowledge base. "
            "Try uploading a PDF or rebuilding the index with `python build_index.py`."
        )
    parts = [
        f"**Extractive answer** (Ollama offline — showing grounded passages for: *{query}*)",
        "",
    ]
    for i, doc in enumerate(documents[:4], 1):
        meta = doc.metadata or {}
        cite = meta.get("citation") or meta.get("title") or meta.get("source") or f"Passage {i}"
        snippet = doc.page_content.strip()
        if len(snippet) > 700:
            snippet = snippet[:700].rstrip() + "…"
        parts.append(f"**{i}. {cite}**\n{snippet}")
        parts.append("")
    parts.append(
        "_Start Ollama (`ollama serve` + `ollama pull llama3.2`) for natural-language answers._"
    )
    return "\n".join(parts).strip()


def _ollama_answer(documents: List[Document], query: str) -> str:
    from langchain_ollama import ChatOllama

    context = get_context(documents) if documents else ""
    if not context.strip():
        return (
            "I do not know based on the current knowledge base. "
            "Upload a PDF or rebuild the legal index."
        )
    llm = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.2,
    )
    prompt = ChatPromptTemplate.from_template(CUSTOM_PROMPT)
    chain = prompt | llm
    result = chain.invoke({"question": query, "context": context})
    if hasattr(result, "content"):
        return str(result.content).strip()
    return str(result).strip()


def answer_query(
    documents: List[Document],
    query: str,
    model=None,  # ignored; kept for older call sites
) -> Tuple[str, str]:
    """
    Returns (answer_text, mode) where mode is 'ollama' or 'extractive'.
    """
    ok, _msg = check_ollama_available()
    if ok:
        try:
            return _ollama_answer(documents, query), "ollama"
        except Exception as exc:  # noqa: BLE001
            fallback = _extractive_answer(documents, query)
            return (
                f"{fallback}\n\n_Ollama error: {exc}_",
                "extractive",
            )
    return _extractive_answer(documents, query), "extractive"


# Soft back-compat for anything still expecting llm_model
llm_model = None
