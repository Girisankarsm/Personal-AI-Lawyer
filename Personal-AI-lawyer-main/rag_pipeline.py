"""RAG answer pipeline — free local Ollama LLM with extractive fallback."""

from __future__ import annotations

import os
import re
from typing import Any, Optional

from langchain_core.prompts import ChatPromptTemplate

from vector_database import (
    ensure_seed_index_from_udhr,
    get_context,
    load_default_vector_store,
    retrieve_docs as _retrieve_docs,
)

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

CUSTOM_PROMPT = """You are a calm, clear personal legal research assistant.
Use ONLY the context below. If the context is insufficient, say you don't know
and suggest what kind of document or statute might help. Do not invent laws.
Always remind the reader this is informational research, not legal advice.

Question: {question}

Context:
{context}

Answer in plain, reassuring language. Cite the most relevant passages briefly.
Answer:"""


def retrieve_docs(query: str, faiss_db=None, k: int = 4):
    """Delegate to vector_database (FAISS + optional free CrossEncoder re-rank)."""
    return _retrieve_docs(query, faiss_db=faiss_db, k=k)


def _strip_think_tags(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()


def check_ollama_available(model: str = OLLAMA_MODEL) -> tuple[bool, str]:
    """Return (ok, message). Uses a lightweight HTTP ping — no paid APIs."""
    try:
        import urllib.request

        req = urllib.request.Request(f"{OLLAMA_BASE_URL.rstrip('/')}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=2) as resp:
            if resp.status != 200:
                return False, "Ollama is not responding. Run `ollama serve` in a terminal."
        # Soft check: model may need pull
        return True, f"Ollama ready (default model: {model})"
    except Exception:
        return False, (
            "Ollama is not running. Install from https://ollama.com then run "
            f"`ollama serve` and `ollama pull {model}`."
        )


def get_ollama_llm(model: str = OLLAMA_MODEL):
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=model,
        base_url=OLLAMA_BASE_URL,
        temperature=0.2,
    )


def extractive_answer(documents, query: str) -> str:
    """Zero-API fallback: present the most relevant retrieved passages clearly."""
    if not documents:
        return (
            "I couldn't find matching passages in the knowledge base yet. "
            "Upload a PDF of laws or wait for the legal corpus index to finish building, "
            "then try again.\n\n"
            "_This is not legal advice._"
        )

    parts = [
        f"Here’s what I found related to your question:\n**“{query.strip()}”**\n",
        "I’ve pulled the closest passages from your local legal index "
        "(no cloud AI was used for this answer):\n",
    ]
    for i, doc in enumerate(documents, start=1):
        meta = doc.metadata or {}
        source = meta.get("source") or meta.get("title") or meta.get("citation") or "local index"
        snippet = doc.page_content.strip()
        if len(snippet) > 700:
            snippet = snippet[:700].rstrip() + "…"
        parts.append(f"**Passage {i}** · _{source}_\n\n{snippet}\n")

    parts.append(
        "---\n"
        "For a fuller written answer in plain language, start Ollama locally "
        f"(`ollama pull {OLLAMA_MODEL}` then `ollama serve`) and ask again.\n\n"
        "_This is informational research only — not legal advice. "
        "Please consult a licensed attorney for your situation._"
    )
    return "\n".join(parts)


def answer_with_ollama(documents, query: str, model: Optional[Any] = None) -> str:
    context = get_context(documents)
    if not context.strip():
        return extractive_answer(documents, query)

    llm = model or get_ollama_llm()
    prompt = ChatPromptTemplate.from_template(CUSTOM_PROMPT)
    chain = prompt | llm
    result = chain.invoke({"question": query, "context": context})
    content = getattr(result, "content", None) or str(result)
    text = _strip_think_tags(content)
    if "not legal advice" not in text.lower():
        text += (
            "\n\n_This is informational research only — not legal advice. "
            "Please consult a licensed attorney for your situation._"
        )
    return text


def answer_query(documents, query: str, model=None, prefer_ollama: bool = True) -> tuple[str, str]:
    """
    Returns (answer_text, mode) where mode is 'ollama' or 'extractive'.
    Never calls paid APIs.
    """
    if prefer_ollama:
        ok, _msg = check_ollama_available()
        if ok:
            try:
                return answer_with_ollama(documents, query, model=model), "ollama"
            except Exception as exc:
                fallback = extractive_answer(documents, query)
                note = (
                    f"\n\n> Ollama was reachable but answering failed ({exc}). "
                    "Showing retrieved passages instead."
                )
                return fallback + note, "extractive"

    return extractive_answer(documents, query), "extractive"


# Lazy convenience for scripts; app.py should call answer_query directly.
llm_model = None
