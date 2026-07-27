"""
Local FAISS retrieval for Personal AI Lawyer.

Public API used by the app / rag_pipeline:
  - retrieve_docs(query, k=None) -> List[Document]
  - get_embedding_model()
  - upload_pdf / load_pdf / create_chunks (for optional user PDF flows)

Does not call paid APIs. Loads vectorstore/db_faiss lazily.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

ROOT = Path(__file__).resolve().parent
PDFS_DIR = ROOT / "pdfs"
FAISS_DB_PATH = str(ROOT / "vectorstore" / "db_faiss")
EMBEDDING_MODEL_NAME = os.environ.get(
    "LEGAL_EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

CHUNK_SIZE = int(os.environ.get("LEGAL_CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.environ.get("LEGAL_CHUNK_OVERLAP", "120"))
DEFAULT_TOP_K = int(os.environ.get("LEGAL_TOP_K", "8"))
FETCH_K = int(os.environ.get("LEGAL_FETCH_K", "24"))  # candidates before re-rank
ENABLE_RERANK = os.environ.get("LEGAL_RERANK", "1") not in {"0", "false", "False"}

pdfs_directory = str(PDFS_DIR) + "/"

_faiss_db: Optional[FAISS] = None
_cross_encoder = None
_cross_encoder_tried = False


def upload_pdf(file) -> None:
    PDFS_DIR.mkdir(parents=True, exist_ok=True)
    with open(PDFS_DIR / file.name, "wb") as f:
        f.write(file.getbuffer())


def load_pdf(file_path: str):
    loader = PDFPlumberLoader(file_path)
    return loader.load()


def create_chunks(
    documents,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


@lru_cache(maxsize=1)
def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def save_faiss_index(chunks: List[Document], embeddings=None, path: Path | str = FAISS_DB_PATH) -> FAISS:
    embeddings = embeddings or get_embedding_model()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(str(path))
    global _faiss_db
    _faiss_db = db
    return db


def index_exists(path: str = FAISS_DB_PATH) -> bool:
    p = Path(path)
    return (p / "index.faiss").exists() and (p / "index.pkl").exists()


def load_faiss_db(path: str = FAISS_DB_PATH, *, allow_dangerous: bool = True) -> FAISS:
    global _faiss_db
    if _faiss_db is not None:
        return _faiss_db
    if not index_exists(path):
        raise FileNotFoundError(
            f"FAISS index not found at {path}. Run: python build_index.py"
        )
    _faiss_db = FAISS.load_local(
        path,
        get_embedding_model(),
        allow_dangerous_deserialization=allow_dangerous,
    )
    return _faiss_db


def _get_cross_encoder():
    """Optional lightweight CPU re-ranker (free Hugging Face model)."""
    global _cross_encoder, _cross_encoder_tried
    if _cross_encoder_tried:
        return _cross_encoder
    _cross_encoder_tried = True
    if not ENABLE_RERANK:
        return None
    try:
        from sentence_transformers import CrossEncoder

        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    except Exception as exc:  # noqa: BLE001
        print(f"[retrieve] re-ranker unavailable ({exc}); using FAISS scores only")
        _cross_encoder = None
    return _cross_encoder


def _format_docs_with_meta(docs: List[Document]) -> List[Document]:
    """Prefix citation/source into page_content so the LLM sees provenance."""
    formatted: List[Document] = []
    for doc in docs:
        meta = doc.metadata or {}
        citation = meta.get("citation") or meta.get("title") or ""
        source = meta.get("source") or ""
        header_bits = [b for b in (citation, source) if b]
        header = " | ".join(header_bits)
        content = doc.page_content
        if header and not content.startswith("["):
            content = f"[{header}]\n{content}"
        formatted.append(Document(page_content=content, metadata=meta))
    return formatted


def retrieve_docs(query: str, k: Optional[int] = None) -> List[Document]:
    """
    Similarity search with optional CrossEncoder re-ranking.
    Stable interface for app.py / rag_pipeline.py / frontend.py.
    """
    top_k = k if k is not None else DEFAULT_TOP_K
    db = load_faiss_db()
    fetch_k = max(top_k, FETCH_K)

    # Prefer score-aware search
    try:
        scored = db.similarity_search_with_relevance_scores(query, k=fetch_k)
        candidates = [doc for doc, _score in scored]
    except Exception:  # noqa: BLE001
        candidates = db.similarity_search(query, k=fetch_k)

    reranker = _get_cross_encoder()
    if reranker is not None and len(candidates) > top_k:
        pairs = [[query, d.page_content] for d in candidates]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: float(x[1]), reverse=True)
        selected = [doc for doc, _ in ranked[:top_k]]
    else:
        selected = candidates[:top_k]

    return _format_docs_with_meta(selected)


def get_context(documents: List[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in documents)


# Back-compat: some older code imported `faiss_db` at module level.
# Provide a lazy proxy so import does not rebuild the index.
class _LazyFaiss:
    def similarity_search(self, query, k=4, **kwargs):
        return load_faiss_db().similarity_search(query, k=k, **kwargs)

    def similarity_search_with_relevance_scores(self, query, k=4, **kwargs):
        return load_faiss_db().similarity_search_with_relevance_scores(query, k=k, **kwargs)


faiss_db = _LazyFaiss()
