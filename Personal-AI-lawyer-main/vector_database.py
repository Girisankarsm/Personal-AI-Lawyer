"""FAISS vector store helpers — Agent 2 legal corpus + PDF upload ingest.

Public API expected by app.py / rag_pipeline.py:
  - retrieve_docs(query, faiss_db=None, k=4)
  - load_default_vector_store / load_faiss / ensure_seed_index_from_udhr
  - index_uploaded_pdf / index_status / USER_FAISS
  - get_embedding_model / create_chunks

Free local only — no paid APIs.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

ROOT = Path(__file__).resolve().parent
PDFS_DIR = ROOT / "pdfs"

# Prefer Agent 2 free legal corpus; fall back to legacy / user indexes
AGENT2_FAISS = ROOT / "vectorstore" / "db_faiss_legal"
LEGACY_FAISS = ROOT / "vectorstore" / "db_faiss"
USER_FAISS = ROOT / "vectorstore" / "db_faiss_user"

# build_index.py writes the free corpus here (and mirrors to LEGACY_FAISS)
FAISS_DB_PATH = str(AGENT2_FAISS)

LEGAL_EMBEDDINGS_DIR = ROOT / "models" / "legal-embeddings"
BASE_EMBEDDING_MODEL = os.environ.get(
    "LEGAL_EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

CHUNK_SIZE = int(os.environ.get("LEGAL_CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.environ.get("LEGAL_CHUNK_OVERLAP", "120"))
DEFAULT_TOP_K = int(os.environ.get("LEGAL_TOP_K", "8"))
FETCH_K = int(os.environ.get("LEGAL_FETCH_K", "24"))
ENABLE_RERANK = os.environ.get("LEGAL_RERANK", "1") not in {"0", "false", "False"}

pdfs_directory = str(PDFS_DIR) + "/"

PDFS_DIR.mkdir(parents=True, exist_ok=True)
(ROOT / "vectorstore").mkdir(parents=True, exist_ok=True)

_cross_encoder = None
_cross_encoder_tried = False


def get_embedding_model():
    """Local free embeddings. Prefer Agent 2 fine-tuned weights when present."""
    if LEGAL_EMBEDDINGS_DIR.exists() and any(LEGAL_EMBEDDINGS_DIR.iterdir()):
        model_name = str(LEGAL_EMBEDDINGS_DIR)
    else:
        model_name = BASE_EMBEDDING_MODEL
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


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


def load_pdf(file_path: str | Path):
    loader = PDFPlumberLoader(str(file_path))
    return loader.load()


def upload_pdf(file) -> None:
    save_uploaded_pdf(file)


def save_uploaded_pdf(file) -> Path:
    PDFS_DIR.mkdir(parents=True, exist_ok=True)
    dest = PDFS_DIR / file.name
    with open(dest, "wb") as f:
        f.write(file.getbuffer())
    return dest


def create_vector_store(text_chunks, db_path: Path | str) -> FAISS:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    faiss_db = FAISS.from_documents(text_chunks, get_embedding_model())
    faiss_db.save_local(str(db_path))
    return faiss_db


def save_faiss_index(chunks: List[Document], embeddings=None, path: Path | str = FAISS_DB_PATH) -> FAISS:
    """Used by build_index.py — writes Agent 2 index and mirrors to legacy path."""
    embeddings = embeddings or get_embedding_model()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(str(path))
    # Mission + Agent 1: keep both db_faiss_legal and db_faiss in sync when building corpus
    if path.resolve() == AGENT2_FAISS.resolve():
        LEGACY_FAISS.parent.mkdir(parents=True, exist_ok=True)
        db.save_local(str(LEGACY_FAISS))
    elif path.resolve() == LEGACY_FAISS.resolve():
        AGENT2_FAISS.parent.mkdir(parents=True, exist_ok=True)
        db.save_local(str(AGENT2_FAISS))
    return db


def load_faiss(db_path: Path | str) -> Optional[FAISS]:
    db_path = Path(db_path)
    index_file = db_path / "index.faiss"
    if not index_file.exists():
        return None
    return FAISS.load_local(
        str(db_path),
        get_embedding_model(),
        allow_dangerous_deserialization=True,
    )


def index_exists(path: str | Path = FAISS_DB_PATH) -> bool:
    p = Path(path)
    return (p / "index.faiss").exists() and (p / "index.pkl").exists()


def resolve_default_faiss_path() -> Optional[Path]:
    """Pick the richest available on-disk index without rebuilding."""
    if (AGENT2_FAISS / "index.faiss").exists():
        return AGENT2_FAISS
    if (USER_FAISS / "index.faiss").exists():
        return USER_FAISS
    if (LEGACY_FAISS / "index.faiss").exists():
        return LEGACY_FAISS
    return None


def load_default_vector_store() -> Optional[FAISS]:
    path = resolve_default_faiss_path()
    if path is None:
        return None
    return load_faiss(path)


def index_uploaded_pdf(file) -> FAISS:
    """Save PDF, chunk, and write/replace the user FAISS index."""
    path = save_uploaded_pdf(file)
    documents = load_pdf(path)
    chunks = create_chunks(documents)
    return create_vector_store(chunks, USER_FAISS)


def ensure_seed_index_from_udhr() -> Optional[FAISS]:
    """If no index exists, seed from bundled UDHR PDF (local, free)."""
    existing = load_default_vector_store()
    if existing is not None:
        return existing

    udhr = ROOT / "universal_declaration_of_human_rights.pdf"
    pdfs_udhr = PDFS_DIR / "universal_declaration_of_human_rights.pdf"
    source = udhr if udhr.exists() else pdfs_udhr if pdfs_udhr.exists() else None
    if source is None:
        return None

    documents = load_pdf(source)
    chunks = create_chunks(documents)
    return create_vector_store(chunks, LEGACY_FAISS)


def index_status() -> dict:
    path = resolve_default_faiss_path()
    return {
        "path": str(path) if path else None,
        "ready": path is not None,
        "agent2": (AGENT2_FAISS / "index.faiss").exists(),
        "user": (USER_FAISS / "index.faiss").exists(),
        "legacy": (LEGACY_FAISS / "index.faiss").exists(),
        "embeddings": (
            "legal-finetuned"
            if LEGAL_EMBEDDINGS_DIR.exists() and any(LEGAL_EMBEDDINGS_DIR.iterdir())
            else BASE_EMBEDDING_MODEL
        ),
    }


def _get_cross_encoder():
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
        print(f"[retrieve] re-ranker unavailable ({exc}); using FAISS only")
        _cross_encoder = None
    return _cross_encoder


def _format_docs_with_meta(docs: List[Document]) -> List[Document]:
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


def retrieve_docs(query: str, faiss_db=None, k: Optional[int] = None) -> List[Document]:
    """
    Similarity search with optional CrossEncoder re-ranking.
    Compatible with app.py: retrieve_docs(question, faiss_db=db, k=4)
    """
    top_k = DEFAULT_TOP_K if k is None else k
    db = faiss_db
    if db is None:
        db = load_default_vector_store() or ensure_seed_index_from_udhr()
    if db is None:
        return []

    fetch_k = max(top_k, FETCH_K)
    # Use plain similarity_search — relevance-score normalization often warns/fails
    # with normalized MiniLM embeddings on small corpora.
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


class _LazyFaiss:
    def similarity_search(self, query, k=4, **kwargs):
        db = load_default_vector_store() or ensure_seed_index_from_udhr()
        if db is None:
            return []
        return db.similarity_search(query, k=k, **kwargs)

    def similarity_search_with_relevance_scores(self, query, k=4, **kwargs):
        db = load_default_vector_store() or ensure_seed_index_from_udhr()
        if db is None:
            return []
        return db.similarity_search_with_relevance_scores(query, k=k, **kwargs)


faiss_db = _LazyFaiss()
