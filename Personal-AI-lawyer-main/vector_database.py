"""FAISS vector store helpers — load Agent 2 indexes, PDF upload ingest."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

ROOT = Path(__file__).resolve().parent
PDFS_DIR = ROOT / "pdfs"
# Prefer Agent 2 legal corpus; fall back to legacy UDHR index
AGENT2_FAISS = ROOT / "vectorstore" / "db_faiss_legal"
LEGACY_FAISS = ROOT / "vectorstore" / "db_faiss"
USER_FAISS = ROOT / "vectorstore" / "db_faiss_user"

# Optional fine-tuned embeddings from Agent 2
LEGAL_EMBEDDINGS_DIR = ROOT / "models" / "legal-embeddings"
BASE_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

PDFS_DIR.mkdir(parents=True, exist_ok=True)
(ROOT / "vectorstore").mkdir(parents=True, exist_ok=True)


def get_embedding_model():
    """Local free embeddings. Prefer Agent 2 fine-tuned weights when present."""
    if LEGAL_EMBEDDINGS_DIR.exists() and any(LEGAL_EMBEDDINGS_DIR.iterdir()):
        model_name = str(LEGAL_EMBEDDINGS_DIR)
    else:
        model_name = BASE_EMBEDDING_MODEL
    return HuggingFaceEmbeddings(model_name=model_name)


def create_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )
    return splitter.split_documents(documents)


def load_pdf(file_path: str | Path):
    loader = PDFPlumberLoader(str(file_path))
    return loader.load()


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
