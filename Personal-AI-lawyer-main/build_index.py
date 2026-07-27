"""
Rebuild the FAISS legal knowledge index from the free corpus cache.

Usage:
  python build_index.py              # ingest if needed, then index
  python build_index.py --force      # re-download + rebuild
  python build_index.py --skip-remote
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path
from typing import List

from langchain_core.documents import Document

from data_ingest import CORPUS_PATH, build_corpus, read_corpus
from vector_database import (
    FAISS_DB_PATH,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    create_chunks,
    get_embedding_model,
    save_faiss_index,
)

ROOT = Path(__file__).resolve().parent


def records_to_documents(records: List[dict]) -> List[Document]:
    docs: List[Document] = []
    for rec in records:
        text = rec.get("text") or ""
        meta = dict(rec.get("metadata") or {})
        if len(text.strip()) < 40:
            continue
        docs.append(Document(page_content=text, metadata=meta))
    return docs


def build_index(*, force: bool = False, skip_remote: bool = False) -> Path:
    build_corpus(force=force, skip_remote=skip_remote)
    records = read_corpus(CORPUS_PATH)
    if not records:
        raise RuntimeError(f"Empty corpus at {CORPUS_PATH}")

    print(f"[index] loaded {len(records)} source documents")
    documents = records_to_documents(records)
    print(f"[index] {len(documents)} documents after filtering")

    chunks = create_chunks(
        documents,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    print(f"[index] {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    out_path = Path(FAISS_DB_PATH)
    if out_path.exists() and force:
        shutil.rmtree(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("[index] embedding with MiniLM and writing FAISS (CPU; may take several minutes)...")
    t0 = time.time()
    embeddings = get_embedding_model()
    save_faiss_index(chunks, embeddings, out_path)
    print(f"[index] saved FAISS to {out_path} in {time.time() - t0:.1f}s")

    # Write a small manifest for debugging
    manifest = out_path / "INDEX_MANIFEST.txt"
    sources = sorted({(d.metadata or {}).get("source", "unknown") for d in documents})
    manifest.write_text(
        "\n".join(
            [
                f"documents={len(documents)}",
                f"chunks={len(chunks)}",
                f"chunk_size={CHUNK_SIZE}",
                f"chunk_overlap={CHUNK_OVERLAP}",
                "sources:",
                *[f"  - {s}" for s in sources],
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS legal index")
    parser.add_argument("--force", action="store_true", help="Force re-ingest and rebuild")
    parser.add_argument("--skip-remote", action="store_true", help="Local PDFs + seeds only")
    args = parser.parse_args()
    build_index(force=args.force, skip_remote=args.skip_remote)


if __name__ == "__main__":
    main()
