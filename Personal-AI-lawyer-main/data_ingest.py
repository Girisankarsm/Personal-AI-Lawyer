"""
Fetch and cache free / public-domain legal text for the Personal AI Lawyer RAG index.

Idempotent: re-running skips network when data/cache/corpus.jsonl already exists
unless force=True.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parent
PDFS_DIR = ROOT / "pdfs"
SEED_DIR = ROOT / "data" / "seed"
CACHE_DIR = ROOT / "data" / "cache"
CORPUS_PATH = CACHE_DIR / "corpus.jsonl"

# Disk-safe caps (full open corpora are multi-GB; keep a useful local slice)
MAX_CONSTITUTIONS = 7762
MAX_STATUTES = 4000
MAX_US_CODE = 3000
MAX_CASE_HOLD = 3000
MAX_PILE_EURLEX = 1500

MIN_TEXT_CHARS = 80


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _record(
    text: str,
    *,
    source: str,
    title: str = "",
    jurisdiction: str = "",
    citation: str = "",
    license_note: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    text = clean_text(text)
    if len(text) < MIN_TEXT_CHARS:
        return None
    meta = {
        "source": source,
        "title": title or source,
        "jurisdiction": jurisdiction,
        "citation": citation,
        "license": license_note,
    }
    if extra:
        meta.update(extra)
    return {"text": text, "metadata": meta}


def load_local_pdfs() -> List[Dict[str, Any]]:
    """Load every PDF under pdfs/ plus the root UDHR PDF if present."""
    records: List[Dict[str, Any]] = []
    try:
        from langchain_community.document_loaders import PDFPlumberLoader
    except ImportError:
        print("[ingest] PDFPlumberLoader unavailable; skipping PDFs")
        return records

    candidates: List[Path] = []
    if PDFS_DIR.exists():
        candidates.extend(sorted(PDFS_DIR.glob("*.pdf")))
    root_udhr = ROOT / "universal_declaration_of_human_rights.pdf"
    if root_udhr.exists() and root_udhr not in candidates:
        candidates.append(root_udhr)

    for pdf_path in candidates:
        try:
            docs = PDFPlumberLoader(str(pdf_path)).load()
        except Exception as exc:  # noqa: BLE001
            print(f"[ingest] PDF failed {pdf_path.name}: {exc}")
            continue
        for i, doc in enumerate(docs):
            rec = _record(
                doc.page_content,
                source=f"local_pdf:{pdf_path.name}",
                title=pdf_path.stem.replace("_", " "),
                jurisdiction="international",
                citation=f"{pdf_path.name} p.{i + 1}",
                license_note="See original document license (UDHR: UN public domain)",
                extra={"page": i + 1},
            )
            if rec:
                records.append(rec)
    print(f"[ingest] local PDFs -> {len(records)} page records")
    return records


def load_seed_texts() -> List[Dict[str, Any]]:
    """Load curated public-domain seed .txt files under data/seed/."""
    records: List[Dict[str, Any]] = []
    if not SEED_DIR.exists():
        return records
    for path in sorted(SEED_DIR.glob("*.txt")):
        raw = path.read_text(encoding="utf-8", errors="ignore")
        # Optional YAML-like header: --- key: value
        meta: Dict[str, str] = {
            "source": f"seed:{path.name}",
            "title": path.stem.replace("_", " "),
            "jurisdiction": "unknown",
            "citation": path.name,
            "license": "public domain / open educational seed",
        }
        body = raw
        if raw.startswith("---"):
            parts = raw.split("---", 2)
            if len(parts) >= 3:
                header, body = parts[1], parts[2]
                for line in header.strip().splitlines():
                    if ":" in line:
                        k, v = line.split(":", 1)
                        meta[k.strip()] = v.strip()
        rec = _record(
            body,
            source=meta.get("source", f"seed:{path.name}"),
            title=meta.get("title", path.stem),
            jurisdiction=meta.get("jurisdiction", ""),
            citation=meta.get("citation", path.name),
            license_note=meta.get("license", ""),
        )
        if rec:
            records.append(rec)
    print(f"[ingest] seed texts -> {len(records)} documents")
    return records


def _stream_hf(
    hf_id: str,
    *,
    name: Optional[str],
    split: str,
    max_rows: int,
    text_builder,
    source_tag: str,
    license_note: str,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install datasets: pip install datasets") from exc

    kwargs: Dict[str, Any] = {"path": hf_id, "split": split, "streaming": True}
    if name:
        kwargs["name"] = name
    print(f"[ingest] streaming {hf_id}" + (f"/{name}" if name else "") + f" (max {max_rows})...")
    ds = load_dataset(**kwargs)
    for i, row in enumerate(ds):
        if i >= max_rows:
            break
        built = text_builder(row)
        if not built:
            continue
        text, title, jurisdiction, citation = built
        rec = _record(
            text,
            source=source_tag,
            title=title,
            jurisdiction=jurisdiction,
            citation=citation,
            license_note=license_note,
            extra={"hf_row": i},
        )
        if rec:
            records.append(rec)
        if (i + 1) % 1000 == 0:
            print(f"  ... {i + 1} rows scanned, {len(records)} kept")
    print(f"[ingest] {source_tag} -> {len(records)} records")
    return records


def fetch_open_us_constitutions(max_rows: int = MAX_CONSTITUTIONS) -> List[Dict[str, Any]]:
    def builder(row: Dict[str, Any]):
        text = row.get("text") or ""
        citation = str(row.get("citation") or "")
        jurisdiction = str(row.get("jurisdiction") or row.get("state") or "US")
        title = citation or f"Constitution section {row.get('act_id', '')}"
        return text, title, jurisdiction, citation

    return _stream_hf(
        "vaquill/open-us-law",
        name="constitutions",
        split="train",
        max_rows=max_rows,
        text_builder=builder,
        source_tag="hf:vaquill/open-us-law/constitutions",
        license_note="US government edicts (public domain); schema CC-BY 4.0",
    )


def fetch_open_us_statutes(max_rows: int = MAX_STATUTES) -> List[Dict[str, Any]]:
    def builder(row: Dict[str, Any]):
        text = row.get("text") or ""
        citation = str(row.get("citation") or "")
        jurisdiction = str(row.get("jurisdiction") or "US")
        title = citation or f"Statute {row.get('act_id', '')}"
        return text, title, jurisdiction, citation

    return _stream_hf(
        "vaquill/open-us-law",
        name="statutes",
        split="train",
        max_rows=max_rows,
        text_builder=builder,
        source_tag="hf:vaquill/open-us-law/statutes",
        license_note="US government edicts (public domain); schema CC-BY 4.0",
    )


def fetch_us_legal_code(max_rows: int = MAX_US_CODE) -> List[Dict[str, Any]]:
    def builder(row: Dict[str, Any]):
        text = row.get("text") or ""
        title = str(row.get("title") or "US Code")
        section = str(row.get("section") or "")
        citation = f"USC {title} § {section}".strip()
        url = str(row.get("url") or "")
        return text, citation, "US-federal", citation if not url else f"{citation} ({url})"

    return _stream_hf(
        "emre570/us-legal-code",
        name=None,
        split="train",
        max_rows=max_rows,
        text_builder=builder,
        source_tag="hf:emre570/us-legal-code",
        license_note="Cornell LII materials: CC-BY-NC-SA 2.5 (non-commercial)",
    )


def fetch_lexglue_case_hold(max_rows: int = MAX_CASE_HOLD) -> List[Dict[str, Any]]:
    def builder(row: Dict[str, Any]):
        context = row.get("context") or ""
        endings = row.get("endings") or []
        label = row.get("label")
        holding = ""
        if isinstance(endings, list) and endings:
            try:
                idx = int(label) if label is not None else 0
                holding = endings[idx] if 0 <= idx < len(endings) else endings[0]
            except (TypeError, ValueError, IndexError):
                holding = endings[0]
        text = f"{context}\n\nHolding: {holding}".strip()
        return text, "CaseHOLD excerpt", "US-case-law", "LexGLUE case_hold"

    return _stream_hf(
        "coastalcph/lex_glue",
        name="case_hold",
        split="train",
        max_rows=max_rows,
        text_builder=builder,
        source_tag="hf:coastalcph/lex_glue/case_hold",
        license_note="LexGLUE / CaseHOLD research dataset",
    )


def fetch_pile_of_law_eurlex(max_rows: int = MAX_PILE_EURLEX) -> List[Dict[str, Any]]:
    def builder(row: Dict[str, Any]):
        text = row.get("text") or ""
        return text, "EUR-Lex document", "EU", "pile-of-law/eurlex"

    try:
        return _stream_hf(
            "pile-of-law/pile-of-law",
            name="eurlex",
            split="train",
            max_rows=max_rows,
            text_builder=builder,
            source_tag="hf:pile-of-law/pile-of-law/eurlex",
            license_note="Pile of Law research license; verify subset terms",
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[ingest] pile-of-law/eurlex skipped: {exc}")
        return []


def write_corpus(records: Iterable[Dict[str, Any]], path: Path = CORPUS_PATH) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_corpus(path: Path = CORPUS_PATH) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def build_corpus(*, force: bool = False, skip_remote: bool = False) -> Path:
    """
    Build data/cache/corpus.jsonl from local + free remote sources.
    If corpus exists and force is False, reuse cache (offline-friendly).
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if CORPUS_PATH.exists() and not force:
        n = sum(1 for _ in CORPUS_PATH.open(encoding="utf-8"))
        print(f"[ingest] using cached corpus ({n} docs): {CORPUS_PATH}")
        return CORPUS_PATH

    started = time.time()
    records: List[Dict[str, Any]] = []
    records.extend(load_seed_texts())
    records.extend(load_local_pdfs())

    if not skip_remote:
        fetchers = [
            ("constitutions", fetch_open_us_constitutions),
            ("statutes", fetch_open_us_statutes),
            ("us_code", fetch_us_legal_code),
            ("case_hold", fetch_lexglue_case_hold),
            ("eurlex", fetch_pile_of_law_eurlex),
        ]
        for label, fn in fetchers:
            try:
                records.extend(fn())
            except Exception as exc:  # noqa: BLE001
                print(f"[ingest] WARNING: {label} failed: {exc}")

    if not records:
        raise RuntimeError("No documents ingested. Add PDFs/seed texts or fix network.")

    n = write_corpus(records)
    print(f"[ingest] wrote {n} documents to {CORPUS_PATH} in {time.time() - started:.1f}s")
    return CORPUS_PATH


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest free legal corpora")
    parser.add_argument("--force", action="store_true", help="Rebuild corpus even if cache exists")
    parser.add_argument("--skip-remote", action="store_true", help="Only local PDFs + seed texts")
    args = parser.parse_args()
    build_corpus(force=args.force, skip_remote=args.skip_remote)
