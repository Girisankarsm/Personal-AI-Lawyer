# Legal data sources (Personal AI Lawyer)

## Disclaimer

**This project is for educational / research RAG demos only. It is not a lawyer, and its outputs are not legal advice.** Laws change; datasets may be incomplete or outdated. Always verify against official primary sources and consult a qualified attorney for real legal matters.

## What this index contains

The free legal corpus is built by `build_index.py` into `vectorstore/db_faiss_legal`
(and mirrored to `vectorstore/db_faiss`) from free / openly available materials:

| Source | License / notes | How loaded |
|---|---|---|
| Local PDFs under `pdfs/` (incl. Universal Declaration of Human Rights) | UDHR: UN text, educational use | `data_ingest.load_local_pdfs` |
| Seed texts in `data/seed/` (US Bill of Rights, selected US Constitution, UDHR articles, ICCPR excerpts) | US constitutional text: public domain; UN texts: educational reproduction | local files |
| [vaquill/open-us-law](https://huggingface.co/datasets/vaquill/open-us-law) constitutions (+ statute sample) | US government edicts (public domain); dataset schema CC-BY 4.0 | Hugging Face streaming |
| [emre570/us-legal-code](https://huggingface.co/datasets/emre570/us-legal-code) (sample) | Cornell LII: **CC-BY-NC-SA 2.5 (non-commercial)** | Hugging Face streaming |
| [coastalcph/lex_glue](https://huggingface.co/datasets/coastalcph/lex_glue) `case_hold` (sample) | LexGLUE / CaseHOLD research dataset | Hugging Face streaming |
| [pile-of-law/pile-of-law](https://huggingface.co/datasets/pile-of-law/pile-of-law) `eurlex` (optional sample) | Pile of Law research license — verify before commercial use | Hugging Face streaming |

Full multi-hundred-GB corpora (entire Pile of Law / MultiLegalPile) are **not** downloaded; disk-safe sample caps are set in `data_ingest.py`.

## Rebuild (offline after first cache)

```bash
cd Personal-AI-lawyer-main
pip install -r requirements.txt

# First run (downloads + caches free HF slices):
python build_index.py --force

# Later runs (uses data/cache/corpus.jsonl; no network needed for ingest):
python build_index.py

# Local-only (PDFs + seeds, no Hugging Face):
python build_index.py --force --skip-remote
```

Cached corpus: `data/cache/corpus.jsonl`  
Index: `vectorstore/db_faiss_legal/` (mirrored to `vectorstore/db_faiss/`)

## Retrieval improvements

- MiniLM embeddings (`sentence-transformers/all-MiniLM-L6-v2`), normalized
- Chunk size 800 / overlap 120 with multi-separator splitting
- Default top-k = 8; fetch 24 candidates then optional CrossEncoder re-rank (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
- Citation / source metadata prefixed into retrieved chunks

Environment knobs: `LEGAL_TOP_K`, `LEGAL_FETCH_K`, `LEGAL_RERANK=0` to disable re-rank, `LEGAL_CHUNK_SIZE`, `LEGAL_CHUNK_OVERLAP`.

## App interface

`vector_database.retrieve_docs(query)` remains the stable retrieval entry point for the UI / LLM layer (Agent 1).
