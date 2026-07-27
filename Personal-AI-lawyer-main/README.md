# Personal AI Lawyer

Calm, private legal research on your device. Ask questions against a local FAISS
knowledge base (public-law corpus and/or your own PDFs). Uses free local
embeddings and optional free local generation via Ollama — **no Groq or paid LLM APIs**.

## What you need

- Python 3.10+
- [Ollama](https://ollama.com) (optional but recommended for natural-language answers)
- Enough disk for the MiniLM embedding model on first run

## Setup

```bash
cd Personal-AI-lawyer-main
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Ollama (free local LLM)

```bash
# Install from https://ollama.com, then:
ollama serve
ollama pull llama3.2
```

Optional env overrides:

```bash
export OLLAMA_MODEL=llama3.2          # or mistral, phi3, etc.
export OLLAMA_BASE_URL=http://localhost:11434
```

If Ollama is not running, the app still works in **extractive mode**: it retrieves
the most relevant passages from your local index and presents them clearly.

## Run

```bash
streamlit run app.py
```

Open the local URL Streamlit prints (usually http://localhost:8501). The UI is
mobile-first — try it on your phone’s browser on the same network if desired.

## How answers work

1. **Retrieve** — FAISS similarity search with `sentence-transformers/all-MiniLM-L6-v2`
   (or Agent 2 fine-tuned weights under `models/legal-embeddings` when present).
2. **Generate** — Ollama (`ChatOllama`) grounded on retrieved context, or extractive fallback.
3. **Indexes used** (first match wins):
   - `vectorstore/db_faiss_legal` — Agent 2 free legal corpus
   - `vectorstore/db_faiss_user` — your uploaded PDF
   - `vectorstore/db_faiss` — bundled / seed index (e.g. UDHR)

Upload a PDF in the app to index it locally for that session’s answers.

### Rebuild the free legal corpus (Agent 2)

```bash
python build_index.py
```

This refreshes `vectorstore/db_faiss_legal` from free public-domain sources.
The app picks it up automatically on the next question.

## Important disclaimer

This project is for **educational research only**. It is not a lawyer and does
not provide legal advice. Consult a licensed attorney for real legal matters.

## Project layout

| Path | Role |
|------|------|
| `app.py` | Primary Streamlit UI |
| `rag_pipeline.py` | Ollama + extractive RAG |
| `vector_database.py` | FAISS load / PDF ingest |
| `frontend.py` | Deprecated redirect → `app.py` |
| `train/` | Agent 2 corpus / training (separate) |
| `vectorstore/` | FAISS indexes |
