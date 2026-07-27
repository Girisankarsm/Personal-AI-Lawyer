"""Training configuration for the Personal AI Lawyer legal knowledge base."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
MODELS_DIR = ROOT / "models"
LEGAL_EMBEDDINGS_DIR = MODELS_DIR / "legal-embeddings"
FAISS_DB_PATH = ROOT / "vectorstore" / "db_faiss_legal"

# Base embedding model (small enough for CPU fine-tuning)
BASE_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Chunking
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

# Disk-safe sample sizes (machine has limited free space; full Pile-of-Law is 256GB+)
MAX_CONSTITUTION_ROWS = 8000
MAX_STATUTE_ROWS = 25000
MAX_PILE_OF_LAW_ROWS = 8000
MAX_LEXGLUE_ROWS = 15000
MAX_FINETUNE_PAIRS = 6000
FINETUNE_EPOCHS = 1
FINETUNE_BATCH_SIZE = 16
FINETUNE_WARMUP = 100

# Free / open legal datasets used for retrieval index + embedding fine-tuning
DATASETS = {
    "constitutions": {
        "hf_id": "vaquill/open-us-law",
        "name": "constitutions",
        "split": "train",
        "text_fields": ["text", "citation", "jurisdiction"],
        "max_rows": MAX_CONSTITUTION_ROWS,
        "license_note": "Public domain / CC-BY (US government edicts)",
    },
    "us_statutes": {
        "hf_id": "vaquill/open-us-law",
        "name": "statutes",
        "split": "train",
        "text_fields": ["text", "citation", "jurisdiction"],
        "max_rows": MAX_STATUTE_ROWS,
        "license_note": "Public domain / CC-BY (US government edicts)",
    },
    "us_code": {
        "hf_id": "emre570/us-legal-code",
        "name": None,
        "split": "train",
        "text_fields": ["text", "title", "section", "url"],
        "max_rows": 20000,
        "license_note": "Cornell LII CC-BY-NC-SA (non-commercial)",
    },
    "pile_of_law_us_code": {
        "hf_id": "pile-of-law/pile-of-law",
        "name": "r_legiscraper.us.code",
        "split": "train",
        "text_fields": ["text"],
        "max_rows": MAX_PILE_OF_LAW_ROWS,
        "license_note": "Pile of Law research license (check subset terms)",
    },
    "pile_of_law_eurlex": {
        "hf_id": "pile-of-law/pile-of-law",
        "name": "eurlex",
        "split": "train",
        "text_fields": ["text"],
        "max_rows": 4000,
        "license_note": "Pile of Law / Eurlex",
    },
    "case_hold": {
        "hf_id": "coastalcph/lex_glue",
        "name": "case_hold",
        "split": "train",
        "text_fields": ["context", "endings", "label"],
        "max_rows": MAX_LEXGLUE_ROWS,
        "license_note": "LexGLUE / CaseHOLD research dataset",
        "for_finetune": True,
    },
}
