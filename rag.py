import os
import io
import uuid
import threading
import logging
import re
import sqlite3
from typing import List

import numpy as np
import requests
import faiss
from docx import Document
from cachetools import TTLCache, cached
from requests.adapters import HTTPAdapter, Retry
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from dotenv import load_dotenv

load_dotenv()

# ─── Configuration & Logging ──────────────────────────────────────────────────

# Persistence paths
INDEX_PATH = "index.faiss"
SQLITE_PATH = "chunks.db"

# Models & Config
EMBED_MODEL_NAME = "paraphrase-MiniLM-L3-v2"   # smaller model
EMBED_BATCH_SIZE = 16                          # smaller batch size to reduce peak RAM
TOP_K = 20
CONTEXT_CHUNK_LIMIT = 5           # max chunks per prompt
QUERY_CACHE = TTLCache(maxsize=32, ttl=3600)   # smaller cache

# Thread lock for FAISS operations
index_lock = threading.Lock()

# Lazy‐loaded embedding model
_embed_model = None
def get_embed_model():
    global _embed_model
    if _embed_model is None:
        logging.info(f"Loading SentenceTransformer model: {EMBED_MODEL_NAME}")
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    return _embed_model

# Set up structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ─── SQLite Setup ──────────────────────────────────────────────────────────────

# Use SQLite for storing (vid, doc_id, chunk_text) instead of an in‑RAM dict
conn = sqlite3.connect(SQLITE_PATH, check_same_thread=False)
cursor = conn.cursor()
cursor.execute(
    """
    CREATE TABLE IF NOT EXISTS chunks (
        vid      INTEGER PRIMARY KEY,
        doc_id   TEXT NOT NULL,
        text     BLOB  NOT NULL
    )
    """
)
conn.commit()

# ─── FAISS Index Initialization ────────────────────────────────────────────────

# Determine embedding dimension by loading model but not actually encoding
_dim = get_embed_model().get_sentence_embedding_dimension()

if os.path.exists(INDEX_PATH):
    # Load with memory‐mapping to avoid pulling full vectors into RAM
    logger.info("Loading existing FAISS index with mmap")
    loaded = faiss.read_index(INDEX_PATH, faiss.IO_FLAG_MMAP)
    if loaded.__class__.__name__ == "IndexIDMap2":
        index = loaded
    else:
        # Wrap a non‐IDMap index in IDMap2
        index = faiss.IndexIDMap2(loaded)
        logger.info("Wrapped loaded index in IndexIDMap2")
else:
    # First run: create a new IndexFlatL2 wrapped in IDMap2
    base_index = faiss.IndexFlatL2(_dim)
    index = faiss.IndexIDMap2(base_index)
    logger.info("Created new FAISS IndexIDMap2")

# ─── HTTP Session for Gemini Calls ────────────────────────────────────────────

def _create_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[502,503,504])
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session

session = _create_session()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or ""
if not GOOGLE_API_KEY:
    raise ValueError("Set GOOGLE_API_KEY in your environment")

# ─── Text Extraction (stop at References) ──────────────────────────────────────

def extract_text(file_bytes: bytes) -> str:
    # Attempt PDF extraction
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        pages = []
        for page in reader.pages:
            txt = page.extract_text() or ""
            if re.match(r"^\s*(References|Bibliography)\s*$", txt, re.IGNORECASE | re.MULTILINE):
                break
            pages.append(txt)
        return "\n".join(pages)
    except Exception:
        pass

    # Attempt DOCX extraction
    try:
        doc = Document(io.BytesIO(file_bytes))
        paras = []
        for p in doc.paragraphs:
            if re.match(r"^\s*(References|Bibliography)\s*$", p.text, re.IGNORECASE):
                break
            paras.append(p.text)
        return "\n".join(paras)
    except Exception:
        pass

    return ""

# ─── Smart Chunk Splitter ──────────────────────────────────────────────────────

def split_chunks(text: str, size: int = 800, overlap: int = 200) -> List[str]:
    # Truncate at first "References" or "Bibliography" header
    m = re.search(r"^\s*(References|Bibliography)\s*$", text, re.IGNORECASE | re.MULTILINE)
    if m:
        text = text[:m.start()]

    paras = re.split(r"\n\s*\n", text)
    chunks = []
    buffer = ""

    for p in paras:
        candidate = (buffer + "\n\n" + p).strip() if buffer else p
        if len(candidate) <= size:
            buffer = candidate
        else:
            if buffer:
                chunks.append(buffer)
            if len(p) > size:
                # Window‐split long paragraph
                for i in range(0, len(p), size - overlap):
                    chunks.append(p[i:i + size])
                buffer = ""
            else:
                buffer = p

    if buffer:
        chunks.append(buffer)

    # Final pass to ensure no chunk > size
    out = []
    for c in chunks:
        if len(c) <= size:
            out.append(c)
        else:
            for i in range(0, len(c), size - overlap):
                out.append(c[i:i + size])
    return out

# ─── Persistence: FAISS index only ──────────────────────────────────────────────

def _persist_index():
    with index_lock:
        faiss.write_index(index, INDEX_PATH)
    logger.info("FAISS index saved to disk")

# ─── Ingestion ─────────────────────────────────────────────────────────────────

def ingest_document(file_bytes: bytes) -> str:
    """
    - Extract text (up to "References").
    - Split into chunks.
    - Embed each chunk.
    - Add each embedding to FAISS under a unique vid.
    - Persist mapping from vid→(doc_id, chunk_text) in SQLite.
    - Persist FAISS index to disk.
    """
    text = extract_text(file_bytes)
    chunks = split_chunks(text)
    if not chunks:
        logger.warning("No text extracted; nothing to index")
        return ""

    model = get_embed_model()
    embs = model.encode(
        chunks,
        convert_to_numpy=True,
        batch_size=EMBED_BATCH_SIZE
    ).astype("float32")

    doc_id = str(uuid.uuid4())
    to_insert = []

    for chunk, emb in zip(chunks, embs):
        # Generate a stable positive int64 for FAISS
        vid = np.int64(uuid.uuid4().int & ((1 << 63) - 1))

        # Add to FAISS
        with index_lock:
            index.add_with_ids(emb.reshape(1, -1), np.array([vid], dtype="int64"))

        # Queue for SQLite insertion
        to_insert.append((int(vid), doc_id, chunk))

    # Bulk‐insert into SQLite
    cursor.executemany(
        "INSERT OR IGNORE INTO chunks (vid, doc_id, text) VALUES (?, ?, ?)",
        to_insert
    )
    conn.commit()

    _persist_index()
    logger.info("Ingested %d chunks for doc %s", len(chunks), doc_id)
    return doc_id

# ─── Query with Caching ────────────────────────────────────────────────────────

@cached(QUERY_CACHE)
def query_doc(doc_id: str, question: str) -> str:
    """
    - Embed the question.
    - Search FAISS to retrieve TOP_K nearest vids.
    - For each vid, pull (doc_id, chunk_text) from SQLite.
    - Filter to only those chunks matching the requested doc_id, up to CONTEXT_CHUNK_LIMIT.
    - Build prompt and call Gemini.
    """
    model = get_embed_model()
    q_emb = model.encode([question], convert_to_numpy=True).astype("float32")

    with index_lock:
        D, I = index.search(q_emb, TOP_K)

    # Flatten list of candidate vids
    candidate_vids = [int(vid) for vid in I[0] if vid >= 0]
    if not candidate_vids:
        return "Not Found"

    # Pull matching chunks from SQLite in one query
    placeholders = ",".join("?" for _ in candidate_vids)
    cursor.execute(
        f"SELECT vid, doc_id, text FROM chunks WHERE vid IN ({placeholders})",
        candidate_vids
    )
    rows = cursor.fetchall()  # list of tuples: (vid, doc_id, text)

    # Filter and collect up to CONTEXT_CHUNK_LIMIT for this doc_id
    seen = set()
    context_chunks = []
    for vid, d_id, chunk_text in rows:
        if d_id == doc_id and vid not in seen:
            context_chunks.append(f"— Chunk {vid} —\n{chunk_text}")
            seen.add(vid)
            if len(context_chunks) >= CONTEXT_CHUNK_LIMIT:
                break

    if not context_chunks:
        return "Not Found"

    prompt = (
        "You are a helpful assistant. Based only on the context below, answer concisely.\n\n"
        + f"Context (showing {len(context_chunks)} chunks):\n\n"
        + "\n\n".join(context_chunks)
        + f"\n\nQuestion: {question}\nAnswer:"
    )
    logger.debug("Prompt length: %d chars", len(prompt))

    # Call Gemini
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "gemini-2.0-flash:generateContent"
        f"?key={GOOGLE_API_KEY}"
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": 200, "temperature": 0.2},
    }

    try:
        res = session.post(url, json=payload, timeout=10)
        res.raise_for_status()
        data = res.json()
        logger.debug("Gemini API response JSON: %s", data)
        candidates = data.get("candidates", [])
        if candidates:
            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            if parts and "text" in parts[0]:
                return parts[0]["text"].strip()
            return "Sorry, no answer generated."
    except Exception as e:
        logger.error("Google API error: %s", e)
        return "Error generating answer."

    return "Error generating answer."
