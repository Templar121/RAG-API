import os
import io
import uuid
import threading
import logging
import re
from typing import List
import certifi

import numpy as np
import requests
import faiss
from docx import Document
from cachetools import TTLCache, cached
from requests.adapters import HTTPAdapter, Retry
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ─── Configuration & Logging ──────────────────────────────────────────────────

# Persistence paths
INDEX_PATH = "index.faiss"

# MongoDB setup for chunk storage
MONGODB_URI = os.getenv("MONGODB_URI")
if not MONGODB_URI:
    raise ValueError("Set MONGODB_URI in your .env")
MONGODB_DB = os.getenv("MONGODB_DB")
if not MONGODB_DB:
    raise ValueError("Set MONGODB_DB (database name) in your .env")
client = MongoClient(
    MONGODB_URI,
    tls=True,
    tlsCAFile=certifi.where(),
    serverSelectionTimeoutMS=20000,
)
db = client[MONGODB_DB]
chunks_coll = db["chunks"]
# Ensure unique index on vid
chunks_coll.create_index("vid", unique=True)

# Models & Config
EMBED_MODEL_NAME = "paraphrase-MiniLM-L3-v2"
EMBED_BATCH_SIZE = 16
TOP_K = 50
CONTEXT_CHUNK_LIMIT = 15
QUERY_CACHE = TTLCache(maxsize=32, ttl=3600)

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

# ─── FAISS Index Initialization ────────────────────────────────────────────────

# Determine embedding dimension
_dim = get_embed_model().get_sentence_embedding_dimension()

if os.path.exists(INDEX_PATH):
    # Load with memory‐mapping
    logger.info("Loading existing FAISS index with mmap")
    loaded = faiss.read_index(INDEX_PATH, faiss.IO_FLAG_MMAP)
    if loaded.__class__.__name__ == "IndexIDMap2":
        index = loaded
    else:
        index = faiss.IndexIDMap2(loaded)
        logger.info("Wrapped loaded index in IndexIDMap2")
else:
    base_index = faiss.IndexFlatL2(_dim)
    index = faiss.IndexIDMap2(base_index)
    logger.info("Created new FAISS IndexIDMap2")

# ─── HTTP Session for LLM Calls ─────────────────────────────────────────────────

def _create_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[502,503,504])
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session

session = _create_session()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or ""
if not GOOGLE_API_KEY:
    raise ValueError("Set GOOGLE_API_KEY in your .env")

# ─── Text Extraction (stop at References) ──────────────────────────────────────

def extract_text(file_bytes: bytes) -> str:
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
    m = re.search(r"^\s*(References|Bibliography)\s*$", text, re.IGNORECASE | re.MULTILINE)
    if m:
        text = text[:m.start()]

    paras = re.split(r"\n\s*\n", text)
    chunks, buffer = [], ""
    for p in paras:
        candidate = (buffer + "\n\n" + p).strip() if buffer else p
        if len(candidate) <= size:
            buffer = candidate
        else:
            if buffer:
                chunks.append(buffer)
            if len(p) > size:
                for i in range(0, len(p), size - overlap):
                    chunks.append(p[i:i + size])
                buffer = ""
            else:
                buffer = p
    if buffer:
        chunks.append(buffer)

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
    mongo_docs = []

    for chunk, emb in zip(chunks, embs):
        vid = np.int64(uuid.uuid4().int & ((1 << 63) - 1))  # 63-bit positive int
        with index_lock:
            index.add_with_ids(emb.reshape(1, -1), np.array([vid], dtype="int64"))
        mongo_docs.append({"vid": int(vid), "doc_id": doc_id, "text": chunk})

    try:
        chunks_coll.insert_many(mongo_docs, ordered=False)
    except Exception as e:
        logger.warning("MongoDB insert warning: %s", e)

    _persist_index()
    logger.info("Ingested %d chunks for doc %s", len(chunks), doc_id)

    return doc_id

# ─── Query with Caching ────────────────────────────────────────────────────────

@cached(QUERY_CACHE)
def query_doc(doc_id: str, question: str) -> str:
    model = get_embed_model()
    q_emb = model.encode([question], convert_to_numpy=True).astype("float32")

    with index_lock:
        D, I = index.search(q_emb, TOP_K)

    candidate_vids = [int(vid) for vid in I[0] if vid >= 0]
    if not candidate_vids:
        return "Not Found"

    cursor = chunks_coll.find(
        {"vid": {"$in": candidate_vids}},
        {"_id": 0, "vid": 1, "doc_id": 1, "text": 1}
    )
    rows = [(d["vid"], d["doc_id"], d["text"]) for d in cursor]

    seen, context_chunks = set(), []
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

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "gemini-2.0-flash:generateContent"
        f"?key={GOOGLE_API_KEY}"
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": 512, "temperature": 0.2},
    }

    try:
        res = session.post(url, json=payload, timeout=10)
        res.raise_for_status()
        data = res.json()
        candidates = data.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            if parts and "text" in parts[0]:
                return parts[0]["text"].strip()
            return "Sorry, no answer generated."
    except Exception as e:
        logger.error("LLM API error: %s", e)
        return "Error generating answer."

    return "Error generating answer."
# ─── Debug Chunks Utility ─────────────────────────────────────────────────────

def debug_chunks(doc_id: str, question: str):
    """
    Retrieve up to CONTEXT_CHUNK_LIMIT raw chunk texts matching the given doc_id for inspection.
    Returns (list_of_texts, warning_str_or_None).
    """
    # Generate question embedding
    model = get_embed_model()
    q_emb = model.encode([question], convert_to_numpy=True).astype("float32")
    # Search FAISS
    with index_lock:
        _, I = index.search(q_emb, TOP_K)
    vids = [int(v) for v in I[0] if v >= 0]
    if not vids:
        return [], "No matching chunks found"
    # Fetch from MongoDB
    docs = list(chunks_coll.find(
        {"vid": {"$in": vids}},
        {"_id": 0, "vid": 1, "doc_id": 1, "text": 1}
    ))
    # Filter by doc_id
    seen = set()
    texts = []
    for d in docs:
        if d.get("doc_id") == doc_id and d["vid"] not in seen:
            texts.append(d["text"])
            seen.add(d["vid"])
            if len(texts) >= CONTEXT_CHUNK_LIMIT:
                break
    if not texts:
        return [], "No matching chunks for this doc_id"
    return texts, None
