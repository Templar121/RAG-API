import os
import io
import uuid
import pickle
import threading
import logging
import re
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

# Configure structured logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Persistence paths
index_path = "index.faiss"
idmap_path = "id_map.pkl"

# Models & Config
EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_BATCH_SIZE = 32
TOP_K = 20
CONTEXT_CHUNK_LIMIT = 5   # max chunks per prompt
QUERY_CACHE = TTLCache(maxsize=128, ttl=3600)  # 1 hr TTL

# Initialize embedding model
embed_model = SentenceTransformer(EMBED_MODEL)
dim = embed_model.get_sentence_embedding_dimension()

if os.path.exists(index_path) and os.path.exists(idmap_path):
    # 1) Read back whatever index was saved
    loaded = faiss.read_index(index_path)

    # 2) If it's already an IndexIDMap2, use it directly
    if loaded.__class__.__name__ == "IndexIDMap2":
        index = loaded
        logger.info("Loaded existing FAISS IndexIDMap2")
    else:
        # 3) Otherwise only wrap if it's truly empty
        if loaded.ntotal != 0:
            raise RuntimeError(
                "Expected an IndexIDMap2 on load but got a non-empty base index"
            )
        index = faiss.IndexIDMap2(loaded)
        logger.info("Wrapped loaded empty index in IndexIDMap2")

    # 4) Load the ID→(doc_id, chunk) map
    with open(idmap_path, "rb") as f:
        id_map = pickle.load(f)
    logger.info("Loaded ID map (%d entries)", len(id_map))

else:
    # First run: create a new base index and wrap it
    base = faiss.IndexFlatL2(dim)
    index = faiss.IndexIDMap2(base)
    id_map = {}
    logger.info("Created new FAISS IndexIDMap2")

# Thread lock for FAISS ops
index_lock = threading.Lock()

# HTTP session with retries for Gemini calls
def _create_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[502,503,504])
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session

session = _create_session()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or ""
if not GOOGLE_API_KEY:
    raise ValueError("Set GOOGLE_API_KEY in your environment")

# ─── Extraction: stop at References ─────────────────────────────────────────────
def extract_text(file_bytes: bytes) -> str:
    # PDF
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

    # DOCX
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

# ─── Smart chunk splitter ───────────────────────────────────────────────────────
def split_chunks(text: str, size: int = 800, overlap: int = 200) -> List[str]:
    # truncate at first References header
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
                # window-split
                for i in range(0, len(p), size - overlap):
                    chunks.append(p[i:i+size])
                buffer = ""
            else:
                buffer = p

    if buffer:
        chunks.append(buffer)

    # final ensure no chunk oversized
    out = []
    for c in chunks:
        if len(c) <= size:
            out.append(c)
        else:
            for i in range(0, len(c), size - overlap):
                out.append(c[i:i+size])
    return out

# ─── Persistence ────────────────────────────────────────────────────────────────
def _persist():
    with index_lock:
        faiss.write_index(index, index_path)
        with open(idmap_path, "wb") as f:
            pickle.dump(id_map, f)
    logger.info("Index and ID map saved")

# ─── Ingestion ─────────────────────────────────────────────────────────────────
def ingest_document(file_bytes: bytes) -> str:
    text = extract_text(file_bytes)
    chunks = split_chunks(text)
    if not chunks:
        logger.warning("No text extracted; nothing to index")
        return ""

    # embed and cast to float32
    embs = embed_model.encode(
        chunks,
        convert_to_numpy=True,
        batch_size=EMBED_BATCH_SIZE
    ).astype("float32")

    # single doc_id for all chunks
    doc_id = str(uuid.uuid4())

    for chunk, emb in zip(chunks, embs):
        # stable positive int64 id
        vid = np.int64(uuid.uuid4().int & ((1<<63)-1))
        with index_lock:
            index.add_with_ids(emb.reshape(1, -1), np.array([vid], dtype="int64"))
        id_map[vid] = (doc_id, chunk)

    _persist()
    logger.info("Ingested %d chunks for doc %s", len(chunks), doc_id)
    return doc_id

# ─── Query with caching ────────────────────────────────────────────────────────
@cached(QUERY_CACHE)
def query_doc(doc_id: str, question: str) -> str:
    q_emb = embed_model.encode([question], convert_to_numpy=True).astype("float32")

    with index_lock:
        D, I = index.search(q_emb, TOP_K)

    # collect up to CONTEXT_CHUNK_LIMIT chunks for this doc
    seen, context = set(), []
    for vid in I[0]:
        if vid in id_map:
            d_id, chunk = id_map[vid]
            if d_id == doc_id and vid not in seen:
                context.append(f"— Chunk {vid} —\n{chunk}")
                seen.add(vid)
                if len(context) >= CONTEXT_CHUNK_LIMIT:
                    break

    if not context:
        return "Not Found"

    prompt = (
        "You are a helpful assistant. Based only on the context below, answer concisely.\n\n"
        + "Context (showing {} chunks):\n\n".format(len(context))
        + "\n\n".join(context)
        + f"\n\nQuestion: {question}\nAnswer:"
    )
    logger.debug("Prompt length: %d chars", len(prompt))

    # call Gemini
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
        logger.debug("API response JSON: %s", data)
        candidates = data.get("candidates", [])
        if candidates:
            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            if parts and "text" in parts[0]:
                return parts[0]["text"].strip()
            else:
                return "Sorry, no answer generated."
    except Exception as e:
        logger.error("Google API error: %s", e)

    return "Error generating answer."
