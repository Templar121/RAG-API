import os
import io
import uuid
import pickle
from typing import Dict, List
from functools import lru_cache

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceApi
import PyPDF2
from docx import Document
from dotenv import load_dotenv

load_dotenv()  # loads HUGGINGFACE_HUB_TOKEN

# Persistence paths
INDEX_PATH = "index.faiss"
IDMAP_PATH  = "id_map.pkl"

# Models & Config
EMBED_MODEL     = "all-MiniLM-L6-v2"
HF_MODEL        = "EleutherAI/gpt-neo-125M"
EMBED_BATCH_SIZE = 32
TOP_K           = 10

# Initialize embedding model
_embed_model = SentenceTransformer(EMBED_MODEL)

# Initialize HF Inference API client
HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")
if not HF_TOKEN:
    raise ValueError("Set HUGGINGFACE_HUB_TOKEN in your environment or .env")
print(f"[INFO] HF_MODEL = {HF_MODEL}")
_chat_infer = InferenceApi(
    repo_id=HF_MODEL,
    token=HF_TOKEN,
    task="text-generation"
)


# Load or initialize FAISS index + ID map
if os.path.exists(INDEX_PATH) and os.path.exists(IDMAP_PATH):
    index = faiss.read_index(INDEX_PATH)
    with open(IDMAP_PATH, "rb") as f:
        id_map: Dict[int, tuple[str, str]] = pickle.load(f)
else:
    dim = _embed_model.get_sentence_embedding_dimension()
    index = faiss.IndexFlatL2(dim)
    id_map = {}


def extract_text(file_bytes: bytes) -> str:
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except:
        pass
    try:
        doc = Document(io.BytesIO(file_bytes))
        return "\n".join(p.text for p in doc.paragraphs)
    except:
        pass
    return file_bytes.decode(errors="ignore")


def split_chunks(text: str, size: int = 1000, overlap: int = 200) -> List[str]:
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        start += size - overlap
    return chunks


def _persist():
    faiss.write_index(index, INDEX_PATH)
    with open(IDMAP_PATH, "wb") as f:
        pickle.dump(id_map, f)


def ingest_document(file_bytes: bytes) -> str:
    text    = extract_text(file_bytes)
    chunks  = split_chunks(text)
    embs    = _embed_model.encode(chunks, convert_to_numpy=True, batch_size=EMBED_BATCH_SIZE)

    start = index.ntotal
    index.add(embs)

    doc_id = str(uuid.uuid4())
    for i, chunk in enumerate(chunks):
        id_map[start + i] = (doc_id, chunk)

    _persist()
    return doc_id


@lru_cache(maxsize=128)
def query_doc(doc_id: str, question: str) -> str:
    # 1️⃣ Encode question and search
    print(f"[DEBUG] Encoding question: {question!r}")
    q_emb = _embed_model.encode([question], convert_to_numpy=True)
    _, inds = index.search(q_emb, TOP_K)
    print(f"[DEBUG] FAISS returned indices: {inds[0].tolist()}")

    # 2️⃣ Gather and dedupe context chunks
    seen, context = set(), []
    for idx in inds[0]:
        if idx in id_map:
            d_id, chunk = id_map[idx]
            print(f"[DEBUG] idx {idx} → doc_id {d_id}")
            if d_id == doc_id and chunk not in seen:
                context.append(chunk)
                seen.add(chunk)
    print(f"[DEBUG] Matched {len(context)} context chunks for doc_id {doc_id}")

    if not context:
        print("[WARN] No context chunks found; returning Not Found")
        return "Not Found"

    # 3️⃣ Build prompt
    prompt = (
        "You are a helpful assistant. Based only on the context below, answer the question concisely.\n\n"
        f"Context:\n{'\n\n'.join(context)}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )
    print(f"[DEBUG] Full prompt:\n{prompt}")

    # 4️⃣ Call HF Inference API
    raw = _chat_infer(inputs=prompt, raw_response=True)
    print(f"[DEBUG] Raw response status code: {raw.status_code}")
    print(f"[DEBUG] Raw response headers: {raw.headers}")
    


    # 5️⃣ Try JSON parsing
    text = None
    try:
        out = raw.json()
        print(f"[DEBUG] JSON output: {out}")
        if isinstance(out, list) and "generated_text" in out[0]:
            text = out[0]["generated_text"]
    except Exception as e:
        print(f"[WARN] JSON parsing failed: {e}")

    # 6️⃣ Fallback to plain text
    if text is None:
        text = raw.text
        print(f"[DEBUG] Fallback raw.text: {text!r}")

    answer = text.strip()
    print(f"[INFO] Final answer: {answer!r}")
    return answer

