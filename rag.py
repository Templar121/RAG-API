import os
import io
import uuid
import pickle
from typing import Dict, List
from functools import lru_cache

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import PyPDF2
from docx import Document
from dotenv import load_dotenv
import requests

load_dotenv()  # loads GOOGLE_API_KEY

# Persistence paths
INDEX_PATH = "index.faiss"
IDMAP_PATH  = "id_map.pkl"

# Models & Config
EMBED_MODEL     = "all-MiniLM-L6-v2"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Set GOOGLE_API_KEY in your environment or .env")

EMBED_BATCH_SIZE = 32
TOP_K           = 20

# Initialize embedding model
_embed_model = SentenceTransformer(EMBED_MODEL)

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

    # 3️⃣ Build prompt text
    prompt_text = (
    "You are a helpful assistant. Based only on the context below, answer the question concisely.\n\n"
    f"Context:\n{'\n\n'.join(context)}\n\n"
    f"Question: {question}\n"
    "Answer:"
)
    print(f"[DEBUG] Full prompt:\n{prompt_text}")

    # 4️⃣ Call Google Gemini API for text generation
    # Use the Gemini API endpoint and model
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GOOGLE_API_KEY}"
    # You can also use other Gemini models like 'gemini-1.0-pro-001', 'gemini-1.5-pro-latest', etc.
    # Refer to the official documentation for available models and their capabilities.

    headers = {
        "Content-Type": "application/json; charset=utf-8",
    }

    payload = {
        "contents": [{
            "parts": [{"text": prompt_text}]
        }],
        # Add the generationConfig object for Gemini API
        "generationConfig": {
            "maxOutputTokens": 200, # Set your desired max output tokens here
            "temperature": 0.2, # Example temperature
            # You can add other parameters like topP, topK here if needed
        }
    }

    response = requests.post(url, headers=headers, json=payload)
    print(f"[DEBUG] Google API response status: {response.status_code}")

    answer = "Sorry, I couldn't generate an answer."

    if response.status_code == 200:
        data = response.json()
        # Response typically has: data['candidates'][0]['message']['content']['text']
        candidates = data.get("candidates")
        if candidates and len(candidates) > 0:
            message = candidates[0].get("message", {})
            content = message.get("content", {})
            answer = content.get("text", "").strip()
        print(f"[INFO] Final answer: {answer!r}")
    else:
        print(f"[ERROR] Google API error: {response.text}")

    return answer

