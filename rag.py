import os
import io
import uuid
import pickle
from typing import Dict, List
from functools import lru_cache

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import PyPDF2
from docx import Document

# Persistence paths
INDEX_PATH = "index.faiss"
IDMAP_PATH = "id_map.pkl"

# Models & Config
EMBED_MODEL = "all-MiniLM-L6-v2"
# switched to a public model – no HF login required
CHAT_MODEL = "EleutherAI/gpt-neo-125M"
EMBED_BATCH_SIZE = 32  # batch size for encoding
TOP_K = 5              # number of chunks to retrieve
MAX_NEW_TOKENS = 200   # limit for generation

# Initialize models
_embed_model = SentenceTransformer(EMBED_MODEL)
_chat_model = pipeline(
    "text-generation",
    model=CHAT_MODEL,
    max_new_tokens=MAX_NEW_TOKENS,
    do_sample=False
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
    # Try PDF
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except:
        pass
    # Try DOCX
    try:
        doc = Document(io.BytesIO(file_bytes))
        return "\n".join(p.text for p in doc.paragraphs)
    except:
        pass
    # Fallback to plain text
    return file_bytes.decode(errors="ignore")

def split_chunks(text: str, size: int = 1000, overlap: int = 200) -> List[str]:
    chunks, start = [], 0
    L = len(text)
    while start < L:
        end = min(start + size, L)
        chunks.append(text[start:end])
        start += size - overlap
    return chunks

def _persist():
    faiss.write_index(index, INDEX_PATH)
    with open(IDMAP_PATH, "wb") as f:
        pickle.dump(id_map, f)

def ingest_document(file_bytes: bytes) -> str:
    text = extract_text(file_bytes)
    chunks = split_chunks(text)
    embeddings = _embed_model.encode(chunks, convert_to_numpy=True, batch_size=EMBED_BATCH_SIZE)

    start = index.ntotal
    index.add(embeddings)

    doc_id = str(uuid.uuid4())
    for i, chunk in enumerate(chunks):
        id_map[start + i] = (doc_id, chunk)

    _persist()
    return doc_id

@lru_cache(maxsize=128)
def query_doc(doc_id: str, question: str) -> str:
    q_emb = _embed_model.encode([question], convert_to_numpy=True)
    _, inds = index.search(q_emb, TOP_K)

    # Collect relevant chunks
    context_chunks = [
        id_map[idx][1]
        for idx in inds[0]
        if id_map.get(idx, (None,))[0] == doc_id
    ]
    prompt = (
        "Context:\n" +
        "\n\n".join(context_chunks) +
        f"\n\nQ: {question}\nA:"
    )

    generation = _chat_model(prompt)
    text = generation[0]["generated_text"]
    # strip everything before the actual answer
    return text.split("A:")[-1].strip()
