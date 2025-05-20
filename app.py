from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sqlite3
import numpy as np

# import your RAG internals
from rag import (
    ingest_document,
    query_doc,
    get_embed_model,
    index,
    TOP_K,
    CONTEXT_CHUNK_LIMIT,
    SQLITE_PATH,
)

app = FastAPI(title="Free RAG API")

# Enable CORS for all origins (adjust as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    doc_id: str
    question: str

# Set up SQLite connection for debugging
conn = sqlite3.connect(SQLITE_PATH, check_same_thread=False)
cursor = conn.cursor()

@app.post("/ingest/", summary="Upload and index a document")
async def ingest(file: UploadFile = File(...)):
    if file.content_type not in (
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "text/plain",
    ):
        raise HTTPException(400, "Unsupported file type")
    data = await file.read()
    doc_id = ingest_document(data)
    if not doc_id:
        raise HTTPException(500, "Failed to ingest document")
    return {"doc_id": doc_id}

@app.post("/query/", summary="Ask a question against an indexed doc")
async def query(req: QueryRequest):
    answer = query_doc(req.doc_id, req.question)
    return {"answer": answer}

@app.get("/", summary="Health check")
def health():
    return {"status": "Running"}

@app.post("/debug_chunks/", summary="Inspect what chunks are retrieved")
async def debug_chunks(req: QueryRequest):
    # Embed and search using the same process as query
    model = get_embed_model()
    q_emb = model.encode([req.question], convert_to_numpy=True).astype("float32")
    D, I = index.search(q_emb, TOP_K)

    candidate_vids = [int(vid) for vid in I[0] if vid >= 0]
    if not candidate_vids:
        return {"chunks": [], "warning": "No matching chunks found for this doc_id"}

    # Pull matching chunks from SQLite
    placeholders = ",".join("?" for _ in candidate_vids)
    cursor.execute(
        f"SELECT vid, doc_id, text FROM chunks WHERE vid IN ({placeholders})",
        candidate_vids
    )
    rows = cursor.fetchall()  # [(vid, doc_id, text), ...]

    # Filter and collect up to CONTEXT_CHUNK_LIMIT for this doc_id
    seen = set()
    context = []
    for vid, d_id, chunk_text in rows:
        if d_id == req.doc_id and vid not in seen:
            context.append(chunk_text)
            seen.add(vid)
            if len(context) >= CONTEXT_CHUNK_LIMIT:
                break

    if not context:
        return {"chunks": [], "warning": "No matching chunks found for this doc_id"}
    return {"chunks": context}
