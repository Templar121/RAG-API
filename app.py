from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# import your RAG internals
from rag import (
    ingest_document,
    query_doc,
    embed_model,
    index,
    id_map,
    TOP_K,
    CONTEXT_CHUNK_LIMIT,
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

@app.get("/status", summary="Health check")
def health():
    return {"status": "ok"}

@app.post("/debug_chunks/", summary="Inspect what chunks are retrieved")
async def debug_chunks(req: QueryRequest):
    # Encode and search
    q_emb = embed_model.encode([req.question], convert_to_numpy=True).astype("float32")
    D, I = index.search(q_emb, TOP_K)

    # Gather up to CONTEXT_CHUNK_LIMIT chunks for this doc
    seen, context = set(), []
    for vid in I[0]:
        if vid in id_map:
            d_id, chunk = id_map[vid]
            if d_id == req.doc_id and vid not in seen:
                context.append(chunk)
                seen.add(vid)
                if len(context) >= CONTEXT_CHUNK_LIMIT:
                    break

    if not context:
        return {"chunks": [], "warning": "No matching chunks found for this doc_id"}
    return {"chunks": context}
