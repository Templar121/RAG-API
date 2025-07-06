import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables for rag.py
load_dotenv()

# Import RAG logic and MongoDB client
from rag import ingest_document, query_doc, debug_chunks as rag_debug_chunks, chunks_coll

# Initialize FastAPI app
app = FastAPI(title="Free RAG API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model for queries
class QueryRequest(BaseModel):
    doc_id: str
    question: str

@app.get("/", summary="Health check")
def health():
    return {"status": "Running"}

@app.get("/db_check", summary="Database connection check")
def db_check():
    """
    Simple endpoint to verify MongoDB connectivity.
    """
    try:
        # Ping the server; raises if unreachable
        chunks_coll.database.client.admin.command('ping')
        return {"db": "connected"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB connection error: {e}")

@app.post("/ingest/", summary="Upload and index a document")
async def ingest_endpoint(file: UploadFile = File(...)):
    # Only allow certain file types
    if file.content_type not in (
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "text/plain",
    ):
        raise HTTPException(status_code=400, detail="Unsupported file type")
    data = await file.read()
    doc_id = ingest_document(data)
    if not doc_id:
        raise HTTPException(status_code=500, detail="Failed to ingest document")
    return {"doc_id": doc_id}

@app.post("/query/", summary="Ask a question against an indexed doc")
async def query_endpoint(req: QueryRequest):
    answer = query_doc(req.doc_id, req.question)
    return {"answer": answer}

@app.post("/debug_chunks/", summary="Inspect retrieved chunks for debugging")
async def debug_chunks_endpoint(req: QueryRequest):
    # Delegate to rag.debug_chunks for chunk inspection
    chunks, warning = rag_debug_chunks(req.doc_id, req.question)
    response = {"chunks": chunks}
    if warning:
        response["warning"] = warning
    return response
