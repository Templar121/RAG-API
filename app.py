from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag import ingest_document, query_doc

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
        "text/plain"
    ):
        raise HTTPException(400, "Unsupported file type")
    data = await file.read()
    doc_id = ingest_document(data)
    return {"doc_id": doc_id}

@app.post("/query/", summary="Ask a question against an indexed doc")
async def query(req: QueryRequest):
    answer = query_doc(req.doc_id, req.question)
    return {"answer": answer}

@app.get("/status", summary="Health check")
def health():
    return {"status": "ok"}