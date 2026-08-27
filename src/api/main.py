import io
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ..parsers import parse_document
from ..summarizer import generate_extractive_summary, generate_abstractive_summary
from ..entities import extract_entities, extract_keywords
from ..analytics import compute_document_analytics
from ..rag import split_document_pages, VectorStore, RAGEngine
from ..exporters import generate_docx_report, generate_pdf_report
from ..evaluation import evaluate_summary, evaluate_dataset
from ..config import PORT, HOST

app = FastAPI(
    title="DocuMind AI NLP Service",
    description="Microservice providing document parsing, dual summarisation, RAG Q&A, NER, analytics, and report exports.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Active in-memory session index store: doc_id -> VectorStore
_session_stores: Dict[str, VectorStore] = {}

class TextRequest(BaseModel):
    text: str
    filename: Optional[str] = "document.txt"

class SummarizeRequest(BaseModel):
    text: str
    extractive_sentences: Optional[int] = 5
    extractive_method: Optional[str] = "lexrank"
    abstractive_format: Optional[str] = "paragraph"
    abstractive_length: Optional[str] = "standard"
    focus_topic: Optional[str] = None

class RAGQueryRequest(BaseModel):
    doc_id: str
    question: str
    top_k: Optional[int] = 4
    chat_history: Optional[List[Dict[str, str]]] = None

class EvaluationRequest(BaseModel):
    reference: str
    candidate: str

class ExportRequest(BaseModel):
    document_info: Dict[str, Any]
    extractive_summary: Optional[Dict[str, Any]] = None
    abstractive_summary: Optional[Dict[str, Any]] = None
    entities_data: Optional[Dict[str, Any]] = None
    keywords_data: Optional[List[Dict[str, Any]]] = None
    analytics_data: Optional[Dict[str, Any]] = None
    qa_history: Optional[List[Dict[str, Any]]] = None

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "documind-nlp"}

@app.post("/api/v1/parse")
async def parse_file(file: UploadFile = File(...)):
    try:
        content = await file.read()
        parsed = parse_document(content, file.filename)
        return {"success": True, "data": parsed}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/v1/summarize")
def summarize(req: SummarizeRequest):
    try:
        extractive = generate_extractive_summary(
            req.text,
            sentence_count=req.extractive_sentences,
            method=req.extractive_method
        )
        abstractive = generate_abstractive_summary(
            req.text,
            format_type=req.abstractive_format,
            max_length=req.abstractive_length,
            focus_topic=req.focus_topic
        )
        return {
            "success": True,
            "data": {
                "extractive": extractive,
                "abstractive": abstractive
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/entities")
def get_entities(req: TextRequest):
    try:
        entities = extract_entities(req.text)
        keywords = extract_keywords(req.text)
        return {
            "success": True,
            "data": {
                "entities": entities,
                "keywords": keywords
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/analytics")
def get_analytics(req: TextRequest):
    try:
        analytics = compute_document_analytics(req.text, req.filename)
        return {"success": True, "data": analytics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/rag/index")
async def index_document_for_rag(file: UploadFile = File(...), doc_id: str = Form(...)):
    try:
        content = await file.read()
        parsed = parse_document(content, file.filename)
        chunks = split_document_pages(parsed["pages"])
        
        vs = VectorStore()
        vs.build_index(chunks)
        _session_stores[doc_id] = vs
        
        return {
            "success": True,
            "data": {
                "doc_id": doc_id,
                "total_chunks": len(chunks),
                "page_count": parsed["page_count"]
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/v1/rag/query")
def query_rag(req: RAGQueryRequest):
    if req.doc_id not in _session_stores:
        raise HTTPException(status_code=404, detail=f"Document session '{req.doc_id}' not found in vector store.")
    
    vs = _session_stores[req.doc_id]
    engine = RAGEngine(vs)
    res = engine.answer_question(req.question, top_k=req.top_k, chat_history=req.chat_history)
    return {"success": True, "data": res}

@app.post("/api/v1/export/docx")
def export_docx(req: ExportRequest):
    try:
        docx_bytes = generate_docx_report(
            document_info=req.document_info,
            extractive_summary=req.extractive_summary,
            abstractive_summary=req.abstractive_summary,
            entities_data=req.entities_data,
            keywords_data=req.keywords_data,
            analytics_data=req.analytics_data,
            qa_history=req.qa_history
        )
        return Response(
            content=docx_bytes,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={"Content-Disposition": f"attachment; filename=DocuMind_{req.document_info.get('filename', 'report')}.docx"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/export/pdf")
def export_pdf(req: ExportRequest):
    try:
        pdf_bytes = generate_pdf_report(
            document_info=req.document_info,
            extractive_summary=req.extractive_summary,
            abstractive_summary=req.abstractive_summary,
            entities_data=req.entities_data,
            keywords_data=req.keywords_data,
            analytics_data=req.analytics_data,
            qa_history=req.qa_history
        )
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=DocuMind_{req.document_info.get('filename', 'report')}.pdf"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/evaluate")
def evaluate(req: EvaluationRequest):
    try:
        scores = evaluate_summary(req.reference, req.candidate)
        return {"success": True, "data": scores}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("documind_nlp.src.api.main:app", host=HOST, port=PORT, reload=True)
