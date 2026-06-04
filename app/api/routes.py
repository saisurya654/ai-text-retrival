from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.models.schemas import ExtractRequest, FeedbackRequest, GenerateRequest, RetrieveRequest
from app.services.pipeline import LegalMindService
from app.utils.config import get_settings

router = APIRouter()
service = LegalMindService()
settings = get_settings()


@router.post("/upload")
async def upload(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".pdf", ".png", ".jpg", ".jpeg"}:
        raise HTTPException(status_code=400, detail="Unsupported file type.")

    temp_path = settings.upload_dir / (file.filename or "uploaded_file")
    content = await file.read()
    temp_path.write_bytes(content)
    return service.upload_document(temp_path)


@router.post("/extract")
def extract(request: ExtractRequest):
    try:
        return service.extract(request.document_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/retrieve")
def retrieve(request: RetrieveRequest):
    return service.retrieve(request.query, top_k=request.top_k, document_id=request.document_id)


@router.post("/generate")
def generate(request: GenerateRequest):
    return service.generate(request.draft_type, request.query, top_k=request.top_k, document_id=request.document_id)


@router.post("/feedback")
def feedback(request: FeedbackRequest):
    return service.record_feedback(request.original, request.edited, request.reason)


@router.get("/history")
def history():
    return service.history()


@router.get("/evaluation")
def evaluation():
    return service.evaluation()


@router.get("/documents")
def documents():
    return service.list_documents()


@router.get("/documents/{document_id}")
def document(document_id: str):
    chunks = service.get_document_chunks(document_id)
    if not chunks:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
    extraction = service.get_understanding(document_id)
    return {"document_id": document_id, "chunks": chunks, "extraction": extraction.model_dump() if extraction else None}
