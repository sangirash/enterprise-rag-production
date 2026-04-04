from __future__ import annotations

from fastapi import APIRouter, File, Form, UploadFile
from pydantic import BaseModel

from ...ingestion.pipeline import IngestionPipeline

router = APIRouter(tags=["ingestion"])
_ingestion = IngestionPipeline()


class IngestResponse(BaseModel):
    source_id: str
    chunks_created: int
    status: str


@router.post("/ingest", response_model=IngestResponse, summary="Ingest a document into the vector store")
async def ingest_file(
    file: UploadFile = File(..., description="Document file (PDF, DOCX, HTML, TXT)"),
    source_id: str = Form(..., description="Unique identifier for this document"),
) -> IngestResponse:
    content = await file.read()
    result = await _ingestion.ingest(
        content=content,
        filename=file.filename or "unknown",
        source_id=source_id,
    )
    return IngestResponse(
        source_id=source_id,
        chunks_created=result["chunks_created"],
        status="success",
    )
