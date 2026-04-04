from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from ...pipeline import RAGPipeline

router = APIRouter(tags=["query"])
_pipeline = RAGPipeline()


class QueryRequest(BaseModel):
    query: str
    top_k: int | None = None
    enable_reranking: bool | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    query: str
    retrieval_count: int


@router.post("/query", response_model=QueryResponse, summary="Answer a question using the RAG pipeline")
async def query_endpoint(request: QueryRequest) -> QueryResponse:
    result = await _pipeline.query(
        query=request.query,
        top_k=request.top_k,
        enable_reranking=request.enable_reranking,
    )
    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"],
        query=request.query,
        retrieval_count=result["retrieval_count"],
    )
