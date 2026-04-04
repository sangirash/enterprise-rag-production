from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

import structlog
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app

from .middleware import RequestLoggingMiddleware
from .routes import ingest, query
from ..config import settings

logger = structlog.get_logger()


def verify_api_key(x_api_key: str = Header(..., alias="X-API-Key")) -> None:
    if x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("rag_api_starting", vector_store=settings.vector_store, model=settings.openai_model)
    yield
    logger.info("rag_api_shutdown")


app = FastAPI(
    title="Enterprise RAG API",
    description="Production-grade Retrieval-Augmented Generation pipeline",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
app.add_middleware(RequestLoggingMiddleware)

app.include_router(query.router, prefix="/api/v1", dependencies=[Depends(verify_api_key)])
app.include_router(ingest.router, prefix="/api/v1", dependencies=[Depends(verify_api_key)])

# Expose Prometheus metrics at /metrics (no auth -- typically firewall-gated)
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


@app.get("/health", tags=["ops"], summary="Health check")
async def health() -> dict:
    return {"status": "ok", "vector_store": settings.vector_store, "version": "0.1.0"}
