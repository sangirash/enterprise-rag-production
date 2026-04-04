from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Literal


class Settings(BaseSettings):
    # LLM
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    openai_model: str = Field("gpt-4o", env="OPENAI_MODEL")
    embedding_model: str = Field("text-embedding-3-small", env="EMBEDDING_MODEL")
    embedding_dimension: int = Field(1536, env="EMBEDDING_DIMENSION")

    # Vector store selection
    vector_store: Literal["pgvector", "pinecone", "weaviate"] = Field("pgvector", env="VECTOR_STORE")

    # pgvector
    database_url: str = Field("postgresql://rag:rag@localhost:5432/ragdb", env="DATABASE_URL")

    # Pinecone
    pinecone_api_key: str = Field("", env="PINECONE_API_KEY")
    pinecone_index: str = Field("rag-production", env="PINECONE_INDEX")

    # Weaviate
    weaviate_url: str = Field("http://localhost:8080", env="WEAVIATE_URL")

    # Chunking
    chunk_strategy: Literal["fixed", "semantic", "recursive"] = Field("recursive", env="CHUNK_STRATEGY")
    chunk_size: int = Field(512, env="CHUNK_SIZE")
    chunk_overlap: int = Field(64, env="CHUNK_OVERLAP")

    # Retrieval
    top_k: int = Field(10, env="TOP_K")
    rerank_top_n: int = Field(4, env="RERANK_TOP_N")
    enable_reranking: bool = Field(True, env="ENABLE_RERANKING")
    enable_hybrid_search: bool = Field(True, env="ENABLE_HYBRID_SEARCH")
    hybrid_alpha: float = Field(0.7, env="HYBRID_ALPHA")  # 1.0 = dense only, 0.0 = BM25 only

    # Observability
    otlp_endpoint: str = Field("http://localhost:4317", env="OTLP_ENDPOINT")
    prometheus_port: int = Field(9090, env="PROMETHEUS_PORT")

    # API
    api_key: str = Field("changeme", env="RAG_API_KEY")
    log_level: str = Field("INFO", env="LOG_LEVEL")

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
