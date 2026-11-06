"""All settings come from environment variables (or .env). Nothing model-specific is hardcoded."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", env_file_encoding="utf-8", extra="ignore")

    # Groq (primary)
    groq_api_key: str = ""
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_large_model: str = "llama-3.3-70b-versatile"
    groq_small_model: str = "llama-3.1-8b-instant"
    groq_daily_token_budget: int = 400_000

    # NVIDIA NIM (fallback, limited credits)
    nvidia_api_key: str = ""
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    nvidia_large_model: str = "meta/llama-3.3-70b-instruct"
    nvidia_small_model: str = "meta/llama-3.1-8b-instruct"
    nvidia_daily_token_budget: int = 100_000

    # Local models
    embed_model: str = "BAAI/bge-small-en-v1.5"
    rerank_model: str = "BAAI/bge-reranker-base"

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "policylens-dev"

    # Retrieval and answering
    retrieve_top_k: int = 10
    rerank_top_n: int = 6
    abstain_threshold: float = 0.45

    # Chunking (tokens, cl100k)
    chunk_max_tokens: int = 700
    chunk_min_tokens: int = 120
    chunk_overlap_tokens: int = 80

    api_url: str = "http://localhost:8000"

    # Paths
    data_dir: Path = ROOT / "data"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"


@lru_cache
def get_settings() -> Settings:
    return Settings()
