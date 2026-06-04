from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "LegalMind AI"
    environment: str = "development"
    data_dir: Path = Path("data")
    upload_dir: Path = Path("data/uploads")
    index_dir: Path = Path("data/index")
    document_store_dir: Path = Path("data/documents")
    extraction_store_dir: Path = Path("data/extractions")
    feedback_store: Path = Path("data/feedback/history.json")
    evaluation_store: Path = Path("samples/outputs/evaluation_report.json")
    chunk_size: int = 800
    chunk_overlap: int = 120
    retrieval_top_k: int = 5
    enable_llm_fallback: bool = False
    openai_api_key: str = ""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="LEGALMIND_")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.index_dir.mkdir(parents=True, exist_ok=True)
    settings.document_store_dir.mkdir(parents=True, exist_ok=True)
    settings.extraction_store_dir.mkdir(parents=True, exist_ok=True)
    settings.feedback_store.parent.mkdir(parents=True, exist_ok=True)
    return settings
