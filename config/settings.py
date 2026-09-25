"""Centralized settings and configuration management."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env file explicitly if it exists
load_dotenv(BASE_DIR / ".env")


class Settings(BaseSettings):
    """Application settings schema with sensible defaults and environment variable overrides."""

    # Project directories
    base_dir: Path = BASE_DIR
    data_dir: Path = BASE_DIR / "data"
    transcripts_dir: Path = BASE_DIR / "data" / "transcripts"
    vectorstores_dir: Path = BASE_DIR / "data" / "vectorstores"

    # LLM Settings
    llm_provider: Literal["groq", "ollama", "huggingface", "openai_compatible", "mock"] = Field(
        default="groq",
        description="LLM provider to use for generating answers."
    )
    model_name: str = Field(
        default="llama-3.1-8b-instant",
        description="Model name to use with the selected provider."
    )
    temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Sampling temperature for the LLM."
    )
    max_tokens: int = Field(
        default=1024,
        ge=64,
        description="Maximum tokens for generation."
    )

    # API Keys & URLs
    groq_api_key: str = Field(default="", description="API key for Groq Cloud")
    huggingface_api_token: str = Field(default="", description="Hugging Face User Access Token")
    openai_api_key: str = Field(default="", description="API key for OpenAI-compatible endpoint")
    openai_base_url: str = Field(default="https://api.together.xyz/v1", description="Base URL for OpenAI-compatible endpoint")
    ollama_base_url: str = Field(default="http://localhost:11434", description="Base URL for local Ollama server")

    # Embedding Settings
    embedding_model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="HuggingFace / Sentence-Transformers model for embeddings."
    )

    # Chunking & RAG Retrieval Settings
    chunk_size: int = Field(
        default=1000,
        ge=100,
        description="Target character size for each transcript chunk."
    )
    chunk_overlap: int = Field(
        default=150,
        ge=0,
        description="Character overlap between consecutive chunks."
    )
    retrieval_candidates: int = Field(
        default=10,
        ge=4,
        le=50,
        description="Number of candidate chunks to search across the transcript before relevance selection."
    )
    top_k: int = Field(
        default=4,
        ge=1,
        le=20,
        description="Maximum number of final supporting source chunks to pass to the LLM and display."
    )

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    def init_directories(self) -> None:
        """Ensure necessary data directories exist on disk."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.transcripts_dir.mkdir(parents=True, exist_ok=True)
        self.vectorstores_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton instance of Settings."""
    settings = Settings()
    settings.init_directories()
    return settings
