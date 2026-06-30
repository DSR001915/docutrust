"""
Centralized configuration for DocuTrust.

Loads from environment variables / a .env file. All CRAG tuning knobs
(thresholds, top-k, iteration caps) live here so they're easy to find
and adjust without hunting through node code.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Provider mode switches ---
    # "mock": deterministic, dependency-free stand-ins (no model downloads,
    #         no API keys, no live network calls) -- the default, and what
    #         this repo's own tests + eval (backend/eval/run_eval.py) run
    #         against.
    # "local"/"anthropic"/"ollama"/"tavily": real providers -- see the
    #         get_embedder() / get_reranker() / get_llm() / get_web_search()
    #         factories in their respective modules for exact behavior and
    #         graceful fallback if initialization fails.
    docutrust_provider_mode: str = "mock"  # "mock" | "local" -- governs embedder + reranker
    llm_provider: str = "mock"             # "mock" | "anthropic" | "ollama"
    web_search_provider: str = "mock"      # "mock" | "tavily"

    # --- LLM provider ---
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-6"
    ollama_model: str = "llama3.1:8b"
    ollama_host: str = "http://localhost:11434"

    # --- Embeddings / reranker ---
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    # --- Vector store ---
    chroma_persist_dir: str = "./chroma_data"
    chroma_collection: str = "docutrust_chunks"

    # --- MongoDB ---
    mongo_uri: str = "mongodb://localhost:27017"
    mongo_db_name: str = "docutrust"

    # --- Web fallback search ---
    tavily_api_key: str = ""

    # --- CRAG tuning knobs ---
    retrieval_top_k: int = 20          # candidates pulled by bi-encoder before reranking
    rerank_top_k: int = 5              # candidates kept after cross-encoder reranking
    relevance_threshold_correct: float = 0.6     # >= this -> grade "correct"
    relevance_threshold_ambiguous: float = 0.3   # >= this (but < correct) -> "ambiguous"
    max_correction_iterations: int = 2           # hard cap on rewrite->retrieve loops

    # --- Chunking ---
    chunk_size: int = 800
    chunk_overlap: int = 120

    # --- Server ---
    app_host: str = "0.0.0.0"
    app_port: int = 8000


@lru_cache
def get_settings() -> Settings:
    return Settings()
