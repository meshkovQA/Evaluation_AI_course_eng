import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""

    # Main settings
    APP_NAME: str = "RAG System API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # API settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8002
    API_PREFIX: str = "/api/v1"

    # Document storage
    DOCUMENTS_STORAGE_PATH: str = "storage/documents"
    VECTOR_DB_PATH: str = "storage/vector_db"

    # File upload limits
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB
    ALLOWED_FILE_EXTENSIONS: list = [
        ".pdf", ".docx", ".txt", ".csv",
        ".json", ".html", ".md", ".xlsx"
    ]

    # Text splitting settings
    DEFAULT_CHUNK_SIZE: int = 1000
    DEFAULT_CHUNK_OVERLAP: int = 200
    # recursive, sentence, paragraph, fixed, markdown
    TEXT_SPLITTER_TYPE: str = "recursive"

    # LLM settings
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    DEFAULT_LLM_PROVIDER: str = "openai"  # openai, anthropic, local
    DEFAULT_LLM_MODEL: str = "gpt-4o-mini"

    # Embedding settings
    EMBEDDING_PROVIDER: str = "openai"  # openai, huggingface, local
    EMBEDDING_MODEL: str = "text-embedding-ada-002"
    EMBEDDING_DIMENSION: int = 1536

    # Vector database
    VECTOR_DB_TYPE: str = "chroma"  # chroma, pinecone, faiss
    CHROMA_PERSIST_DIRECTORY: str = "storage/vector_db/chroma"

    # Pinecone settings (if used)
    PINECONE_API_KEY: Optional[str] = None
    PINECONE_ENVIRONMENT: Optional[str] = None
    PINECONE_INDEX_NAME: str = "rag-documents"

    # CORS settings
    CORS_ORIGINS: list = ["*"]
    CORS_CREDENTIALS: bool = True
    CORS_METHODS: list = ["*"]
    CORS_HEADERS: list = ["*"]

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Database (if used later)
    DATABASE_URL: Optional[str] = None

    # Redis (for caching, if needed)
    REDIS_URL: Optional[str] = None

    # Security
    SECRET_KEY: str = "supersecretkey"

    # RAG settings
    MAX_RELEVANT_CHUNKS: int = 5
    SIMILARITY_THRESHOLD: float = 0.0
    MAX_CONTEXT_LENGTH: int = 4000

    # Async processing
    MAX_CONCURRENT_UPLOADS: int = 5
    PROCESSING_TIMEOUT: int = 300  # 5 minutes

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Create a global settings instance
settings = Settings()


def get_storage_paths():
    """Creates the directories required for storage"""
    paths = [
        settings.DOCUMENTS_STORAGE_PATH,
        settings.VECTOR_DB_PATH,
        settings.CHROMA_PERSIST_DIRECTORY
    ]

    for path in paths:
        os.makedirs(path, exist_ok=True)

    return paths


def validate_settings():
    """Validates the settings"""
    errors = []

    # Check API keys for LLM
    if settings.DEFAULT_LLM_PROVIDER == "openai" and not settings.OPENAI_API_KEY:
        errors.append(
            "OPENAI_API_KEY is not set, but OpenAI is selected as provider")

    if settings.DEFAULT_LLM_PROVIDER == "anthropic" and not settings.ANTHROPIC_API_KEY:
        errors.append(
            "ANTHROPIC_API_KEY is not set, but Anthropic is selected as provider")

    # Check embeddings settings
    if settings.EMBEDDING_PROVIDER == "openai" and not settings.OPENAI_API_KEY:
        errors.append("OPENAI_API_KEY is not set for embeddings")

    # Check Pinecone settings
    if settings.VECTOR_DB_TYPE == "pinecone":
        if not settings.PINECONE_API_KEY:
            errors.append("PINECONE_API_KEY is not set")
        if not settings.PINECONE_ENVIRONMENT:
            errors.append("PINECONE_ENVIRONMENT is not set")

    # Check chunk sizes
    if settings.DEFAULT_CHUNK_SIZE <= settings.DEFAULT_CHUNK_OVERLAP:
        errors.append("CHUNK_SIZE must be greater than CHUNK_OVERLAP")

    if errors:
        raise ValueError("Configuration errors:\n" + "\n".join(errors))

    return True


# Mapping from file extensions to MIME types
FILE_TYPE_MAPPING = {
    '.pdf': 'application/pdf',
    '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    '.doc': 'application/vnd.ms-word',
    '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    '.xls': 'application/vnd.ms-excel',
    '.txt': 'text/plain',
    '.csv': 'text/csv',
    '.json': 'application/json',
    '.html': 'text/html',
    '.htm': 'text/html',
    '.md': 'text/markdown',
    '.markdown': 'text/x-markdown'
}


def get_file_mime_type(filename: str) -> Optional[str]:
    """Returns the MIME type of a file based on its extension"""
    ext = os.path.splitext(filename)[1].lower()
    return FILE_TYPE_MAPPING.get(ext)
