from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import uvicorn
from datetime import datetime
from contextlib import asynccontextmanager

from app.config import settings, get_storage_paths, validate_settings
from app.api.documents import router as documents_router
from app.api.chat import router as chat_router


# Logging configuration
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format=settings.LOG_FORMAT
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management"""
    # Startup
    logger.info("Starting RAG System API...")

    try:
        # Validate settings
        validate_settings()
        logger.info("✓ Settings validated")

        # Creating required directories
        get_storage_paths()
        logger.info("✓ Storage directories created")

        logger.info("🚀 RAG System API started successfully")

    except Exception as e:
        logger.error(f"❌ Startup error: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down RAG System API...")


# Create the FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="RAG System API for document upload and chat",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS configuration - ALLOW ALL
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)


# Main routes
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "RAG System API",
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """System health check"""
    try:
        services_status = {}

        # Check the embedding service
        try:
            from app.services.embedding_service import get_embedding_service
            embedding_service = get_embedding_service()
            services_status["embeddings"] = await embedding_service.test_connection()
        except Exception as e:
            services_status["embeddings"] = False
            logger.error(f"Embedding service error: {e}")

        # Check the vector store
        try:
            from app.services.vector_store import get_vector_store_service
            vector_store = get_vector_store_service()
            services_status["vector_store"] = await vector_store.test_connection()
        except Exception as e:
            services_status["vector_store"] = False
            logger.error(f"Vector store error: {e}")

        # Check the LLM service
        try:
            from app.services.llm_service import get_llm_service
            llm_service = get_llm_service()
            services_status["llm"] = await llm_service.test_connection()
        except Exception as e:
            services_status["llm"] = False
            logger.error(f"LLM service error: {e}")

        # Determine overall status
        all_healthy = all(services_status.values())
        status_code = 200 if all_healthy else 503

        return JSONResponse(
            status_code=status_code,
            content={
                "status": "healthy" if all_healthy else "degraded",
                "version": settings.APP_VERSION,
                "timestamp": datetime.utcnow().isoformat(),
                "services": {
                    "api": "running",
                    "storage": "available",
                    "embeddings": "connected" if services_status.get("embeddings") else "disconnected",
                    "vector_store": "connected" if services_status.get("vector_store") else "disconnected",
                    "llm": "connected" if services_status.get("llm") else "disconnected"
                }
            }
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e)
            }
        )


@app.get("/info")
async def get_info():
    """System information"""
    return {
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "supported_formats": settings.ALLOWED_FILE_EXTENSIONS,
        "max_file_size": f"{settings.MAX_FILE_SIZE / (1024 * 1024):.1f} MB",
        "chunk_size": settings.DEFAULT_CHUNK_SIZE,
        "chunk_overlap": settings.DEFAULT_CHUNK_OVERLAP,
        "text_splitter": settings.TEXT_SPLITTER_TYPE,
        "llm_provider": settings.DEFAULT_LLM_PROVIDER,
        "embedding_provider": settings.EMBEDDING_PROVIDER,
        "vector_db": settings.VECTOR_DB_TYPE
    }


# Register routes
app.include_router(documents_router, prefix=settings.API_PREFIX)
app.include_router(chat_router, prefix=settings.API_PREFIX)


if __name__ == "__main__":
    # Development server startup
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )