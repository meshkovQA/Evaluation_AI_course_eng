from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional, Dict, Any
from datetime import datetime
from app.config import settings

from app.models.chat import (
    ChatRequest, ChatResponse,
    RelevantChunk
)
from app.services.embedding_service import get_embedding_service
from app.services.vector_store import get_vector_store_service
from app.services.llm_service import get_llm_service
from app.services.document_service import DocumentService, get_document_service as get_global_service
import logging
from app.utils.auth import verify_api_key

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])

def get_document_service() -> DocumentService:
    return get_global_service()


@router.post("/", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    embedding_service=Depends(get_embedding_service),
    vector_store=Depends(get_vector_store_service),
    llm_service=Depends(get_llm_service),
    document_service: DocumentService = Depends(get_document_service),
    api_key: str = Depends(verify_api_key)
):
    """
    Sends a message to the RAG chat

    - **message**: message text
    - **document_ids**: specific documents to search (optional)
    - **max_relevant_chunks**: maximum number of relevant chunks
    - **similarity_threshold**: minimum similarity threshold
    - **temperature**: LLM generation temperature
    - **llm_provider**: LLM provider (openai, anthropic)
    - **model_name**: specific model (gpt-4, claude-3-sonnet-20240229, etc.)
    """

    if not api_key:
        raise HTTPException(status_code=401, detail="Unauthorized")
    try:
        relevant_chunks = await find_relevant_chunks(
            request.message,
            request,
            embedding_service,
            vector_store,
            document_service
        )

        rag_response = await llm_service.generate_rag_response(
            query=request.message,
            relevant_chunks=relevant_chunks,
            temperature=request.temperature or 0.7,
            max_tokens=None,
            llm_provider=request.llm_provider.value if request.llm_provider else None,
            model_name=request.model_name
        )

        sources = []
        for chunk in relevant_chunks:
            metadata = chunk.get('metadata', {})
            sources.append({
                "chunk_id": chunk.get('chunk_id'),
                "document_id": metadata.get('document_id'),
                "document_title": metadata.get('source', 'Unknown source'),
                "similarity": chunk.get('similarity', 0),
                "content": chunk['text']  # Full text without truncation
            })

        return ChatResponse(
            message_id="msg_" + str(int(datetime.utcnow().timestamp())),
            session_id="single_session",
            content=rag_response["answer"],
            sources=sources,
            metadata={
                "sources_count": len(sources),
                "context_length": rag_response["context_length"],
                "llm_provider": rag_response["llm_provider"],
                "model_used": rag_response["model_used"],
                "token_usage": rag_response["token_usage"]
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing query: {str(e)}")


async def find_relevant_chunks(
    query: str,
    request: ChatRequest,
    embedding_service,
    vector_store,
    document_service: DocumentService,
    api_key: str = Depends(verify_api_key)
) -> List[Dict[str, Any]]:
    """Finds relevant chunks for a query"""

    if not api_key:
        raise HTTPException(status_code=401, detail="Unauthorized")
    try:
        threshold = request.similarity_threshold if request.similarity_threshold is not None else settings.SIMILARITY_THRESHOLD
        max_chunks = request.max_relevant_chunks if request.max_relevant_chunks is not None else settings.MAX_RELEVANT_CHUNKS

        logger.info(f"🔍 Search: '{query}' (threshold={threshold}, max={max_chunks})")

        query_embedding = await embedding_service.create_embedding_for_query(query)
        logger.info(f"✅ Embedding created: {len(query_embedding)} dimensions")

        relevant_chunks = await vector_store.search_similar_chunks(
            query_embedding=query_embedding,
            top_k=max_chunks,
            similarity_threshold=threshold
        )

        logger.info(f"📊 Found chunks: {len(relevant_chunks)}")

        if request.document_ids:
            relevant_chunks = [
                chunk for chunk in relevant_chunks
                if chunk.get('metadata', {}).get('document_id') in request.document_ids
            ]
            logger.info(f"📊 After filtering: {len(relevant_chunks)}")

        for chunk in relevant_chunks:
            doc_id = chunk.get('metadata', {}).get('document_id')
            document = document_service.get_document(doc_id)
            if document:
                chunk['metadata']['source'] = document.title
                chunk['metadata']['document_type'] = document.document_type.value

            logger.info(f"   Chunk: similarity={chunk.get('similarity', 0):.4f}, doc={chunk['metadata'].get('source', 'Unknown')}")

        return relevant_chunks

    except Exception as e:
        logger.error(f"❌ Search error: {e}", exc_info=True)
        return []
