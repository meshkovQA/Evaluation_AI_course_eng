from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Query
from typing import List, Optional
from app.models.document import (
    Document, DocumentUploadRequest, DocumentUpdateRequest,
    DocumentUploadResponse, DocumentListResponse, DocumentDetailResponse,
    DocumentType, DocumentStatus, DocumentSearchRequest
)
from app.utils.auth import verify_api_key
from app.services.document_service import DocumentService, get_document_service as get_global_service

router = APIRouter(prefix="/documents", tags=["documents"])


def get_document_service() -> DocumentService:
    return get_global_service()


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    title: Optional[str] = None,
    description: Optional[str] = None,
    tags: Optional[str] = None,  # Tags as comma-separated string
    service: DocumentService = Depends(get_document_service),
    api_key: str = Depends(verify_api_key)
):
    """
    Uploads a document

    - **file**: file to upload (PDF, DOCX, TXT, CSV, JSON, HTML, MD, XLSX)
    - **title**: document title (optional)
    - **description**: document description (optional)
    - **tags**: tags as comma-separated string (optional)
    """

    if not api_key:
        raise HTTPException(status_code=401, detail="Unauthorized")

    tags_list = []
    if tags:
        tags_list = [tag.strip() for tag in tags.split(",") if tag.strip()]

    request = DocumentUploadRequest(
        title=title,
        description=description,
        tags=tags_list
    )

    document = await service.upload_document(file, request)

    return DocumentUploadResponse(
        document_id=document.id,
        filename=document.metadata.filename,
        status=document.status,
        message="Document successfully uploaded and is being processed"
    )


@router.get("/", response_model=DocumentListResponse)
async def get_documents(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Page size"),
    status: Optional[DocumentStatus] = Query(
        None, description="Filter by status"),
    document_type: Optional[DocumentType] = Query(
        None, description="Filter by document type"),
    service: DocumentService = Depends(get_document_service),
    api_key: str = Depends(verify_api_key)
):
    """
    Gets list of all documents with pagination and filtering

    - **page**: page number (starting from 1)
    - **page_size**: number of documents per page
    - **status**: filter by status (uploading, processing, ready, error)
    - **document_type**: filter by document type
    """

    if not api_key:
        raise HTTPException(status_code=401, detail="Unauthorized")
    result = service.get_documents(
        page=page,
        page_size=page_size,
        status=status,
        document_type=document_type
    )

    return DocumentListResponse(
        documents=result['documents'],
        total=result['total'],
        page=page,
        page_size=page_size
    )


@router.get("/{document_id}", response_model=DocumentDetailResponse)
async def get_document(
    document_id: str,
    service: DocumentService = Depends(get_document_service),
    api_key: str = Depends(verify_api_key)
):
    """
    Gets detailed information about a document

    - **document_id**: document ID
    """

    if not api_key:
        raise HTTPException(status_code=401, detail="Unauthorized")
    document = service.get_document(document_id)

    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    return DocumentDetailResponse(
        document=document,
        chunks_count=len(document.chunks),
        processing_info={
            "total_chunks": len(document.chunks),
            "status": document.status.value,
            "processed_at": document.processed_at
        }
    )


@router.put("/{document_id}", response_model=Document)
async def update_document(
    document_id: str,
    request: DocumentUpdateRequest,
    service: DocumentService = Depends(get_document_service),
    api_key: str = Depends(verify_api_key)
):
    """
    Updates document metadata

    - **document_id**: document ID
    - **title**: new title
    - **description**: new description
    - **tags**: new tags
    """
    if not api_key:
        raise HTTPException(status_code=401, detail="Unauthorized")

    document = service.update_document(document_id, request)

    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    return document


@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    service: DocumentService = Depends(get_document_service),
    api_key: str = Depends(verify_api_key)

):
    """
    Deletes a document and its associated file

    - **document_id**: document ID
    """
    if not api_key:
        raise HTTPException(status_code=401, detail="Unauthorized")

    success = service.delete_document(document_id)

    if not success:
        raise HTTPException(status_code=404, detail="Document not found")

    return {"message": "Document successfully deleted"}


@router.get("/search/")
async def search_documents(
    query: str = Query(..., description="Search query"),
    service: DocumentService = Depends(get_document_service),
    api_key: str = Depends(verify_api_key)
):
    """
    Searches documents by content and metadata

    - **query**: search query
    """
    if not api_key:
        raise HTTPException(status_code=401, detail="Unauthorized")

    documents = service.search_documents(query)

    return {
        "query": query,
        "results": documents,
        "total": len(documents)
    }


@router.get("/{document_id}/chunks")
async def get_document_chunks(
    document_id: str,
    service: DocumentService = Depends(get_document_service),
    api_key: str = Depends(verify_api_key)

):
    """
    Gets document chunks for vector search

    - **document_id**: document ID
    """
    if not api_key:
        raise HTTPException(status_code=401, detail="Unauthorized")

    document = service.get_document(document_id)

    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    chunks = await service.get_document_chunks(document_id)

    return {
        "document_id": document_id,
        "document_title": document.title,
        "chunks": chunks,
        "total_chunks": len(chunks)
    }


@router.get("/{document_id}/content")
async def get_document_content(
    document_id: str,
    service: DocumentService = Depends(get_document_service),
    api_key: str = Depends(verify_api_key)
):
    """
    Gets the full content of a document

    - **document_id**: document ID
    """
    if not api_key:
        raise HTTPException(status_code=401, detail="Unauthorized")

    document = service.get_document(document_id)

    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    if document.status != DocumentStatus.READY:
        raise HTTPException(
            status_code=400,
            detail=f"Document has not been processed yet. Status: {document.status.value}"
        )

    return {
        "document_id": document_id,
        "title": document.title,
        "content": document.content,
        "metadata": document.metadata,
        "word_count": document.metadata.word_count,
        "char_count": document.metadata.char_count
    }


@router.post("/bulk-upload")
async def bulk_upload_documents(
    files: List[UploadFile] = File(...),
    service: DocumentService = Depends(get_document_service),
    api_key: str = Depends(verify_api_key)

):
    """
    Bulk upload of documents

    - **files**: list of files to upload
    """
    if not api_key:
        raise HTTPException(status_code=401, detail="Unauthorized")

    results = []

    for file in files:
        try:
            request = DocumentUploadRequest(title=None)
            document = await service.upload_document(file, request)

            results.append({
                "filename": file.filename,
                "document_id": document.id,
                "status": "success",
                "message": "Document uploaded"
            })

        except Exception as e:
            results.append({
                "filename": file.filename,
                "status": "error",
                "message": str(e)
            })

    return {
        "results": results,
        "total_files": len(files),
        "successful": len([r for r in results if r["status"] == "success"]),
        "failed": len([r for r in results if r["status"] == "error"])
    }
