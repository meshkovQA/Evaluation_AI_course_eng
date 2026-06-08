import os
import shutil
import logging
import hashlib
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path
import asyncio
import json
from fastapi import UploadFile, HTTPException

from app.models.document import (
    Document, DocumentType, DocumentStatus, DocumentMetadata,
    DocumentChunk, DocumentUploadRequest, DocumentUpdateRequest
)
from app.utils.file_parser import FileParser
from app.utils.text_splitter import TextSplitter
from app.config import settings

logger = logging.getLogger(__name__)


class DocumentService:
    """Service for working with documents"""

    def __init__(self, storage_path: str = None):
        self.storage_path = storage_path or settings.DOCUMENTS_STORAGE_PATH
        self.file_parser = FileParser()
        self.text_splitter = TextSplitter()

        Path(self.storage_path).mkdir(parents=True, exist_ok=True)

        # In-memory document storage (in production — use a database)
        self.documents: Dict[str, Document] = {}

        self._restore_documents_from_metadata()

        logger.info(f"✅ DocumentService initialized: {len(self.documents)} documents in memory")

    def _save_documents_metadata(self):
        """Saves document metadata to a JSON file"""
        metadata_file = os.path.join(self.storage_path, "documents_metadata.json")

        try:
            metadata = {}
            for doc_id, doc in self.documents.items():
                metadata[doc_id] = {
                    "id": doc.id,
                    "title": doc.title,
                    "file_path": doc.file_path,
                    "document_type": doc.document_type.value,
                    "status": doc.status.value,
                    "description": doc.description,
                    "tags": doc.tags,
                    "content": doc.content,
                    "uploaded_at": doc.uploaded_at.isoformat() if doc.uploaded_at else None,
                    "processed_at": doc.processed_at.isoformat() if doc.processed_at else None,
                    "error_message": doc.error_message,
                    "metadata": {
                        "filename": doc.metadata.filename,
                        "file_size": doc.metadata.file_size,
                        "file_extension": doc.metadata.file_extension,
                        "word_count": doc.metadata.word_count,
                        "char_count": doc.metadata.char_count,
                        "created_at": doc.metadata.created_at,
                        "modified_at": doc.metadata.modified_at,
                    },
                    "chunks_count": len(doc.chunks)
                }

            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

            logger.info(f"💾 Saved document metadata: {len(metadata)}")

        except Exception as e:
            logger.error(f"❌ Error saving metadata: {e}")

    def _load_documents_metadata(self) -> Dict[str, Any]:
        """Loads document metadata from a JSON file"""
        metadata_file = os.path.join(self.storage_path, "documents_metadata.json")

        if not os.path.exists(metadata_file):
            logger.info("📄 Metadata file not found, starting with empty storage")
            return {}

        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            logger.info(f"📂 Loaded metadata: {len(metadata)} documents")
            return metadata

        except Exception as e:
            logger.error(f"❌ Error loading metadata: {e}")
            return {}

    def _restore_documents_from_metadata(self):
        """
        Restores documents from saved metadata.
        Chunks are NOT restored to memory but remain in ChromaDB for search.
        """
        try:
            metadata = self._load_documents_metadata()

            if not metadata:
                logger.info("ℹ️  No saved documents to restore")
                return

            restored_count = 0
            for doc_id, doc_meta in metadata.items():
                try:
                    if not os.path.exists(doc_meta['file_path']):
                        logger.warning(f"⚠️  File not found: {doc_meta['file_path']}, skipping document {doc_id}")
                        continue

                    file_metadata = DocumentMetadata(
                        filename=doc_meta['metadata']['filename'],
                        file_size=doc_meta['metadata']['file_size'],
                        file_extension=doc_meta['metadata']['file_extension'],
                        created_at=doc_meta['metadata'].get('created_at'),
                        modified_at=doc_meta['metadata'].get('modified_at'),
                        word_count=doc_meta['metadata'].get('word_count', 0),
                        char_count=doc_meta['metadata'].get('char_count', 0),
                    )

                    uploaded_at = None
                    processed_at = None
                    if doc_meta.get('uploaded_at'):
                        uploaded_at = datetime.fromisoformat(doc_meta['uploaded_at'])
                    if doc_meta.get('processed_at'):
                        processed_at = datetime.fromisoformat(doc_meta['processed_at'])

                    # Create document WITHOUT chunks (they are in ChromaDB)
                    document = Document(
                        id=doc_id,
                        title=doc_meta['title'],
                        file_path=doc_meta['file_path'],
                        document_type=DocumentType(doc_meta['document_type']),
                        status=DocumentStatus(doc_meta['status']),
                        metadata=file_metadata,
                        description=doc_meta.get('description'),
                        tags=doc_meta.get('tags', []),
                        content=doc_meta.get('content', ''),
                        uploaded_at=uploaded_at,
                        processed_at=processed_at,
                        error_message=doc_meta.get('error_message'),
                    )

                    # Chunks are empty in memory but available in ChromaDB
                    document.chunks = []

                    self.documents[doc_id] = document
                    restored_count += 1

                    logger.debug(f"✅ Restored: {doc_meta['title']} (had {doc_meta['chunks_count']} chunks)")

                except Exception as e:
                    logger.error(f"❌ Error restoring document {doc_id}: {e}")

            logger.info(f"✅ Restored documents: {restored_count}/{len(metadata)}")
            logger.info("💡 Chunks are NOT loaded into memory but available for search in ChromaDB")

        except Exception as e:
            logger.error(f"❌ Error restoring documents: {e}")

    def _generate_stable_id(self, file_path: str, filename: str) -> str:
        """Generates a stable ID based on the file"""
        file_stat = os.stat(file_path)
        hash_input = f"{filename}_{file_stat.st_size}_{file_stat.st_mtime}"

        hash_object = hashlib.md5(hash_input.encode())
        short_hash = hash_object.hexdigest()[:8]

        return f"doc_{short_hash}"

    async def upload_document(
        self,
        file: UploadFile,
        request: DocumentUploadRequest
    ) -> Document:
        """
        Uploads and processes a document

        Args:
            file: file to upload
            request: additional parameters

        Returns:
            Processed document
        """
        if not self.file_parser.is_supported(file.filename):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format: {file.filename}"
            )

        file_extension = Path(file.filename).suffix
        safe_filename = self._generate_safe_filename(file.filename)
        file_path = os.path.join(self.storage_path, safe_filename)

        try:
            await self._save_uploaded_file(file, file_path)

            metadata = self._create_metadata(file_path, file.filename)

            stable_id = self._generate_stable_id(file_path, safe_filename)

            document = Document(
                id=stable_id,
                title=request.title or Path(file.filename).stem,
                file_path=file_path,
                document_type=DocumentType(
                    self.file_parser.get_file_type(file_path)),
                status=DocumentStatus.PROCESSING,
                metadata=metadata,
                description=request.description,
                tags=request.tags
            )

            self.documents[document.id] = document

            asyncio.create_task(self._process_document(document.id))

            self._save_documents_metadata()

            return document

        except Exception as e:
            if os.path.exists(file_path):
                os.remove(file_path)
            raise HTTPException(
                status_code=500,
                detail=f"Error uploading file: {str(e)}"
            )

    async def _process_document(self, document_id: str):
        """Async document processing with detailed logging"""
        document = self.documents.get(document_id)
        if not document:
            logger.error(f"❌ Document {document_id} not found for processing")
            return

        logger.info(f"🔄 Starting document processing: {document.title} (ID: {document_id})")

        try:
            # 1. Parse file content
            logger.info(f"📄 Parsing file: {document.file_path}")
            parse_result = self.file_parser.parse_file(document.file_path)

            if not parse_result['success']:
                logger.error(f"❌ Parse error: {parse_result['error']}")
                document.status = DocumentStatus.ERROR
                document.error_message = parse_result['error']
                return

            # 2. Update document content
            document.content = parse_result['text']
            logger.info(f"✅ Text extracted: {len(document.content)} characters")

            if not document.content or len(document.content.strip()) < 10:
                logger.error(f"❌ Document is empty or too short")
                document.status = DocumentStatus.ERROR
                document.error_message = "Document is empty or contains too little text"
                return

            # 3. Update metadata
            document.metadata.word_count = len(document.content.split())
            document.metadata.char_count = len(document.content)
            logger.info(f"📊 Stats: {document.metadata.word_count} words, {document.metadata.char_count} characters")

            # 4. Create chunks for vector search
            logger.info(f"✂️  Splitting into chunks...")
            chunks_text = self.text_splitter.split_text(document.content)
            logger.info(f"✅ Created {len(chunks_text)} chunks")

            if not chunks_text:
                logger.error(f"❌ Failed to create chunks")
                document.status = DocumentStatus.ERROR
                document.error_message = "Failed to split text into chunks"
                return

            document.chunks = [
                DocumentChunk(
                    document_id=document.id,
                    chunk_index=i,
                    text=chunk,
                    metadata={
                        'source': document.title,
                        'document_type': document.document_type.value,
                        'chunk_size': len(chunk)
                    }
                )
                for i, chunk in enumerate(chunks_text)
            ]
            logger.info(f"📦 Created {len(document.chunks)} DocumentChunk objects")

            # 5. Create embeddings for chunks
            logger.info(f"🧮 Creating embeddings...")
            try:
                await self._create_embeddings_for_chunks(document)

                chunks_with_embeddings = [c for c in document.chunks if c.embedding is not None]
                logger.info(f"✅ Embeddings created for {len(chunks_with_embeddings)}/{len(document.chunks)} chunks")

                if len(chunks_with_embeddings) == 0:
                    logger.error(f"❌ No embeddings created")
                    document.status = DocumentStatus.ERROR
                    document.error_message = "Failed to create embeddings"
                    return

            except Exception as e:
                logger.error(f"❌ Error creating embeddings: {e}", exc_info=True)
                document.status = DocumentStatus.ERROR
                document.error_message = f"Error creating embeddings: {str(e)}"
                return

            # 6. Save to vector storage
            logger.info(f"💾 Saving to vector storage...")
            try:
                await self._save_to_vector_store(document)
                logger.info(f"✅ Chunks saved to vector storage")
            except Exception as e:
                logger.error(f"❌ Error saving to vector storage: {e}", exc_info=True)
                document.error_message = f"Warning: error saving to vector storage: {str(e)}"

            # 7. Mark as ready
            document.status = DocumentStatus.READY
            document.processed_at = datetime.utcnow()
            logger.info(f"✅ Document {document.title} successfully processed!")
            logger.info(f"📊 Total: {len(document.chunks)} chunks, status: {document.status.value}")

            self._save_documents_metadata()

        except Exception as e:
            logger.error(f"❌ Critical error processing document {document_id}: {e}", exc_info=True)
            document.status = DocumentStatus.ERROR
            document.error_message = f"Critical error: {str(e)}"

            self._save_documents_metadata()

    async def _create_embeddings_for_chunks(self, document: Document):
        """Creates embeddings for document chunks"""
        try:
            logger.info(f"🔧 Initializing embedding service...")
            from app.services.embedding_service import get_embedding_service

            embedding_service = get_embedding_service()
            logger.info(f"✅ Embedding service initialized")

            chunk_texts = [chunk.text for chunk in document.chunks]
            logger.info(f"📝 Prepared {len(chunk_texts)} texts for embedding")

            if chunk_texts:
                logger.info(f"🚀 Sending request to create embeddings...")
                embeddings = await embedding_service.create_embeddings_for_chunks(chunk_texts)
                logger.info(f"✅ Received {len(embeddings)} embeddings")

                for i, (chunk, embedding) in enumerate(zip(document.chunks, embeddings)):
                    chunk.embedding = embedding
                    logger.debug(f"   Chunk {i}: embedding size {len(embedding)}")

                logger.info(f"✅ Embeddings assigned to all chunks")

        except ImportError as e:
            logger.error(f"❌ Error importing embedding service: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ Error creating embeddings for document {document.id}: {e}", exc_info=True)
            raise

    async def _save_to_vector_store(self, document: Document):
        """Saves document chunks to vector storage"""
        try:
            logger.info(f"🔧 Initializing vector storage...")
            from app.services.vector_store import get_vector_store_service

            vector_store = get_vector_store_service()
            logger.info(f"✅ Vector storage initialized")

            chunks_with_embeddings = [
                chunk for chunk in document.chunks
                if chunk.embedding is not None
            ]

            logger.info(f"📦 Chunks with embeddings: {len(chunks_with_embeddings)}/{len(document.chunks)}")

            if chunks_with_embeddings:
                logger.info(f"💾 Saving {len(chunks_with_embeddings)} chunks to vector storage...")
                success = await vector_store.add_document_chunks(chunks_with_embeddings)

                if not success:
                    logger.warning(f"⚠️  Failed to save chunks for document {document.id} to vector storage")
                    raise Exception("Vector storage returned success=False")
                else:
                    logger.info(f"✅ Chunks successfully saved to vector storage")

        except ImportError as e:
            logger.error(f"❌ Error importing vector storage: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ Error saving to vector storage for document {document.id}: {e}", exc_info=True)
            raise

    def get_document(self, document_id: str) -> Optional[Document]:
        """Gets a document by ID"""
        return self.documents.get(document_id)

    def get_documents(
        self,
        page: int = 1,
        page_size: int = 20,
        status: Optional[DocumentStatus] = None,
        document_type: Optional[DocumentType] = None
    ) -> Dict[str, Any]:
        """
        Gets list of documents with pagination and filtering

        Returns:
            Dict with documents and pagination metadata
        """
        documents = list(self.documents.values())

        if status:
            documents = [doc for doc in documents if doc.status == status]

        if document_type:
            documents = [
                doc for doc in documents if doc.document_type == document_type]

        # Sort by upload date (newest first)
        documents.sort(key=lambda x: x.uploaded_at, reverse=True)

        total = len(documents)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        page_documents = documents[start_idx:end_idx]

        return {
            'documents': page_documents,
            'total': total,
            'page': page,
            'page_size': page_size,
            'total_pages': (total + page_size - 1) // page_size
        }

    def update_document(
        self,
        document_id: str,
        request: DocumentUpdateRequest
    ) -> Optional[Document]:
        """Updates document metadata"""
        document = self.documents.get(document_id)
        if not document:
            return None

        if request.title is not None:
            document.title = request.title
        if request.description is not None:
            document.description = request.description
        if request.tags is not None:
            document.tags = request.tags

        self._save_documents_metadata()

        return document

    def delete_document(self, document_id: str) -> bool:
        """Deletes a document and its associated file"""
        document = self.documents.get(document_id)
        if not document:
            return False

        try:
            asyncio.create_task(self._delete_from_vector_store(document_id))

            if os.path.exists(document.file_path):
                os.remove(document.file_path)

            del self.documents[document_id]

            self._save_documents_metadata()
            return True

        except Exception:
            return False

    async def _delete_from_vector_store(self, document_id: str):
        """Deletes a document from vector storage"""
        try:
            from app.services.vector_store import get_vector_store_service

            vector_store = get_vector_store_service()
            await vector_store.delete_document_chunks(document_id)

        except Exception as e:
            logger.error(f"Error deleting from vector storage: {e}")

    def search_documents(self, query: str) -> List[Document]:
        """Simple document search by title and content"""
        query_lower = query.lower()
        results = []

        for document in self.documents.values():
            if (query_lower in document.title.lower() or
                query_lower in document.content.lower() or
                    any(query_lower in tag.lower() for tag in document.tags)):
                results.append(document)

        return results

    async def get_document_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        """Gets chunks: first from memory, then from ChromaDB"""
        document = self.documents.get(document_id)
        if not document:
            return []

        if document.chunks:
            return [
                {
                    "id": chunk.id,
                    "document_id": chunk.document_id,
                    "chunk_index": chunk.chunk_index,
                    "text": chunk.text,
                    "metadata": chunk.metadata,
                    "has_embedding": chunk.embedding is not None
                }
                for chunk in document.chunks
            ]

        logger.info(f"📂 Loading chunks from ChromaDB for {document_id}")

        try:
            from app.services.vector_store import get_vector_store_service
            vector_store = get_vector_store_service()
            chunks = await vector_store.get_document_chunks_from_store(document_id)
            logger.info(f"✅ Loaded {len(chunks)} chunks")
            return chunks
        except Exception as e:
            logger.error(f"❌ Error: {e}", exc_info=True)
            return []

    async def _save_uploaded_file(self, file: UploadFile, file_path: str):
        """Saves an uploaded file to disk"""
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

    def _generate_safe_filename(self, filename: str) -> str:
        """Generates a safe unique filename"""
        extension = Path(filename).suffix
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = f"{timestamp}_{filename.replace(' ', '_')}"
        return safe_name

    def _create_metadata(self, file_path: str, original_filename: str) -> DocumentMetadata:
        """Creates metadata for a document"""
        file_stat = os.stat(file_path)

        return DocumentMetadata(
            filename=original_filename,
            file_size=file_stat.st_size,
            file_extension=Path(file_path).suffix,
            created_at=file_stat.st_ctime,
            modified_at=file_stat.st_mtime
        )


# Global singleton instance of the document service
_document_service_instance: Optional[DocumentService] = None


def get_document_service() -> DocumentService:
    """
    Returns the global document service instance.
    Ensures background processing tasks are not lost.
    """
    global _document_service_instance
    if _document_service_instance is None:
        _document_service_instance = DocumentService()
        logger.info("✅ Global DocumentService instance created")
    return _document_service_instance
