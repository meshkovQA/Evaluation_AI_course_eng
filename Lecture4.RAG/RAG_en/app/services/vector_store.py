import json
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
import logging
from pathlib import Path

# ChromaDB
try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
except ImportError:
    chromadb = None

# Pinecone
try:
    import pinecone
except ImportError:
    pinecone = None

# FAISS
try:
    import faiss
    import numpy as np
except ImportError:
    faiss = None
    np = None

from app.config import settings
from app.models.document import DocumentChunk

logger = logging.getLogger(__name__)


class BaseVectorStore(ABC):
    """Base class for vector stores"""

    @abstractmethod
    async def add_documents(self, chunks: List[DocumentChunk]) -> bool:
        """Adds documents to vector storage"""
        pass

    @abstractmethod
    async def search(self, query_embedding: List[float], top_k: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """Executes search in vector storage"""
        pass

    @abstractmethod
    async def delete_document(self, document_id: str) -> bool:
        """Deletes all chunks of a document"""
        pass

    @abstractmethod
    async def get_collection_info(self) -> Dict[str, Any]:
        """Returns information about the collection"""
        pass


class ChromaVectorStore(BaseVectorStore):
    """Vector storage based on ChromaDB"""

    def __init__(self, persist_directory: str = None, collection_name: str = "rag_documents"):
        if not chromadb:
            raise ImportError("Install chromadb: pip install chromadb")

        self.persist_directory = persist_directory or settings.CHROMA_PERSIST_DIRECTORY
        self.collection_name = collection_name

        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=ChromaSettings(anonymized_telemetry=False)
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "RAG documents collection"}
        )

        logger.info(f"ChromaDB initialized: {self.persist_directory}")

    async def add_documents(self, chunks: List[DocumentChunk]) -> bool:
        """Adds document chunks to ChromaDB"""
        if not chunks:
            return True

        try:
            ids = []
            embeddings = []
            documents = []
            metadatas = []

            for chunk in chunks:
                if chunk.embedding is None:
                    logger.warning(f"Chunk {chunk.id} has no embedding")
                    continue

                ids.append(chunk.id)
                embeddings.append(chunk.embedding)
                documents.append(chunk.text)
                metadatas.append({
                    "document_id": chunk.document_id,
                    "chunk_index": chunk.chunk_index,
                    **chunk.metadata
                })

            if not ids:
                logger.warning("No chunks with embeddings to add")
                return False

            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas
                )
            )

            logger.info(f"Added {len(ids)} chunks to ChromaDB")
            return True

        except Exception as e:
            logger.error(f"Error adding to ChromaDB: {e}")
            return False

    async def search(self, query_embedding: List[float], top_k: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """Executes search in ChromaDB"""
        try:
            total_count = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.collection.count()
            )
            logger.info(f"🔍 Search in ChromaDB: total {total_count} chunks in collection")

            if total_count == 0:
                logger.warning("⚠️  ChromaDB collection is EMPTY!")
                return []

            logger.info(f"🔎 Search query: top_k={top_k}, embedding_dim={len(query_embedding)}")
            results = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=min(top_k, total_count),
                    include=["documents", "metadatas", "distances"]
                )
            )

            search_results = []
            if results['ids'] and results['ids'][0]:
                logger.info(f"📊 ChromaDB returned {len(results['ids'][0])} results")

                for i in range(len(results['ids'][0])):
                    distance = results['distances'][0][i]
                    # ChromaDB default space is squared L2. For unit-normalized
                    # embeddings (OpenAI), L2^2 ∈ [0, 4] and cosine_similarity = 1 - L2^2/2.
                    similarity = max(0.0, 1.0 - distance / 2.0)

                    result = {
                        "chunk_id": results['ids'][0][i],
                        "text": results['documents'][0][i],
                        "metadata": results['metadatas'][0][i],
                        "distance": distance,
                        "similarity": similarity
                    }

                    logger.info(f"   {i+1}. similarity={similarity:.4f}, distance={distance:.4f}, doc={results['metadatas'][0][i].get('source', 'Unknown')}")

                    search_results.append(result)
            else:
                logger.warning("⚠️  ChromaDB returned 0 results")

            return search_results

        except Exception as e:
            logger.error(f"❌ Error searching in ChromaDB: {e}", exc_info=True)
            return []

    async def delete_document(self, document_id: str) -> bool:
        """Deletes all document chunks from ChromaDB"""
        try:
            results = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.collection.get(
                    where={"document_id": document_id},
                    include=["metadatas"]
                )
            )

            if results['ids']:
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.collection.delete(ids=results['ids'])
                )
                logger.info(
                    f"Deleted {len(results['ids'])} chunks of document {document_id}")

            return True

        except Exception as e:
            logger.error(f"Error deleting document from ChromaDB: {e}")
            return False

    async def get_collection_info(self) -> Dict[str, Any]:
        """Returns information about the ChromaDB collection"""
        try:
            count = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.collection.count()
            )

            return {
                "type": "chromadb",
                "collection_name": self.collection_name,
                "total_chunks": count,
                "persist_directory": self.persist_directory
            }

        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {"type": "chromadb", "error": str(e)}

    async def get_document_chunks_from_db(self, document_id: str) -> List[Dict[str, Any]]:
        """Gets all document chunks from ChromaDB"""
        try:
            logger.info(f"📂 Getting chunks for document {document_id} from ChromaDB")

            results = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.collection.get(
                    where={"document_id": document_id},
                    include=["documents", "metadatas"]
                )
            )

            chunks = []
            if results['ids']:
                logger.info(f"✅ Found {len(results['ids'])} chunks")

                for i, chunk_id in enumerate(results['ids']):
                    chunks.append({
                        "id": chunk_id,
                        "document_id": document_id,
                        "chunk_index": results['metadatas'][i].get('chunk_index', i),
                        "text": results['documents'][i],
                        "metadata": results['metadatas'][i],
                        "has_embedding": True
                    })

                chunks.sort(key=lambda x: x.get('chunk_index', 0))

            return chunks

        except Exception as e:
            logger.error(f"❌ Error: {e}", exc_info=True)
            return []


class PineconeVectorStore(BaseVectorStore):
    """Vector storage based on Pinecone"""

    def __init__(self, api_key: str = None, environment: str = None, index_name: str = None):
        if not pinecone:
            raise ImportError(
                "Install pinecone: pip install pinecone-client")

        self.api_key = api_key or settings.PINECONE_API_KEY
        self.environment = environment or settings.PINECONE_ENVIRONMENT
        self.index_name = index_name or settings.PINECONE_INDEX_NAME

        if not all([self.api_key, self.environment, self.index_name]):
            raise ValueError(
                "PINECONE_API_KEY, PINECONE_ENVIRONMENT and PINECONE_INDEX_NAME must be set")

        pinecone.init(
            api_key=self.api_key,
            environment=self.environment
        )

        self.index = pinecone.Index(self.index_name)
        logger.info(f"Pinecone initialized: {self.index_name}")

    async def add_documents(self, chunks: List[DocumentChunk]) -> bool:
        """Adds chunks to Pinecone"""
        if not chunks:
            return True

        try:
            vectors = []
            for chunk in chunks:
                if chunk.embedding is None:
                    continue

                vectors.append({
                    "id": chunk.id,
                    "values": chunk.embedding,
                    "metadata": {
                        "document_id": chunk.document_id,
                        "chunk_index": chunk.chunk_index,
                        "text": chunk.text,
                        **chunk.metadata
                    }
                })

            if vectors:
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.index.upsert(vectors=vectors)
                )
                logger.info(f"Added {len(vectors)} chunks to Pinecone")

            return True

        except Exception as e:
            logger.error(f"Error adding to Pinecone: {e}")
            return False

    async def search(self, query_embedding: List[float], top_k: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """Executes search in Pinecone"""
        try:
            results = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.index.query(
                    vector=query_embedding,
                    top_k=top_k,
                    include_metadata=True
                )
            )

            search_results = []
            for match in results['matches']:
                search_results.append({
                    "chunk_id": match['id'],
                    "text": match['metadata'].get('text', ''),
                    "metadata": match['metadata'],
                    "similarity": match['score']
                })

            return search_results

        except Exception as e:
            logger.error(f"Error searching in Pinecone: {e}")
            return []

    async def delete_document(self, document_id: str) -> bool:
        """Deletes a document from Pinecone"""
        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.index.delete(filter={"document_id": document_id})
            )
            return True

        except Exception as e:
            logger.error(f"Error deleting from Pinecone: {e}")
            return False

    async def get_collection_info(self) -> Dict[str, Any]:
        """Returns information about the Pinecone index"""
        try:
            stats = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.index.describe_index_stats()
            )

            return {
                "type": "pinecone",
                "index_name": self.index_name,
                "total_vectors": stats.get('total_vector_count', 0),
                "dimension": stats.get('dimension', 0)
            }

        except Exception as e:
            logger.error(f"Error getting Pinecone stats: {e}")
            return {"type": "pinecone", "error": str(e)}


class VectorStoreService:
    """Service for working with vector stores"""

    def __init__(self, store: Optional[BaseVectorStore] = None):
        self.store = store or self._create_default_store()

    def _create_default_store(self) -> BaseVectorStore:
        """Creates the default vector storage"""
        if settings.VECTOR_DB_TYPE == "chroma":
            return ChromaVectorStore()
        elif settings.VECTOR_DB_TYPE == "pinecone":
            return PineconeVectorStore()
        else:
            raise ValueError(
                f"Unsupported vector DB type: {settings.VECTOR_DB_TYPE}")

    async def add_document_chunks(self, chunks: List[DocumentChunk]) -> bool:
        """Adds document chunks to vector storage"""
        return await self.store.add_documents(chunks)

    async def search_similar_chunks(
        self,
        query_embedding: List[float],
        top_k: int = None,
        similarity_threshold: float = None
    ) -> List[Dict[str, Any]]:
        """
        Searches for similar chunks

        Args:
            query_embedding: query embedding
            top_k: number of results
            similarity_threshold: minimum similarity threshold

        Returns:
            List of found chunks with metadata
        """
        top_k = top_k or settings.MAX_RELEVANT_CHUNKS
        similarity_threshold = similarity_threshold or settings.SIMILARITY_THRESHOLD

        results = await self.store.search(query_embedding, top_k)

        filtered_results = []
        for result in results:
            similarity = result.get('similarity', 0)
            if similarity >= similarity_threshold:
                filtered_results.append(result)

        return filtered_results

    async def delete_document_chunks(self, document_id: str) -> bool:
        """Deletes all document chunks"""
        return await self.store.delete_document(document_id)

    async def get_store_info(self) -> Dict[str, Any]:
        """Returns information about the vector storage"""
        return await self.store.get_collection_info()

    async def get_document_chunks_from_store(self, document_id: str) -> List[Dict[str, Any]]:
        """Gets document chunks from vector storage"""
        if isinstance(self.store, ChromaVectorStore):
            return await self.store.get_document_chunks_from_db(document_id)
        else:
            logger.warning(f"Not implemented for {type(self.store).__name__}")
            return []

    async def test_connection(self) -> bool:
        """Tests connection to vector storage"""
        try:
            info = await self.get_store_info()
            return not info.get('error')
        except Exception as e:
            logger.error(f"Error testing vector storage: {e}")
            return False


# Global service instance
_vector_store_service: Optional[VectorStoreService] = None


def get_vector_store_service() -> VectorStoreService:
    """Returns the global vector storage service instance"""
    global _vector_store_service
    if _vector_store_service is None:
        _vector_store_service = VectorStoreService()
    return _vector_store_service
