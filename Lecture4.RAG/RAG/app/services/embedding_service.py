import asyncio
from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod
import numpy as np
import logging

# OpenAI
try:
    import openai
except ImportError:
    openai = None

# HuggingFace
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from app.config import settings

logger = logging.getLogger(__name__)


class BaseEmbeddingProvider(ABC):
    """Base class for embedding providers"""

    @abstractmethod
    async def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Creates embeddings for a list of texts"""
        pass

    @abstractmethod
    async def create_embedding(self, text: str) -> List[float]:
        """Creates an embedding for a single text"""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Vector dimensionality"""
        pass


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """Embedding provider via OpenAI API"""

    def __init__(self, api_key: str, model: str = "text-embedding-ada-002"):
        if not openai:
            raise ImportError("Install openai: pip install openai")

        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self._dimension = 1536 if "ada-002" in model else 1536

    async def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Creates embeddings for a list of texts"""
        try:
            # OpenAI API supports batches up to 2048 texts
            batch_size = 100
            all_embeddings = []

            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]

                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: self.client.embeddings.create(
                        input=batch,
                        model=self.model
                    )
                )

                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)

                # Small delay between batches
                if i + batch_size < len(texts):
                    await asyncio.sleep(0.1)

            return all_embeddings

        except Exception as e:
            logger.error(f"Error creating embeddings via OpenAI: {e}")
            raise

    async def create_embedding(self, text: str) -> List[float]:
        """Creates an embedding for a single text"""
        embeddings = await self.create_embeddings([text])
        return embeddings[0]

    @property
    def dimension(self) -> int:
        return self._dimension


class HuggingFaceEmbeddingProvider(BaseEmbeddingProvider):
    """Embedding provider via HuggingFace models"""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        if not SentenceTransformer:
            raise ImportError(
                "Install sentence-transformers: pip install sentence-transformers")

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self._dimension = self.model.get_sentence_embedding_dimension()

    async def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Creates embeddings for a list of texts"""
        try:
            # Run in executor since model works synchronously
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None,
                lambda: self.model.encode(texts, convert_to_numpy=True)
            )

            return embeddings.tolist()

        except Exception as e:
            logger.error(f"Error creating embeddings via HuggingFace: {e}")
            raise

    async def create_embedding(self, text: str) -> List[float]:
        """Creates an embedding for a single text"""
        embeddings = await self.create_embeddings([text])
        return embeddings[0]

    @property
    def dimension(self) -> int:
        return self._dimension


class EmbeddingService:
    """Service for working with embeddings"""

    def __init__(self, provider: Optional[BaseEmbeddingProvider] = None):
        self.provider = provider or self._create_default_provider()

    def _create_default_provider(self) -> BaseEmbeddingProvider:
        """Creates the default provider based on settings"""
        if settings.EMBEDDING_PROVIDER == "openai":
            if not settings.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY is not set")
            return OpenAIEmbeddingProvider(
                api_key=settings.OPENAI_API_KEY,
                model=settings.EMBEDDING_MODEL
            )
        elif settings.EMBEDDING_PROVIDER == "huggingface":
            return HuggingFaceEmbeddingProvider(
                model_name=settings.EMBEDDING_MODEL
            )
        else:
            raise ValueError(
                f"Unsupported provider: {settings.EMBEDDING_PROVIDER}")

    async def create_embeddings_for_chunks(self, chunks: List[str]) -> List[List[float]]:
        """
        Creates embeddings for a list of chunks

        Args:
            chunks: list of text chunks

        Returns:
            List of embedding vectors
        """
        if not chunks:
            return []

        logger.info(f"Creating embeddings for {len(chunks)} chunks")

        try:
            embeddings = await self.provider.create_embeddings(chunks)
            logger.info(f"Successfully created {len(embeddings)} embeddings")
            return embeddings

        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            raise

    async def create_embedding_for_query(self, query: str) -> List[float]:
        """
        Creates an embedding for a search query

        Args:
            query: search query

        Returns:
            Embedding vector
        """
        logger.debug(f"Creating embedding for query: {query[:100]}...")

        try:
            embedding = await self.provider.create_embedding(query)
            return embedding

        except Exception as e:
            logger.error(f"Error creating embedding for query: {e}")
            raise

    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Computes cosine similarity between two vectors

        Args:
            embedding1: first vector
            embedding2: second vector

        Returns:
            Similarity value from 0 to 1
        """
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = dot_product / (norm1 * norm2)
        return float(max(0, similarity))

    def find_most_similar(
        self,
        query_embedding: List[float],
        embeddings: List[List[float]],
        top_k: int = 5,
        threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Finds the most similar embeddings

        Args:
            query_embedding: query embedding
            embeddings: list of embeddings to compare
            top_k: number of results
            threshold: minimum similarity threshold

        Returns:
            List of dicts with indices and similarity scores
        """
        similarities = []

        for i, embedding in enumerate(embeddings):
            similarity = self.calculate_similarity(query_embedding, embedding)
            if similarity >= threshold:
                similarities.append({
                    'index': i,
                    'similarity': similarity
                })

        similarities.sort(key=lambda x: x['similarity'], reverse=True)

        return similarities[:top_k]

    @property
    def embedding_dimension(self) -> int:
        """Returns vector dimensionality"""
        return self.provider.dimension

    async def test_connection(self) -> bool:
        """Tests connection to the embedding provider"""
        try:
            test_embedding = await self.provider.create_embedding("test")
            return len(test_embedding) == self.embedding_dimension
        except Exception as e:
            logger.error(f"Error testing connection: {e}")
            return False


# Global service instance
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Returns the global embedding service instance"""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
