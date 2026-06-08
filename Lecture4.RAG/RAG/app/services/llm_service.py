import asyncio
from typing import List, Dict, Any, Optional, AsyncGenerator, Tuple
from abc import ABC, abstractmethod
import logging
import json

# OpenAI
try:
    import openai
except ImportError:
    openai = None

# Anthropic
try:
    import anthropic
except ImportError:
    anthropic = None

from app.config import settings

logger = logging.getLogger(__name__)


class BaseLLMProvider(ABC):
    """Base class for LLM providers"""

    @abstractmethod
    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generates a response based on messages"""
        pass

    @abstractmethod
    async def generate_streaming_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generates a streaming response"""
        pass


class OpenAILLMProvider(BaseLLMProvider):
    """LLM provider via OpenAI API"""

    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        if not openai:
            raise ImportError("Install openai: pip install openai")

        self.client = openai.OpenAI(api_key=api_key)
        self.model = model

    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generates a response via OpenAI API"""
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
            )

            usage = response.usage
            return {
                "content": response.choices[0].message.content,
                "token_usage": {
                    "input_tokens": usage.prompt_tokens if usage else 0,
                    "output_tokens": usage.completion_tokens if usage else 0,
                    "total_tokens": usage.total_tokens if usage else 0,
                }
            }

        except Exception as e:
            logger.error(f"Error generating OpenAI response: {e}")
            raise

    async def generate_streaming_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generates a streaming response via OpenAI API"""
        try:
            loop = asyncio.get_event_loop()
            stream = await loop.run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True,
                    **kwargs
                )
            )

            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"Error streaming OpenAI response: {e}")
            raise


class AnthropicLLMProvider(BaseLLMProvider):
    """LLM provider via Anthropic API"""

    def __init__(self, api_key: str, model: str = "claude-3-sonnet-20240229"):
        if not anthropic:
            raise ImportError("Install anthropic: pip install anthropic")

        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.model = model

    def _convert_messages(self, messages: List[Dict[str, str]]) -> Tuple[str, List[Dict[str, str]]]:
        """Converts OpenAI message format to Anthropic format"""
        system_message = ""
        converted_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                converted_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

        return system_message, converted_messages

    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generates a response via Anthropic API"""
        try:
            system_message, converted_messages = self._convert_messages(messages)

            response = await self.client.messages.create(
                model=self.model,
                messages=converted_messages,
                system=system_message,
                temperature=temperature,
                max_tokens=max_tokens or 4000,
                **kwargs
            )

            return {
                "content": response.content[0].text,
                "token_usage": {
                    "input_tokens": response.usage.input_tokens if response.usage else 0,
                    "output_tokens": response.usage.output_tokens if response.usage else 0,
                    "total_tokens": (response.usage.input_tokens + response.usage.output_tokens) if response.usage else 0,
                }
            }

        except Exception as e:
            logger.error(f"Error generating Anthropic response: {e}")
            raise

    async def generate_streaming_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generates a streaming response via Anthropic API"""
        try:
            system_message, converted_messages = self._convert_messages(messages)

            async with self.client.messages.stream(
                model=self.model,
                messages=converted_messages,
                system=system_message,
                temperature=temperature,
                max_tokens=max_tokens or 4000,
                **kwargs
            ) as stream:
                async for text in stream.text_stream:
                    yield text

        except Exception as e:
            logger.error(f"Error streaming Anthropic response: {e}")
            raise


class LLMService:
    """Service for working with LLM"""

    def __init__(self, provider: Optional[BaseLLMProvider] = None):
        self.default_provider = provider or self._create_default_provider()

        self.system_prompt = """You are a helpful AI assistant that answers questions based on the provided documentation.

IMPORTANT RULES:
1. Provide accurate and helpful answers
2. Quote relevant parts of documents when appropriate
3. Always respond in English

Context from documents:
{context}

Answer the following question using only information from the provided context."""

    def _create_default_provider(self) -> BaseLLMProvider:
        """Creates the default provider"""
        if settings.DEFAULT_LLM_PROVIDER == "openai":
            if not settings.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY is not set")
            return OpenAILLMProvider(
                api_key=settings.OPENAI_API_KEY,
                model=settings.DEFAULT_LLM_MODEL
            )
        elif settings.DEFAULT_LLM_PROVIDER == "anthropic":
            if not settings.ANTHROPIC_API_KEY:
                raise ValueError("ANTHROPIC_API_KEY is not set")
            return AnthropicLLMProvider(
                api_key=settings.ANTHROPIC_API_KEY,
                model=settings.DEFAULT_LLM_MODEL
            )
        else:
            raise ValueError(
                f"Unsupported LLM provider: {settings.DEFAULT_LLM_PROVIDER}")

    def _get_provider(self, llm_provider: Optional[str] = None, model_name: Optional[str] = None) -> BaseLLMProvider:
        """Gets the LLM provider for a specific request"""
        if not llm_provider:
            return self.default_provider

        if llm_provider == "openai":
            if not settings.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY is not set for OpenAI")
            return OpenAILLMProvider(
                api_key=settings.OPENAI_API_KEY,
                model=model_name or "gpt-4o-mini"
            )
        elif llm_provider == "anthropic":
            if not settings.ANTHROPIC_API_KEY:
                raise ValueError("ANTHROPIC_API_KEY is not set for Anthropic")
            return AnthropicLLMProvider(
                api_key=settings.ANTHROPIC_API_KEY,
                model=model_name or "claude-3-sonnet-20240229"
            )
        else:
            raise ValueError(f"Unsupported provider: {llm_provider}")

    def _build_context(self, relevant_chunks: List[Dict[str, Any]]) -> str:
        """Builds context from relevant chunks"""
        if not relevant_chunks:
            return "No context found."

        context_parts = []
        for i, chunk in enumerate(relevant_chunks, 1):
            metadata = chunk.get('metadata', {})
            source = metadata.get('source', 'Unknown source')

            context_parts.append(
                f"[Source {i}: {source}]\n"
                f"{chunk['text']}\n"
            )

        return "\n".join(context_parts)

    def _prepare_messages(self, query: str, context: str) -> List[Dict[str, str]]:
        """Prepares messages for LLM"""
        system_message = self.system_prompt.format(context=context)

        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": query}
        ]

    async def generate_rag_response(
        self,
        query: str,
        relevant_chunks: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        llm_provider: Optional[str] = None,
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generates a RAG response based on the query and relevant chunks

        Args:
            query: user query
            relevant_chunks: relevant document chunks
            temperature: generation temperature
            max_tokens: maximum token count
            llm_provider: LLM provider (openai, anthropic)
            model_name: specific model

        Returns:
            Dict with response and metadata
        """
        try:
            provider = self._get_provider(llm_provider, model_name)

            context = self._build_context(relevant_chunks)

            if len(context) > settings.MAX_CONTEXT_LENGTH:
                context = context[:settings.MAX_CONTEXT_LENGTH] + \
                    "...\n[Context truncated]"

            messages = self._prepare_messages(query, context)

            result = await provider.generate_response(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )

            used_model = model_name or (
                provider.model if hasattr(provider, 'model')
                else f"{llm_provider}_model"
            )

            return {
                "answer": result["content"],
                "sources_used": len(relevant_chunks),
                "context_length": len(context),
                "query": query,
                "llm_provider": llm_provider or settings.DEFAULT_LLM_PROVIDER,
                "model_used": used_model,
                "token_usage": result["token_usage"]
            }

        except Exception as e:
            logger.error(f"Error generating RAG response: {e}")
            raise

    async def generate_rag_streaming_response(
        self,
        query: str,
        relevant_chunks: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Generates a streaming RAG response

        Yields:
            Dicts with response parts and metadata
        """
        try:
            context = self._build_context(relevant_chunks)

            if len(context) > settings.MAX_CONTEXT_LENGTH:
                context = context[:settings.MAX_CONTEXT_LENGTH] + \
                    "...\n[Context truncated]"

            messages = self._prepare_messages(query, context)

            yield {
                "type": "metadata",
                "sources_used": len(relevant_chunks),
                "context_length": len(context),
                "query": query
            }

            async for chunk in self.provider.generate_streaming_response(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            ):
                yield {
                    "type": "content",
                    "content": chunk
                }

        except Exception as e:
            logger.error(f"Error streaming RAG response: {e}")
            yield {
                "type": "error",
                "error": str(e)
            }

    async def summarize_document(self, text: str, max_tokens: int = 500) -> str:
        """
        Creates a brief document summary

        Args:
            text: document text
            max_tokens: maximum summary length

        Returns:
            Brief summary
        """
        try:
            if len(text) > settings.MAX_CONTEXT_LENGTH:
                text = text[:settings.MAX_CONTEXT_LENGTH] + "..."

            messages = [
                {
                    "role": "system",
                    "content": "Create a concise and informative summary of the provided text. Highlight the main ideas and key points."
                },
                {
                    "role": "user",
                    "content": f"Create a summary of this text:\n\n{text}"
                }
            ]

            result = await self.default_provider.generate_response(
                messages=messages,
                temperature=0.3,
                max_tokens=max_tokens
            )

            return result["content"]

        except Exception as e:
            logger.error(f"Error creating summary: {e}")
            return "Failed to create document summary."

    async def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """
        Extracts keywords from text

        Args:
            text: text to analyze
            max_keywords: maximum number of keywords

        Returns:
            List of keywords
        """
        try:
            if len(text) > settings.MAX_CONTEXT_LENGTH:
                text = text[:settings.MAX_CONTEXT_LENGTH] + "..."

            messages = [
                {
                    "role": "system",
                    "content": f"Extract the {max_keywords} most important keywords or phrases from the text. Return them as a JSON array of strings."
                },
                {
                    "role": "user",
                    "content": f"Text to analyze:\n\n{text}"
                }
            ]

            result = await self.default_provider.generate_response(
                messages=messages,
                temperature=0.1,
                max_tokens=200
            )
            response = result["content"]

            try:
                keywords = json.loads(response)
                if isinstance(keywords, list):
                    return keywords[:max_keywords]
            except json.JSONDecodeError:
                lines = response.strip().split('\n')
                keywords = []
                for line in lines:
                    line = line.strip().strip('-').strip('*').strip()
                    if line:
                        keywords.append(line)
                return keywords[:max_keywords]

            return []

        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            return []

    async def test_connection(self) -> bool:
        """Tests connection to the LLM provider"""
        try:
            test_messages = [
                {"role": "user", "content": "Hello! Reply with one word: working"}
            ]
            result = await self.default_provider.generate_response(test_messages, max_tokens=10)
            return bool(result["content"] and result["content"].strip())
        except Exception as e:
            logger.error(f"Error testing LLM: {e}")
            return False


# Global service instance
_llm_service: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    """Returns the global LLM service instance"""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service
