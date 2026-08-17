from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class LLMProvider(str, Enum):
    """Available LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class ChatRequest(BaseModel):
    """Simplified chat request"""
    message: str
    document_ids: Optional[List[str]] = None  # Specific documents to search
    max_relevant_chunks: Optional[int] = None
    similarity_threshold: Optional[float] = None
    temperature: Optional[float] = None

    # Parameters for model selection
    llm_provider: Optional[LLMProvider] = None  # openai or anthropic
    model_name: Optional[str] = None  # specific model (gpt-4, claude-3-sonnet-20240229, etc.)


class ChatResponse(BaseModel):
    """Simplified chat response"""
    message_id: str
    session_id: str  # Placeholder for compatibility
    content: str
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class RelevantChunk(BaseModel):
    """Relevant document chunk"""
    chunk_id: str
    document_id: str
    document_title: str
    content: str  # Full content without truncation
    similarity: float
    metadata: Dict[str, Any] = Field(default_factory=dict)
