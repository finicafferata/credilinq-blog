"""Document management data models."""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class DocumentUploadResponse(BaseModel):
    """Response model for document upload."""
    id: str = Field(..., description="Unique document identifier")
    filename: str = Field(..., description="Original filename")
    status: str = Field(..., description="Processing status (processing, completed, failed)")
    message: str = Field(..., description="Status message")


class DocumentSummary(BaseModel):
    """Summary model for document listings."""
    id: str
    filename: str
    file_size: int
    mime_type: str
    description: Optional[str] = None
    status: str
    created_at: str
    updated_at: str


class DocumentDetail(DocumentSummary):
    """Detailed document model with processing information."""
    original_filename: str
    chunks_count: Optional[int] = 0
    processing_completed_at: Optional[str] = None
    error_message: Optional[str] = None
    temp_file_path: Optional[str] = None


class DocumentListResponse(BaseModel):
    """Response model for document listing."""
    documents: List[DocumentSummary]
    total_count: int
    limit: int
    offset: int


class DocumentProcessingStatus(BaseModel):
    """Document processing status model."""
    document_id: str
    status: str
    progress_percentage: Optional[float] = None
    current_step: Optional[str] = None
    chunks_processed: Optional[int] = None
    total_chunks: Optional[int] = None
    error_message: Optional[str] = None
    updated_at: str


class DocumentChunk(BaseModel):
    """Model for document chunks."""
    id: str
    document_id: str
    chunk_index: int
    content: str
    embedding: Optional[List[float]] = None
    metadata: Optional[dict] = None
    created_at: str


class DocumentSearchRequest(BaseModel):
    """Request model for document search."""
    query: str = Field(..., min_length=1, max_length=500, description="Search query")
    limit: int = Field(default=10, ge=1, le=50, description="Maximum number of results")
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Minimum similarity score")


class DocumentSearchResult(BaseModel):
    """Individual search result."""
    chunk_id: str
    document_id: str
    document_filename: str
    content: str
    similarity_score: float
    chunk_index: int
    metadata: Optional[dict] = None


class DocumentSearchResponse(BaseModel):
    """Response model for document search."""
    query: str
    results: List[DocumentSearchResult]
    total_found: int
    processing_time_ms: float