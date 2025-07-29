"""Document upload and knowledge base management endpoints."""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks
from typing import List, Optional
import uuid
import datetime
import os
import tempfile
import mimetypes
from pathlib import Path

from ...config.database import db_config
from ...core.security import InputValidator, SecurityValidator
from ...core.exceptions import (
    convert_to_http_exception, FileUploadError, FileValidationError,
    FileSizeExceededError, SecurityException
)
from ..models.document import (
    DocumentUploadResponse, DocumentListResponse, DocumentDetail,
    DocumentProcessingStatus, DocumentSearchRequest, DocumentSearchResponse
)

router = APIRouter()


@router.post("/documents/upload", response_model=List[DocumentUploadResponse])
async def upload_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    description: Optional[str] = Form(None)
):
    """
    Upload one or more documents to the knowledge base.
    Files will be processed asynchronously for vector embeddings.
    """
    try:
        # Validate description if provided
        if description:
            InputValidator.validate_string_input(description, "description", max_length=500)
        
        uploaded_docs = []
        
        for file in files:
            # Validate file
            validation_result = await _validate_uploaded_file(file)
            if not validation_result["valid"]:
                raise FileValidationError(file.filename, validation_result["error"])
            
            # Create document record
            doc_id = str(uuid.uuid4())
            created_at = datetime.datetime.utcnow().isoformat()
            
            # Save file temporarily
            temp_file_path = await _save_uploaded_file(file, doc_id)
            
            # Create database record
            doc_data = {
                "id": doc_id,
                "filename": file.filename,
                "original_filename": file.filename,
                "file_size": validation_result["file_size"],
                "mime_type": validation_result["mime_type"],
                "description": description,
                "status": "processing",
                "temp_file_path": temp_file_path,
                "created_at": created_at,
                "updated_at": created_at
            }
            
            # Insert into documents table
            result = db_config.supabase.table("documents").insert(doc_data).execute()
            if not result or getattr(result, "status_code", 200) >= 400:
                raise HTTPException(status_code=500, detail=f"Database error: {result}")
            
            # Schedule background processing
            background_tasks.add_task(_process_document_async, doc_id, temp_file_path)
            
            uploaded_docs.append(DocumentUploadResponse(
                id=doc_id,
                filename=file.filename,
                status="processing",
                message="Document uploaded successfully and queued for processing"
            ))
        
        return uploaded_docs
        
    except (FileValidationError, FileSizeExceededError, SecurityException) as e:
        raise convert_to_http_exception(e)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.get("/documents", response_model=DocumentListResponse)
def list_documents(limit: int = 50, offset: int = 0):
    """List all uploaded documents with their processing status."""
    try:
        # Validate pagination parameters
        if limit < 1 or limit > 100:
            raise HTTPException(status_code=400, detail="Limit must be between 1 and 100")
        if offset < 0:
            raise HTTPException(status_code=400, detail="Offset must be non-negative")
        
        # Query documents
        query = db_config.supabase.table("documents")\
            .select("id, filename, file_size, mime_type, description, status, created_at, updated_at")\
            .order("created_at", desc=True)\
            .range(offset, offset + limit - 1)
        
        result = query.execute()
        documents = getattr(result, 'data', [])
        
        # Get total count
        count_result = db_config.supabase.table("documents").select("id", count="exact").execute()
        total_count = getattr(count_result, 'count', 0)
        
        return DocumentListResponse(
            documents=documents,
            total_count=total_count,
            limit=limit,
            offset=offset
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")


@router.get("/documents/{doc_id}", response_model=DocumentDetail)
def get_document(doc_id: str):
    """Get detailed information about a specific document."""
    try:
        # Validate UUID
        InputValidator.validate_uuid(doc_id, "doc_id")
        
        # Query document
        result = db_config.supabase.table("documents")\
            .select("*")\
            .eq("id", doc_id)\
            .single()\
            .execute()
        
        document = getattr(result, 'data', None)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Get processing chunks info if available
        chunks_result = db_config.supabase.table("document_chunks")\
            .select("id")\
            .eq("document_id", doc_id)\
            .execute()
        
        chunks_count = len(getattr(chunks_result, 'data', []))
        
        return DocumentDetail(
            **document,
            chunks_count=chunks_count
        )
        
    except HTTPException:
        raise
    except SecurityException as e:
        raise convert_to_http_exception(e)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get document: {str(e)}")


@router.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    """Delete a document and all its associated chunks."""
    try:
        # Validate UUID
        InputValidator.validate_uuid(doc_id, "doc_id")
        
        # Check if document exists
        doc_result = db_config.supabase.table("documents")\
            .select("id, filename")\
            .eq("id", doc_id)\
            .single()\
            .execute()
        
        document = getattr(doc_result, 'data', None)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Delete chunks first (foreign key constraint)
        db_config.supabase.table("document_chunks")\
            .delete()\
            .eq("document_id", doc_id)\
            .execute()
        
        # Delete document
        db_config.supabase.table("documents")\
            .delete()\
            .eq("id", doc_id)\
            .execute()
        
        return {
            "message": f"Document '{document['filename']}' deleted successfully",
            "document_id": doc_id
        }
        
    except HTTPException:
        raise
    except SecurityException as e:
        raise convert_to_http_exception(e)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")


@router.post("/documents/{doc_id}/reprocess")
def reprocess_document(doc_id: str, background_tasks: BackgroundTasks):
    """Reprocess a document (regenerate embeddings)."""
    try:
        # Validate UUID
        InputValidator.validate_uuid(doc_id, "doc_id")
        
        # Check if document exists
        doc_result = db_config.supabase.table("documents")\
            .select("id, filename, temp_file_path")\
            .eq("id", doc_id)\
            .single()\
            .execute()
        
        document = getattr(doc_result, 'data', None)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Update status to processing
        db_config.supabase.table("documents")\
            .update({"status": "processing", "updated_at": datetime.datetime.utcnow().isoformat()})\
            .eq("id", doc_id)\
            .execute()
        
        # Delete existing chunks
        db_config.supabase.table("document_chunks")\
            .delete()\
            .eq("document_id", doc_id)\
            .execute()
        
        # Schedule reprocessing
        temp_file_path = document.get("temp_file_path")
        if temp_file_path and os.path.exists(temp_file_path):
            background_tasks.add_task(_process_document_async, doc_id, temp_file_path)
        else:
            # If temp file is gone, mark as failed
            db_config.supabase.table("documents")\
                .update({
                    "status": "failed",
                    "error_message": "Original file no longer available for reprocessing",
                    "updated_at": datetime.datetime.utcnow().isoformat()
                })\
                .eq("id", doc_id)\
                .execute()
            
            raise HTTPException(status_code=400, detail="Original file no longer available for reprocessing")
        
        return {
            "message": f"Document '{document['filename']}' queued for reprocessing",
            "document_id": doc_id
        }
        
    except HTTPException:
        raise
    except SecurityException as e:
        raise convert_to_http_exception(e)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reprocess document: {str(e)}")


async def _validate_uploaded_file(file: UploadFile) -> dict:
    """Validate uploaded file for security and size constraints."""
    try:
        # Get file size
        content = await file.read()
        file_size = len(content)
        await file.seek(0)  # Reset file pointer
        
        # Check file size (10MB limit from settings)
        from ...config.settings import get_settings
        settings = get_settings()
        max_size_bytes = settings.max_file_size_mb * 1024 * 1024
        
        if file_size > max_size_bytes:
            return {
                "valid": False,
                "error": f"File size ({file_size / 1024 / 1024:.1f}MB) exceeds limit ({settings.max_file_size_mb}MB)"
            }
        
        # Check file type
        mime_type, _ = mimetypes.guess_type(file.filename)
        allowed_types = ["text/plain", "text/markdown", "application/pdf", "text/csv"]
        
        if mime_type not in allowed_types:
            return {
                "valid": False,
                "error": f"File type '{mime_type}' not allowed. Supported types: {', '.join(allowed_types)}"
            }
        
        # Validate filename
        try:
            InputValidator.sanitize_filename(file.filename)
        except Exception as e:
            return {
                "valid": False,
                "error": f"Invalid filename: {str(e)}"
            }
        
        # Basic content validation
        if file_size == 0:
            return {
                "valid": False,
                "error": "File is empty"
            }
        
        return {
            "valid": True,
            "file_size": file_size,
            "mime_type": mime_type
        }
        
    except Exception as e:
        return {
            "valid": False,
            "error": f"File validation failed: {str(e)}"
        }


async def _save_uploaded_file(file: UploadFile, doc_id: str) -> str:
    """Save uploaded file to temporary storage."""
    try:
        # Create temp directory if it doesn't exist
        temp_dir = Path(tempfile.gettempdir()) / "credilinq_uploads"
        temp_dir.mkdir(exist_ok=True)
        
        # Generate safe filename
        safe_filename = InputValidator.sanitize_filename(file.filename)
        temp_file_path = temp_dir / f"{doc_id}_{safe_filename}"
        
        # Save file
        content = await file.read()
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(content)
        
        return str(temp_file_path)
        
    except Exception as e:
        raise FileUploadError(file.filename, f"Failed to save file: {str(e)}")


async def _process_document_async(doc_id: str, file_path: str):
    """Process document in the background - chunk and create embeddings."""
    try:
        from ...agents.specialized.document_processor import DocumentProcessorAgent
        
        # Create processor agent
        processor = DocumentProcessorAgent()
        
        # Process the document
        result = processor.execute_safe({
            "document_id": doc_id,
            "file_path": file_path
        })
        
        if result.success:
            # Update document status
            db_config.supabase.table("documents")\
                .update({
                    "status": "completed",
                    "chunks_count": result.data.get("chunks_created", 0),
                    "processing_completed_at": datetime.datetime.utcnow().isoformat(),
                    "updated_at": datetime.datetime.utcnow().isoformat()
                })\
                .eq("id", doc_id)\
                .execute()
        else:
            # Update with error
            db_config.supabase.table("documents")\
                .update({
                    "status": "failed",
                    "error_message": result.error_message,
                    "updated_at": datetime.datetime.utcnow().isoformat()
                })\
                .eq("id", doc_id)\
                .execute()
        
        # Clean up temp file
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Warning: Failed to clean up temp file {file_path}: {e}")
            
    except Exception as e:
        # Update document with error status
        db_config.supabase.table("documents")\
            .update({
                "status": "failed",
                "error_message": str(e),
                "updated_at": datetime.datetime.utcnow().isoformat()
            })\
            .eq("id", doc_id)\
            .execute()


@router.post("/documents/search", response_model=DocumentSearchResponse)
def search_documents(request: DocumentSearchRequest):
    """Search documents using vector similarity."""
    try:
        import time
        start_time = time.time()
        
        from ...agents.specialized.document_processor import DocumentProcessorAgent
        
        # Create processor agent for search
        processor = DocumentProcessorAgent()
        
        # Perform search
        result = processor.search_documents(
            query=request.query,
            limit=request.limit,
            similarity_threshold=request.similarity_threshold
        )
        
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        if result.success:
            return DocumentSearchResponse(
                query=request.query,
                results=result.data["results"],
                total_found=result.data["total_found"],
                processing_time_ms=processing_time
            )
        else:
            raise HTTPException(status_code=500, detail=f"Search failed: {result.error_message}")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")