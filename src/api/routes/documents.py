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
            
            # Insert into Document table using PostgreSQL
            with db_config.get_db_connection() as conn:
                cur = conn.cursor()
                cur.execute("""
                    INSERT INTO "Document" (id, title, "storagePath", "uploadedAt")
                    VALUES (%s, %s, %s, %s)
                """, (doc_id, file.filename, temp_file_path, created_at))
                conn.commit()
            
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
        
        # Query documents using PostgreSQL
        with db_config.get_db_connection() as conn:
            cur = conn.cursor()
            
            # Get documents with pagination
            cur.execute("""
                SELECT id, title as filename, "storagePath", 'text/plain' as mime_type, 
                       title as description, 'completed' as status, 
                       "uploadedAt" as created_at, "uploadedAt" as updated_at
                FROM "Document"
                ORDER BY "uploadedAt" DESC
                LIMIT %s OFFSET %s
            """, (limit, offset))
            
            documents = []
            for row in cur.fetchall():
                # Calculate file size from the actual file
                file_size = 0
                storage_path = row[2]  # storagePath
                if storage_path and os.path.exists(storage_path):
                    try:
                        file_size = os.path.getsize(storage_path)
                    except OSError:
                        file_size = 0
                
                # Determine mime type from file extension
                mime_type = 'text/plain'
                if storage_path:
                    mime_type_detected, _ = mimetypes.guess_type(storage_path)
                    if mime_type_detected:
                        mime_type = mime_type_detected
                
                documents.append({
                    'id': row[0],
                    'filename': row[1],
                    'file_size': file_size,
                    'mime_type': mime_type,
                    'description': row[4],
                    'status': row[5],
                    'created_at': row[6].isoformat() if row[6] else None,
                    'updated_at': row[7].isoformat() if row[7] else None
                })
            
            # Get total count
            cur.execute('SELECT COUNT(*) FROM "Document"')
            total_count = cur.fetchone()[0]
        
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
        
        # Query document using PostgreSQL
        with db_config.get_db_connection() as conn:
            cur = conn.cursor()
            
            cur.execute("""
                SELECT id, title, "storagePath", "uploadedAt"
                FROM "Document"
                WHERE id = %s
            """, (doc_id,))
            
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Document not found")
            
            # Calculate file size and mime type
            storage_path = row[2]  # storagePath
            file_size = 0
            mime_type = 'text/plain'
            
            if storage_path and os.path.exists(storage_path):
                try:
                    file_size = os.path.getsize(storage_path)
                except OSError:
                    file_size = 0
                
                mime_type_detected, _ = mimetypes.guess_type(storage_path)
                if mime_type_detected:
                    mime_type = mime_type_detected
            
            document = {
                'id': row[0],
                'filename': row[1],
                'original_filename': row[1],
                'file_size': file_size,
                'mime_type': mime_type,
                'description': row[1],
                'status': 'completed',
                'created_at': row[3].isoformat() if row[3] else None,
                'updated_at': row[3].isoformat() if row[3] else None,
                'temp_file_path': row[2]
            }
            
            # Get processing chunks info if available
            cur.execute("""
                SELECT COUNT(*) FROM "DocumentChunk" WHERE "documentId" = %s
            """, (doc_id,))
            
            chunks_count = cur.fetchone()[0]
        
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
        
        # Check if document exists and delete using PostgreSQL
        with db_config.get_db_connection() as conn:
            cur = conn.cursor()
            
            cur.execute("""
                SELECT id, title FROM "Document" WHERE id = %s
            """, (doc_id,))
            
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Document not found")
            
            document = {'id': row[0], 'filename': row[1]}
            
            # Delete chunks first (foreign key constraint)
            cur.execute('DELETE FROM "DocumentChunk" WHERE "documentId" = %s', (doc_id,))
            
            # Delete document
            cur.execute('DELETE FROM "Document" WHERE id = %s', (doc_id,))
            
            conn.commit()
        
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


@router.get("/documents/stats")
def get_knowledge_base_stats():
    """Get comprehensive Knowledge Base statistics."""
    try:
        from ...utils.knowledge_base_manager import KnowledgeBaseManager
        
        manager = KnowledgeBaseManager()
        stats = manager.get_document_statistics()
        
        if 'error' in stats:
            raise HTTPException(status_code=500, detail=f"Failed to get statistics: {stats['error']}")
        
        return stats
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")


@router.post("/documents/cleanup")
def cleanup_knowledge_base(dry_run: bool = True):
    """Clean up orphaned documents and missing files."""
    try:
        from ...utils.knowledge_base_manager import KnowledgeBaseManager
        
        manager = KnowledgeBaseManager()
        result = manager.cleanup_orphaned_documents(dry_run=dry_run)
        
        if 'error' in result:
            raise HTTPException(status_code=500, detail=f"Cleanup failed: {result['error']}")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")


@router.post("/documents/bulk-upload")
async def bulk_upload_from_directory(
    background_tasks: BackgroundTasks,
    directory_path: str,
    file_pattern: str = "*",
    recursive: bool = False,
    description_prefix: str = "Bulk uploaded document"
):
    """
    Bulk upload documents from a directory.
    
    Args:
        directory_path: Path to directory containing documents
        file_pattern: File pattern to match (e.g., "*.pdf", "*.txt")
        recursive: Whether to search subdirectories
        description_prefix: Prefix for document descriptions
    """
    try:
        import glob
        from pathlib import Path
        
        # Validate directory path
        directory = Path(directory_path)
        if not directory.exists() or not directory.is_dir():
            raise HTTPException(status_code=400, detail="Directory not found or not a directory")
        
        # Find matching files
        if recursive:
            pattern = str(directory / "**" / file_pattern)
            files = glob.glob(pattern, recursive=True)
        else:
            pattern = str(directory / file_pattern)
            files = glob.glob(pattern)
        
        # Filter to only include files (not directories)
        files = [f for f in files if Path(f).is_file()]
        
        if not files:
            return {
                "message": "No matching files found",
                "directory": str(directory),
                "pattern": file_pattern,
                "files_found": 0
            }
        
        # Validate file types and sizes
        from ...config.settings import get_settings
        settings = get_settings()
        max_size_bytes = settings.max_file_size_mb * 1024 * 1024
        allowed_extensions = [".txt", ".md", ".pdf", ".csv"]
        
        valid_files = []
        skipped_files = []
        
        for file_path in files:
            file_path = Path(file_path)
            
            # Check extension
            if file_path.suffix.lower() not in allowed_extensions:
                skipped_files.append({"file": str(file_path), "reason": "unsupported_type"})
                continue
            
            # Check size
            if file_path.stat().st_size > max_size_bytes:
                skipped_files.append({"file": str(file_path), "reason": "too_large"})
                continue
            
            valid_files.append(file_path)
        
        if not valid_files:
            return {
                "message": "No valid files found for upload",
                "files_found": len(files),
                "skipped_files": skipped_files,
                "valid_files": 0
            }
        
        # Process valid files
        uploaded_docs = []
        
        for file_path in valid_files:
            try:
                # Create document record
                doc_id = str(uuid.uuid4())
                created_at = datetime.datetime.utcnow().isoformat()
                
                # Create database record
                doc_data = {
                    "id": doc_id,
                    "filename": file_path.name,
                    "original_filename": file_path.name,
                    "file_size": file_path.stat().st_size,
                    "mime_type": mimetypes.guess_type(str(file_path))[0],
                    "description": f"{description_prefix}: {file_path.name}",
                    "status": "processing",
                    "temp_file_path": str(file_path),  # Use original path
                    "created_at": created_at,
                    "updated_at": created_at
                }
                
                # Insert into Document table using PostgreSQL
                with db_config.get_db_connection() as conn:
                    cur = conn.cursor()
                    cur.execute("""
                        INSERT INTO "Document" (id, title, "storagePath", "uploadedAt")
                        VALUES (%s, %s, %s, %s)
                    """, (doc_id, file_path.name, str(file_path), created_at))
                    conn.commit()
                
                # Schedule background processing
                background_tasks.add_task(_process_document_async, doc_id, str(file_path))
                
                uploaded_docs.append({
                    "id": doc_id,
                    "filename": file_path.name,
                    "status": "processing",
                    "file_path": str(file_path)
                })
                
            except Exception as e:
                skipped_files.append({"file": str(file_path), "reason": f"error: {str(e)}"})
                continue
        
        return {
            "message": f"Bulk upload initiated for {len(uploaded_docs)} files",
            "directory": str(directory),
            "pattern": file_pattern,
            "files_found": len(files),
            "valid_files": len(valid_files),
            "uploaded_count": len(uploaded_docs),
            "skipped_count": len(skipped_files),
            "uploaded_documents": uploaded_docs,
            "skipped_files": skipped_files
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Bulk upload failed: {str(e)}")