"""
Document Processor Agent - Handles document chunking and vector embedding generation.
"""

from typing import Dict, List, Optional, Any
import os
import uuid
import hashlib
from pathlib import Path
import mimetypes

from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, UnstructuredPDFLoader
from langchain.schema import Document

from ..core.base_agent import BaseAgent, AgentResult, AgentExecutionContext, AgentMetadata, AgentType
from ...core.security import SecurityValidator
from ...core.exceptions import AgentExecutionError


class DocumentProcessorAgent(BaseAgent[Dict[str, Any]]):
    """
    Agent responsible for processing uploaded documents into searchable chunks with embeddings.
    """
    
    def __init__(self, metadata: Optional[AgentMetadata] = None):
        if metadata is None:
            metadata = AgentMetadata(
                agent_type=AgentType.DOCUMENT_PROCESSOR,
                name="DocumentProcessorAgent",
                description="Processes documents into chunks with vector embeddings for knowledge base",
                capabilities=[
                    "document_chunking",
                    "vector_embedding_generation",
                    "content_extraction",
                    "metadata_processing",
                    "multiple_file_formats"
                ],
                version="2.0.0"
            )
        
        super().__init__(metadata)
        self.security_validator = SecurityValidator()
        self.embeddings_model = None
        self.text_splitter = None
        
    def _initialize(self):
        """Initialize the embeddings model and text splitter."""
        try:
            from ...config.settings import get_settings
            settings = get_settings()
            
            # Initialize embeddings model
            self.embeddings_model = OpenAIEmbeddings(
                openai_api_key=settings.OPENAI_API_KEY,
                model="text-embedding-ada-002"
            )
            
            # Initialize text splitter with optimal settings
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,  # Characters per chunk
                chunk_overlap=200,  # Overlap between chunks
                length_function=len,
                separators=["\\n\\n", "\\n", ". ", " ", ""]
            )
            
            self.logger.info("DocumentProcessorAgent initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize DocumentProcessorAgent: {str(e)}")
            raise
    
    def _validate_input(self, input_data: Dict[str, Any]) -> None:
        """Validate input data for document processing."""
        super()._validate_input(input_data)
        
        required_fields = ["document_id", "file_path"]
        for field in required_fields:
            if field not in input_data:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate file exists
        file_path = input_data["file_path"]
        if not os.path.exists(file_path):
            raise ValueError(f"File not found: {file_path}")
        
        # Security validation
        self.security_validator.validate_input(str(input_data["document_id"]))
    
    def execute(
        self, 
        input_data: Dict[str, Any], 
        context: Optional[AgentExecutionContext] = None,
        **kwargs
    ) -> AgentResult:
        """
        Process a document into chunks with embeddings.
        
        Args:
            input_data: Dictionary containing:
                - document_id: ID of the document
                - file_path: Path to the document file
                - custom_chunk_size: Optional custom chunk size (optional)
                - custom_overlap: Optional custom overlap size (optional)
            context: Execution context
            
        Returns:
            AgentResult: Processing results with chunk information
        """
        try:
            document_id = input_data["document_id"]
            file_path = input_data["file_path"]
            custom_chunk_size = input_data.get("custom_chunk_size")
            custom_overlap = input_data.get("custom_overlap")
            
            self.logger.info(f"Processing document {document_id}: {file_path}")
            
            # Load document content
            document_content = self._load_document(file_path)
            
            # Create custom text splitter if parameters provided
            text_splitter = self.text_splitter
            if custom_chunk_size or custom_overlap:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=custom_chunk_size or 1000,
                    chunk_overlap=custom_overlap or 200,
                    length_function=len,
                    separators=["\\n\\n", "\\n", ". ", " ", ""]
                )
            
            # Split document into chunks
            chunks = text_splitter.split_documents([document_content])
            
            self.logger.info(f"Split document into {len(chunks)} chunks")
            
            # Generate embeddings for chunks
            processed_chunks = []
            for i, chunk in enumerate(chunks):
                try:
                    # Generate embedding
                    embedding = self.embeddings_model.embed_query(chunk.page_content)
                    
                    # Validate embedding
                    self.security_validator.validate_vector_embedding(embedding)
                    
                    # Create chunk record
                    chunk_data = {
                        "id": str(uuid.uuid4()),
                        "document_id": document_id,
                        "chunk_index": i,
                        "content": chunk.page_content,
                        "embedding": embedding,
                        "metadata": {
                            "source": file_path,
                            "chunk_size": len(chunk.page_content),
                            "chunk_hash": hashlib.md5(chunk.page_content.encode()).hexdigest(),
                            **chunk.metadata
                        },
                        "created_at": context.started_at.isoformat() if context else None
                    }
                    
                    processed_chunks.append(chunk_data)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to process chunk {i}: {str(e)}")
                    continue
            
            # Store chunks in database
            chunks_stored = self._store_chunks(processed_chunks)
            
            result_data = {
                "document_id": document_id,
                "chunks_created": chunks_stored,
                "total_chunks_attempted": len(chunks),
                "success_rate": chunks_stored / len(chunks) if chunks else 0,
                "processing_summary": {
                    "original_content_length": len(document_content.page_content),
                    "average_chunk_size": sum(len(chunk["content"]) for chunk in processed_chunks) / len(processed_chunks) if processed_chunks else 0,
                    "chunk_size_used": text_splitter.chunk_size,
                    "overlap_used": text_splitter.chunk_overlap
                }
            }
            
            self.logger.info(f"Successfully processed document {document_id}: {chunks_stored}/{len(chunks)} chunks stored")
            
            return AgentResult(
                success=True,
                data=result_data,
                metadata={
                    "agent_type": "document_processor",
                    "document_id": document_id,
                    "chunks_created": chunks_stored,
                    "file_processed": os.path.basename(file_path)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Document processing failed: {str(e)}")
            return AgentResult(
                success=False,
                error_message=str(e),
                error_code="DOCUMENT_PROCESSING_FAILED"
            )
    
    def _load_document(self, file_path: str) -> Document:
        """Load document content based on file type."""
        file_path = Path(file_path)
        mime_type, _ = mimetypes.guess_type(file_path)
        
        try:
            if mime_type == "application/pdf":
                loader = UnstructuredPDFLoader(str(file_path))
            elif mime_type in ["text/plain", "text/markdown"]:
                loader = TextLoader(str(file_path), encoding="utf-8")
            else:
                # Fallback to text loader
                self.logger.warning(f"Unknown mime type {mime_type}, treating as text")
                loader = TextLoader(str(file_path), encoding="utf-8")
            
            documents = loader.load()
            
            if not documents:
                raise ValueError("No content could be extracted from the document")
            
            # Combine all pages/sections into a single document
            combined_content = "\\n\\n".join([doc.page_content for doc in documents])
            
            return Document(
                page_content=combined_content,
                metadata={
                    "source": str(file_path),
                    "filename": file_path.name,
                    "mime_type": mime_type,
                    "total_pages": len(documents)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to load document {file_path}: {str(e)}")
            raise AgentExecutionError("document_processor", "document_loading", str(e))
    
    def _store_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """Store processed chunks in the database."""
        try:
            from ...config.database import db_config
            
            stored_count = 0
            
            for chunk in chunks:
                try:
                    # Convert embedding to the format expected by pgvector
                    chunk_data = {
                        "id": chunk["id"],
                        "document_id": chunk["document_id"],
                        "chunk_index": chunk["chunk_index"],
                        "content": chunk["content"],
                        "embedding": chunk["embedding"],  # pgvector will handle the conversion
                        "metadata": chunk["metadata"],
                        "created_at": chunk["created_at"]
                    }
                    
                    # Insert chunk
                    result = db_config.supabase.table("document_chunks").insert(chunk_data).execute()
                    
                    if result and getattr(result, "status_code", 200) < 400:
                        stored_count += 1
                    else:
                        self.logger.warning(f"Failed to store chunk {chunk['id']}: {result}")
                        
                except Exception as e:
                    self.logger.warning(f"Failed to store chunk {chunk['id']}: {str(e)}")
                    continue
            
            return stored_count
            
        except Exception as e:
            self.logger.error(f"Failed to store chunks: {str(e)}")
            raise
    
    def search_documents(
        self, 
        query: str, 
        limit: int = 10, 
        similarity_threshold: float = 0.7
    ) -> AgentResult:
        """
        Search documents using vector similarity.
        
        Args:
            query: Search query
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score
            
        Returns:
            AgentResult: Search results
        """
        try:
            # Generate query embedding
            query_embedding = self.embeddings_model.embed_query(query)
            
            # Validate embedding
            self.security_validator.validate_vector_embedding(query_embedding)
            
            # Perform vector search
            search_results = self._vector_search(query_embedding, limit, similarity_threshold)
            
            result_data = {
                "query": query,
                "results": search_results,
                "total_found": len(search_results),
                "search_parameters": {
                    "limit": limit,
                    "similarity_threshold": similarity_threshold
                }
            }
            
            return AgentResult(
                success=True,
                data=result_data,
                metadata={
                    "agent_type": "document_search",
                    "query_length": len(query),
                    "results_count": len(search_results)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Document search failed: {str(e)}")
            return AgentResult(
                success=False,
                error_message=str(e),
                error_code="DOCUMENT_SEARCH_FAILED"
            )
    
    def _vector_search(
        self, 
        query_embedding: List[float], 
        limit: int, 
        similarity_threshold: float
    ) -> List[Dict[str, Any]]:
        """Perform vector similarity search in the database."""
        try:
            from ...config.database import db_config
            
            # Convert embedding to string format for SQL
            embedding_str = '[' + ','.join([str(x) for x in query_embedding]) + ']'
            
            # Perform similarity search using pgvector
            # Note: This uses cosine distance (<#>) - lower values mean higher similarity
            query = f"""
                SELECT 
                    dc.id as chunk_id,
                    dc.document_id,
                    dc.content,
                    dc.chunk_index,
                    dc.metadata,
                    d.filename as document_filename,
                    (dc.embedding <#> %s::vector) as distance
                FROM document_chunks dc
                JOIN documents d ON dc.document_id = d.id
                WHERE d.status = 'completed'
                ORDER BY distance ASC
                LIMIT %s;
            """
            
            # Execute search
            with db_config.get_db_connection() as conn:
                cur = conn.cursor()
                cur.execute(query, (embedding_str, limit))
                rows = cur.fetchall()
            
            # Convert distance to similarity score and filter
            results = []
            for row in rows:
                # Convert cosine distance to similarity (1 - distance)
                similarity_score = 1 - row["distance"]
                
                if similarity_score >= similarity_threshold:
                    results.append({
                        "chunk_id": row["chunk_id"],
                        "document_id": row["document_id"],
                        "document_filename": row["document_filename"],
                        "content": row["content"],
                        "chunk_index": row["chunk_index"],
                        "similarity_score": similarity_score,
                        "metadata": row["metadata"]
                    })
            
            return results
            
        except Exception as e:
            self.logger.error(f"Vector search failed: {str(e)}")
            raise
    
    def get_document_stats(self, document_id: str) -> AgentResult:
        """Get statistics about a processed document."""
        try:
            from ...config.database import db_config
            
            # Get chunk statistics
            stats_query = """
                SELECT 
                    COUNT(*) as total_chunks,
                    AVG(LENGTH(content)) as avg_chunk_length,
                    MIN(LENGTH(content)) as min_chunk_length,
                    MAX(LENGTH(content)) as max_chunk_length,
                    SUM(LENGTH(content)) as total_content_length
                FROM document_chunks 
                WHERE document_id = %s
            """
            
            with db_config.get_db_connection() as conn:
                cur = conn.cursor()
                cur.execute(stats_query, (document_id,))
                stats = cur.fetchone()
            
            if not stats or stats["total_chunks"] == 0:
                raise ValueError(f"No chunks found for document {document_id}")
            
            result_data = {
                "document_id": document_id,
                "statistics": {
                    "total_chunks": stats["total_chunks"],
                    "average_chunk_length": round(stats["avg_chunk_length"], 2),
                    "min_chunk_length": stats["min_chunk_length"],
                    "max_chunk_length": stats["max_chunk_length"],
                    "total_content_length": stats["total_content_length"]
                }
            }
            
            return AgentResult(
                success=True,
                data=result_data,
                metadata={
                    "agent_type": "document_stats",
                    "document_id": document_id
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get document stats: {str(e)}")
            return AgentResult(
                success=False,
                error_message=str(e),
                error_code="DOCUMENT_STATS_FAILED"
            )