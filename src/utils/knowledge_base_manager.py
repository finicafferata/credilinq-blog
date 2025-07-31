"""
Knowledge Base Management Utility
Provides comprehensive document management functionality for the Knowledge Base system.
"""

import os
import json
import asyncio
import psycopg2.extras
from typing import List, Dict, Optional, Any
from datetime import datetime
from pathlib import Path

from ..config.database import db_config
from ..agents.specialized.document_processor import DocumentProcessorAgent
from ..api.models.document import DocumentSearchRequest


class KnowledgeBaseManager:
    """
    Comprehensive Knowledge Base management utility that provides:
    - Document listing with metadata
    - Batch operations
    - Search functionality
    - Statistics and analytics
    - Document refresh capabilities
    """
    
    def __init__(self):
        self.db_config = db_config
        self.processor = DocumentProcessorAgent()
        
    def list_all_documents(self, include_chunks_info: bool = True) -> List[Dict[str, Any]]:
        """
        List all documents in the Knowledge Base with comprehensive metadata.
        
        Args:
            include_chunks_info: Whether to include chunk count and processing info
            
        Returns:
            List of document records with metadata
        """
        try:
            with self.db_config.get_db_connection() as conn:
                cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                
                if include_chunks_info:
                    query = """
                        SELECT 
                            d.id,
                            d.title,
                            d."storagePath",
                            d."uploadedAt",
                            COUNT(dc.id) as chunks_count,
                            AVG(LENGTH(dc.content)) as avg_chunk_length,
                            MAX(dc.id) as last_chunk_id
                        FROM "Document" d
                        LEFT JOIN "DocumentChunk" dc ON d.id = dc."documentId"
                        GROUP BY d.id, d.title, d."storagePath", d."uploadedAt"
                        ORDER BY d."uploadedAt" DESC
                    """
                else:
                    query = """
                        SELECT id, title, "storagePath", "uploadedAt"
                        FROM "Document"
                        ORDER BY "uploadedAt" DESC
                    """
                
                cur.execute(query)
                documents = cur.fetchall()
                
                # Convert to list of dictionaries
                result = []
                for doc in documents:
                    doc_dict = dict(doc)
                    
                    # Convert datetime to ISO string if present
                    if doc_dict.get('uploadedAt'):
                        doc_dict['uploadedAt'] = doc_dict['uploadedAt'].isoformat()
                    
                    # Add file size if file exists
                    storage_path = doc_dict.get('storagePath')
                    if storage_path and os.path.exists(storage_path):
                        doc_dict['file_size'] = os.path.getsize(storage_path)
                        doc_dict['file_exists'] = True
                    else:
                        doc_dict['file_size'] = None
                        doc_dict['file_exists'] = False
                    
                    result.append(doc_dict)
                    
                return result
                
        except Exception as e:
            print(f"Error listing documents: {e}")
            return []
    
    def get_document_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the Knowledge Base.
        
        Returns:
            Dictionary with various statistics
        """
        try:
            with self.db_config.get_db_connection() as conn:
                cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                
                # Document statistics
                cur.execute("""
                    SELECT 
                        COUNT(*) as total_documents,
                        COUNT(CASE WHEN "uploadedAt" > NOW() - INTERVAL '7 days' THEN 1 END) as recent_documents,
                        MIN("uploadedAt") as oldest_document,
                        MAX("uploadedAt") as newest_document
                    FROM "Document"
                """)
                doc_stats = cur.fetchone()
                
                # Chunk statistics
                cur.execute("""
                    SELECT 
                        COUNT(*) as total_chunks,
                        AVG(LENGTH(content)) as avg_chunk_length,
                        MIN(LENGTH(content)) as min_chunk_length,
                        MAX(LENGTH(content)) as max_chunk_length,
                        SUM(LENGTH(content)) as total_content_length
                    FROM "DocumentChunk"
                """)
                chunk_stats = cur.fetchone()
                
                # Documents without chunks (processing issues)
                cur.execute("""
                    SELECT COUNT(*) as orphaned_documents
                    FROM "Document" d
                    LEFT JOIN "DocumentChunk" dc ON d.id = dc."documentId"
                    WHERE dc.id IS NULL
                """)
                orphaned_count = cur.fetchone()
                
                return {
                    "document_statistics": {
                        "total_documents": doc_stats["total_documents"],
                        "recent_documents": doc_stats["recent_documents"],
                        "oldest_document": doc_stats["oldest_document"].isoformat() if doc_stats["oldest_document"] else None,
                        "newest_document": doc_stats["newest_document"].isoformat() if doc_stats["newest_document"] else None,
                        "orphaned_documents": orphaned_count["orphaned_documents"]
                    },
                    "chunk_statistics": {
                        "total_chunks": chunk_stats["total_chunks"],
                        "average_chunk_length": round(chunk_stats["avg_chunk_length"], 2) if chunk_stats["avg_chunk_length"] else 0,
                        "min_chunk_length": chunk_stats["min_chunk_length"],
                        "max_chunk_length": chunk_stats["max_chunk_length"],
                        "total_content_length": chunk_stats["total_content_length"]
                    },
                    "health_metrics": {
                        "documents_with_chunks": doc_stats["total_documents"] - orphaned_count["orphaned_documents"],
                        "processing_success_rate": round(
                            ((doc_stats["total_documents"] - orphaned_count["orphaned_documents"]) / doc_stats["total_documents"] * 100)
                            if doc_stats["total_documents"] > 0 else 0, 2
                        )
                    },
                    "timestamp": datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            print(f"Error getting statistics: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def search_knowledge_base(self, query: str, limit: int = 10, similarity_threshold: float = 0.7) -> Dict[str, Any]:
        """
        Search the Knowledge Base using vector similarity.
        
        Args:
            query: Search query
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score
            
        Returns:
            Search results with metadata
        """
        try:
            search_request = DocumentSearchRequest(
                query=query,
                limit=limit,
                similarity_threshold=similarity_threshold
            )
            
            result = self.processor.search_documents(
                query=query,
                limit=limit,
                similarity_threshold=similarity_threshold
            )
            
            if result.success:
                return {
                    "success": True,
                    "query": query,
                    "results": result.data["results"],
                    "total_found": result.data["total_found"],
                    "search_parameters": {
                        "limit": limit,
                        "similarity_threshold": similarity_threshold
                    },
                    "timestamp": datetime.utcnow().isoformat()
                }
            else:
                return {
                    "success": False,
                    "error": result.error_message,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def cleanup_orphaned_documents(self, dry_run: bool = True) -> Dict[str, Any]:
        """
        Clean up documents that have no associated chunks or missing files.
        
        Args:
            dry_run: If True, only report what would be cleaned up
            
        Returns:
            Cleanup report
        """
        try:
            with self.db_config.get_db_connection() as conn:
                cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                
                # Find orphaned documents (no chunks)
                cur.execute("""
                    SELECT d.id, d.title, d."storagePath"
                    FROM "Document" d
                    LEFT JOIN "DocumentChunk" dc ON d.id = dc."documentId"
                    WHERE dc.id IS NULL
                """)
                orphaned_docs = cur.fetchall()
                
                # Find documents with missing files
                cur.execute("""
                    SELECT id, title, "storagePath"
                    FROM "Document"
                """)
                all_docs = cur.fetchall()
                
                missing_files = []
                for doc in all_docs:
                    if doc["storagePath"] and not os.path.exists(doc["storagePath"]):
                        missing_files.append(dict(doc))
                
                cleanup_report = {
                    "dry_run": dry_run,
                    "orphaned_documents": [dict(doc) for doc in orphaned_docs],
                    "missing_files": missing_files,
                    "total_to_clean": len(orphaned_docs) + len(missing_files),
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                if not dry_run and (orphaned_docs or missing_files):
                    # Actually perform cleanup
                    deleted_count = 0
                    
                    # Delete orphaned documents
                    for doc in orphaned_docs:
                        cur.execute('DELETE FROM "Document" WHERE id = %s', (doc["id"],))
                        deleted_count += 1
                    
                    # Delete documents with missing files
                    for doc in missing_files:
                        # First delete any chunks
                        cur.execute('DELETE FROM "DocumentChunk" WHERE "documentId" = %s', (doc["id"],))
                        # Then delete document
                        cur.execute('DELETE FROM "Document" WHERE id = %s', (doc["id"],))
                        deleted_count += 1
                    
                    conn.commit()
                    cleanup_report["deleted_count"] = deleted_count
                    cleanup_report["status"] = "completed"
                else:
                    cleanup_report["status"] = "dry_run" if dry_run else "no_cleanup_needed"
                
                return cleanup_report
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def reprocess_document(self, document_id: str) -> Dict[str, Any]:
        """
        Reprocess a specific document (regenerate chunks and embeddings).
        
        Args:
            document_id: ID of the document to reprocess
            
        Returns:
            Processing result
        """
        try:
            with self.db_config.get_db_connection() as conn:
                cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                
                # Get document info
                cur.execute("""
                    SELECT id, title, "storagePath"
                    FROM "Document"
                    WHERE id = %s
                """, (document_id,))
                
                document = cur.fetchone()
                if not document:
                    return {
                        "success": False,
                        "error": "Document not found",
                        "document_id": document_id
                    }
                
                storage_path = document["storagePath"]
                if not storage_path or not os.path.exists(storage_path):
                    return {
                        "success": False,
                        "error": "Document file not found",
                        "document_id": document_id,
                        "storage_path": storage_path
                    }
                
                # Delete existing chunks
                cur.execute('DELETE FROM "DocumentChunk" WHERE "documentId" = %s', (document_id,))
                conn.commit()
                
                # Reprocess document
                result = self.processor.execute_safe({
                    "document_id": document_id,
                    "file_path": storage_path
                })
                
                return {
                    "success": result.success,
                    "document_id": document_id,
                    "document_title": document["title"],
                    "chunks_created": result.data.get("chunks_created", 0) if result.success else 0,
                    "error": result.error_message if not result.success else None,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "document_id": document_id,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def export_knowledge_base_summary(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Export a comprehensive summary of the Knowledge Base to JSON.
        
        Args:
            output_path: Optional path to save the summary file
            
        Returns:
            Summary data and export status
        """
        try:
            # Gather all data
            documents = self.list_all_documents(include_chunks_info=True)
            statistics = self.get_document_statistics()
            
            summary = {
                "export_timestamp": datetime.utcnow().isoformat(),
                "knowledge_base_summary": {
                    "total_documents": len(documents),
                    "documents": documents,
                    "statistics": statistics
                },
                "metadata": {
                    "export_version": "1.0",
                    "system": "CrediLinQ Knowledge Base",
                    "generated_by": "KnowledgeBaseManager"
                }
            }
            
            # Save to file if path provided
            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_path, 'w') as f:
                    json.dump(summary, f, indent=2, default=str)
                
                summary["export_file"] = str(output_path)
                summary["file_size"] = output_path.stat().st_size
            
            return {
                "success": True,
                "summary": summary
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }


# Convenience functions for direct usage
async def list_knowledge_base_documents():
    """Quick function to list all documents in the Knowledge Base."""
    manager = KnowledgeBaseManager()
    return manager.list_all_documents()

async def get_knowledge_base_stats():
    """Quick function to get Knowledge Base statistics."""
    manager = KnowledgeBaseManager()
    return manager.get_document_statistics()

async def search_knowledge_base(query: str, limit: int = 10):
    """Quick function to search the Knowledge Base."""
    manager = KnowledgeBaseManager()
    return manager.search_knowledge_base(query, limit)


if __name__ == "__main__":
    import psycopg2.extras
    
    # Example usage
    async def main():
        manager = KnowledgeBaseManager()
        
        print("=== Knowledge Base Summary ===")
        
        # List documents
        documents = manager.list_all_documents()
        print(f"\nFound {len(documents)} documents:")
        for doc in documents:
            print(f"- {doc['title']} (ID: {doc['id'][:8]}...)")
            print(f"  Chunks: {doc.get('chunks_count', 0)}")
            print(f"  File exists: {doc.get('file_exists', False)}")
        
        # Get statistics
        stats = manager.get_document_statistics()
        print(f"\n=== Statistics ===")
        print(f"Total documents: {stats['document_statistics']['total_documents']}")
        print(f"Total chunks: {stats['chunk_statistics']['total_chunks']}")
        print(f"Processing success rate: {stats['health_metrics']['processing_success_rate']}%")
        
        # Example search
        if stats['chunk_statistics']['total_chunks'] > 0:
            search_result = manager.search_knowledge_base("content", limit=3)
            print(f"\n=== Sample Search Results ===")
            print(f"Found {search_result.get('total_found', 0)} results for 'content'")
    
    # Run example
    asyncio.run(main())