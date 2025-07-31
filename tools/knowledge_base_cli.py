#!/usr/bin/env python3
"""
Knowledge Base CLI Tool
Command-line interface for managing the CrediLinQ Knowledge Base.

Usage:
    python tools/knowledge_base_cli.py list
    python tools/knowledge_base_cli.py stats
    python tools/knowledge_base_cli.py search "your query"
    python tools/knowledge_base_cli.py cleanup --dry-run
    python tools/knowledge_base_cli.py reprocess <document_id>
    python tools/knowledge_base_cli.py export [output_file.json]
"""

import sys
import os
import asyncio
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.knowledge_base_manager import KnowledgeBaseManager


def format_file_size(size_bytes):
    """Convert bytes to human readable format."""
    if size_bytes is None:
        return "Unknown"
    
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def print_document_table(documents):
    """Print documents in a formatted table."""
    if not documents:
        print("No documents found in the Knowledge Base.")
        return
    
    print(f"\n{'ID':<12} {'Title':<30} {'Chunks':<8} {'File Size':<12} {'Status':<12} {'Upload Date':<12}")
    print("-" * 100)
    
    for doc in documents:
        doc_id = doc['id'][:8] + "..." if len(doc['id']) > 8 else doc['id']
        title = (doc['title'][:27] + "...") if len(doc['title']) > 30 else doc['title']
        chunks = str(doc.get('chunks_count', 0))
        file_size = format_file_size(doc.get('file_size'))
        status = "✅ OK" if doc.get('file_exists') else "❌ Missing"
        
        upload_date = "Unknown"
        if doc.get('uploadedAt'):
            try:
                dt = datetime.fromisoformat(doc['uploadedAt'].replace('Z', '+00:00'))
                upload_date = dt.strftime('%Y-%m-%d')
            except:
                upload_date = doc['uploadedAt'][:10]
        
        print(f"{doc_id:<12} {title:<30} {chunks:<8} {file_size:<12} {status:<12} {upload_date:<12}")


def print_statistics(stats):
    """Print Knowledge Base statistics."""
    print("\n=== Knowledge Base Statistics ===")
    
    doc_stats = stats.get('document_statistics', {})
    chunk_stats = stats.get('chunk_statistics', {})
    health_stats = stats.get('health_metrics', {})
    
    print(f"\n📄 Documents:")
    print(f"  Total documents: {doc_stats.get('total_documents', 0)}")
    print(f"  Recent documents (7 days): {doc_stats.get('recent_documents', 0)}")
    print(f"  Orphaned documents: {doc_stats.get('orphaned_documents', 0)}")
    
    if doc_stats.get('oldest_document'):
        oldest = datetime.fromisoformat(doc_stats['oldest_document'].replace('Z', '+00:00'))
        print(f"  Oldest document: {oldest.strftime('%Y-%m-%d')}")
    
    if doc_stats.get('newest_document'):
        newest = datetime.fromisoformat(doc_stats['newest_document'].replace('Z', '+00:00'))
        print(f"  Newest document: {newest.strftime('%Y-%m-%d')}")
    
    print(f"\n🔍 Content Chunks:")
    print(f"  Total chunks: {chunk_stats.get('total_chunks', 0)}")
    print(f"  Average chunk length: {chunk_stats.get('average_chunk_length', 0)} characters")
    print(f"  Total content: {format_file_size(chunk_stats.get('total_content_length', 0))}")
    
    print(f"\n📊 Health Metrics:")
    print(f"  Documents with chunks: {health_stats.get('documents_with_chunks', 0)}")
    print(f"  Processing success rate: {health_stats.get('processing_success_rate', 0)}%")


def print_search_results(results):
    """Print search results."""
    if not results.get('success'):
        print(f"❌ Search failed: {results.get('error', 'Unknown error')}")
        return
    
    search_results = results.get('results', [])
    total_found = results.get('total_found', 0)
    query = results.get('query', '')
    
    print(f"\n🔍 Search results for: '{query}'")
    print(f"Found {total_found} results")
    
    if search_results:
        print(f"\n{'Score':<8} {'Document':<25} {'Content Preview':<50}")
        print("-" * 90)
        
        for result in search_results:
            score = f"{result.get('similarity_score', 0):.3f}"
            doc_name = result.get('document_filename', 'Unknown')[:22]
            if len(result.get('document_filename', '')) > 22:
                doc_name += "..."
            
            content = result.get('content', '')[:47]
            if len(result.get('content', '')) > 47:
                content += "..."
            
            # Replace newlines with spaces for clean display
            content = content.replace('\n', ' ').replace('\r', '')
            
            print(f"{score:<8} {doc_name:<25} {content:<50}")
    else:
        print("No results found.")


async def cmd_list(args):
    """List all documents in the Knowledge Base."""
    manager = KnowledgeBaseManager()
    documents = manager.list_all_documents()
    
    print(f"📚 Knowledge Base Documents ({len(documents)} total)")
    print_document_table(documents)


async def cmd_stats(args):
    """Show Knowledge Base statistics."""
    manager = KnowledgeBaseManager()
    stats = manager.get_document_statistics()
    
    if 'error' in stats:
        print(f"❌ Error getting statistics: {stats['error']}")
        return
    
    print_statistics(stats)


async def cmd_search(args):
    """Search the Knowledge Base."""
    if not args.query:
        print("❌ Please provide a search query")
        return
    
    manager = KnowledgeBaseManager()
    results = manager.search_knowledge_base(
        query=args.query,
        limit=args.limit,
        similarity_threshold=args.threshold
    )
    
    print_search_results(results)


async def cmd_cleanup(args):
    """Clean up orphaned documents and missing files."""
    manager = KnowledgeBaseManager()
    result = manager.cleanup_orphaned_documents(dry_run=args.dry_run)
    
    if 'error' in result:
        print(f"❌ Cleanup failed: {result['error']}")
        return
    
    orphaned = result.get('orphaned_documents', [])
    missing = result.get('missing_files', [])
    total_to_clean = result.get('total_to_clean', 0)
    
    print(f"\n🧹 Knowledge Base Cleanup Report")
    print(f"Mode: {'Dry run (preview only)' if args.dry_run else 'Live cleanup'}")
    
    if orphaned:
        print(f"\n📄 Orphaned documents (no chunks): {len(orphaned)}")
        for doc in orphaned:
            print(f"  - {doc['title']} (ID: {doc['id'][:8]}...)")
    
    if missing:
        print(f"\n📁 Documents with missing files: {len(missing)}")
        for doc in missing:
            print(f"  - {doc['title']} (Path: {doc['storagePath']})")
    
    if total_to_clean == 0:
        print("\n✅ No cleanup needed - Knowledge Base is healthy!")
    else:
        if args.dry_run:
            print(f"\n⚠️  Would clean up {total_to_clean} items.")
            print("Run without --dry-run to perform actual cleanup.")
        else:
            deleted = result.get('deleted_count', 0)
            print(f"\n✅ Cleaned up {deleted} items successfully.")


async def cmd_reprocess(args):
    """Reprocess a specific document."""
    if not args.document_id:
        print("❌ Please provide a document ID")
        return
    
    manager = KnowledgeBaseManager()
    print(f"🔄 Reprocessing document {args.document_id}...")
    
    result = await manager.reprocess_document(args.document_id)
    
    if result.get('success'):
        print(f"✅ Successfully reprocessed '{result['document_title']}'")
        print(f"   Created {result['chunks_created']} chunks")
    else:
        print(f"❌ Reprocessing failed: {result.get('error', 'Unknown error')}")


async def cmd_export(args):
    """Export Knowledge Base summary to JSON."""
    manager = KnowledgeBaseManager()
    
    output_file = args.output_file
    if not output_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"knowledge_base_export_{timestamp}.json"
    
    print(f"📤 Exporting Knowledge Base summary to {output_file}...")
    
    result = manager.export_knowledge_base_summary(output_file)
    
    if result.get('success'):
        summary = result['summary']
        file_size = format_file_size(summary.get('file_size', 0))
        total_docs = summary['knowledge_base_summary']['total_documents']
        
        print(f"✅ Export completed successfully!")
        print(f"   File: {output_file}")
        print(f"   Size: {file_size}")
        print(f"   Documents exported: {total_docs}")
    else:
        print(f"❌ Export failed: {result.get('error', 'Unknown error')}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Knowledge Base CLI Tool for CrediLinQ",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/knowledge_base_cli.py list
  python tools/knowledge_base_cli.py stats
  python tools/knowledge_base_cli.py search "artificial intelligence" --limit 5
  python tools/knowledge_base_cli.py cleanup --dry-run
  python tools/knowledge_base_cli.py reprocess 12345678-1234-1234-1234-123456789012
  python tools/knowledge_base_cli.py export kb_backup.json
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List all documents')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show Knowledge Base statistics')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search the Knowledge Base')
    search_parser.add_argument('query', help='Search query')
    search_parser.add_argument('--limit', type=int, default=10, help='Maximum results (default: 10)')
    search_parser.add_argument('--threshold', type=float, default=0.7, help='Similarity threshold (default: 0.7)')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up orphaned documents')
    cleanup_parser.add_argument('--dry-run', action='store_true', help='Preview only, do not make changes')
    
    # Reprocess command
    reprocess_parser = subparsers.add_parser('reprocess', help='Reprocess a specific document')
    reprocess_parser.add_argument('document_id', help='Document ID to reprocess')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export Knowledge Base summary')
    export_parser.add_argument('output_file', nargs='?', help='Output JSON file (optional)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Map commands to functions
    commands = {
        'list': cmd_list,
        'stats': cmd_stats,
        'search': cmd_search,
        'cleanup': cmd_cleanup,
        'reprocess': cmd_reprocess,
        'export': cmd_export
    }
    
    if args.command in commands:
        try:
            asyncio.run(commands[args.command](args))
        except KeyboardInterrupt:
            print("\n⚠️  Operation cancelled by user")
        except Exception as e:
            print(f"❌ Error: {e}")
            sys.exit(1)
    else:
        print(f"❌ Unknown command: {args.command}")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()