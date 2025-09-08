"""
LangGraph-enhanced Document Processor Workflow for intelligent RAG knowledge base management.
"""

import json
import asyncio
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, TypedDict, Annotated
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import mimetypes
import uuid

# Import LangGraph components with version compatibility
from src.agents.core.langgraph_compat import StateGraph, END, add_messages
from langgraph.checkpoint.memory import MemorySaver
try:
    from langgraph.checkpoint.postgres import PostgresSaver
except ImportError:
    PostgresSaver = None  # PostgreSQL checkpointing not available
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain.schema import Document

from ..core.langgraph_base import (
    LangGraphWorkflowBase, WorkflowState, WorkflowStatus, 
    CheckpointStrategy, LangGraphExecutionContext
)
from ..core.base_agent import AgentType, AgentResult, AgentExecutionContext
# Removed broken import: from .document_processor import DocumentProcessorAgent
# from ...config.database import DatabaseConnection  # Temporarily disabled


class DocumentType(Enum):
    """Supported document types."""
    PDF = "pdf"
    TEXT = "text"
    MARKDOWN = "markdown"
    HTML = "html"
    DOCX = "docx"
    SPREADSHEET = "spreadsheet"
    PRESENTATION = "presentation"
    UNKNOWN = "unknown"


class ProcessingStrategy(Enum):
    """Document processing strategies."""
    STANDARD = "standard"              # Default chunking and embedding
    SEMANTIC = "semantic"              # Semantic boundary detection
    HIERARCHICAL = "hierarchical"      # Preserve document structure
    CUSTOM = "custom"                  # User-defined parameters
    INTELLIGENT = "intelligent"        # AI-optimized chunking


class ChunkingQuality(Enum):
    """Chunk quality assessment levels."""
    EXCELLENT = "excellent"    # Semantically coherent, optimal size
    GOOD = "good"             # Mostly coherent, acceptable size
    FAIR = "fair"             # Some fragmentation, usable
    POOR = "poor"             # Fragmented, needs improvement


@dataclass
class DocumentMetadata:
    """Enhanced document metadata."""
    document_id: str
    file_name: str
    file_path: str
    file_size: int
    mime_type: str
    document_type: DocumentType
    creation_date: Optional[datetime] = None
    modification_date: Optional[datetime] = None
    author: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    language: str = "en"
    encoding: str = "utf-8"


@dataclass
class ChunkMetadata:
    """Metadata for document chunks."""
    chunk_id: str
    document_id: str
    chunk_index: int
    start_char: int
    end_char: int
    chunk_size: int
    chunk_hash: str
    semantic_density: float = 0.0
    quality_score: float = 0.0
    has_tables: bool = False
    has_images: bool = False
    has_code: bool = False
    section_title: Optional[str] = None
    page_number: Optional[int] = None


@dataclass
class ProcessingMetrics:
    """Document processing performance metrics."""
    total_chunks: int = 0
    successful_chunks: int = 0
    failed_chunks: int = 0
    average_chunk_size: float = 0.0
    processing_time_seconds: float = 0.0
    embedding_generation_time: float = 0.0
    storage_time: float = 0.0
    total_tokens: int = 0
    quality_score: float = 0.0
    semantic_coherence: float = 0.0


class DocumentProcessorState(WorkflowState):
    """Enhanced state for document processing workflow."""
    
    # Input parameters
    document_id: str = ""
    file_path: str = ""
    processing_strategy: ProcessingStrategy = ProcessingStrategy.STANDARD
    custom_chunk_size: Optional[int] = None
    custom_chunk_overlap: Optional[int] = None
    
    # Document analysis
    document_metadata: Optional[DocumentMetadata] = None
    document_content: Optional[str] = None
    document_structure: Dict[str, Any] = field(default_factory=dict)
    content_analysis: Dict[str, Any] = field(default_factory=dict)
    
    # Processing configuration
    chunking_parameters: Dict[str, Any] = field(default_factory=dict)
    embedding_config: Dict[str, Any] = field(default_factory=dict)
    quality_thresholds: Dict[str, float] = field(default_factory=dict)
    
    # Chunking results
    document_chunks: List[Dict[str, Any]] = field(default_factory=list)
    chunk_metadata: List[ChunkMetadata] = field(default_factory=list)
    chunk_quality_scores: Dict[str, float] = field(default_factory=dict)
    
    # Embeddings
    chunk_embeddings: Dict[str, List[float]] = field(default_factory=dict)
    embedding_model_info: Dict[str, Any] = field(default_factory=dict)
    
    # Quality assurance
    quality_assessment: Dict[str, Any] = field(default_factory=dict)
    validation_results: Dict[str, Any] = field(default_factory=dict)
    optimization_suggestions: List[str] = field(default_factory=list)
    
    # Knowledge graph integration
    knowledge_graph_nodes: List[Dict[str, Any]] = field(default_factory=list)
    knowledge_graph_edges: List[Dict[str, Any]] = field(default_factory=list)
    entity_extraction: Dict[str, List[str]] = field(default_factory=dict)
    
    # Storage and indexing
    storage_status: Dict[str, Any] = field(default_factory=dict)
    index_updates: List[Dict[str, Any]] = field(default_factory=list)
    search_optimization: Dict[str, Any] = field(default_factory=dict)
    
    # Performance metrics
    processing_metrics: ProcessingMetrics = field(default_factory=ProcessingMetrics)
    
    # Messages for communication between nodes
    messages: Annotated[List[BaseMessage], add_messages] = field(default_factory=list)


class DocumentProcessorWorkflow(LangGraphWorkflowBase[DocumentProcessorState]):
    """LangGraph workflow for advanced document processing with RAG optimization."""
    
    def __init__(
        self, 
        workflow_name: str = "document_processor_workflow",
        checkpoint_strategy: CheckpointStrategy = CheckpointStrategy.DATABASE_PERSISTENT,
        enable_human_in_loop: bool = False
    ):
        # Initialize the legacy agent for core functionality
        self.legacy_agent = DocumentProcessorAgent()
        # Initialize the agent's components
        self.legacy_agent._initialize()
        
        super().__init__(
            workflow_name=workflow_name,
            checkpoint_strategy=checkpoint_strategy,
            enable_human_in_loop=enable_human_in_loop
        )
    
    def _create_initial_state(self, context: Dict[str, Any]) -> DocumentProcessorState:
        """Create initial workflow state from context."""
        return DocumentProcessorState(
            workflow_id=context.get("workflow_id", f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
            document_id=context.get("document_id", str(uuid.uuid4())),
            file_path=context.get("file_path", ""),
            processing_strategy=ProcessingStrategy(context.get("processing_strategy", "standard")),
            custom_chunk_size=context.get("custom_chunk_size"),
            custom_chunk_overlap=context.get("custom_chunk_overlap"),
            created_at=datetime.utcnow()
        )
    
    def _create_workflow_graph(self) -> StateGraph:
        """Create the document processing workflow graph."""
        workflow = StateGraph(DocumentProcessorState)
        
        # Define workflow nodes
        workflow.add_node("validate_document", self._validate_document_node)
        workflow.add_node("analyze_document", self._analyze_document_node)
        workflow.add_node("determine_strategy", self._determine_strategy_node)
        workflow.add_node("chunk_document", self._chunk_document_node)
        workflow.add_node("assess_quality", self._assess_quality_node)
        workflow.add_node("generate_embeddings", self._generate_embeddings_node)
        workflow.add_node("extract_knowledge", self._extract_knowledge_node)
        workflow.add_node("optimize_search", self._optimize_search_node)
        workflow.add_node("store_chunks", self._store_chunks_node)
        workflow.add_node("update_index", self._update_index_node)
        workflow.add_node("finalize_processing", self._finalize_processing_node)
        
        # Define workflow edges
        workflow.add_edge("validate_document", "analyze_document")
        workflow.add_edge("analyze_document", "determine_strategy")
        workflow.add_edge("determine_strategy", "chunk_document")
        workflow.add_edge("chunk_document", "assess_quality")
        
        # Conditional routing for quality improvement
        workflow.add_conditional_edges(
            "assess_quality",
            self._check_quality_threshold,
            {
                "rechunk": "chunk_document",
                "proceed": "generate_embeddings"
            }
        )
        
        workflow.add_edge("generate_embeddings", "extract_knowledge")
        workflow.add_edge("extract_knowledge", "optimize_search")
        workflow.add_edge("optimize_search", "store_chunks")
        workflow.add_edge("store_chunks", "update_index")
        workflow.add_edge("update_index", "finalize_processing")
        workflow.add_edge("finalize_processing", END)
        
        # Set entry point
        workflow.set_entry_point("validate_document")
        
        return workflow
    
    async def _validate_document_node(self, state: DocumentProcessorState) -> DocumentProcessorState:
        """Validate document and extract basic metadata."""
        try:
            self._log_progress("Validating document and extracting metadata")
            
            validation_errors = []
            
            # Validate file path
            if not state.file_path:
                validation_errors.append("File path is required")
            
            file_path = Path(state.file_path)
            if not file_path.exists():
                validation_errors.append(f"File not found: {state.file_path}")
            
            if validation_errors:
                state.status = WorkflowStatus.FAILED
                state.error_message = "; ".join(validation_errors)
                return state
            
            # Extract file metadata
            file_stats = file_path.stat()
            mime_type, _ = mimetypes.guess_type(str(file_path))
            
            # Determine document type
            document_type = self._determine_document_type(mime_type, file_path.suffix)
            
            # Create document metadata
            document_metadata = DocumentMetadata(
                document_id=state.document_id,
                file_name=file_path.name,
                file_path=str(file_path),
                file_size=file_stats.st_size,
                mime_type=mime_type or "application/octet-stream",
                document_type=document_type,
                creation_date=datetime.fromtimestamp(file_stats.st_ctime),
                modification_date=datetime.fromtimestamp(file_stats.st_mtime)
            )
            
            # Validate file size
            max_file_size = 50 * 1024 * 1024  # 50MB limit
            if document_metadata.file_size > max_file_size:
                validation_errors.append(f"File too large: {document_metadata.file_size / 1024 / 1024:.2f}MB (max: 50MB)")
            
            # Set quality thresholds based on document type
            quality_thresholds = self._get_quality_thresholds(document_type)
            
            if validation_errors:
                state.status = WorkflowStatus.FAILED
                state.error_message = "; ".join(validation_errors)
            else:
                state.document_metadata = document_metadata
                state.quality_thresholds = quality_thresholds
                state.status = WorkflowStatus.IN_PROGRESS
                state.progress_percentage = 10.0
                
                state.messages.append(HumanMessage(
                    content=f"Document validated: {document_metadata.file_name} "
                           f"({document_metadata.file_size / 1024:.2f}KB, type: {document_type.value})"
                ))
            
            return state
            
        except Exception as e:
            state.status = WorkflowStatus.FAILED
            state.error_message = f"Document validation failed: {str(e)}"
            return state
    
    async def _analyze_document_node(self, state: DocumentProcessorState) -> DocumentProcessorState:
        """Analyze document content and structure."""
        try:
            self._log_progress("Analyzing document content and structure")
            
            # Load document content using legacy agent
            document = self.legacy_agent._load_document(state.file_path)
            state.document_content = document.page_content
            
            # Analyze document structure
            document_structure = self._analyze_document_structure(
                document.page_content, state.document_metadata.document_type
            )
            
            # Analyze content characteristics
            content_analysis = {
                "total_length": len(document.page_content),
                "line_count": document.page_content.count('\n'),
                "word_count": len(document.page_content.split()),
                "paragraph_count": len(document.page_content.split('\n\n')),
                "has_headings": self._detect_headings(document.page_content),
                "has_lists": self._detect_lists(document.page_content),
                "has_tables": self._detect_tables(document.page_content),
                "has_code": self._detect_code_blocks(document.page_content),
                "language": self._detect_language(document.page_content),
                "complexity_score": self._calculate_complexity_score(document.page_content),
                "semantic_density": self._calculate_semantic_density(document.page_content)
            }
            
            # Identify key sections
            if document_structure.get("sections"):
                content_analysis["key_sections"] = [
                    section["title"] for section in document_structure["sections"][:10]
                ]
            
            # Detect content type patterns
            content_patterns = self._detect_content_patterns(document.page_content)
            content_analysis["content_patterns"] = content_patterns
            
            state.document_structure = document_structure
            state.content_analysis = content_analysis
            state.progress_percentage = 20.0
            
            state.messages.append(SystemMessage(
                content=f"Document analysis completed. Content: {content_analysis['word_count']} words, "
                       f"{content_analysis['paragraph_count']} paragraphs. "
                       f"Complexity: {content_analysis['complexity_score']:.2f}, "
                       f"Semantic density: {content_analysis['semantic_density']:.2f}"
            ))
            
            return state
            
        except Exception as e:
            state.status = WorkflowStatus.FAILED
            state.error_message = f"Document analysis failed: {str(e)}"
            return state
    
    async def _determine_strategy_node(self, state: DocumentProcessorState) -> DocumentProcessorState:
        """Determine optimal chunking strategy based on document analysis."""
        try:
            self._log_progress("Determining optimal processing strategy")
            
            # Override with custom strategy if specified
            if state.processing_strategy == ProcessingStrategy.CUSTOM:
                chunking_parameters = {
                    "chunk_size": state.custom_chunk_size or 1000,
                    "chunk_overlap": state.custom_chunk_overlap or 200,
                    "strategy": "custom",
                    "separators": ["\n\n", "\n", ". ", " ", ""]
                }
            else:
                # Determine optimal strategy based on document characteristics
                optimal_strategy = self._determine_optimal_strategy(
                    state.content_analysis,
                    state.document_structure,
                    state.document_metadata.document_type
                )
                
                # Get chunking parameters for the strategy
                chunking_parameters = self._get_chunking_parameters(
                    optimal_strategy,
                    state.content_analysis
                )
                
                # Update processing strategy if it was auto-determined
                if state.processing_strategy == ProcessingStrategy.STANDARD:
                    state.processing_strategy = optimal_strategy
            
            # Set embedding configuration
            embedding_config = {
                "model": "text-embedding-ada-002",
                "batch_size": 50,
                "max_retries": 3,
                "normalize": True
            }
            
            # Adjust parameters based on document size
            if state.content_analysis["total_length"] > 100000:
                chunking_parameters["chunk_size"] = min(chunking_parameters["chunk_size"], 800)
                embedding_config["batch_size"] = 25
            
            state.chunking_parameters = chunking_parameters
            state.embedding_config = embedding_config
            state.progress_percentage = 30.0
            
            state.messages.append(SystemMessage(
                content=f"Processing strategy determined: {state.processing_strategy.value}. "
                       f"Chunk size: {chunking_parameters['chunk_size']}, "
                       f"Overlap: {chunking_parameters['chunk_overlap']}"
            ))
            
            return state
            
        except Exception as e:
            state.status = WorkflowStatus.FAILED
            state.error_message = f"Strategy determination failed: {str(e)}"
            return state
    
    async def _chunk_document_node(self, state: DocumentProcessorState) -> DocumentProcessorState:
        """Chunk document using determined strategy."""
        try:
            self._log_progress("Chunking document with optimized parameters")
            
            # Create text splitter with determined parameters
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=state.chunking_parameters["chunk_size"],
                chunk_overlap=state.chunking_parameters["chunk_overlap"],
                separators=state.chunking_parameters.get("separators", ["\n\n", "\n", ". ", " ", ""])
            )
            
            # Create document object
            doc = Document(
                page_content=state.document_content,
                metadata={"source": state.file_path}
            )
            
            # Split document into chunks
            chunks = text_splitter.split_documents([doc])
            
            # Process chunks and create metadata
            document_chunks = []
            chunk_metadata_list = []
            chunk_quality_scores = {}
            
            current_position = 0
            for i, chunk in enumerate(chunks):
                chunk_id = f"{state.document_id}_chunk_{i}"
                chunk_content = chunk.page_content
                chunk_size = len(chunk_content)
                
                # Calculate chunk position in original document
                start_char = state.document_content.find(chunk_content, current_position)
                end_char = start_char + chunk_size if start_char >= 0 else -1
                current_position = end_char
                
                # Create chunk metadata
                metadata = ChunkMetadata(
                    chunk_id=chunk_id,
                    document_id=state.document_id,
                    chunk_index=i,
                    start_char=start_char,
                    end_char=end_char,
                    chunk_size=chunk_size,
                    chunk_hash=hashlib.md5(chunk_content.encode()).hexdigest(),
                    semantic_density=self._calculate_chunk_semantic_density(chunk_content),
                    quality_score=self._assess_chunk_quality(chunk_content),
                    has_tables=self._detect_tables(chunk_content),
                    has_code=self._detect_code_blocks(chunk_content)
                )
                
                # Detect section title if applicable
                section_title = self._extract_section_title(chunk_content, state.document_structure)
                if section_title:
                    metadata.section_title = section_title
                
                # Store chunk data
                chunk_data = {
                    "chunk_id": chunk_id,
                    "content": chunk_content,
                    "metadata": metadata,
                    "position": {
                        "start": start_char,
                        "end": end_char,
                        "index": i,
                        "total_chunks": len(chunks)
                    }
                }
                
                document_chunks.append(chunk_data)
                chunk_metadata_list.append(metadata)
                chunk_quality_scores[chunk_id] = metadata.quality_score
            
            state.document_chunks = document_chunks
            state.chunk_metadata = chunk_metadata_list
            state.chunk_quality_scores = chunk_quality_scores
            state.progress_percentage = 45.0
            
            # Calculate average quality
            avg_quality = sum(chunk_quality_scores.values()) / len(chunk_quality_scores) if chunk_quality_scores else 0
            
            state.messages.append(SystemMessage(
                content=f"Document chunked into {len(chunks)} chunks. "
                       f"Average chunk size: {sum(m.chunk_size for m in chunk_metadata_list) / len(chunk_metadata_list):.0f} chars. "
                       f"Average quality score: {avg_quality:.2f}"
            ))
            
            return state
            
        except Exception as e:
            state.status = WorkflowStatus.FAILED
            state.error_message = f"Document chunking failed: {str(e)}"
            return state
    
    async def _assess_quality_node(self, state: DocumentProcessorState) -> DocumentProcessorState:
        """Assess quality of chunking and determine if rechunking is needed."""
        try:
            self._log_progress("Assessing chunk quality and coherence")
            
            quality_assessment = {
                "total_chunks": len(state.document_chunks),
                "average_quality": sum(state.chunk_quality_scores.values()) / len(state.chunk_quality_scores) if state.chunk_quality_scores else 0,
                "quality_distribution": self._calculate_quality_distribution(state.chunk_quality_scores),
                "semantic_coherence": self._assess_semantic_coherence(state.document_chunks),
                "size_consistency": self._assess_size_consistency(state.chunk_metadata),
                "boundary_quality": self._assess_boundary_quality(state.document_chunks),
                "information_preservation": self._assess_information_preservation(
                    state.document_content, state.document_chunks
                )
            }
            
            # Identify problematic chunks
            problematic_chunks = []
            for chunk_id, quality_score in state.chunk_quality_scores.items():
                if quality_score < state.quality_thresholds.get("min_chunk_quality", 0.6):
                    problematic_chunks.append({
                        "chunk_id": chunk_id,
                        "quality_score": quality_score,
                        "issue": self._identify_chunk_issue(chunk_id, state.document_chunks)
                    })
            
            quality_assessment["problematic_chunks"] = problematic_chunks
            
            # Generate optimization suggestions
            optimization_suggestions = []
            
            if quality_assessment["average_quality"] < 0.7:
                optimization_suggestions.append("Consider adjusting chunk size for better semantic coherence")
            
            if quality_assessment["size_consistency"] < 0.6:
                optimization_suggestions.append("Chunk sizes vary significantly - consider using semantic boundaries")
            
            if len(problematic_chunks) > len(state.document_chunks) * 0.2:
                optimization_suggestions.append("Many low-quality chunks detected - consider alternative chunking strategy")
            
            # Calculate overall quality level
            overall_quality = self._calculate_overall_quality(quality_assessment)
            quality_assessment["overall_quality"] = overall_quality
            quality_assessment["quality_level"] = self._determine_quality_level(overall_quality)
            
            state.quality_assessment = quality_assessment
            state.optimization_suggestions = optimization_suggestions
            state.progress_percentage = 55.0
            
            state.messages.append(SystemMessage(
                content=f"Quality assessment completed. Overall quality: {overall_quality:.2f} ({quality_assessment['quality_level']}). "
                       f"Problematic chunks: {len(problematic_chunks)}/{len(state.document_chunks)}"
            ))
            
            return state
            
        except Exception as e:
            state.status = WorkflowStatus.FAILED
            state.error_message = f"Quality assessment failed: {str(e)}"
            return state
    
    async def _generate_embeddings_node(self, state: DocumentProcessorState) -> DocumentProcessorState:
        """Generate embeddings for all chunks."""
        try:
            self._log_progress("Generating embeddings for document chunks")
            
            start_time = datetime.utcnow()
            chunk_embeddings = {}
            failed_chunks = []
            
            # Generate embeddings in batches
            batch_size = state.embedding_config["batch_size"]
            total_batches = (len(state.document_chunks) + batch_size - 1) // batch_size
            
            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(state.document_chunks))
                batch_chunks = state.document_chunks[start_idx:end_idx]
                
                self._log_progress(f"Processing embedding batch {batch_idx + 1}/{total_batches}")
                
                for chunk_data in batch_chunks:
                    try:
                        # Generate embedding using legacy agent's model
                        embedding = self.legacy_agent.embeddings_model.embed_query(chunk_data["content"])
                        
                        # Validate embedding
                        if not embedding or len(embedding) == 0:
                            raise ValueError("Empty embedding generated")
                        
                        # Normalize if requested
                        if state.embedding_config.get("normalize", True):
                            embedding = self._normalize_embedding(embedding)
                        
                        chunk_embeddings[chunk_data["chunk_id"]] = embedding
                        
                    except Exception as e:
                        self._log_error(f"Failed to generate embedding for chunk {chunk_data['chunk_id']}: {str(e)}")
                        failed_chunks.append(chunk_data["chunk_id"])
                        continue
            
            # Store embedding model information
            embedding_model_info = {
                "model_name": state.embedding_config["model"],
                "embedding_dimension": len(next(iter(chunk_embeddings.values()))) if chunk_embeddings else 0,
                "total_embeddings": len(chunk_embeddings),
                "failed_embeddings": len(failed_chunks),
                "normalization_applied": state.embedding_config.get("normalize", True)
            }
            
            # Calculate embedding generation time
            embedding_time = (datetime.utcnow() - start_time).total_seconds()
            state.processing_metrics.embedding_generation_time = embedding_time
            
            state.chunk_embeddings = chunk_embeddings
            state.embedding_model_info = embedding_model_info
            state.progress_percentage = 70.0
            
            success_rate = len(chunk_embeddings) / len(state.document_chunks) * 100 if state.document_chunks else 0
            
            state.messages.append(SystemMessage(
                content=f"Embedding generation completed. Success rate: {success_rate:.1f}%. "
                       f"Generated {len(chunk_embeddings)} embeddings in {embedding_time:.2f} seconds"
            ))
            
            return state
            
        except Exception as e:
            state.status = WorkflowStatus.FAILED
            state.error_message = f"Embedding generation failed: {str(e)}"
            return state
    
    async def _extract_knowledge_node(self, state: DocumentProcessorState) -> DocumentProcessorState:
        """Extract knowledge graph elements and entities from chunks."""
        try:
            self._log_progress("Extracting knowledge graph elements and entities")
            
            knowledge_graph_nodes = []
            knowledge_graph_edges = []
            entity_extraction = {
                "people": set(),
                "organizations": set(),
                "locations": set(),
                "concepts": set(),
                "technologies": set(),
                "dates": set()
            }
            
            # Process each chunk for knowledge extraction
            for chunk_data in state.document_chunks:
                chunk_content = chunk_data["content"]
                chunk_id = chunk_data["chunk_id"]
                
                # Extract entities (simplified - could use NER models)
                entities = self._extract_entities(chunk_content)
                
                # Add to entity collection
                for entity_type, entity_list in entities.items():
                    if entity_type in entity_extraction:
                        entity_extraction[entity_type].update(entity_list)
                
                # Create knowledge graph nodes for important entities
                for entity in entities.get("concepts", [])[:5]:  # Top 5 concepts per chunk
                    node = {
                        "id": f"node_{hashlib.md5(entity.encode()).hexdigest()[:8]}",
                        "label": entity,
                        "type": "concept",
                        "chunk_id": chunk_id,
                        "relevance_score": self._calculate_entity_relevance(entity, chunk_content)
                    }
                    knowledge_graph_nodes.append(node)
                
                # Create edges between related concepts
                concepts = list(entities.get("concepts", []))
                for i, concept1 in enumerate(concepts):
                    for concept2 in concepts[i+1:]:
                        if self._are_concepts_related(concept1, concept2, chunk_content):
                            edge = {
                                "source": f"node_{hashlib.md5(concept1.encode()).hexdigest()[:8]}",
                                "target": f"node_{hashlib.md5(concept2.encode()).hexdigest()[:8]}",
                                "relationship": "related",
                                "chunk_id": chunk_id,
                                "strength": self._calculate_relationship_strength(concept1, concept2, chunk_content)
                            }
                            knowledge_graph_edges.append(edge)
            
            # Convert sets to lists for storage
            for key in entity_extraction:
                entity_extraction[key] = list(entity_extraction[key])[:50]  # Limit to top 50 per category
            
            # Deduplicate nodes and edges
            unique_nodes = {node["id"]: node for node in knowledge_graph_nodes}
            knowledge_graph_nodes = list(unique_nodes.values())
            
            unique_edges = {f"{edge['source']}_{edge['target']}": edge for edge in knowledge_graph_edges}
            knowledge_graph_edges = list(unique_edges.values())
            
            state.knowledge_graph_nodes = knowledge_graph_nodes
            state.knowledge_graph_edges = knowledge_graph_edges
            state.entity_extraction = entity_extraction
            state.progress_percentage = 80.0
            
            total_entities = sum(len(entities) for entities in entity_extraction.values())
            
            state.messages.append(SystemMessage(
                content=f"Knowledge extraction completed. Extracted {total_entities} unique entities, "
                       f"created {len(knowledge_graph_nodes)} knowledge nodes and {len(knowledge_graph_edges)} edges"
            ))
            
            return state
            
        except Exception as e:
            state.status = WorkflowStatus.FAILED
            state.error_message = f"Knowledge extraction failed: {str(e)}"
            return state
    
    async def _optimize_search_node(self, state: DocumentProcessorState) -> DocumentProcessorState:
        """Optimize chunks for search and retrieval."""
        try:
            self._log_progress("Optimizing chunks for search and retrieval")
            
            search_optimization = {
                "indexing_strategy": self._determine_indexing_strategy(state),
                "retrieval_weights": {},
                "search_keywords": [],
                "query_expansion_terms": [],
                "relevance_boosters": {}
            }
            
            # Calculate retrieval weights for each chunk based on quality and relevance
            for chunk_data in state.document_chunks:
                chunk_id = chunk_data["chunk_id"]
                chunk_metadata = next((m for m in state.chunk_metadata if m.chunk_id == chunk_id), None)
                
                if chunk_metadata:
                    # Calculate retrieval weight
                    weight = self._calculate_retrieval_weight(
                        chunk_metadata.quality_score,
                        chunk_metadata.semantic_density,
                        chunk_data.get("position", {}).get("index", 0),
                        len(state.document_chunks)
                    )
                    search_optimization["retrieval_weights"][chunk_id] = weight
                    
                    # Identify relevance boosters (important sections, headers, etc.)
                    if chunk_metadata.section_title:
                        search_optimization["relevance_boosters"][chunk_id] = {
                            "section_title": chunk_metadata.section_title,
                            "boost_factor": 1.2
                        }
            
            # Extract search keywords from high-quality chunks
            high_quality_chunks = [
                chunk for chunk in state.document_chunks
                if state.chunk_quality_scores.get(chunk["chunk_id"], 0) > 0.7
            ]
            
            for chunk in high_quality_chunks[:10]:  # Process top 10 high-quality chunks
                keywords = self._extract_keywords(chunk["content"])
                search_optimization["search_keywords"].extend(keywords)
            
            # Deduplicate and limit keywords
            search_optimization["search_keywords"] = list(set(search_optimization["search_keywords"]))[:100]
            
            # Generate query expansion terms
            search_optimization["query_expansion_terms"] = self._generate_query_expansion_terms(
                search_optimization["search_keywords"],
                state.entity_extraction
            )
            
            # Determine optimal search configuration
            search_optimization["search_config"] = {
                "similarity_threshold": self._calculate_optimal_similarity_threshold(state),
                "max_results": 10,
                "reranking_enabled": True,
                "hybrid_search": len(search_optimization["search_keywords"]) > 20
            }
            
            state.search_optimization = search_optimization
            state.progress_percentage = 85.0
            
            state.messages.append(SystemMessage(
                content=f"Search optimization completed. Indexed {len(search_optimization['retrieval_weights'])} chunks, "
                       f"extracted {len(search_optimization['search_keywords'])} keywords, "
                       f"optimized with {search_optimization['indexing_strategy']} strategy"
            ))
            
            return state
            
        except Exception as e:
            state.status = WorkflowStatus.FAILED
            state.error_message = f"Search optimization failed: {str(e)}"
            return state
    
    async def _store_chunks_node(self, state: DocumentProcessorState) -> DocumentProcessorState:
        """Store chunks and embeddings in the database."""
        try:
            self._log_progress("Storing chunks and embeddings in database")
            
            start_time = datetime.utcnow()
            storage_status = {
                "chunks_to_store": len(state.document_chunks),
                "chunks_stored": 0,
                "embeddings_stored": 0,
                "metadata_stored": 0,
                "storage_errors": []
            }
            
            # Prepare chunks for storage
            chunks_for_storage = []
            for chunk_data in state.document_chunks:
                chunk_id = chunk_data["chunk_id"]
                
                # Get embedding if available
                embedding = state.chunk_embeddings.get(chunk_id)
                if not embedding:
                    storage_status["storage_errors"].append(f"No embedding for chunk {chunk_id}")
                    continue
                
                # Get metadata
                chunk_metadata = next((m for m in state.chunk_metadata if m.chunk_id == chunk_id), None)
                
                # Prepare storage record
                storage_record = {
                    "id": chunk_id,
                    "document_id": state.document_id,
                    "chunk_index": chunk_data["position"]["index"],
                    "content": chunk_data["content"],
                    "embedding": embedding,
                    "metadata": {
                        "chunk_size": len(chunk_data["content"]),
                        "chunk_hash": chunk_metadata.chunk_hash if chunk_metadata else "",
                        "quality_score": chunk_metadata.quality_score if chunk_metadata else 0,
                        "semantic_density": chunk_metadata.semantic_density if chunk_metadata else 0,
                        "section_title": chunk_metadata.section_title if chunk_metadata else None,
                        "position": chunk_data["position"],
                        "search_weight": state.search_optimization["retrieval_weights"].get(chunk_id, 1.0)
                    },
                    "created_at": datetime.utcnow().isoformat()
                }
                
                chunks_for_storage.append(storage_record)
            
            # Store chunks using legacy agent's storage method
            try:
                chunks_stored = self.legacy_agent._store_chunks(chunks_for_storage)
                storage_status["chunks_stored"] = chunks_stored
                storage_status["embeddings_stored"] = chunks_stored  # Embeddings stored with chunks
                storage_status["metadata_stored"] = chunks_stored
            except Exception as e:
                storage_status["storage_errors"].append(f"Database storage failed: {str(e)}")
                self._log_error(f"Failed to store chunks: {str(e)}")
            
            # Calculate storage time
            storage_time = (datetime.utcnow() - start_time).total_seconds()
            state.processing_metrics.storage_time = storage_time
            
            state.storage_status = storage_status
            state.progress_percentage = 90.0
            
            success_rate = (storage_status["chunks_stored"] / storage_status["chunks_to_store"] * 100) if storage_status["chunks_to_store"] > 0 else 0
            
            state.messages.append(SystemMessage(
                content=f"Storage completed. Stored {storage_status['chunks_stored']}/{storage_status['chunks_to_store']} chunks "
                       f"(success rate: {success_rate:.1f}%) in {storage_time:.2f} seconds"
            ))
            
            return state
            
        except Exception as e:
            state.status = WorkflowStatus.FAILED
            state.error_message = f"Chunk storage failed: {str(e)}"
            return state
    
    async def _update_index_node(self, state: DocumentProcessorState) -> DocumentProcessorState:
        """Update search indices and metadata."""
        try:
            self._log_progress("Updating search indices and metadata")
            
            index_updates = []
            
            # Update document metadata index
            document_index_update = {
                "index_type": "document_metadata",
                "document_id": state.document_id,
                "updates": {
                    "total_chunks": len(state.document_chunks),
                    "processing_strategy": state.processing_strategy.value,
                    "quality_score": state.quality_assessment.get("average_quality", 0),
                    "entities": state.entity_extraction,
                    "keywords": state.search_optimization.get("search_keywords", [])[:20],
                    "processing_date": datetime.utcnow().isoformat()
                },
                "status": "completed"
            }
            index_updates.append(document_index_update)
            
            # Update knowledge graph index if nodes were created
            if state.knowledge_graph_nodes:
                kg_index_update = {
                    "index_type": "knowledge_graph",
                    "document_id": state.document_id,
                    "updates": {
                        "node_count": len(state.knowledge_graph_nodes),
                        "edge_count": len(state.knowledge_graph_edges),
                        "primary_concepts": [node["label"] for node in state.knowledge_graph_nodes[:10]]
                    },
                    "status": "completed"
                }
                index_updates.append(kg_index_update)
            
            # Update search optimization index
            search_index_update = {
                "index_type": "search_optimization",
                "document_id": state.document_id,
                "updates": {
                    "indexing_strategy": state.search_optimization.get("indexing_strategy", "standard"),
                    "search_keywords_count": len(state.search_optimization.get("search_keywords", [])),
                    "query_expansion_terms": state.search_optimization.get("query_expansion_terms", [])[:10],
                    "optimal_similarity_threshold": state.search_optimization.get("search_config", {}).get("similarity_threshold", 0.7)
                },
                "status": "completed"
            }
            index_updates.append(search_index_update)
            
            state.index_updates = index_updates
            state.progress_percentage = 95.0
            
            state.messages.append(SystemMessage(
                content=f"Index updates completed. Updated {len(index_updates)} indices for optimized search and retrieval"
            ))
            
            return state
            
        except Exception as e:
            state.status = WorkflowStatus.FAILED
            state.error_message = f"Index update failed: {str(e)}"
            return state
    
    async def _finalize_processing_node(self, state: DocumentProcessorState) -> DocumentProcessorState:
        """Finalize processing and generate summary."""
        try:
            self._log_progress("Finalizing document processing")
            
            # Calculate final metrics
            processing_metrics = ProcessingMetrics(
                total_chunks=len(state.document_chunks),
                successful_chunks=state.storage_status.get("chunks_stored", 0),
                failed_chunks=len(state.document_chunks) - state.storage_status.get("chunks_stored", 0),
                average_chunk_size=sum(m.chunk_size for m in state.chunk_metadata) / len(state.chunk_metadata) if state.chunk_metadata else 0,
                processing_time_seconds=(datetime.utcnow() - state.created_at).total_seconds(),
                embedding_generation_time=state.processing_metrics.embedding_generation_time,
                storage_time=state.processing_metrics.storage_time,
                total_tokens=self._estimate_total_tokens(state.document_chunks),
                quality_score=state.quality_assessment.get("average_quality", 0),
                semantic_coherence=state.quality_assessment.get("semantic_coherence", 0)
            )
            
            # Generate validation results
            validation_results = {
                "document_integrity": self._validate_document_integrity(state),
                "chunk_coverage": self._validate_chunk_coverage(state),
                "embedding_completeness": len(state.chunk_embeddings) / len(state.document_chunks) * 100 if state.document_chunks else 0,
                "storage_success_rate": (state.storage_status.get("chunks_stored", 0) / len(state.document_chunks) * 100) if state.document_chunks else 0,
                "quality_threshold_met": state.quality_assessment.get("average_quality", 0) >= state.quality_thresholds.get("min_average_quality", 0.6)
            }
            
            state.processing_metrics = processing_metrics
            state.validation_results = validation_results
            state.status = WorkflowStatus.COMPLETED
            state.progress_percentage = 100.0
            state.completed_at = datetime.utcnow()
            
            state.messages.append(SystemMessage(
                content=f"Document processing completed successfully! "
                       f"Processed {processing_metrics.total_chunks} chunks in {processing_metrics.processing_time_seconds:.2f} seconds. "
                       f"Quality score: {processing_metrics.quality_score:.2f}, "
                       f"Storage success rate: {validation_results['storage_success_rate']:.1f}%"
            ))
            
            return state
            
        except Exception as e:
            state.status = WorkflowStatus.FAILED
            state.error_message = f"Processing finalization failed: {str(e)}"
            return state
    
    def _check_quality_threshold(self, state: DocumentProcessorState) -> str:
        """Check if quality threshold is met or rechunking is needed."""
        min_quality = state.quality_thresholds.get("min_average_quality", 0.6)
        current_quality = state.quality_assessment.get("average_quality", 0)
        
        # Allow one rechunking attempt
        rechunk_attempts = getattr(state, "_rechunk_attempts", 0)
        
        if current_quality < min_quality and rechunk_attempts == 0:
            # Adjust chunking parameters for rechunking
            state.chunking_parameters["chunk_size"] = int(state.chunking_parameters["chunk_size"] * 0.8)
            state.chunking_parameters["chunk_overlap"] = int(state.chunking_parameters["chunk_overlap"] * 1.2)
            state._rechunk_attempts = 1
            
            self._log_progress(f"Quality below threshold ({current_quality:.2f} < {min_quality}), rechunking with adjusted parameters")
            return "rechunk"
        
        return "proceed"
    
    # Helper methods for document analysis and processing
    
    def _determine_document_type(self, mime_type: str, file_extension: str) -> DocumentType:
        """Determine document type from mime type and extension."""
        mime_type_mapping = {
            "application/pdf": DocumentType.PDF,
            "text/plain": DocumentType.TEXT,
            "text/markdown": DocumentType.MARKDOWN,
            "text/html": DocumentType.HTML,
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": DocumentType.DOCX,
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": DocumentType.SPREADSHEET,
            "application/vnd.openxmlformats-officedocument.presentationml.presentation": DocumentType.PRESENTATION
        }
        
        if mime_type in mime_type_mapping:
            return mime_type_mapping[mime_type]
        
        # Fallback to extension-based detection
        extension_mapping = {
            ".pdf": DocumentType.PDF,
            ".txt": DocumentType.TEXT,
            ".md": DocumentType.MARKDOWN,
            ".html": DocumentType.HTML,
            ".docx": DocumentType.DOCX,
            ".xlsx": DocumentType.SPREADSHEET,
            ".pptx": DocumentType.PRESENTATION
        }
        
        return extension_mapping.get(file_extension.lower(), DocumentType.UNKNOWN)
    
    def _get_quality_thresholds(self, document_type: DocumentType) -> Dict[str, float]:
        """Get quality thresholds based on document type."""
        base_thresholds = {
            "min_chunk_quality": 0.6,
            "min_average_quality": 0.7,
            "min_semantic_coherence": 0.65,
            "max_chunk_size_variance": 0.3
        }
        
        # Adjust thresholds based on document type
        if document_type == DocumentType.PDF:
            # PDFs often have complex layouts, slightly lower thresholds
            base_thresholds["min_chunk_quality"] = 0.55
            base_thresholds["min_average_quality"] = 0.65
        elif document_type in [DocumentType.TEXT, DocumentType.MARKDOWN]:
            # Plain text should have higher quality thresholds
            base_thresholds["min_chunk_quality"] = 0.7
            base_thresholds["min_average_quality"] = 0.75
        
        return base_thresholds
    
    def _analyze_document_structure(self, content: str, document_type: DocumentType) -> Dict[str, Any]:
        """Analyze document structure and identify sections."""
        structure = {
            "sections": [],
            "hierarchy_depth": 0,
            "has_toc": False,
            "structural_elements": []
        }
        
        # Detect sections based on common patterns
        lines = content.split('\n')
        current_section = None
        section_level = 0
        
        for i, line in enumerate(lines):
            # Detect markdown headers
            if line.startswith('#'):
                header_level = len(line) - len(line.lstrip('#'))
                title = line.lstrip('#').strip()
                if title:
                    structure["sections"].append({
                        "title": title,
                        "level": header_level,
                        "line_number": i,
                        "char_position": sum(len(l) + 1 for l in lines[:i])
                    })
                    structure["hierarchy_depth"] = max(structure["hierarchy_depth"], header_level)
            
            # Detect numbered sections (1., 1.1, etc.)
            elif self._is_numbered_section(line):
                structure["sections"].append({
                    "title": line.strip(),
                    "level": line.count('.'),
                    "line_number": i,
                    "char_position": sum(len(l) + 1 for l in lines[:i])
                })
        
        # Detect structural elements
        if '```' in content:
            structure["structural_elements"].append("code_blocks")
        if '|' in content and content.count('|') > 5:
            structure["structural_elements"].append("tables")
        if any(line.strip().startswith(('- ', '* ', '+ ')) for line in lines):
            structure["structural_elements"].append("lists")
        
        return structure
    
    def _detect_headings(self, content: str) -> bool:
        """Detect if content has headings."""
        return any(line.startswith('#') for line in content.split('\n')) or \
               any(self._is_numbered_section(line) for line in content.split('\n'))
    
    def _detect_lists(self, content: str) -> bool:
        """Detect if content has lists."""
        lines = content.split('\n')
        return any(line.strip().startswith(('- ', '* ', '+ ', '• ')) for line in lines) or \
               any(line.strip() and line.strip()[0].isdigit() and '. ' in line[:5] for line in lines)
    
    def _detect_tables(self, content: str) -> bool:
        """Detect if content has tables."""
        lines = content.split('\n')
        # Simple table detection based on pipe characters
        return any(line.count('|') >= 3 for line in lines)
    
    def _detect_code_blocks(self, content: str) -> bool:
        """Detect if content has code blocks."""
        return '```' in content or '    ' in content  # Markdown code blocks or indented code
    
    def _detect_language(self, content: str) -> str:
        """Detect language of content (simplified)."""
        # This is a placeholder - in production, use langdetect or similar
        return "en"
    
    def _calculate_complexity_score(self, content: str) -> float:
        """Calculate document complexity score (0-100)."""
        words = content.split()
        sentences = content.split('.')
        
        if not words or not sentences:
            return 0.0
        
        # Simple complexity metrics
        avg_word_length = sum(len(word) for word in words) / len(words)
        avg_sentence_length = len(words) / len(sentences)
        
        # Normalize to 0-100 scale
        complexity = min(100, (avg_word_length * 10) + (avg_sentence_length * 2))
        return complexity
    
    def _calculate_semantic_density(self, content: str) -> float:
        """Calculate semantic density of content."""
        # Simplified semantic density based on unique words ratio
        words = content.lower().split()
        if not words:
            return 0.0
        
        unique_words = set(words)
        density = len(unique_words) / len(words)
        return min(1.0, density * 1.5)  # Scale up slightly
    
    def _detect_content_patterns(self, content: str) -> List[str]:
        """Detect content patterns for better chunking."""
        patterns = []
        
        if "Introduction" in content or "Abstract" in content:
            patterns.append("academic")
        if "Figure" in content or "Table" in content:
            patterns.append("scientific")
        if "def " in content or "class " in content or "function " in content:
            patterns.append("technical")
        if content.count("$") > 5:
            patterns.append("financial")
        if "Chapter" in content:
            patterns.append("book")
        
        return patterns
    
    def _determine_optimal_strategy(self, content_analysis: Dict[str, Any], 
                                  document_structure: Dict[str, Any],
                                  document_type: DocumentType) -> ProcessingStrategy:
        """Determine optimal processing strategy based on document characteristics."""
        
        # High complexity documents benefit from semantic chunking
        if content_analysis.get("complexity_score", 0) > 70:
            return ProcessingStrategy.SEMANTIC
        
        # Documents with clear structure benefit from hierarchical processing
        if len(document_structure.get("sections", [])) > 5:
            return ProcessingStrategy.HIERARCHICAL
        
        # Technical documents with code need special handling
        if content_analysis.get("has_code", False):
            return ProcessingStrategy.INTELLIGENT
        
        # Default to standard for simple documents
        return ProcessingStrategy.STANDARD
    
    def _get_chunking_parameters(self, strategy: ProcessingStrategy, 
                                content_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Get chunking parameters based on strategy."""
        base_params = {
            ProcessingStrategy.STANDARD: {
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "separators": ["\n\n", "\n", ". ", " ", ""]
            },
            ProcessingStrategy.SEMANTIC: {
                "chunk_size": 800,
                "chunk_overlap": 150,
                "separators": ["\n\n", "\n", ". ", "! ", "? ", "; "]
            },
            ProcessingStrategy.HIERARCHICAL: {
                "chunk_size": 1200,
                "chunk_overlap": 100,
                "separators": ["\n## ", "\n### ", "\n\n", "\n", ". "]
            },
            ProcessingStrategy.INTELLIGENT: {
                "chunk_size": 900,
                "chunk_overlap": 250,
                "separators": ["\n\n", "\n", "```", ". ", " "]
            }
        }
        
        params = base_params.get(strategy, base_params[ProcessingStrategy.STANDARD])
        
        # Adjust based on document characteristics
        if content_analysis.get("word_count", 0) > 10000:
            # Larger documents can use slightly larger chunks
            params["chunk_size"] = int(params["chunk_size"] * 1.1)
        
        return params
    
    def _calculate_chunk_semantic_density(self, chunk_content: str) -> float:
        """Calculate semantic density for a chunk."""
        return self._calculate_semantic_density(chunk_content)
    
    def _assess_chunk_quality(self, chunk_content: str) -> float:
        """Assess quality of an individual chunk."""
        quality_score = 0.0
        
        # Length quality (not too short, not too long)
        length = len(chunk_content)
        if 200 <= length <= 1500:
            quality_score += 0.3
        elif 100 <= length <= 2000:
            quality_score += 0.2
        else:
            quality_score += 0.1
        
        # Completeness (ends with sentence)
        if chunk_content.rstrip().endswith(('.', '!', '?')):
            quality_score += 0.2
        
        # Semantic coherence (has reasonable word diversity)
        words = chunk_content.split()
        if words:
            unique_ratio = len(set(words)) / len(words)
            quality_score += min(0.3, unique_ratio)
        
        # Structure preservation
        if '\n' in chunk_content:  # Preserves some structure
            quality_score += 0.1
        
        # No truncated code blocks
        if chunk_content.count('```') % 2 == 0:
            quality_score += 0.1
        
        return min(1.0, quality_score)
    
    def _extract_section_title(self, chunk_content: str, document_structure: Dict[str, Any]) -> Optional[str]:
        """Extract section title for a chunk if applicable."""
        # Check if chunk starts with a section header
        first_line = chunk_content.split('\n')[0] if '\n' in chunk_content else chunk_content[:50]
        
        for section in document_structure.get("sections", []):
            if section["title"] in first_line:
                return section["title"]
        
        return None
    
    def _calculate_quality_distribution(self, quality_scores: Dict[str, float]) -> Dict[str, int]:
        """Calculate distribution of quality scores."""
        distribution = {
            "excellent": 0,  # >= 0.8
            "good": 0,       # >= 0.6
            "fair": 0,       # >= 0.4
            "poor": 0        # < 0.4
        }
        
        for score in quality_scores.values():
            if score >= 0.8:
                distribution["excellent"] += 1
            elif score >= 0.6:
                distribution["good"] += 1
            elif score >= 0.4:
                distribution["fair"] += 1
            else:
                distribution["poor"] += 1
        
        return distribution
    
    def _assess_semantic_coherence(self, chunks: List[Dict[str, Any]]) -> float:
        """Assess overall semantic coherence of chunks."""
        if not chunks:
            return 0.0
        
        # Simple coherence based on content overlap
        total_overlap = 0
        for i in range(len(chunks) - 1):
            words1 = set(chunks[i]["content"].lower().split())
            words2 = set(chunks[i + 1]["content"].lower().split())
            
            if words1 and words2:
                overlap = len(words1 & words2) / min(len(words1), len(words2))
                total_overlap += overlap
        
        avg_overlap = total_overlap / (len(chunks) - 1) if len(chunks) > 1 else 1.0
        
        # Ideal overlap is around 0.1-0.3 (some continuity but not too much repetition)
        if 0.1 <= avg_overlap <= 0.3:
            return 1.0
        elif avg_overlap < 0.1:
            return 0.5 + avg_overlap * 5  # Too little overlap
        else:
            return max(0.3, 1.0 - (avg_overlap - 0.3) * 2)  # Too much overlap
    
    def _assess_size_consistency(self, chunk_metadata: List[ChunkMetadata]) -> float:
        """Assess consistency of chunk sizes."""
        if not chunk_metadata:
            return 0.0
        
        sizes = [m.chunk_size for m in chunk_metadata]
        if not sizes:
            return 0.0
        
        avg_size = sum(sizes) / len(sizes)
        variance = sum((s - avg_size) ** 2 for s in sizes) / len(sizes)
        std_dev = variance ** 0.5
        
        # Coefficient of variation (lower is better)
        cv = std_dev / avg_size if avg_size > 0 else 1.0
        
        # Convert to 0-1 score (lower CV = higher score)
        return max(0, 1.0 - cv)
    
    def _assess_boundary_quality(self, chunks: List[Dict[str, Any]]) -> float:
        """Assess quality of chunk boundaries."""
        if not chunks:
            return 0.0
        
        good_boundaries = 0
        
        for chunk in chunks:
            content = chunk["content"]
            # Check if chunk starts and ends reasonably
            if content and not content[0].islower() and content.rstrip().endswith(('.', '!', '?', ':', '\n')):
                good_boundaries += 1
        
        return good_boundaries / len(chunks) if chunks else 0.0
    
    def _assess_information_preservation(self, original_content: str, chunks: List[Dict[str, Any]]) -> float:
        """Assess how well information is preserved in chunks."""
        if not original_content or not chunks:
            return 0.0
        
        # Reconstruct content from chunks
        reconstructed = " ".join(chunk["content"] for chunk in chunks)
        
        # Simple preservation check based on content length
        preservation_ratio = len(reconstructed) / len(original_content)
        
        # Ideal is close to 1.0 (accounting for overlap)
        if 0.9 <= preservation_ratio <= 1.2:
            return 1.0
        else:
            return max(0, 1.0 - abs(1.0 - preservation_ratio))
    
    def _identify_chunk_issue(self, chunk_id: str, chunks: List[Dict[str, Any]]) -> str:
        """Identify specific issue with a chunk."""
        chunk = next((c for c in chunks if c["chunk_id"] == chunk_id), None)
        if not chunk:
            return "Chunk not found"
        
        content = chunk["content"]
        
        if len(content) < 100:
            return "Chunk too short"
        elif len(content) > 2000:
            return "Chunk too long"
        elif not content.rstrip().endswith(('.', '!', '?')):
            return "Incomplete sentence"
        elif content.count('```') % 2 != 0:
            return "Truncated code block"
        else:
            return "Low semantic coherence"
    
    def _calculate_overall_quality(self, quality_assessment: Dict[str, Any]) -> float:
        """Calculate overall quality score from assessment."""
        weights = {
            "average_quality": 0.3,
            "semantic_coherence": 0.25,
            "size_consistency": 0.15,
            "boundary_quality": 0.15,
            "information_preservation": 0.15
        }
        
        total_score = 0.0
        for metric, weight in weights.items():
            if metric in quality_assessment:
                total_score += quality_assessment[metric] * weight
        
        return total_score
    
    def _determine_quality_level(self, quality_score: float) -> str:
        """Determine quality level from score."""
        if quality_score >= 0.8:
            return ChunkingQuality.EXCELLENT.value
        elif quality_score >= 0.6:
            return ChunkingQuality.GOOD.value
        elif quality_score >= 0.4:
            return ChunkingQuality.FAIR.value
        else:
            return ChunkingQuality.POOR.value
    
    def _normalize_embedding(self, embedding: List[float]) -> List[float]:
        """Normalize embedding vector to unit length."""
        import math
        
        magnitude = math.sqrt(sum(x ** 2 for x in embedding))
        if magnitude > 0:
            return [x / magnitude for x in embedding]
        return embedding
    
    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract entities from text (simplified version)."""
        import re
        
        entities = {
            "people": [],
            "organizations": [],
            "locations": [],
            "concepts": [],
            "technologies": [],
            "dates": []
        }
        
        # Simple pattern-based extraction (could be replaced with NER)
        # Extract capitalized sequences as potential entities
        capitalized_words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        
        for word in capitalized_words[:20]:  # Limit to prevent overload
            # Simple heuristic classification
            if any(title in word.lower() for title in ["mr", "mrs", "dr", "prof"]):
                entities["people"].append(word)
            elif any(org in word.lower() for org in ["inc", "corp", "llc", "company"]):
                entities["organizations"].append(word)
            elif len(word.split()) == 1 and len(word) > 3:
                entities["concepts"].append(word)
        
        # Extract dates (simple pattern)
        date_patterns = re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b', text)
        entities["dates"] = date_patterns[:10]
        
        # Extract technologies (common patterns)
        tech_keywords = ["API", "AI", "ML", "AWS", "Azure", "Docker", "Kubernetes", "Python", "Java", "React"]
        for tech in tech_keywords:
            if tech in text:
                entities["technologies"].append(tech)
        
        return entities
    
    def _calculate_entity_relevance(self, entity: str, context: str) -> float:
        """Calculate relevance score for an entity."""
        # Simple frequency-based relevance
        occurrences = context.lower().count(entity.lower())
        return min(1.0, occurrences / 10)
    
    def _are_concepts_related(self, concept1: str, concept2: str, context: str) -> bool:
        """Determine if two concepts are related in the context."""
        # Simple proximity-based relationship
        text_lower = context.lower()
        if concept1.lower() in text_lower and concept2.lower() in text_lower:
            # Check if they appear near each other
            pos1 = text_lower.find(concept1.lower())
            pos2 = text_lower.find(concept2.lower())
            
            if pos1 >= 0 and pos2 >= 0:
                distance = abs(pos1 - pos2)
                return distance < 500  # Within 500 characters
        
        return False
    
    def _calculate_relationship_strength(self, concept1: str, concept2: str, context: str) -> float:
        """Calculate strength of relationship between concepts."""
        text_lower = context.lower()
        pos1 = text_lower.find(concept1.lower())
        pos2 = text_lower.find(concept2.lower())
        
        if pos1 >= 0 and pos2 >= 0:
            distance = abs(pos1 - pos2)
            # Closer concepts have stronger relationships
            return max(0, 1.0 - (distance / 1000))
        
        return 0.0
    
    def _determine_indexing_strategy(self, state: DocumentProcessorState) -> str:
        """Determine optimal indexing strategy."""
        if len(state.knowledge_graph_nodes) > 20:
            return "hybrid_graph_vector"
        elif state.content_analysis.get("has_tables", False):
            return "structured_hybrid"
        elif state.processing_strategy == ProcessingStrategy.HIERARCHICAL:
            return "hierarchical_index"
        else:
            return "standard_vector"
    
    def _calculate_retrieval_weight(self, quality_score: float, semantic_density: float, 
                                  chunk_index: int, total_chunks: int) -> float:
        """Calculate retrieval weight for a chunk."""
        # Base weight on quality
        base_weight = quality_score
        
        # Boost for semantic density
        density_boost = semantic_density * 0.2
        
        # Position boost (earlier chunks often contain important context)
        position_factor = 1.0 - (chunk_index / total_chunks) * 0.3
        
        return min(1.5, base_weight + density_boost) * position_factor
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text."""
        # Simple keyword extraction (could be enhanced with TF-IDF or RAKE)
        import re
        from collections import Counter
        
        # Remove common words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", 
                     "of", "with", "by", "from", "as", "is", "was", "are", "were", "be"}
        
        # Extract words
        words = re.findall(r'\b[a-z]+\b', text.lower())
        words = [w for w in words if w not in stop_words and len(w) > 3]
        
        # Get most common words as keywords
        word_counts = Counter(words)
        keywords = [word for word, _ in word_counts.most_common(20)]
        
        return keywords
    
    def _generate_query_expansion_terms(self, keywords: List[str], 
                                       entities: Dict[str, List[str]]) -> List[str]:
        """Generate query expansion terms for better search."""
        expansion_terms = []
        
        # Add variations of keywords
        for keyword in keywords[:10]:
            if keyword.endswith('ing'):
                expansion_terms.append(keyword[:-3])
            elif keyword.endswith('ed'):
                expansion_terms.append(keyword[:-2])
        
        # Add key entities as expansion terms
        for entity_type in ["concepts", "technologies"]:
            expansion_terms.extend(entities.get(entity_type, [])[:5])
        
        return list(set(expansion_terms))[:20]
    
    def _calculate_optimal_similarity_threshold(self, state: DocumentProcessorState) -> float:
        """Calculate optimal similarity threshold for search."""
        # Base threshold on quality and semantic density
        avg_quality = state.quality_assessment.get("average_quality", 0.7)
        avg_semantic_density = sum(m.semantic_density for m in state.chunk_metadata) / len(state.chunk_metadata) if state.chunk_metadata else 0.5
        
        # Higher quality chunks can use stricter similarity thresholds
        base_threshold = 0.7
        quality_adjustment = (avg_quality - 0.7) * 0.2
        density_adjustment = (avg_semantic_density - 0.5) * 0.1
        
        return max(0.5, min(0.85, base_threshold + quality_adjustment + density_adjustment))
    
    def _is_numbered_section(self, line: str) -> bool:
        """Check if line is a numbered section header."""
        import re
        # Match patterns like "1.", "1.1", "1.1.1", etc.
        return bool(re.match(r'^\d+(\.\d+)*\.?\s+\w+', line.strip()))
    
    def _estimate_total_tokens(self, chunks: List[Dict[str, Any]]) -> int:
        """Estimate total tokens for all chunks."""
        # Rough estimation: 1 token ≈ 4 characters
        total_chars = sum(len(chunk["content"]) for chunk in chunks)
        return total_chars // 4
    
    def _validate_document_integrity(self, state: DocumentProcessorState) -> bool:
        """Validate document processing integrity."""
        # Check if all chunks were processed
        return len(state.document_chunks) > 0 and len(state.chunk_embeddings) > 0
    
    def _validate_chunk_coverage(self, state: DocumentProcessorState) -> float:
        """Validate that chunks cover the document content."""
        if not state.document_content or not state.document_chunks:
            return 0.0
        
        total_chunk_chars = sum(len(chunk["content"]) for chunk in state.document_chunks)
        original_chars = len(state.document_content)
        
        # Coverage should be 100-120% (accounting for overlap)
        coverage = total_chunk_chars / original_chars if original_chars > 0 else 0
        
        if 0.9 <= coverage <= 1.3:
            return 100.0
        else:
            return max(0, 100 - abs(100 - coverage * 100))
    
    async def execute_workflow(
        self,
        document_id: str,
        file_path: str,
        processing_strategy: ProcessingStrategy = ProcessingStrategy.STANDARD,
        custom_chunk_size: Optional[int] = None,
        custom_chunk_overlap: Optional[int] = None,
        execution_context: Optional[LangGraphExecutionContext] = None
    ) -> AgentResult:
        """Execute the document processing workflow."""
        
        context = {
            "document_id": document_id,
            "file_path": file_path,
            "processing_strategy": processing_strategy.value if isinstance(processing_strategy, ProcessingStrategy) else processing_strategy,
            "custom_chunk_size": custom_chunk_size,
            "custom_chunk_overlap": custom_chunk_overlap,
            "workflow_id": f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }
        
        if execution_context:
            context["execution_context"] = execution_context.__dict__
        
        try:
            final_state = await self.run_workflow(context)
            
            if final_state.status == WorkflowStatus.COMPLETED:
                # Prepare successful result
                result_data = {
                    "document_id": final_state.document_id,
                    "chunks_created": final_state.processing_metrics.successful_chunks,
                    "total_chunks": final_state.processing_metrics.total_chunks,
                    "success_rate": (final_state.processing_metrics.successful_chunks / final_state.processing_metrics.total_chunks * 100) if final_state.processing_metrics.total_chunks > 0 else 0,
                    "processing_summary": {
                        "original_content_length": final_state.content_analysis.get("total_length", 0),
                        "average_chunk_size": final_state.processing_metrics.average_chunk_size,
                        "chunk_size_used": final_state.chunking_parameters.get("chunk_size", 0),
                        "overlap_used": final_state.chunking_parameters.get("chunk_overlap", 0),
                        "processing_strategy": final_state.processing_strategy.value,
                        "quality_score": final_state.processing_metrics.quality_score,
                        "semantic_coherence": final_state.processing_metrics.semantic_coherence
                    },
                    "knowledge_extraction": {
                        "entities_extracted": sum(len(entities) for entities in final_state.entity_extraction.values()),
                        "knowledge_nodes": len(final_state.knowledge_graph_nodes),
                        "knowledge_edges": len(final_state.knowledge_graph_edges),
                        "key_concepts": final_state.entity_extraction.get("concepts", [])[:10]
                    },
                    "search_optimization": {
                        "indexing_strategy": final_state.search_optimization.get("indexing_strategy", "standard"),
                        "search_keywords": len(final_state.search_optimization.get("search_keywords", [])),
                        "optimal_similarity_threshold": final_state.search_optimization.get("search_config", {}).get("similarity_threshold", 0.7)
                    },
                    "quality_metrics": {
                        "overall_quality": final_state.quality_assessment.get("overall_quality", 0),
                        "quality_level": final_state.quality_assessment.get("quality_level", "unknown"),
                        "validation_results": final_state.validation_results
                    },
                    "performance_metrics": {
                        "total_processing_time": final_state.processing_metrics.processing_time_seconds,
                        "embedding_generation_time": final_state.processing_metrics.embedding_generation_time,
                        "storage_time": final_state.processing_metrics.storage_time,
                        "total_tokens_processed": final_state.processing_metrics.total_tokens
                    }
                }
                
                return AgentResult(
                    success=True,
                    data=result_data,
                    execution_time_ms=final_state.processing_metrics.processing_time_seconds * 1000,
                    metadata={
                        "workflow_id": final_state.workflow_id,
                        "document_id": final_state.document_id,
                        "chunks_created": final_state.processing_metrics.successful_chunks,
                        "quality_score": final_state.processing_metrics.quality_score
                    }
                )
            else:
                return AgentResult(
                    success=False,
                    error_message=final_state.error_message or "Workflow failed",
                    metadata={
                        "workflow_id": final_state.workflow_id,
                        "final_status": final_state.status.value
                    }
                )
                
        except Exception as e:
            return AgentResult(
                success=False,
                error_message=f"Workflow execution failed: {str(e)}",
                metadata={"error_type": "workflow_execution_error"}
            )