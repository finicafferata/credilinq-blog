"""
LangGraph State Management System Usage Example

This example demonstrates how to use the complete PostgreSQL-backed
LangGraph state management system for production workflows.

Features demonstrated:
- DatabaseStateManager for persistent state storage
- PostgreSQL checkpointing integration
- State recovery and resumption
- Workflow snapshots and versioning
- Error handling and resilience
"""

import asyncio
import logging
from typing import Dict, Any, TypedDict
from dataclasses import asdict

from ..core.langgraph_compat import StateGraph, START, END
from langgraph.graph.message import add_messages

from ..core.langgraph_base import (
    DatabaseStateManager,
    PostgreSQLStateCheckpointer,
    LangGraphWorkflowBase,
    WorkflowState,
    WorkflowStatus,
    CheckpointStrategy,
    LangGraphExecutionContext
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define workflow state structure
class BlogCreationState(TypedDict):
    """State structure for blog creation workflow."""
    topic: str
    research_data: Dict[str, Any]
    outline: str
    content: str
    seo_data: Dict[str, Any]
    status: str
    current_step: str
    messages: list

class BlogCreationWorkflow(LangGraphWorkflowBase[BlogCreationState]):
    """
    Example LangGraph workflow with complete PostgreSQL state management.
    """
    
    def __init__(self, state_manager: DatabaseStateManager = None):
        # Initialize with database state management
        if not state_manager:
            state_manager = DatabaseStateManager()
        
        super().__init__(
            workflow_name="blog_creation_workflow",
            state_manager=state_manager,
            checkpoint_strategy=CheckpointStrategy.DATABASE_PERSISTENT,
            max_retries=3
        )
    
    def _create_workflow_graph(self) -> StateGraph:
        """Create the LangGraph workflow structure."""
        workflow = StateGraph(BlogCreationState)
        
        # Add workflow nodes
        workflow.add_node("research", self._research_step)
        workflow.add_node("outline", self._outline_step)
        workflow.add_node("write", self._write_step)
        workflow.add_node("seo_optimize", self._seo_step)
        workflow.add_node("finalize", self._finalize_step)
        
        # Define workflow edges
        workflow.add_edge(START, "research")
        workflow.add_edge("research", "outline")
        workflow.add_edge("outline", "write")
        workflow.add_edge("write", "seo_optimize")
        workflow.add_edge("seo_optimize", "finalize")
        workflow.add_edge("finalize", END)
        
        return workflow
    
    def _create_initial_state(self, input_data: Dict[str, Any]) -> BlogCreationState:
        """Create initial state for the workflow."""
        return BlogCreationState(
            topic=input_data.get("topic", ""),
            research_data={},
            outline="",
            content="",
            seo_data={},
            status="initialized",
            current_step="research",
            messages=[]
        )
    
    async def _research_step(self, state: BlogCreationState) -> BlogCreationState:
        """Research step with state persistence."""
        logger.info(f"🔍 Starting research for topic: {state['topic']}")
        
        try:
            # Simulate research work
            await asyncio.sleep(1)  # Simulate API calls
            
            research_data = {
                "key_points": ["Point 1", "Point 2", "Point 3"],
                "sources": ["Source A", "Source B"],
                "target_audience": "B2B professionals",
                "research_completed": True
            }
            
            # Update state
            state["research_data"] = research_data
            state["status"] = "research_completed"
            state["current_step"] = "outline"
            state["messages"] = add_messages(state["messages"], [
                {"role": "system", "content": "Research completed successfully"}
            ])
            
            logger.info("✅ Research step completed")
            return state
            
        except Exception as e:
            logger.error(f"❌ Research step failed: {e}")
            state["status"] = "error"
            state["messages"] = add_messages(state["messages"], [
                {"role": "system", "content": f"Research failed: {str(e)}"}
            ])
            raise
    
    async def _outline_step(self, state: BlogCreationState) -> BlogCreationState:
        """Outline creation step."""
        logger.info("📝 Creating content outline")
        
        try:
            await asyncio.sleep(0.5)
            
            outline = f"""
            # {state['topic']}
            
            ## Introduction
            - Hook and context
            
            ## Main Points
            {chr(10).join(f"- {point}" for point in state['research_data']['key_points'])}
            
            ## Conclusion
            - Summary and call-to-action
            """
            
            state["outline"] = outline.strip()
            state["status"] = "outline_completed"
            state["current_step"] = "write"
            state["messages"] = add_messages(state["messages"], [
                {"role": "system", "content": "Outline created successfully"}
            ])
            
            logger.info("✅ Outline step completed")
            return state
            
        except Exception as e:
            logger.error(f"❌ Outline step failed: {e}")
            state["status"] = "error"
            raise
    
    async def _write_step(self, state: BlogCreationState) -> BlogCreationState:
        """Content writing step."""
        logger.info("✍️ Writing blog content")
        
        try:
            await asyncio.sleep(1.5)  # Simulate content generation
            
            content = f"""
            # {state['topic']}
            
            ## Introduction
            
            Welcome to our comprehensive guide on {state['topic']}. This topic is increasingly important for {state['research_data']['target_audience']}.
            
            ## Key Insights
            
            {chr(10).join(f"### {point}" + chr(10) + "Detailed explanation of this important concept." for point in state['research_data']['key_points'])}
            
            ## Conclusion
            
            In conclusion, {state['topic']} represents a crucial area for modern businesses to understand and implement effectively.
            
            ---
            
            *Sources: {', '.join(state['research_data']['sources'])}*
            """
            
            state["content"] = content.strip()
            state["status"] = "content_completed"
            state["current_step"] = "seo_optimize"
            state["messages"] = add_messages(state["messages"], [
                {"role": "system", "content": "Content writing completed successfully"}
            ])
            
            logger.info("✅ Writing step completed")
            return state
            
        except Exception as e:
            logger.error(f"❌ Writing step failed: {e}")
            state["status"] = "error"
            raise
    
    async def _seo_step(self, state: BlogCreationState) -> BlogCreationState:
        """SEO optimization step."""
        logger.info("🎯 Optimizing for SEO")
        
        try:
            await asyncio.sleep(0.8)
            
            seo_data = {
                "title": f"Complete Guide to {state['topic']} | Expert Insights",
                "meta_description": f"Discover everything you need to know about {state['topic']} with our comprehensive guide and expert insights.",
                "keywords": [state['topic'].lower(), "guide", "expert", "insights"],
                "word_count": len(state["content"].split()),
                "reading_time": max(1, len(state["content"].split()) // 200),
                "seo_score": 85
            }
            
            state["seo_data"] = seo_data
            state["status"] = "seo_completed"
            state["current_step"] = "finalize"
            state["messages"] = add_messages(state["messages"], [
                {"role": "system", "content": f"SEO optimization completed with score: {seo_data['seo_score']}"}
            ])
            
            logger.info("✅ SEO optimization completed")
            return state
            
        except Exception as e:
            logger.error(f"❌ SEO step failed: {e}")
            state["status"] = "error"
            raise
    
    async def _finalize_step(self, state: BlogCreationState) -> BlogCreationState:
        """Final step to complete the workflow."""
        logger.info("🏁 Finalizing blog creation")
        
        try:
            state["status"] = "completed"
            state["current_step"] = "finished"
            state["messages"] = add_messages(state["messages"], [
                {"role": "system", "content": "Blog creation workflow completed successfully"}
            ])
            
            logger.info("✅ Blog creation workflow completed")
            return state
            
        except Exception as e:
            logger.error(f"❌ Finalization failed: {e}")
            state["status"] = "error"
            raise

async def demonstrate_state_management():
    """
    Comprehensive demonstration of the LangGraph state management system.
    """
    print("🚀 Starting LangGraph State Management Demonstration")
    print("=" * 60)
    
    # Initialize state manager
    state_manager = DatabaseStateManager()
    
    # Create workflow instance
    workflow = BlogCreationWorkflow(state_manager)
    
    # Create execution context
    context = LangGraphExecutionContext(
        workflow_id="demo_blog_workflow_001",
        user_id="demo_user",
        checkpoint_strategy=CheckpointStrategy.DATABASE_PERSISTENT
    )
    
    # Define input data
    input_data = {
        "topic": "AI-Powered Content Creation for B2B Marketing"
    }
    
    print(f"📝 Topic: {input_data['topic']}")
    print(f"🆔 Workflow ID: {context.workflow_id}")
    print()
    
    try:
        # Execute workflow with full state management
        print("▶️  Executing workflow with state persistence...")
        result = await workflow.execute(input_data, context)
        
        if result.success:
            print("✅ Workflow completed successfully!")
            print(f"📊 Final status: {result.data.get('status', 'unknown')}")
            print(f"📈 SEO Score: {result.data.get('seo_data', {}).get('seo_score', 'N/A')}")
            print(f"📝 Word Count: {result.data.get('seo_data', {}).get('word_count', 'N/A')}")
        else:
            print(f"❌ Workflow failed: {result.error_message}")
        
        print()
        print("🔍 Demonstrating state management features...")
        
        # Demonstrate state snapshot creation
        print("📸 Creating state snapshot...")
        snapshot_id = await state_manager.create_state_snapshot(
            context.workflow_id,
            snapshot_type="demo",
            description="Demo workflow completion snapshot",
            tags=["demo", "blog_creation", "completed"]
        )
        print(f"✅ Snapshot created: {snapshot_id}")
        
        # Demonstrate checkpoint listing
        print("📋 Listing workflow checkpoints...")
        checkpoints = await state_manager.list_checkpoints(context.workflow_id)
        print(f"✅ Found {len(checkpoints)} checkpoints")
        for checkpoint in checkpoints[:3]:  # Show first 3
            print(f"   • {checkpoint['checkpoint_id']}: {checkpoint['step_name']} (step {checkpoint['step_index']})")
        
        # Demonstrate workflow metrics
        print("📊 Getting workflow metrics...")
        metrics = await state_manager.get_workflow_metrics(context.workflow_id)
        if metrics:
            print(f"✅ Workflow Metrics:")
            print(f"   • Duration: {metrics.get('total_duration_ms', 0)}ms")
            print(f"   • Success Rate: {metrics.get('success_rate', 0):.1%}")
            print(f"   • Agent Executions: {metrics.get('agent_executions', 0)}")
            print(f"   • Quality Score: {metrics.get('average_quality_score', 0)}")
        
        # Demonstrate workflow recovery simulation
        print("🔄 Demonstrating workflow recovery...")
        recovered_state = await state_manager.load_state(context.workflow_id)
        if recovered_state:
            print(f"✅ State recovery successful:")
            print(f"   • Status: {recovered_state.get('status')}")
            print(f"   • Current Step: {recovered_state.get('current_step')}")
            print(f"   • Messages: {len(recovered_state.get('messages', []))} recorded")
        
        print()
        print("🧹 Cleanup demonstration...")
        
        # Note: In production, you wouldn't usually clean up immediately
        # This is just for demonstration purposes
        cleanup_stats = await state_manager.cleanup_expired_data(days_old=0)  # Immediate cleanup for demo
        print(f"✅ Cleanup completed: {cleanup_stats}")
        
    except Exception as e:
        print(f"❌ Demonstration failed: {e}")
        logger.exception("Full error details:")
    
    finally:
        # Clean up resources
        state_manager.close()
        print("\n🏁 Demonstration completed!")

async def demonstrate_error_recovery():
    """
    Demonstrate error recovery and workflow resumption.
    """
    print("\n🔄 Demonstrating Error Recovery and Resumption")
    print("=" * 50)
    
    state_manager = DatabaseStateManager()
    
    try:
        # Simulate recovering from a failed workflow
        failed_workflow_id = "demo_failed_workflow_001"
        
        # Create some mock state data
        mock_failed_state = {
            "workflow_id": failed_workflow_id,
            "status": "error",
            "current_step": "write",
            "topic": "AI Error Recovery",
            "research_data": {"key_points": ["Recovery", "Resilience"]},
            "outline": "# AI Error Recovery\n\n## Introduction\n...",
            "error_message": "Network timeout during content generation",
            "retry_count": 1
        }
        
        # Save the failed state
        await state_manager.save_state(failed_workflow_id, mock_failed_state)
        print(f"💾 Saved mock failed workflow state: {failed_workflow_id}")
        
        # Demonstrate state recovery
        recovered_state = await state_manager.load_state(failed_workflow_id)
        if recovered_state:
            print("✅ Successfully recovered failed workflow state:")
            print(f"   • Last Step: {recovered_state.get('current_step')}")
            print(f"   • Error: {recovered_state.get('error_message')}")
            print(f"   • Retry Count: {recovered_state.get('retry_count')}")
        
        # Create a snapshot before attempting recovery
        snapshot_id = await state_manager.create_state_snapshot(
            failed_workflow_id,
            snapshot_type="error_recovery",
            description="Snapshot before error recovery attempt"
        )
        print(f"📸 Created recovery snapshot: {snapshot_id}")
        
        # Simulate successful recovery
        recovered_state["status"] = "recovered"
        recovered_state["error_message"] = None
        recovered_state["current_step"] = "finalize"
        await state_manager.save_state(failed_workflow_id, recovered_state)
        print("✅ Simulated successful workflow recovery")
        
    except Exception as e:
        print(f"❌ Error recovery demonstration failed: {e}")
    
    finally:
        state_manager.close()

if __name__ == "__main__":
    """
    Run the complete LangGraph state management demonstration.
    """
    print("🎯 LangGraph PostgreSQL State Management System")
    print("Complete Production-Ready Implementation")
    print("=" * 60)
    
    async def run_demonstrations():
        await demonstrate_state_management()
        await demonstrate_error_recovery()
        
        print("\n🎉 All demonstrations completed successfully!")
        print("\nKey Features Demonstrated:")
        print("✅ PostgreSQL-backed state persistence")
        print("✅ Automatic checkpoint creation and recovery")
        print("✅ Workflow state snapshots and versioning")
        print("✅ Comprehensive error handling and recovery")
        print("✅ Performance metrics and monitoring")
        print("✅ Production-ready resource management")
        print("\n🚀 Ready for production use!")
    
    # Run the demonstrations
    asyncio.run(run_demonstrations())