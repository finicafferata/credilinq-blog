"""
Structured Blog Workflow - Modern replacement for the legacy blog_workflow.py
"""

from typing import Dict, Any, Optional, List
import uuid
import datetime
import json
import os
from dotenv import load_dotenv
load_dotenv()

from ..core.base_agent import (
    WorkflowAgent, AgentResult, AgentExecutionContext, AgentMetadata, 
    AgentType, AgentStatus
)
from ..core.agent_factory import BlogWorkflowAgentFactory, create_agent, register_agent
from ..specialized.planner_agent import PlannerAgent
from ..specialized.researcher_agent import ResearcherAgent
from ..specialized.writer_agent import WriterAgent
from ..specialized.editor_agent import EditorAgent
from ..specialized.image_prompt_agent import ImagePromptAgent
from ..specialized.video_prompt_agent import VideoPromptAgent
from ..specialized.seo_agent import SEOAgent
from ..specialized.social_media_agent import SocialMediaAgent
from ...core.exceptions import AgentExecutionError, WorkflowExecutionError
from ...core.security import SecurityValidator


class BlogWorkflow(WorkflowAgent):
    """
    Structured blog workflow orchestrating multiple specialized agents.
    """
    
    def __init__(self, metadata: Optional[AgentMetadata] = None):
        if metadata is None:
            metadata = AgentMetadata(
                agent_type=AgentType.WORKFLOW_ORCHESTRATOR,
                name="BlogWorkflow",
                description="Orchestrates blog content creation using specialized agents",
                capabilities=[
                    "blog_generation",
                    "linkedin_post_creation", 
                    "article_writing",
                    "multi_agent_orchestration",
                    "quality_assurance",
                    "revision_management"
                ],
                version="2.0.0",
                max_retries=2
            )
        
        super().__init__(metadata)
        self.security_validator = SecurityValidator()
        self.db_service = None
        
        # Initialize specialized agents
        self._initialize_agents()
    
    def _register_workflow_agents(self):
        """Register all workflow agents with the global registry."""
        try:
            # Register agents if not already registered
            from ..core.agent_factory import _global_registry
            
            if not _global_registry.is_registered(AgentType.PLANNER):
                register_agent(AgentType.PLANNER, PlannerAgent)
            
            if not _global_registry.is_registered(AgentType.RESEARCHER):
                register_agent(AgentType.RESEARCHER, ResearcherAgent)
            
            if not _global_registry.is_registered(AgentType.WRITER):
                register_agent(AgentType.WRITER, WriterAgent)
            
            if not _global_registry.is_registered(AgentType.EDITOR):
                register_agent(AgentType.EDITOR, EditorAgent)
            
            if not _global_registry.is_registered(AgentType.IMAGE_PROMPT):
                register_agent(AgentType.IMAGE_PROMPT, ImagePromptAgent)
            
            if not _global_registry.is_registered(AgentType.VIDEO_PROMPT):
                register_agent(AgentType.VIDEO_PROMPT, VideoPromptAgent)
            
            if not _global_registry.is_registered(AgentType.SEO):
                register_agent(AgentType.SEO, SEOAgent)
            
            if not _global_registry.is_registered(AgentType.SOCIAL_MEDIA):
                register_agent(AgentType.SOCIAL_MEDIA, SocialMediaAgent)
                
            self.logger.info("Workflow agents registered successfully")
            
        except Exception as e:
            self.logger.warning(f"Agent registration failed: {str(e)}")
            # Continue anyway, we'll handle missing agents gracefully
    
    def _initialize_agents(self):
        """Initialize all specialized agents for the workflow."""
        try:
            # Register agents if not already registered
            self._register_workflow_agents()
            
            # Create specialized agents with custom metadata
            self.planner_agent = create_agent(
                AgentType.PLANNER,
                AgentMetadata(
                    agent_type=AgentType.PLANNER,
                    name="BlogPlanner",
                    description="Creates structured outlines for blog content"
                )
            )
            
            self.researcher_agent = create_agent(
                AgentType.RESEARCHER,
                AgentMetadata(
                    agent_type=AgentType.RESEARCHER,
                    name="BlogResearcher",
                    description="Researches information for blog content"
                )
            )
            
            self.writer_agent = create_agent(
                AgentType.WRITER,
                AgentMetadata(
                    agent_type=AgentType.WRITER,
                    name="BlogWriter",
                    description="Generates blog content from research"
                )
            )
            
            self.editor_agent = create_agent(
                AgentType.EDITOR,
                AgentMetadata(
                    agent_type=AgentType.EDITOR,
                    name="BlogEditor",
                    description="Reviews and refines blog content"
                )
            )
            
            # Add to child agents list
            self.add_agent(self.planner_agent)
            self.add_agent(self.researcher_agent)
            self.add_agent(self.writer_agent)
            self.add_agent(self.editor_agent)
            
            # Initialize database service
            from ..core.database_service import get_db_service
            self.db_service = get_db_service()
            
            self.logger.info("Blog workflow agents initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize workflow agents: {str(e)}")
            raise AgentExecutionError("BlogWorkflow", "initialization", str(e))
    
    def execute_workflow(
        self, 
        input_data: Dict[str, Any], 
        context: Optional[AgentExecutionContext] = None
    ) -> AgentResult:
        """
        Execute the complete blog generation workflow.
        
        Args:
            input_data: Dictionary containing:
                - title: Blog title
                - company_context: Company context
                - content_type: Type of content (blog, linkedin, article)
            context: Execution context
            
        Returns:
            AgentResult: Complete workflow result
        """
        if context is None:
            context = AgentExecutionContext(
                workflow_id=str(uuid.uuid4())
            )
        
        try:
            # Validate and sanitize input
            validated_input = self._validate_and_sanitize_input(input_data)
            
            # Create initial blog record
            blog_id = self._create_initial_blog_record(validated_input, context)
            
            # Execute workflow steps
            workflow_result = self._execute_workflow_steps(validated_input, context, blog_id)
            
            # Update final blog record
            self._update_final_blog_record(blog_id, workflow_result)
            
            # Prepare final result
            final_result = {
                "blog_id": blog_id,
                "content_markdown": workflow_result.get("final_content"),
                "workflow_metadata": {
                    "steps_completed": workflow_result.get("steps_completed", []),
                    "total_execution_time": workflow_result.get("total_execution_time"),
                    "revision_count": workflow_result.get("revision_count", 0),
                    "quality_score": workflow_result.get("quality_score"),
                    "agent_performance": workflow_result.get("agent_performance", {})
                }
            }
            
            return AgentResult(
                success=True,
                data=final_result,
                metadata={
                    "workflow_id": context.workflow_id,
                    "blog_id": blog_id,
                    "content_type": validated_input["content_type"],
                    "agents_used": len(self.child_agents)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Workflow execution failed: {str(e)}")
            return AgentResult(
                success=False,
                error_message=str(e),
                error_code="WORKFLOW_EXECUTION_FAILED"
            )
    
    def _validate_and_sanitize_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and sanitize workflow input."""
        required_fields = ["title", "company_context", "content_type"]
        
        for field in required_fields:
            if field not in input_data:
                raise ValueError(f"Missing required field: {field}")
        
        # Security validation (relaxed for NL content)
        try:
            self.security_validator.validate_content(str(input_data["title"]).strip(), "title")
            self.security_validator.validate_content(str(input_data["company_context"]).strip(), "company_context")
        except Exception as e:
            self.logger.error(f"Workflow input validation failed: title='{str(input_data['title'])[:80]}', error={e}")
            raise
        
        # Validate content type
        valid_content_types = ["blog", "linkedin", "article"]
        content_type = input_data["content_type"].lower()
        if content_type not in valid_content_types:
            raise ValueError(f"Invalid content_type. Must be one of: {valid_content_types}")
        
        return {
            "blog_title": input_data["title"].strip(),
            "company_context": input_data["company_context"].strip(), 
            "content_type": content_type
        }
    
    def _create_initial_blog_record(
        self, 
        validated_input: Dict[str, Any], 
        context: AgentExecutionContext
    ) -> str:
        """Create initial blog record in database."""
        try:
            blog_id = str(uuid.uuid4())
            blog_data = {
                "id": blog_id,
                "title": validated_input["blog_title"],
                "status": "draft",
                "content_markdown": "",
                "initial_prompt": json.dumps({
                    "title": validated_input["blog_title"],
                    "company_context": validated_input["company_context"],
                    "content_type": validated_input["content_type"],
                    "workflow_id": context.workflow_id
                }),
                "created_at": datetime.datetime.utcnow().isoformat()
            }
            
            # Insert into database
            if self.db_service.use_supabase and self.db_service.supabase:
                result = self.db_service.supabase.table("blog_posts").insert(blog_data).execute()
                
                if not result or getattr(result, "status_code", 200) >= 400:
                    raise Exception(f"Database insertion failed: {result}")
            else:
                # For local PostgreSQL, use direct connection
                with self.db_service.get_db_connection() as conn:
                    cur = conn.cursor()
                    cur.execute("""
                        INSERT INTO blog_posts (id, title, status, content_markdown, initial_prompt, "createdAt", "updatedAt")
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, (
                        blog_data["id"],
                        blog_data["title"],
                        blog_data["status"],
                        blog_data["content_markdown"],
                        blog_data["initial_prompt"],
                        blog_data["created_at"],
                        blog_data["created_at"]  # Use same timestamp for updatedAt
                    ))
                    conn.commit()
            
            self.logger.info(f"Created initial blog record: {blog_id}")
            return blog_id
            
        except Exception as e:
            raise AgentExecutionError("BlogWorkflow", "create_blog_record", f"Failed to create blog record: {str(e)}")
    
    def _execute_workflow_steps(
        self, 
        validated_input: Dict[str, Any], 
        context: AgentExecutionContext,
        blog_id: str
    ) -> Dict[str, Any]:
        """Execute the main workflow steps."""
        workflow_state = {
            "steps_completed": [],
            "current_step": None,
            "revision_count": 0,
            "max_revisions": 2,
            "agent_performance": {},
            "total_execution_time": 0
        }
        
        start_time = datetime.datetime.utcnow()
        
        try:
            # Step 1: Planning
            outline_result = self._execute_planning_step(validated_input, context)
            workflow_state["steps_completed"].append("planning")
            workflow_state["agent_performance"]["planner"] = {
                "execution_time": outline_result.execution_time_ms,
                "success": outline_result.success
            }
            
            if not outline_result.success:
                raise WorkflowExecutionError("BlogWorkflow", "planning", f"Planning step failed: {outline_result.error_message}")
            
            # Step 2: Research
            research_result = self._execute_research_step(
                validated_input, outline_result.data, context
            )
            workflow_state["steps_completed"].append("research")
            workflow_state["agent_performance"]["researcher"] = {
                "execution_time": research_result.execution_time_ms,
                "success": research_result.success
            }
            
            if not research_result.success:
                raise WorkflowExecutionError("BlogWorkflow", "research", f"Research step failed: {research_result.error_message}")
            
            # Step 3: Writing with potential revisions
            final_content, revision_data = self._execute_writing_revision_loop(
                validated_input, outline_result.data, research_result.data, 
                context, workflow_state
            )
            
            workflow_state["revision_count"] = revision_data["revision_count"]
            workflow_state["quality_score"] = revision_data["final_quality_score"]
            
            # Step 4: Enhance content with creative prompts
            enhanced_content = self._enhance_content_with_prompts(
                final_content, validated_input, context, workflow_state
            )
            
            # Calculate total execution time
            end_time = datetime.datetime.utcnow()
            workflow_state["total_execution_time"] = (end_time - start_time).total_seconds() * 1000
            
            return {
                **workflow_state,
                "final_content": enhanced_content,
                "outline": outline_result.data["outline"],
                "research": research_result.data["research"]
            }
            
        except Exception as e:
            self.logger.error(f"Workflow step execution failed: {str(e)}")
            raise
    
    def _execute_planning_step(
        self, 
        validated_input: Dict[str, Any], 
        context: AgentExecutionContext
    ) -> AgentResult:
        """Execute the planning step."""
        self.logger.info("Executing planning step")
        
        planning_input = {
            "blog_title": validated_input["blog_title"],
            "company_context": validated_input["company_context"],
            "content_type": validated_input["content_type"]
        }
        
        return self.planner_agent.execute_safe(planning_input, context)
    
    def _execute_research_step(
        self, 
        validated_input: Dict[str, Any], 
        outline_data: Dict[str, Any], 
        context: AgentExecutionContext
    ) -> AgentResult:
        """Execute the research step."""
        self.logger.info("Executing research step")
        
        research_input = {
            "outline": outline_data["outline"],
            "blog_title": validated_input["blog_title"],
            "company_context": validated_input["company_context"]
        }
        
        return self.researcher_agent.execute_safe(research_input, context)
    
    def _execute_writing_revision_loop(
        self,
        validated_input: Dict[str, Any],
        outline_data: Dict[str, Any],
        research_data: Dict[str, Any],
        context: AgentExecutionContext,
        workflow_state: Dict[str, Any]
    ) -> tuple[str, Dict[str, Any]]:
        """Execute writing and revision loop."""
        self.logger.info("Executing writing and revision loop")
        
        revision_count = 0
        max_revisions = workflow_state["max_revisions"]
        current_content = None
        review_notes = None
        
        while revision_count <= max_revisions:
            # Writing step
            writing_input = {
                "outline": outline_data["outline"],
                "research": research_data["research"],
                "blog_title": validated_input["blog_title"],
                "company_context": validated_input["company_context"],
                "content_type": validated_input["content_type"],
                "review_notes": review_notes
            }
            
            writing_result = self.writer_agent.execute_safe(writing_input, context)
            
            if not writing_result.success:
                raise WorkflowExecutionError("BlogWorkflow", "writing", f"Writing step failed: {writing_result.error_message}")
            
            current_content = writing_result.data["content"]
            
            # Update performance tracking
            step_name = f"writing_attempt_{revision_count + 1}"
            workflow_state["agent_performance"][step_name] = {
                "execution_time": writing_result.execution_time_ms,
                "success": writing_result.success,
                "word_count": writing_result.data.get("word_count", 0)
            }
            
            # Editorial review
            review_input = {
                "content": current_content,
                "blog_title": validated_input["blog_title"],
                "company_context": validated_input["company_context"],
                "content_type": validated_input["content_type"]
            }
            
            review_result = self.editor_agent.execute_safe(review_input, context)
            
            if not review_result.success:
                self.logger.warning(f"Editorial review failed: {review_result.error_message}")
                # Continue with content but note the issue
                break
            
            # Update performance tracking
            step_name = f"editing_attempt_{revision_count + 1}"
            workflow_state["agent_performance"][step_name] = {
                "execution_time": review_result.execution_time_ms,
                "success": review_result.success,
                "quality_score": review_result.data.get("review_score", 0)
            }
            
            # Check if content is approved
            if review_result.data["approved"]:
                self.logger.info(f"Content approved after {revision_count} revisions")
                workflow_state["steps_completed"].append(f"writing_approved_revision_{revision_count}")
                break
            
            # Prepare for revision
            review_notes = review_result.data.get("revision_notes")
            revision_count += 1
            
            if revision_count > max_revisions:
                self.logger.warning(f"Maximum revisions ({max_revisions}) reached, using current content")
                workflow_state["steps_completed"].append("writing_max_revisions_reached")
                break
            
            self.logger.info(f"Content needs revision (attempt {revision_count}/{max_revisions})")
            workflow_state["steps_completed"].append(f"writing_revision_{revision_count}")
        
        revision_data = {
            "revision_count": revision_count,
            "final_quality_score": review_result.data.get("review_score", 70) if review_result.success else 70,
            "final_approved": review_result.data.get("approved", False) if review_result.success else False
        }
        
        return current_content, revision_data
    
    def _enhance_content_with_prompts(
        self,
        content: str,
        validated_input: Dict[str, Any],
        context: AgentExecutionContext,
        workflow_state: Dict[str, Any]
    ) -> str:
        """Enhance content with image and video prompt suggestions."""
        try:
            self.logger.info("Enhancing content with creative prompts")
            
            # Initialize prompt agents
            image_prompt_agent = ImagePromptAgent()
            video_prompt_agent = VideoPromptAgent()
            
            # Prepare input for prompt agents
            prompt_input = {
                "content": content,
                "blog_title": validated_input["blog_title"],
                "company_context": validated_input["company_context"],
                "content_type": validated_input["content_type"]
            }
            
            # Generate image prompts
            image_result = image_prompt_agent.execute_safe(prompt_input, context)
            image_prompts = []
            if image_result.success:
                image_prompts = image_result.data.get("prompts", [])
                workflow_state["agent_performance"]["image_prompt"] = {
                    "execution_time": image_result.execution_time_ms,
                    "success": image_result.success,
                    "prompts_generated": len(image_prompts)
                }
            else:
                self.logger.warning(f"Image prompt generation failed: {image_result.error_message}")
            
            # Generate video prompts
            video_result = video_prompt_agent.execute_safe(prompt_input, context)
            video_prompts = []
            if video_result.success:
                video_prompts = video_result.data.get("prompts", [])
                workflow_state["agent_performance"]["video_prompt"] = {
                    "execution_time": video_result.execution_time_ms,
                    "success": video_result.success,
                    "prompts_generated": len(video_prompts)
                }
            else:
                self.logger.warning(f"Video prompt generation failed: {video_result.error_message}")
            
            # Enhance content with prompts
            if image_prompts or video_prompts:
                enhanced_content = self._integrate_prompts_into_content(
                    content, image_prompts, video_prompts, validated_input["content_type"]
                )
                workflow_state["steps_completed"].append("content_enhancement_with_prompts")
                return enhanced_content
            else:
                self.logger.info("No prompts generated, returning original content")
                return content
                
        except Exception as e:
            self.logger.error(f"Content enhancement failed: {str(e)}")
            # Return original content if enhancement fails
            return content
    
    def _integrate_prompts_into_content(
        self,
        content: str,
        image_prompts: List[Dict[str, Any]],
        video_prompts: List[Dict[str, Any]],
        content_type: str
    ) -> str:
        """Integrate prompts strategically into the content."""
        try:
            # Split content into sections for strategic placement
            sections = content.split('\n\n')
            enhanced_sections = []
            
            # Add prompts throughout the content
            prompt_positions = self._calculate_prompt_positions(len(sections), len(image_prompts), len(video_prompts))
            
            for i, section in enumerate(sections):
                enhanced_sections.append(section)
                
                # Add image prompt suggestions at calculated positions
                if i in prompt_positions.get("image_positions", []) and image_prompts:
                    prompt_idx = prompt_positions["image_positions"].index(i)
                    if prompt_idx < len(image_prompts):
                        image_suggestion = self._format_image_prompt_suggestion(
                            image_prompts[prompt_idx], content_type
                        )
                        enhanced_sections.append(image_suggestion)
                
                # Add video prompt suggestions at calculated positions
                if i in prompt_positions.get("video_positions", []) and video_prompts:
                    prompt_idx = prompt_positions["video_positions"].index(i)
                    if prompt_idx < len(video_prompts):
                        video_suggestion = self._format_video_prompt_suggestion(
                            video_prompts[prompt_idx], content_type
                        )
                        enhanced_sections.append(video_suggestion)
            
            # Add comprehensive creative assets section at the end
            creative_section = self._create_creative_assets_section(
                image_prompts, video_prompts, content_type
            )
            enhanced_sections.append(creative_section)
            
            return '\n\n'.join(enhanced_sections)
            
        except Exception as e:
            self.logger.error(f"Prompt integration failed: {str(e)}")
            return content
    
    def _calculate_prompt_positions(
        self, 
        section_count: int, 
        image_count: int, 
        video_count: int
    ) -> Dict[str, List[int]]:
        """Calculate optimal positions for prompt suggestions."""
        positions = {"image_positions": [], "video_positions": []}
        
        if section_count < 3:
            return positions
        
        # Place image prompts in the first third and middle sections
        if image_count > 0:
            mid_point = section_count // 2
            positions["image_positions"] = [min(2, section_count - 1)]  # Early position
            if image_count > 1 and section_count > 4:
                positions["image_positions"].append(mid_point)  # Middle position
        
        # Place video prompts in the latter sections
        if video_count > 0 and section_count > 3:
            late_position = max(section_count - 3, section_count // 2 + 1)
            positions["video_positions"] = [late_position]
        
        return positions
    
    def _format_image_prompt_suggestion(
        self, 
        image_prompt: Dict[str, Any], 
        content_type: str
    ) -> str:
        """Format an image prompt suggestion for content integration."""
        prompt_text = image_prompt.get("prompt", "")
        style_info = image_prompt.get("style_guidance", {})
        
        suggestion = f"\n**💡 Image Suggestion:**\n"
        suggestion += f"*{prompt_text}*\n"
        
        if style_info:
            style_notes = []
            if style_info.get("style"):
                style_notes.append(f"Style: {style_info['style']}")
            if style_info.get("colors"):
                style_notes.append(f"Colors: {', '.join(style_info['colors'])}")
            if style_info.get("mood"):
                style_notes.append(f"Mood: {style_info['mood']}")
            
            if style_notes:
                suggestion += f"*({' | '.join(style_notes)})*\n"
        
        return suggestion
    
    def _format_video_prompt_suggestion(
        self, 
        video_prompt: Dict[str, Any], 
        content_type: str
    ) -> str:
        """Format a video prompt suggestion for content integration."""
        prompt_text = video_prompt.get("prompt", "")
        duration = video_prompt.get("duration", "30-60 seconds")
        
        suggestion = f"\n**🎬 Video Suggestion:**\n"
        suggestion += f"*{prompt_text}*\n"
        suggestion += f"*Duration: {duration}*\n"
        
        return suggestion
    
    def _create_creative_assets_section(
        self,
        image_prompts: List[Dict[str, Any]],
        video_prompts: List[Dict[str, Any]],
        content_type: str
    ) -> str:
        """Create a comprehensive creative assets section."""
        section = "\n---\n\n## 🎨 Creative Assets Suggestions\n\n"
        
        if image_prompts:
            section += "### Image Prompts\n\n"
            for i, prompt in enumerate(image_prompts, 1):
                section += f"**Image {i}:**\n"
                section += f"- **Prompt:** {prompt.get('prompt', 'N/A')}\n"
                section += f"- **Platform:** {prompt.get('platform', 'General')}\n"
                section += f"- **Style:** {prompt.get('style_guidance', {}).get('style', 'Professional')}\n\n"
        
        if video_prompts:
            section += "### Video Prompts\n\n"
            for i, prompt in enumerate(video_prompts, 1):
                section += f"**Video {i}:**\n"
                section += f"- **Concept:** {prompt.get('prompt', 'N/A')}\n"
                section += f"- **Duration:** {prompt.get('duration', '30-60 seconds')}\n"
                section += f"- **Type:** {prompt.get('video_type', 'Explainer')}\n"
                
                scenes = prompt.get('scenes', [])
                if scenes:
                    section += f"- **Scene Breakdown:**\n"
                    for j, scene in enumerate(scenes, 1):
                        section += f"  - Scene {j}: {scene.get('description', 'N/A')}\n"
                section += "\n"
        
        if not image_prompts and not video_prompts:
            section += "*No creative assets generated for this content.*\n\n"
        
        section += "---\n"
        
        return section
    
    def _update_final_blog_record(self, blog_id: str, workflow_result: Dict[str, Any]):
        """Update the blog record with final content."""
        try:
            update_data = {
                "content_markdown": workflow_result["final_content"],
                "status": "completed",
                "updated_at": datetime.datetime.utcnow().isoformat()
            }
            
            if self.db_service.use_supabase and self.db_service.supabase:
                result = self.db_service.supabase.table("blog_posts").update(update_data).eq("id", blog_id).execute()
                
                if not result or getattr(result, "status_code", 200) >= 400:
                    self.logger.warning(f"Failed to update blog record: {result}")
                else:
                    self.logger.info(f"Updated blog record {blog_id} with final content")
            else:
                # For local PostgreSQL, use direct connection
                with self.db_service.get_db_connection() as conn:
                    cur = conn.cursor()
                    cur.execute("""
                        UPDATE blog_posts 
                        SET content_markdown = %s, status = %s, "updatedAt" = %s
                        WHERE id = %s
                    """, (
                        update_data["content_markdown"],
                        update_data["status"],
                        update_data["updated_at"],
                        blog_id
                    ))
                    conn.commit()
                    self.logger.info(f"Updated blog record {blog_id} with final content")
                
        except Exception as e:
            self.logger.error(f"Failed to update blog record: {str(e)}")
            # Don't raise exception here as the main workflow succeeded
    
    def execute(
        self, 
        input_data: Dict[str, Any], 
        context: Optional[AgentExecutionContext] = None,
        **kwargs
    ) -> AgentResult:
        """Execute the workflow (implements abstract method)."""
        return self.execute_workflow(input_data, context)
    
    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get the current status of a workflow execution."""
        # This could be enhanced to track workflow state in database
        return {
            "workflow_id": workflow_id,
            "status": self.state.status.value,
            "progress": self.state.progress_percentage,
            "current_operation": self.state.current_operation,
            "last_updated": self.state.last_updated.isoformat()
        }
    
    def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel a running workflow."""
        try:
            self.state.status = AgentStatus.CANCELLED
            self.logger.info(f"Workflow {workflow_id} cancelled")
            return True
        except Exception as e:
            self.logger.error(f"Failed to cancel workflow {workflow_id}: {str(e)}")
            return False
    
    def get_workflow_metrics(self) -> Dict[str, Any]:
        """Get workflow performance metrics."""
        base_metrics = self.get_metrics()
        
        # Add workflow-specific metrics
        workflow_metrics = {
            **base_metrics,
            "agent_count": len(self.child_agents),
            "workflow_type": "blog_generation",
            "supported_content_types": ["blog", "linkedin", "article"],
            "max_revisions": 2,
            "average_workflow_time": base_metrics.get("average_execution_time_ms", 0)
        }
        
        return workflow_metrics


# Backward compatibility class
class BlogWorkflowCompatibility:
    """
    Provides backward compatibility with the legacy blog_workflow.py interface.
    """
    
    def __init__(self):
        self.workflow = BlogWorkflow()
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow with legacy interface."""
        try:
            # Convert legacy input format
            if "blog_title" in input_data:
                title = input_data["blog_title"]
            elif "title" in input_data:
                title = input_data["title"]
            else:
                raise ValueError("Missing title field")
            
            workflow_input = {
                "title": title,
                "company_context": input_data.get("company_context", ""),
                "content_type": input_data.get("content_type", "blog")
            }
            
            result = self.workflow.execute_safe(workflow_input)
            
            if result.success:
                # Return in legacy format
                return {
                    "final_post": result.data["content_markdown"],
                    "blog_id": result.data["blog_id"],
                    "success": True
                }
            else:
                return {
                    "final_post": f"Error: {result.error_message}",
                    "success": False,
                    "error": result.error_message
                }
                
        except Exception as e:
            return {
                "final_post": f"Workflow error: {str(e)}",
                "success": False,
                "error": str(e)
            }


# For backward compatibility, expose the legacy interface
def create_blog_workflow_app():
    """Create a blog workflow app with legacy interface compatibility."""
    return BlogWorkflowCompatibility()

# Export for legacy compatibility
blog_agent_app = create_blog_workflow_app()