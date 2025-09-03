"""
Unified LLM client module for LangGraph agents.
Uses Google Gemini instead of OpenAI for all LLM operations.
Enhanced with automatic performance tracking and usage analytics.
"""

import os
import google.generativeai as genai
from typing import List, Dict, Any, Optional
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
import logging

# Import performance tracking
from .gemini_performance_tracker import global_gemini_tracker, GeminiUsageMetrics

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Wrapper for Google Gemini to provide consistent interface across agents.
    Automatically tracks performance, cost, and token usage.
    """
    
    def __init__(
        self,
        model: str = "gemini-1.5-flash",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        api_key: Optional[str] = None,
        enable_tracking: bool = True,
        agent_name: Optional[str] = None,
        agent_type: Optional[str] = None
    ):
        """Initialize the LLM client with Google Gemini and performance tracking."""
        # Map common OpenAI model names to Gemini equivalents
        model_mapping = {
            "gpt-3.5-turbo": "gemini-1.5-flash",
            "gpt-4": "gemini-1.5-pro",
            "gpt-4-turbo": "gemini-1.5-pro",
        }
        
        self.model_name = model_mapping.get(model, model)
        self.temperature = temperature
        self.max_tokens = max_tokens or 2000
        self.enable_tracking = enable_tracking
        self.agent_name = agent_name or "unknown_agent"
        self.agent_type = agent_type or "llm_client"
        
        # Performance tracking state
        self.current_execution_id = None
        self.campaign_id = None
        self.blog_post_id = None
        self.workflow_id = None
        
        # Get API key from parameter or environment
        api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Gemini API key not found. Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable.")
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        
        # Initialize the model with generation config
        self.generation_config = genai.GenerationConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
        )
        
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=self.generation_config
        )
    
    def set_tracking_context(
        self,
        campaign_id: Optional[str] = None,
        blog_post_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        agent_type: Optional[str] = None
    ):
        """Set tracking context for performance monitoring."""
        if campaign_id:
            self.campaign_id = campaign_id
        if blog_post_id:
            self.blog_post_id = blog_post_id
        if workflow_id:
            self.workflow_id = workflow_id
        if agent_name:
            self.agent_name = agent_name
        if agent_type:
            self.agent_type = agent_type
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current usage statistics for this client."""
        return {
            "model_name": self.model_name,
            "agent_name": self.agent_name,
            "agent_type": self.agent_type,
            "campaign_id": self.campaign_id,
            "blog_post_id": self.blog_post_id,
            "workflow_id": self.workflow_id,
            "tracking_enabled": self.enable_tracking
        }
    
    def invoke(self, messages: List[BaseMessage]) -> AIMessage:
        """
        Invoke the LLM with a list of messages.
        Compatible with langchain message format with automatic performance tracking.
        """
        # Convert langchain messages to Gemini format
        # Gemini expects a single prompt or a conversation history
        prompt_parts = []
        
        for msg in messages:
            if isinstance(msg, SystemMessage):
                # Add system message as context
                prompt_parts.append(f"System: {msg.content}\n")
            elif isinstance(msg, HumanMessage):
                prompt_parts.append(f"User: {msg.content}\n")
            elif isinstance(msg, AIMessage):
                prompt_parts.append(f"Assistant: {msg.content}\n")
            else:
                prompt_parts.append(f"{msg.content}\n")
        
        # Combine all parts into a single prompt
        full_prompt = "".join(prompt_parts)
        
        # Add instruction for response
        full_prompt += "Assistant: "
        
        execution_id = None
        
        try:
            # Start performance tracking if enabled
            if self.enable_tracking:
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # Create task for non-blocking tracking
                        task = asyncio.create_task(
                            global_gemini_tracker.track_gemini_execution_start(
                                agent_name=self.agent_name,
                                agent_type=self.agent_type,
                                model_name=self.model_name,
                                prompt_text=full_prompt,
                                campaign_id=self.campaign_id,
                                blog_post_id=self.blog_post_id,
                                workflow_id=self.workflow_id
                            )
                        )
                        # Don't await - let it run async
                        execution_id = "async_tracked"
                except Exception as tracking_error:
                    logger.debug(f"Performance tracking start failed: {tracking_error}")
            
            # Generate response
            response = self.model.generate_content(full_prompt)
            
            # Complete performance tracking if enabled
            if self.enable_tracking and execution_id:
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # Create task for non-blocking completion tracking
                        asyncio.create_task(
                            global_gemini_tracker.track_gemini_execution_end(
                                execution_id, response, "success"
                            )
                        )
                except Exception as tracking_error:
                    logger.debug(f"Performance tracking end failed: {tracking_error}")
            
            # Extract text from response
            if response.text:
                content = response.text
            else:
                content = "I couldn't generate a response. Please try again."
            
            return AIMessage(content=content)
            
        except Exception as e:
            # Complete performance tracking with error if enabled
            if self.enable_tracking and execution_id:
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.create_task(
                            global_gemini_tracker.track_gemini_execution_end(
                                execution_id, None, "failed", str(e)
                            )
                        )
                except Exception as tracking_error:
                    logger.debug(f"Error tracking failed: {tracking_error}")
            
            logger.error(f"Error invoking Gemini: {str(e)}")
            # Fallback response
            return AIMessage(content=f"Error generating response: {str(e)}")
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Simple generation with optional system prompt.
        Returns string directly.
        """
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))
        
        response = self.invoke(messages)
        return response.content.strip()
    
    def chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        **kwargs
    ) -> str:
        """
        Direct chat completion compatible with OpenAI format.
        """
        try:
            # Convert dict messages to prompt
            prompt_parts = []
            for msg in messages:
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                
                if role == 'system':
                    prompt_parts.append(f"System: {content}\n")
                elif role == 'user':
                    prompt_parts.append(f"User: {content}\n")
                elif role == 'assistant':
                    prompt_parts.append(f"Assistant: {content}\n")
            
            full_prompt = "".join(prompt_parts) + "Assistant: "
            
            # Update generation config if needed
            temp = kwargs.get('temperature', self.temperature)
            max_tokens = kwargs.get('max_tokens', self.max_tokens)
            
            # Create temporary model with custom config
            gen_config = genai.GenerationConfig(
                temperature=temp,
                max_output_tokens=max_tokens,
            )
            
            model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=gen_config
            )
            
            response = model.generate_content(full_prompt)
            return response.text if response.text else ""
            
        except Exception as e:
            logger.error(f"Error in chat completion: {str(e)}")
            raise


class EmbeddingsClient:
    """Wrapper for Google embeddings."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize embeddings client with Gemini."""
        api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Gemini API key not found.")
        
        genai.configure(api_key=api_key)
        
        # Use the embedding model
        self.model_name = "models/text-embedding-004"
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents."""
        try:
            embeddings = []
            for text in texts:
                # Generate embedding for each text
                result = genai.embed_content(
                    model=self.model_name,
                    content=text,
                    task_type="retrieval_document"
                )
                embeddings.append(result['embedding'])
            return embeddings
        except Exception as e:
            logger.error(f"Error creating embeddings: {str(e)}")
            # Return dummy embeddings as fallback
            return [[0.0] * 768 for _ in texts]
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        try:
            result = genai.embed_content(
                model=self.model_name,
                content=text,
                task_type="retrieval_query"
            )
            return result['embedding']
        except Exception as e:
            logger.error(f"Error creating query embedding: {str(e)}")
            # Return dummy embedding as fallback
            return [0.0] * 768


# Factory functions for backward compatibility
def create_llm(
    model: str = "gemini-1.5-flash", 
    temperature: float = 0.7,
    agent_name: Optional[str] = None,
    agent_type: Optional[str] = None,
    enable_tracking: bool = True,
    **kwargs
) -> LLMClient:
    """Create an LLM client instance with Gemini and performance tracking."""
    return LLMClient(
        model=model, 
        temperature=temperature,
        agent_name=agent_name,
        agent_type=agent_type,
        enable_tracking=enable_tracking,
        **kwargs
    )


def create_embeddings(**kwargs) -> EmbeddingsClient:
    """Create an embeddings client instance with Gemini."""
    # Map api_key parameter for compatibility
    if 'api_key' in kwargs:
        kwargs['api_key'] = kwargs.get('api_key')
    
    return EmbeddingsClient(**kwargs)