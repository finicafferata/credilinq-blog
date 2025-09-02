"""
Unified LLM client module for LangGraph agents.
Uses Google Gemini instead of OpenAI for all LLM operations.
"""

import os
import google.generativeai as genai
from typing import List, Dict, Any, Optional
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
import logging

logger = logging.getLogger(__name__)


class LLMClient:
    """Wrapper for Google Gemini to provide consistent interface across agents."""
    
    def __init__(
        self,
        model: str = "gemini-1.5-flash",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        api_key: Optional[str] = None
    ):
        """Initialize the LLM client with Google Gemini."""
        # Map common OpenAI model names to Gemini equivalents
        model_mapping = {
            "gpt-3.5-turbo": "gemini-1.5-flash",
            "gpt-4": "gemini-1.5-pro",
            "gpt-4-turbo": "gemini-1.5-pro",
        }
        
        self.model_name = model_mapping.get(model, model)
        self.temperature = temperature
        self.max_tokens = max_tokens or 2000
        
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
    
    def invoke(self, messages: List[BaseMessage]) -> AIMessage:
        """
        Invoke the LLM with a list of messages.
        Compatible with langchain message format.
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
        
        try:
            # Generate response
            response = self.model.generate_content(full_prompt)
            
            # Extract text from response
            if response.text:
                content = response.text
            else:
                content = "I couldn't generate a response. Please try again."
            
            return AIMessage(content=content)
            
        except Exception as e:
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
def create_llm(model: str = "gemini-1.5-flash", temperature: float = 0.7, **kwargs) -> LLMClient:
    """Create an LLM client instance with Gemini."""
    # Map api_key parameter for compatibility
    if 'api_key' in kwargs and 'api_key' not in kwargs:
        kwargs['api_key'] = kwargs.pop('api_key')
    
    return LLMClient(model=model, temperature=temperature, **kwargs)


def create_embeddings(**kwargs) -> EmbeddingsClient:
    """Create an embeddings client instance with Gemini."""
    # Map api_key parameter for compatibility
    if 'api_key' in kwargs:
        kwargs['api_key'] = kwargs.get('api_key')
    
    return EmbeddingsClient(**kwargs)