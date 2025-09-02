"""
Unified LLM client module for LangGraph agents.
Uses native OpenAI client instead of langchain_openai.
"""

import os
import openai
from typing import List, Dict, Any, Optional
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
import logging

logger = logging.getLogger(__name__)


class LLMClient:
    """Wrapper for OpenAI client to provide consistent interface across agents."""
    
    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        api_key: Optional[str] = None
    ):
        """Initialize the LLM client with OpenAI."""
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens or 2000
        
        # Get API key from parameter or environment
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        self.client = openai.OpenAI(api_key=api_key)
    
    def invoke(self, messages: List[BaseMessage]) -> AIMessage:
        """
        Invoke the LLM with a list of messages.
        Compatible with langchain message format.
        """
        # Convert langchain messages to OpenAI format
        openai_messages = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                openai_messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                openai_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                openai_messages.append({"role": "assistant", "content": msg.content})
            else:
                # Default to user message
                openai_messages.append({"role": "user", "content": str(msg.content)})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=openai_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            content = response.choices[0].message.content
            return AIMessage(content=content)
            
        except Exception as e:
            logger.error(f"Error invoking OpenAI: {str(e)}")
            raise
    
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
            response = self.client.chat.completions.create(
                model=kwargs.get('model', self.model),
                messages=messages,
                temperature=kwargs.get('temperature', self.temperature),
                max_tokens=kwargs.get('max_tokens', self.max_tokens)
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error in chat completion: {str(e)}")
            raise


class EmbeddingsClient:
    """Wrapper for OpenAI embeddings to replace OpenAIEmbeddings."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize embeddings client."""
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found.")
        
        self.client = openai.OpenAI(api_key=api_key)
        self.model = "text-embedding-ada-002"
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents."""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=texts
            )
            return [embedding.embedding for embedding in response.data]
        except Exception as e:
            logger.error(f"Error creating embeddings: {str(e)}")
            raise
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=[text]
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error creating query embedding: {str(e)}")
            raise


# Factory functions for backward compatibility
def create_llm(model: str = "gpt-3.5-turbo", temperature: float = 0.7, **kwargs) -> LLMClient:
    """Create an LLM client instance."""
    return LLMClient(model=model, temperature=temperature, **kwargs)


def create_embeddings(**kwargs) -> EmbeddingsClient:
    """Create an embeddings client instance."""
    return EmbeddingsClient(**kwargs)