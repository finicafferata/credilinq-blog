"""
AI utilities for easy integration with multiple AI providers.
Provides backward-compatible methods for existing agents.
"""

import logging
from typing import Dict, Any, Optional, List

from .ai_client_factory import get_ai_client, generate_ai_content, generate_ai_content_with_system

logger = logging.getLogger(__name__)


async def call_ai_model(prompt: str, 
                       temperature: Optional[float] = None,
                       max_tokens: Optional[int] = None,
                       provider: Optional[str] = None) -> str:
    """
    Backward-compatible AI model calling function.
    Used by legacy agents that call _call_ai_model().
    """
    try:
        # Import here to avoid circular import
        from ..config.settings import get_settings
        
        # Get the client to show provider info
        client = get_ai_client(provider)
        actual_provider = provider or get_settings().primary_ai_provider
        logger.info(f"🤖 AI Request - Provider: {actual_provider.upper()}, Model: {client.model}")
        
        kwargs = {}
        if temperature is not None:
            kwargs['temperature'] = temperature
        if max_tokens is not None:
            kwargs['max_tokens'] = max_tokens
        
        return await generate_ai_content(prompt, provider, **kwargs)
        
    except Exception as e:
        logger.error(f"AI model call failed: {e}")
        raise


async def call_ai_model_with_system(system_prompt: str,
                                   user_prompt: str,
                                   temperature: Optional[float] = None,
                                   max_tokens: Optional[int] = None,
                                   provider: Optional[str] = None) -> str:
    """
    AI model calling function with system prompt.
    """
    try:
        # Import here to avoid circular import
        from ..config.settings import get_settings
        
        # Get the client to show provider info
        client = get_ai_client(provider)
        actual_provider = provider or get_settings().primary_ai_provider
        logger.info(f"🤖 AI Request (System) - Provider: {actual_provider.upper()}, Model: {client.model}")
        
        kwargs = {}
        if temperature is not None:
            kwargs['temperature'] = temperature
        if max_tokens is not None:
            kwargs['max_tokens'] = max_tokens
        
        return await generate_ai_content_with_system(system_prompt, user_prompt, provider, **kwargs)
        
    except Exception as e:
        logger.error(f"AI model call with system prompt failed: {e}")
        raise


def get_langchain_llm(provider: Optional[str] = None, **kwargs):
    """
    Get a LangChain-compatible LLM instance.
    """
    from ..config.settings import get_settings
    settings = get_settings()
    
    # Use provided provider or default to primary
    provider = provider or settings.primary_ai_provider
    
    try:
        if provider == "openai":
            from langchain_openai import ChatOpenAI
            
            return ChatOpenAI(
                openai_api_key=settings.openai_api_key,
                model_name=kwargs.get('model', settings.openai_model),
                temperature=kwargs.get('temperature', settings.openai_temperature),
                max_tokens=kwargs.get('max_tokens', settings.openai_max_tokens),
                **{k: v for k, v in kwargs.items() if k not in ['model', 'temperature', 'max_tokens']}
            )
            
        elif provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            
            api_key = settings.gemini_api_key or settings.google_api_key
            return ChatGoogleGenerativeAI(
                google_api_key=api_key,
                model=kwargs.get('model', settings.gemini_model),
                temperature=kwargs.get('temperature', settings.gemini_temperature),
                max_output_tokens=kwargs.get('max_tokens', settings.gemini_max_tokens),
                **{k: v for k, v in kwargs.items() if k not in ['model', 'temperature', 'max_tokens']}
            )
            
        else:
            raise ValueError(f"Unsupported provider: {provider}")
            
    except ImportError as e:
        logger.error(f"Required library not installed for {provider}: {e}")
        # Fallback to OpenAI if available
        if provider != "openai" and settings.openai_api_key:
            logger.warning(f"Falling back to OpenAI due to missing {provider} dependencies")
            return get_langchain_llm("openai", **kwargs)
        raise
    except Exception as e:
        logger.error(f"Failed to create {provider} LLM: {e}")
        raise


def get_embeddings_model(provider: Optional[str] = None):
    """
    Get an embeddings model for the specified provider.
    """
    from ..config.settings import get_settings
    settings = get_settings()
    
    # Use provided provider or default to primary
    provider = provider or settings.primary_ai_provider
    
    try:
        if provider == "openai":
            from langchain_openai import OpenAIEmbeddings
            
            return OpenAIEmbeddings(
                openai_api_key=settings.openai_api_key,
                model="text-embedding-ada-002"
            )
            
        elif provider == "gemini":
            # For Gemini, we'll use our custom client
            client = get_ai_client(provider)
            
            class GeminiEmbeddings:
                """Wrapper for Gemini embeddings to match LangChain interface."""
                
                def __init__(self, client):
                    self.client = client
                
                def embed_query(self, text: str) -> List[float]:
                    return self.client.get_embedding(text)
                
                def embed_documents(self, texts: List[str]) -> List[List[float]]:
                    return [self.client.get_embedding(text) for text in texts]
            
            return GeminiEmbeddings(client)
            
        else:
            raise ValueError(f"Unsupported provider: {provider}")
            
    except ImportError as e:
        logger.error(f"Required library not installed for {provider}: {e}")
        # Fallback to OpenAI if available
        if provider != "openai" and settings.openai_api_key:
            logger.warning(f"Falling back to OpenAI embeddings due to missing {provider} dependencies")
            return get_embeddings_model("openai")
        raise
    except Exception as e:
        logger.error(f"Failed to create {provider} embeddings model: {e}")
        raise


# Backward compatibility functions for legacy agents
async def _call_ai_model(prompt: str, **kwargs) -> str:
    """Backward compatibility function for legacy agents."""
    return await call_ai_model(prompt, **kwargs)


class AIModelMixin:
    """
    Mixin class to add AI capabilities to agents.
    Provides backward-compatible methods for existing agents.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ai_provider = kwargs.get('ai_provider')
    
    async def _call_ai_model(self, prompt: str, **kwargs) -> str:
        """Call AI model with the configured provider."""
        return await call_ai_model(prompt, provider=self._ai_provider, **kwargs)
    
    async def _call_ai_model_with_system(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        """Call AI model with system prompt."""
        return await call_ai_model_with_system(system_prompt, user_prompt, provider=self._ai_provider, **kwargs)
    
    def get_llm(self, **kwargs):
        """Get LangChain LLM for this agent."""
        return get_langchain_llm(self._ai_provider, **kwargs)
    
    def get_embeddings(self):
        """Get embeddings model for this agent."""
        return get_embeddings_model(self._ai_provider)


# Provider status checking utilities
def check_provider_availability() -> Dict[str, bool]:
    """Check which AI providers are available."""
    from ..config.settings import get_settings
    settings = get_settings()
    
    availability = {}
    
    # Check OpenAI
    try:
        import openai
        availability["openai"] = bool(settings.openai_api_key)
    except ImportError:
        availability["openai"] = False
    
    # Check Gemini
    try:
        import google.generativeai
        api_key = settings.gemini_api_key or settings.google_api_key
        availability["gemini"] = bool(api_key)
    except ImportError:
        availability["gemini"] = False
    
    return availability


def get_recommended_provider() -> str:
    """Get the recommended AI provider based on availability."""
    from ..config.settings import get_settings
    settings = get_settings()
    
    availability = check_provider_availability()
    
    # First check if primary provider is available
    if availability.get(settings.primary_ai_provider, False):
        return settings.primary_ai_provider
    
    # Fallback to any available provider
    for provider, available in availability.items():
        if available:
            return provider
    
    # No providers available
    raise RuntimeError("No AI providers are configured and available")


# Environment setup utilities
def setup_ai_environment():
    """Setup AI environment with appropriate providers."""
    logger.info("Setting up AI environment...")
    
    availability = check_provider_availability()
    logger.info(f"AI provider availability: {availability}")
    
    if not any(availability.values()):
        logger.error("No AI providers are available! Please configure API keys.")
        raise RuntimeError("No AI providers configured")
    
    recommended = get_recommended_provider()
    logger.info(f"Using AI provider: {recommended}")
    
    return recommended