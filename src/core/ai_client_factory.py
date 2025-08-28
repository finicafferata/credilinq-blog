"""
AI Client Factory for managing multiple AI providers (OpenAI, Gemini).
"""

import logging
from typing import Dict, Any, Optional, List, Union
from abc import ABC, abstractmethod

from ..config.settings import get_settings

logger = logging.getLogger(__name__)


class AIClientError(Exception):
    """Custom exception for AI client errors."""
    pass


class BaseAIClient(ABC):
    """Abstract base class for AI clients."""
    
    def __init__(self, api_key: str, model: str, temperature: float = 0.7, max_tokens: int = 4000):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
    
    @abstractmethod
    async def generate_text(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt."""
        pass
    
    @abstractmethod
    async def generate_text_with_system(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        """Generate text with system and user prompts."""
        pass
    
    @abstractmethod
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text."""
        pass


class OpenAIClient(BaseAIClient):
    """OpenAI client implementation."""
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo", temperature: float = 0.7, max_tokens: int = 4000):
        super().__init__(api_key, model, temperature, max_tokens)
        self._client = None
        self._embeddings_client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize OpenAI client."""
        try:
            from openai import OpenAI
            from langchain_openai import OpenAIEmbeddings
            
            self._client = OpenAI(api_key=self.api_key)
            self._embeddings_client = OpenAIEmbeddings(
                openai_api_key=self.api_key,
                model="text-embedding-ada-002"
            )
            self.logger.info(f"OpenAI client initialized with model: {self.model}")
            
        except ImportError as e:
            self.logger.error(f"OpenAI library not installed: {e}")
            raise AIClientError("OpenAI library not available")
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI client: {e}")
            raise AIClientError(f"OpenAI client initialization failed: {e}")
    
    async def generate_text(self, prompt: str, **kwargs) -> str:
        """Generate text using OpenAI."""
        try:
            temperature = kwargs.get('temperature', self.temperature)
            max_tokens = kwargs.get('max_tokens', self.max_tokens)
            
            self.logger.debug(f"🔵 OpenAI generating text with model: {self.model}")
            
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            result = response.choices[0].message.content.strip()
            self.logger.debug(f"✅ OpenAI generation complete - {len(result)} characters")
            return result
            
        except Exception as e:
            self.logger.error(f"OpenAI text generation failed: {e}")
            raise AIClientError(f"OpenAI generation failed: {e}")
    
    async def generate_text_with_system(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        """Generate text with system and user prompts."""
        try:
            temperature = kwargs.get('temperature', self.temperature)
            max_tokens = kwargs.get('max_tokens', self.max_tokens)
            
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            self.logger.error(f"OpenAI text generation with system prompt failed: {e}")
            raise AIClientError(f"OpenAI generation failed: {e}")
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding using OpenAI."""
        try:
            return self._embeddings_client.embed_query(text)
        except Exception as e:
            self.logger.error(f"OpenAI embedding generation failed: {e}")
            raise AIClientError(f"OpenAI embedding failed: {e}")


class GeminiClient(BaseAIClient):
    """Google Gemini client implementation."""
    
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash", temperature: float = 0.7, max_tokens: int = 4000):
        super().__init__(api_key, model, temperature, max_tokens)
        self._client = None
        self._embeddings_client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Gemini client."""
        try:
            import google.generativeai as genai
            
            genai.configure(api_key=self.api_key)
            self._client = genai.GenerativeModel(self.model)
            
            # For embeddings, we'll use a specific embedding model
            self._embeddings_client = genai.GenerativeModel('models/embedding-001')
            
            self.logger.info(f"Gemini client initialized with model: {self.model}")
            
        except ImportError as e:
            self.logger.error(f"Google Generative AI library not installed: {e}")
            raise AIClientError("Google Generative AI library not available")
        except Exception as e:
            self.logger.error(f"Failed to initialize Gemini client: {e}")
            raise AIClientError(f"Gemini client initialization failed: {e}")
    
    async def generate_text(self, prompt: str, **kwargs) -> str:
        """Generate text using Gemini."""
        try:
            # Configure generation parameters
            generation_config = {
                "temperature": kwargs.get('temperature', self.temperature),
                "max_output_tokens": kwargs.get('max_tokens', self.max_tokens),
            }
            
            self.logger.debug(f"🟢 Gemini generating text with model: {self.model}")
            
            response = self._client.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            result = response.text.strip() if response.text else ""
            self.logger.debug(f"✅ Gemini generation complete - {len(result)} characters")
            return result
            
        except Exception as e:
            self.logger.error(f"Gemini text generation failed: {e}")
            raise AIClientError(f"Gemini generation failed: {e}")
    
    async def generate_text_with_system(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        """Generate text with system and user prompts."""
        try:
            # Gemini handles system prompts differently - combine them
            combined_prompt = f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"
            
            generation_config = {
                "temperature": kwargs.get('temperature', self.temperature),
                "max_output_tokens": kwargs.get('max_tokens', self.max_tokens),
            }
            
            response = self._client.generate_content(
                combined_prompt,
                generation_config=generation_config
            )
            
            return response.text.strip() if response.text else ""
            
        except Exception as e:
            self.logger.error(f"Gemini text generation with system prompt failed: {e}")
            raise AIClientError(f"Gemini generation failed: {e}")
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding using Gemini."""
        try:
            import google.generativeai as genai
            
            # Use the embedding API
            response = genai.embed_content(
                model="models/embedding-001",
                content=text,
                task_type="retrieval_document"
            )
            
            return response['embedding']
            
        except Exception as e:
            self.logger.error(f"Gemini embedding generation failed: {e}")
            # Fallback: create a dummy embedding for development
            self.logger.warning("Using dummy embedding - install google-generativeai for real embeddings")
            import hashlib
            hash_obj = hashlib.md5(text.encode())
            # Create a simple 1536-dimensional vector (same as OpenAI)
            dummy_embedding = [float(int(hash_obj.hexdigest()[i:i+2], 16)) / 255.0 
                              for i in range(0, min(len(hash_obj.hexdigest()), 32), 2)]
            # Pad to 1536 dimensions
            while len(dummy_embedding) < 1536:
                dummy_embedding.extend(dummy_embedding)
            return dummy_embedding[:1536]


class AIClientFactory:
    """Factory for creating AI clients based on configuration."""
    
    _clients: Dict[str, BaseAIClient] = {}
    
    @classmethod
    def get_client(cls, provider: Optional[str] = None) -> BaseAIClient:
        """Get AI client for the specified provider."""
        settings = get_settings()
        
        # Use provided provider or default to primary
        provider = provider or settings.primary_ai_provider
        
        # Return cached client if available
        if provider in cls._clients:
            client = cls._clients[provider]
            logger.info(f"🤖 Using {provider.upper()} AI - Model: {client.model}")
            return client
        
        # Create new client
        client = cls._create_client(provider, settings)
        cls._clients[provider] = client
        logger.info(f"🚀 Initialized {provider.upper()} AI - Model: {client.model}")
        return client
    
    @classmethod
    def _create_client(cls, provider: str, settings) -> BaseAIClient:
        """Create a new AI client."""
        try:
            if provider == "openai":
                if not settings.openai_api_key:
                    raise AIClientError("OpenAI API key not configured")
                
                return OpenAIClient(
                    api_key=settings.openai_api_key,
                    model=settings.openai_model,
                    temperature=settings.openai_temperature,
                    max_tokens=settings.openai_max_tokens
                )
                
            elif provider == "gemini":
                api_key = settings.gemini_api_key or settings.google_api_key
                if not api_key:
                    raise AIClientError("Gemini/Google API key not configured")
                
                return GeminiClient(
                    api_key=api_key,
                    model=settings.gemini_model,
                    temperature=settings.gemini_temperature,
                    max_tokens=settings.gemini_max_tokens
                )
                
            else:
                raise AIClientError(f"Unsupported AI provider: {provider}")
                
        except Exception as e:
            logger.error(f"Failed to create {provider} client: {e}")
            raise
    
    @classmethod
    def get_primary_client(cls) -> BaseAIClient:
        """Get the primary AI client."""
        return cls.get_client()
    
    @classmethod
    def clear_cache(cls):
        """Clear the client cache."""
        cls._clients.clear()
    
    @classmethod
    def get_available_providers(cls) -> List[str]:
        """Get list of available AI providers."""
        settings = get_settings()
        providers = []
        
        if settings.openai_api_key:
            providers.append("openai")
        
        if settings.gemini_api_key or settings.google_api_key:
            providers.append("gemini")
        
        return providers


# Convenience functions for backward compatibility
def get_ai_client(provider: Optional[str] = None) -> BaseAIClient:
    """Get AI client (convenience function)."""
    return AIClientFactory.get_client(provider)


def get_primary_ai_client() -> BaseAIClient:
    """Get the primary AI client (convenience function)."""
    return AIClientFactory.get_primary_client()


async def generate_ai_content(prompt: str, provider: Optional[str] = None, **kwargs) -> str:
    """Generate AI content using the specified provider."""
    client = get_ai_client(provider)
    return await client.generate_text(prompt, **kwargs)


async def generate_ai_content_with_system(system_prompt: str, user_prompt: str, 
                                        provider: Optional[str] = None, **kwargs) -> str:
    """Generate AI content with system prompt using the specified provider."""
    client = get_ai_client(provider)
    return await client.generate_text_with_system(system_prompt, user_prompt, **kwargs)