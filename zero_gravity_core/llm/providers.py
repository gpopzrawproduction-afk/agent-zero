"""
LLM Provider Abstraction Layer for ZeroGravity

This module provides a unified interface for different LLM providers
(OpenAI, Anthropic, etc.) to enable provider switching without
changing agent logic.
"""
import os
import time
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Generator
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class LLMProviderType(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    CUSTOM = "custom"


class LLMResponse:
    """Standardized response object for LLM calls"""
    def __init__(self, content: str, model: str, usage: Dict[str, int] = None, 
                 finish_reason: str = None):
        self.content = content
        self.model = model
        self.usage = usage or {}
        self.finish_reason = finish_reason

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "model": self.model,
            "usage": self.usage,
            "finish_reason": self.finish_reason
        }


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    def __init__(self, api_key: str = None, base_url: str = None, **kwargs):
        self.api_key = api_key or os.getenv(self.get_api_key_env_var())
        self.base_url = base_url
        self.config = kwargs
        
    @abstractmethod
    def get_api_key_env_var(self) -> str:
        """Return the environment variable name for the API key"""
        pass
    
    @abstractmethod
    def call(self, messages: List[Dict[str, str]], model: str, **kwargs) -> LLMResponse:
        """Make a synchronous call to the LLM"""
        pass
    
    @abstractmethod
    def stream(self, messages: List[Dict[str, str]], model: str, **kwargs) -> Generator[str, None, None]:
        """Stream responses from the LLM"""
        pass


class OpenAILLMProvider(BaseLLMProvider):
    """OpenAI implementation of LLM provider"""
    
    def __init__(self, api_key: str = None, base_url: str = None, **kwargs):
        super().__init__(api_key, base_url, **kwargs)
        try:
            import openai
            self.openai = openai
            if base_url:
                self.openai.base_url = base_url
            if self.api_key:
                self.openai.api_key = self.api_key
        except ImportError:
            raise ImportError("Please install openai: pip install openai")
    
    def get_api_key_env_var(self) -> str:
        return "OPENAI_API_KEY"
    
    def call(self, messages: List[Dict[str, str]], model: str = "gpt-4", **kwargs) -> LLMResponse:
        """Make a synchronous call to OpenAI API"""
        if not self.api_key:
            raise ValueError(f"OpenAI API key not found. Set {self.get_api_key_env_var()}")
        
        try:
            # Prepare the call with default parameters
            params = {
                "model": model,
                "messages": messages,
                "temperature": kwargs.get("temperature", 0.7),
                "max_tokens": kwargs.get("max_tokens", 1000),
            }
            
            # Add any additional parameters
            for key, value in kwargs.items():
                if key not in params:
                    params[key] = value
            
            response = self.openai.chat.completions.create(**params)
            
            return LLMResponse(
                content=response.choices[0].message.content,
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                finish_reason=response.choices[0].finish_reason
            )
        except Exception as e:
            logger.error(f"OpenAI API call failed: {str(e)}")
            raise
    
    def stream(self, messages: List[Dict[str, str]], model: str = "gpt-4", **kwargs) -> Generator[str, None, None]:
        """Stream responses from OpenAI API"""
        if not self.api_key:
            raise ValueError(f"OpenAI API key not found. Set {self.get_api_key_env_var()}")
        
        try:
            # Prepare the call with default parameters
            params = {
                "model": model,
                "messages": messages,
                "temperature": kwargs.get("temperature", 0.7),
                "max_tokens": kwargs.get("max_tokens", 1000),
                "stream": True
            }
            
            # Add any additional parameters
            for key, value in kwargs.items():
                if key not in params:
                    params[key] = value
            
            response = self.openai.chat.completions.create(**params)
            
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"OpenAI API stream failed: {str(e)}")
            raise


class AnthropicLLMProvider(BaseLLMProvider):
    """Anthropic implementation of LLM provider"""
    
    def __init__(self, api_key: str = None, **kwargs):
        super().__init__(api_key, **kwargs)
        try:
            import anthropic
            self.anthropic = anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError("Please install anthropic: pip install anthropic")
    
    def get_api_key_env_var(self) -> str:
        return "ANTHROPIC_API_KEY"
    
    def call(self, messages: List[Dict[str, str]], model: str = "claude-3-opus-20240229", **kwargs) -> LLMResponse:
        """Make a synchronous call to Anthropic API"""
        if not self.api_key:
            raise ValueError(f"Anthropic API key not found. Set {self.get_api_key_env_var()}")
        
        try:
            # Convert messages to Anthropic format
            # Anthropic requires a different message format
            system_message = None
            user_messages = []
            
            for msg in messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                else:
                    user_messages.append(msg)
            
            # Prepare the call with default parameters
            params = {
                "model": model,
                "messages": user_messages,
                "max_tokens": kwargs.get("max_tokens", 1000),
            }
            
            if system_message:
                params["system"] = system_message
            
            # Add any additional parameters
            for key, value in kwargs.items():
                if key not in params:
                    params[key] = value
            
            response = self.client.messages.create(**params)
            
            return LLMResponse(
                content=response.content[0].text if response.content else "",
                model=response.model,
                usage={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens
                },
                finish_reason=response.stop_reason
            )
        except Exception as e:
            logger.error(f"Anthropic API call failed: {str(e)}")
            raise
    
    def stream(self, messages: List[Dict[str, str]], model: str = "claude-3-opus-20240229", **kwargs) -> Generator[str, None, None]:
        """Stream responses from Anthropic API"""
        if not self.api_key:
            raise ValueError(f"Anthropic API key not found. Set {self.get_api_key_env_var()}")
        
        try:
            # Convert messages to Anthropic format
            system_message = None
            user_messages = []
            
            for msg in messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                else:
                    user_messages.append(msg)
            
            # Prepare the call with default parameters
            params = {
                "model": model,
                "messages": user_messages,
                "max_tokens": kwargs.get("max_tokens", 1000),
            }
            
            if system_message:
                params["system"] = system_message
            
            # Add any additional parameters
            for key, value in kwargs.items():
                if key not in params:
                    params[key] = value
            
            with self.client.messages.stream(**params) as stream:
                for text in stream.text_stream:
                    yield text
        except Exception as e:
            logger.error(f"Anthropic API stream failed: {str(e)}")
            raise


class LLMManager:
    """Manages multiple LLM providers and provides a unified interface"""
    
    def __init__(self):
        self.providers: Dict[LLMProviderType, BaseLLMProvider] = {}
        self.default_provider: Optional[LLMProviderType] = None
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize available LLM providers"""
        # Initialize OpenAI provider if API key is available
        if os.getenv("OPENAI_API_KEY"):
            try:
                self.providers[LLMProviderType.OPENAI] = OpenAILLMProvider()
                if self.default_provider is None:
                    self.default_provider = LLMProviderType.OPENAI
            except ImportError:
                logger.warning("OpenAI library not installed. Skipping OpenAI provider.")
        
        # Initialize Anthropic provider if API key is available
        if os.getenv("ANTHROPIC_API_KEY"):
            try:
                self.providers[LLMProviderType.ANTHROPIC] = AnthropicLLMProvider()
                if self.default_provider is None:
                    self.default_provider = LLMProviderType.ANTHROPIC
            except ImportError:
                logger.warning("Anthropic library not installed. Skipping Anthropic provider.")
        
        # Set a default if none were initialized
        if self.default_provider is None and self.providers:
            self.default_provider = next(iter(self.providers))
    
    def add_provider(self, provider_type: LLMProviderType, provider: BaseLLMProvider):
        """Add a custom provider to the manager"""
        self.providers[provider_type] = provider
        if self.default_provider is None:
            self.default_provider = provider_type
    
    def set_default_provider(self, provider_type: LLMProviderType):
        """Set the default provider to use"""
        if provider_type not in self.providers:
            raise ValueError(f"Provider {provider_type} not registered")
        self.default_provider = provider_type
    
    def call(self, messages: List[Dict[str, str]], model: str = None, 
             provider_type: LLMProviderType = None, **kwargs) -> LLMResponse:
        """Make a call using the specified or default provider"""
        if provider_type is None:
            provider_type = self.default_provider
        
        if provider_type not in self.providers:
            raise ValueError(f"Provider {provider_type} not available")
        
        provider = self.providers[provider_type]
        
        # Use a default model if none specified
        if model is None:
            if provider_type == LLMProviderType.OPENAI:
                model = "gpt-4-turbo-preview"
            elif provider_type == LLMProviderType.ANTHROPIC:
                model = "claude-3-sonnet-20240229"
        
        return provider.call(messages, model, **kwargs)
    
    def stream(self, messages: List[Dict[str, str]], model: str = None,
               provider_type: LLMProviderType = None, **kwargs) -> Generator[str, None, None]:
        """Stream responses using the specified or default provider"""
        if provider_type is None:
            provider_type = self.default_provider
        
        if provider_type not in self.providers:
            raise ValueError(f"Provider {provider_type} not available")
        
        provider = self.providers[provider_type]
        
        # Use a default model if none specified
        if model is None:
            if provider_type == LLMProviderType.OPENAI:
                model = "gpt-4-turbo-preview"
            elif provider_type == LLMProviderType.ANTHROPIC:
                model = "claude-3-sonnet-20240229"
        
        return provider.stream(messages, model, **kwargs)
    
    def get_available_providers(self) -> List[LLMProviderType]:
        """Get list of available providers"""
        return list(self.providers.keys())


# Global LLM manager instance
llm_manager = LLMManager()
