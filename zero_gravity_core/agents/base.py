# zero_gravity_core/agents/base.py

import json
import asyncio
import functools
from typing import Any, Dict, Optional, List, Generator
from ..llm.providers import llm_manager, LLMProviderType, LLMResponse
from ..llm.cache import cache_manager


class BaseAgent:
    """
    Base class for all ZeroGravity agents.
    Handles common functions like memory and recording.
    """

    def __init__(self, base_dir: str = None, role: str = "base", system_prompt: str = None, coordinator: Any = None):
        self.base_dir = base_dir
        self.role = role
        self.system_prompt = system_prompt
        self.coordinator = coordinator
        self.memory = []
        self.llm_manager = llm_manager
        self.cache_manager = cache_manager

    def record(self, data: Dict[str, Any]):
        """Save something to agent memory"""
        self.memory.append(data)

    def get_memory(self):
        """Retrieve memory entries"""
        return self.memory

    def get_system_prompt(self) -> Optional[str]:
        """Retrieve the system prompt for this agent"""
        return self.system_prompt

    def prepare_messages(self, input_data: Any) -> List[Dict[str, str]]:
        """Prepare messages for LLM call"""
        system_prompt = self.get_system_prompt()
        if not system_prompt:
            system_prompt = f"You are a {self.role} agent in the ZeroGravity platform."
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": str(input_data)}
        ]
        
        return messages

    def execute_with_llm(self, input_data: Any, model: str = None, 
                         provider_type: LLMProviderType = None, 
                         use_cache: bool = True, ttl_seconds: int = 3600, 
                         **kwargs) -> str:
        """
        Execute agent logic with LLM integration.
        Uses caching by default to reduce API costs and improve performance.
        """
        # Prepare messages for the LLM
        messages = self.prepare_messages(input_data)
        
        # Try to get cached response first
        if use_cache:
            cached_response = self.cache_manager.get_cached_response(
                messages=messages,
                model=model or "gpt-4-turbo-preview",
                ttl_seconds=ttl_seconds,
                **kwargs
            )
            
            if cached_response is not None:
                return cached_response
        
        # Call the LLM if not cached
        try:
            llm_response: LLMResponse = self.llm_manager.call(
                messages=messages,
                model=model,
                provider_type=provider_type,
                **kwargs
            )
            
            # Cache the response
            if use_cache:
                self.cache_manager.cache_response(
                    messages=messages,
                    model=model or llm_response.model,
                    response=llm_response.content,
                    ttl_seconds=ttl_seconds,
                    **kwargs
                )
            
            return llm_response.content
            
        except Exception as e:
            # Log error and return structured fallback response
            error_msg = f"LLM call failed: {str(e)}"
            if self.coordinator:
                self.coordinator.log(error_msg, "ERROR")
            
            # Fallback to structured response based on system prompt
            return self._generate_structured_fallback_response(input_data)

    async def execute_with_llm_streaming(self, input_data: Any, model: str = None, 
                                        provider_type: LLMProviderType = None, 
                                        **kwargs) -> Generator[str, None, None]:
        """
        Execute agent logic with streaming LLM response.
        """
        messages = self.prepare_messages(input_data)
        
        try:
            stream = self.llm_manager.stream(
                messages=messages,
                model=model,
                provider_type=provider_type,
                **kwargs
            )
            
            for chunk in stream:
                yield chunk
                
        except Exception as e:
            error_msg = f"Streaming LLM call failed: {str(e)}"
            if self.coordinator:
                self.coordinator.log(error_msg, "ERROR")
            
            # Fallback to non-streaming call
            yield self.execute_with_llm(input_data, model, provider_type, **kwargs)

    def _generate_structured_fallback_response(self, input_data: Any) -> str:
        """Generate a structured fallback response when LLM calls fail"""
        system_prompt = self.get_system_prompt()
        if not system_prompt:
            system_prompt = f"You are a {self.role} agent in the ZeroGravity platform."
        
        # Create a simple structured response based on the role and input
        fallback_response = f"LLM call failed. As a {self.role} agent: {str(input_data)}"
        return fallback_response

    def format_response(self, response: str, expected_format: str = "json") -> Any:
        """
        Format the response according to the expected format.
        """
        if expected_format.lower() == "json":
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                # If parsing fails, return the raw response
                return response
        return response
