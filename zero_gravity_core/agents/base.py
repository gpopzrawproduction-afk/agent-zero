# zero_gravity_core/agents/base.py

import json
from typing import Any, Dict, Optional

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

    def record(self, data: Dict[str, Any]):
        """Save something to agent memory"""
        self.memory.append(data)

    def get_memory(self):
        """Retrieve memory entries"""
        return self.memory

    def get_system_prompt(self) -> Optional[str]:
        """Retrieve the system prompt for this agent"""
        return self.system_prompt

    def execute_with_llm(self, input_data: Any) -> Any:
        """
        Placeholder method for executing agent logic with LLM integration.
        This should be overridden by subclasses to implement actual LLM reasoning.
        For now, this method uses basic string formatting to simulate LLM behavior.
        """
        # This is a placeholder implementation that simulates LLM behavior
        # In a real implementation, this would call an actual LLM API
        system_prompt = self.get_system_prompt()
        if not system_prompt:
            system_prompt = f"You are a {self.role} agent in the ZeroGravity platform."
        
        # Create a simple prompt for the "LLM"
        prompt = f"{system_prompt}\n\nInput: {input_data}\n\nResponse:"
        
        # For now, return the input data as is, but in a real implementation
        # this would call an actual LLM API and return the response
        return input_data

    def format_response(self, response: str, expected_format: str = "json") -> Any:
        """
        Format the response according to the expected format.
        In a real implementation, this would parse LLM responses.
        """
        if expected_format.lower() == "json":
            try:
                # This is a placeholder - in a real implementation, this would
                # parse the actual LLM response
                return json.loads(response)
            except json.JSONDecodeError:
                # If parsing fails, return the raw response
                return response
        return response
