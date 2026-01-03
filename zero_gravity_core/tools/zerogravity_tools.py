# zero_gravity_core/tools/zerogravity_tools.py

"""
ZeroGravity Tools Module

This module contains tools that can be used by agents in the ZeroGravity platform.
These tools provide various capabilities like file operations, API calls, data analysis, etc.
"""

import json
import os
from typing import Any, Dict, List, Optional
from pathlib import Path


class ToolError(Exception):
    """Custom exception for tool-related errors"""
    pass


def file_reader(file_path: str) -> str:
    """
    Read the contents of a file.
    
    Args:
        file_path (str): Path to the file to read
        
    Returns:
        str: Contents of the file
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        raise ToolError(f"File not found: {file_path}")
    except Exception as e:
        raise ToolError(f"Error reading file {file_path}: {str(e)}")


def file_writer(file_path: str, content: str) -> bool:
    """
    Write content to a file.
    
    Args:
        file_path (str): Path to the file to write
        content (str): Content to write to the file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        directory = os.path.dirname(file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    except Exception as e:
        raise ToolError(f"Error writing file {file_path}: {str(e)}")


def search_files(directory: str, pattern: str) -> List[str]:
    """
    Search for files in a directory matching a pattern.
    
    Args:
        directory (str): Directory to search in
        pattern (str): Pattern to match (supports * and ? wildcards)
        
    Returns:
        List[str]: List of matching file paths
    """
    import fnmatch
    
    try:
        matches = []
        for root, dirs, files in os.walk(directory):
            for file in fnmatch.filter(files, pattern):
                matches.append(os.path.join(root, file))
        return matches
    except Exception as e:
        raise ToolError(f"Error searching files in {directory}: {str(e)}")


def execute_command(command: str) -> Dict[str, Any]:
    """
    Execute a shell command safely.
    
    Args:
        command (str): Command to execute
        
    Returns:
        Dict[str, Any]: Result with stdout, stderr, and return code
    """
    import subprocess
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30  # 30 second timeout
        )
        
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": result.returncode
        }
    except subprocess.TimeoutExpired:
        raise ToolError(f"Command timed out: {command}")
    except Exception as e:
        raise ToolError(f"Error executing command {command}: {str(e)}")


def json_validator(json_string: str) -> Dict[str, Any]:
    """
    Validate and parse a JSON string.
    
    Args:
        json_string (str): JSON string to validate and parse
        
    Returns:
        Dict[str, Any]: Parsed JSON object
    """
    try:
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        raise ToolError(f"Invalid JSON: {str(e)}")


def web_scraper(url: str) -> str:
    """
    Basic web scraping function (placeholder).
    
    Args:
        url (str): URL to scrape
        
    Returns:
        str: Content of the webpage (placeholder implementation)
    """
    # This is a placeholder - in a real implementation, you'd use requests or similar
    # For now, return a message indicating this is a placeholder
    return f"Web scraping tool would fetch content from: {url}"


def calculate(expression: str) -> float:
    """
    Safely evaluate a mathematical expression.
    
    Args:
        expression (str): Mathematical expression to evaluate
        
    Returns:
        float: Result of the calculation
    """
    import re
    
    # Only allow numbers, operators, parentheses, and decimal points
    if not re.match(r'^[0-9+\-*/().\s]+$', expression):
        raise ToolError("Invalid expression: only numbers and basic operators allowed")
    
    try:
        # Use eval safely by restricting allowed characters (in a real app, use a proper parser)
        result = eval(expression, {"__builtins__": {}}, {})
        return float(result)
    except Exception as e:
        raise ToolError(f"Error evaluating expression '{expression}': {str(e)}")


# Tool registry - maps tool names to functions
TOOLS = {
    "file_reader": file_reader,
    "file_writer": file_writer,
    "search_files": search_files,
    "execute_command": execute_command,
    "json_validator": json_validator,
    "web_scraper": web_scraper,
    "calculate": calculate
}


def get_tool(name: str):
    """
    Get a tool function by name.
    
    Args:
        name (str): Name of the tool to retrieve
        
    Returns:
        function: The tool function
    """
    if name not in TOOLS:
        raise ToolError(f"Tool '{name}' not found. Available tools: {list(TOOLS.keys())}")
    
    return TOOLS[name]


def list_tools() -> List[str]:
    """
    Get a list of all available tools.
    
    Returns:
        List[str]: List of tool names
    """
    return list(TOOLS.keys())


def execute_tool(name: str, **kwargs) -> Any:
    """
    Execute a tool with the given arguments.
    
    Args:
        name (str): Name of the tool to execute
        **kwargs: Arguments to pass to the tool
        
    Returns:
        Any: Result of the tool execution
    """
    tool = get_tool(name)
    return tool(**kwargs)
