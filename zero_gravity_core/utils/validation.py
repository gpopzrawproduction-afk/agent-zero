"""
Input Sanitization and Validation System for ZeroGravity

This module provides input validation and sanitization using Pydantic
for all endpoints and agents in the ZeroGravity platform.
"""
import re
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, validator, root_validator
from enum import Enum
import html
import json
import urllib.parse
from pathlib import Path


class InputType(Enum):
    """Types of input that need validation"""
    OBJECTIVE = "objective"
    TEXT = "text"
    JSON = "json"
    URL = "url"
    FILE_PATH = "file_path"
    CODE = "code"
    COMMAND = "command"


class SanitizedInput(BaseModel):
    """Model for sanitized input"""
    original_input: str
    sanitized_input: str
    input_type: InputType
    is_valid: bool = True
    validation_errors: List[str] = Field(default_factory=list)
    
    class Config:
        arbitrary_types_allowed = True


class ObjectiveInput(BaseModel):
    """Validation model for objectives"""
    objective: str = Field(..., min_length=1, max_length=5000)
    priority: str = Field(default="normal", regex=r"^(low|normal|high)$")
    callback_url: Optional[str] = None
    
    @validator('objective')
    def validate_objective(cls, v):
        # Check for dangerous patterns
        dangerous_patterns = [
            r'<script.*?>.*?</script>',  # XSS
            r'(\.\.\/)+',               # Directory traversal
            r'javascript:',             # JavaScript protocol
            r'vbscript:',              # VBScript protocol
            r'on\w+\s*=',              # Event handlers
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, v, re.IGNORECASE):
                raise ValueError(f'Dangerous pattern detected in objective: {pattern}')
        
        # Sanitize HTML
        v = html.escape(v)
        
        return v
    
    @validator('callback_url')
    def validate_callback_url(cls, v):
        if v is None:
            return v
        
        # Validate URL format
        try:
            parsed = urllib.parse.urlparse(v)
            if not all([parsed.scheme, parsed.netloc]):
                raise ValueError('Invalid URL format')
            
            # Only allow http/https
            if parsed.scheme not in ['http', 'https']:
                raise ValueError('Only HTTP/HTTPS URLs allowed')
            
            return v
        except Exception:
            raise ValueError('Invalid callback URL')


class AgentInput(BaseModel):
    """Validation model for agent inputs"""
    input_data: Union[str, Dict[str, Any], List[Any]]
    agent_role: str = Field(..., regex=r"^(architect|engineer|designer|operator|coordinator)$")
    session_id: Optional[str] = None
    
    @validator('input_data')
    def validate_input_data(cls, v):
        # Convert to string for pattern checking
        input_str = str(v) if not isinstance(v, str) else v
        
        # Check for dangerous patterns
        dangerous_patterns = [
            r'<script.*?>.*?</script>',  # XSS
            r'(\.\.\/)+',               # Directory traversal
            r'javascript:',             # JavaScript protocol
            r'vbscript:',              # VBScript protocol
            r'on\w+\s*=',              # Event handlers
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, input_str, re.IGNORECASE):
                raise ValueError(f'Dangerous pattern detected in input: {pattern}')
        
        # If it's a string, sanitize HTML
        if isinstance(v, str):
            v = html.escape(v)
        
        return v


class APIRequestInput(BaseModel):
    """Validation model for API requests"""
    endpoint: str
    method: str = Field(..., regex=r"^(GET|POST|PUT|DELETE|PATCH)$")
    headers: Dict[str, str] = Field(default_factory=dict)
    body: Optional[Union[Dict[str, Any], str]] = None
    query_params: Dict[str, str] = Field(default_factory=dict)
    
    @validator('endpoint')
    def validate_endpoint(cls, v):
        # Check for dangerous patterns in endpoint
        dangerous_patterns = [
            r'(\.\.\/)+',               # Directory traversal
            r'<script',                 # Potential XSS
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, v, re.IGNORECASE):
                raise ValueError(f'Dangerous pattern detected in endpoint: {pattern}')
        
        return v


class FileInput(BaseModel):
    """Validation model for file inputs"""
    file_path: str
    allowed_extensions: List[str] = Field(default_factory=list)
    max_size: int = 10 * 1024 * 1024  # 10MB default
    
    @validator('file_path')
    def validate_file_path(cls, v):
        # Normalize the path to prevent directory traversal
        path = Path(v).resolve()
        base_path = Path.cwd().resolve()
        
        # Ensure the path is within the allowed base directory
        try:
            path.relative_to(base_path)
        except ValueError:
            raise ValueError("File path is outside allowed directory")
        
        # Check for dangerous patterns
        dangerous_patterns = [
            r'\.\./',      # Directory traversal
            r'\.\.\\',     # Directory traversal (Windows)
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, v, re.IGNORECASE):
                raise ValueError(f'Dangerous pattern detected in file path: {pattern}')
        
        return str(path)
    
    @validator('file_path')
    def check_file_extension(cls, v, values):
        if 'allowed_extensions' in values:
            allowed_exts = values['allowed_extensions']
            if allowed_exts and allowed_exts != []:
                file_ext = Path(v).suffix.lower()
                if file_ext not in [ext.lower() for ext in allowed_exts]:
                    raise ValueError(f'File extension not allowed: {file_ext}')
        
        return v


class CodeInput(BaseModel):
    """Validation model for code inputs"""
    code: str
    language: str = Field(..., regex=r"^(python|javascript|html|css|json|markdown|text)$")
    
    @validator('code')
    def validate_code(cls, v):
        # Check for dangerous patterns in code
        dangerous_patterns = [
            r'import\s+os\s+|from\s+os\s+import',     # OS module imports
            r'import\s+subprocess\s+|from\s+subprocess\s+import',  # Subprocess imports
            r'exec\s*\(',                            # exec() function
            r'eval\s*\(',                            # eval() function
            r'__import__',                          # Import magic method
            r'open\s*\([^)]*\)\.write',              # File write operations
            r'import\s+sys\s+|from\s+sys\s+import',   # Sys module imports
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, v, re.IGNORECASE):
                raise ValueError(f'Dangerous pattern detected in code: {pattern}')
        
        return v


class CommandInput(BaseModel):
    """Validation model for command inputs"""
    command: str
    allowed_commands: List[str] = Field(default_factory=list)
    
    @validator('command')
    def validate_command(cls, v):
        # Sanitize command to prevent command injection
        dangerous_patterns = [
            r'[;&|]',                 # Command separators
            r'\$\(.*?\)',             # Command substitution
            r'`.*?`',                 # Backtick command substitution
            r'>&',                    # Output redirection to file descriptor
            r'>>?',                   # Output redirection
            r'<',                     # Input redirection
            r'\$\(',                  # Process substitution
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, v, re.IGNORECASE):
                raise ValueError(f'Dangerous pattern detected in command: {pattern}')
        
        # Basic command validation - ensure it's a simple command
        parts = v.strip().split()
        if parts and parts[0].lower() in ['rm', 'mv', 'cp', 'chmod', 'chown']:
            raise ValueError(f'Dangerous command not allowed: {parts[0]}')
        
        return v


class InputSanitizer:
    """Main class for input sanitization and validation"""
    
    def __init__(self):
        self.validation_errors = []
    
    def sanitize_input(self, input_data: Any, input_type: InputType = InputType.TEXT) -> SanitizedInput:
        """Sanitize input based on type"""
        original_input = str(input_data)
        
        try:
            if input_type == InputType.TEXT:
                sanitized = self._sanitize_text(original_input)
            elif input_type == InputType.URL:
                sanitized = self._sanitize_url(original_input)
            elif input_type == InputType.CODE:
                sanitized = self._sanitize_code(original_input)
            elif input_type == InputType.COMMAND:
                sanitized = self._sanitize_command(original_input)
            else:
                sanitized = self._sanitize_text(original_input)
            
            return SanitizedInput(
                original_input=original_input,
                sanitized_input=sanitized,
                input_type=input_type,
                is_valid=True
            )
        except Exception as e:
            return SanitizedInput(
                original_input=original_input,
                sanitized_input=original_input,  # Return original if sanitization fails
                input_type=input_type,
                is_valid=False,
                validation_errors=[str(e)]
            )
    
    def _sanitize_text(self, text: str) -> str:
        """Sanitize general text input"""
        # Remove dangerous patterns
        dangerous_patterns = [
            r'<script.*?>.*?</script>',  # XSS
            r'(\.\.\/)+',               # Directory traversal
            r'javascript:',             # JavaScript protocol
            r'vbscript:',              # VBScript protocol
            r'on\w+\s*=',              # Event handlers
        ]
        
        for pattern in dangerous_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Escape HTML
        text = html.escape(text)
        
        return text
    
    def _sanitize_url(self, url: str) -> str:
        """Sanitize URL input"""
        try:
            parsed = urllib.parse.urlparse(url)
            if not all([parsed.scheme, parsed.netloc]):
                raise ValueError('Invalid URL format')
            
            # Only allow http/https
            if parsed.scheme not in ['http', 'https']:
                raise ValueError('Only HTTP/HTTPS URLs allowed')
            
            return url
        except Exception:
            raise ValueError('Invalid URL')
    
    def _sanitize_code(self, code: str) -> str:
        """Sanitize code input"""
        # Check for dangerous patterns
        dangerous_patterns = [
            r'import\s+os\s+|from\s+os\s+import',     # OS module imports
            r'import\s+subprocess\s+|from\s+subprocess\s+import',  # Subprocess imports
            r'exec\s*\(',                            # exec() function
            r'eval\s*\(',                            # eval() function
            r'__import__',                          # Import magic method
            r'open\s*\([^)]*\)\.write',              # File write operations
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                raise ValueError(f'Dangerous pattern detected in code: {pattern}')
        
        return code
    
    def _sanitize_command(self, command: str) -> str:
        """Sanitize command input"""
        # Sanitize command to prevent command injection
        dangerous_patterns = [
            r'[;&|]',                 # Command separators
            r'\$\(.*?\)',             # Command substitution
            r'`.*?`',                 # Backtick command substitution
            r'>&',                    # Output redirection to file descriptor
            r'>>?',                   # Output redirection
            r'<',                     # Input redirection
            r'\$\(',                  # Process substitution
        ]
        
        for pattern in dangerous_patterns:
            command = re.sub(pattern, '', command, flags=re.IGNORECASE)
        
        return command
    
    def validate_objective_input(self, objective: str, priority: str = "normal", callback_url: str = None) -> ObjectiveInput:
        """Validate objective input"""
        try:
            return ObjectiveInput(
                objective=objective,
                priority=priority,
                callback_url=callback_url
            )
        except Exception as e:
            raise ValueError(f"Invalid objective input: {str(e)}")
    
    def validate_agent_input(self, input_data: Union[str, Dict[str, Any], List[Any]], 
                           agent_role: str, session_id: str = None) -> AgentInput:
        """Validate agent input"""
        try:
            return AgentInput(
                input_data=input_data,
                agent_role=agent_role,
                session_id=session_id
            )
        except Exception as e:
            raise ValueError(f"Invalid agent input: {str(e)}")
    
    def validate_api_request(self, endpoint: str, method: str, headers: Dict[str, str] = None,
                           body: Union[Dict[str, Any], str] = None, 
                           query_params: Dict[str, str] = None) -> APIRequestInput:
        """Validate API request input"""
        try:
            return APIRequestInput(
                endpoint=endpoint,
                method=method,
                headers=headers or {},
                body=body,
                query_params=query_params or {}
            )
        except Exception as e:
            raise ValueError(f"Invalid API request input: {str(e)}")


# Global sanitizer instance
input_sanitizer = InputSanitizer()


def validate_input(input_data: Any, input_type: InputType = InputType.TEXT) -> SanitizedInput:
    """Convenience function to validate and sanitize input"""
    return input_sanitizer.sanitize_input(input_data, input_type)


def validate_objective(objective: str, priority: str = "normal", callback_url: str = None) -> ObjectiveInput:
    """Convenience function to validate objective"""
    return input_sanitizer.validate_objective_input(objective, priority, callback_url)


def validate_agent_input(input_data: Union[str, Dict[str, Any], List[Any]], 
                        agent_role: str, session_id: str = None) -> AgentInput:
    """Convenience function to validate agent input"""
    return input_sanitizer.validate_agent_input(input_data, agent_role, session_id)
