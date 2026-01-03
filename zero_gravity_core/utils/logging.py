"""
Centralized Logging System for ZeroGravity

This module provides centralized logging using structlog for
unified error, execution, and security logs across the platform.
"""
import structlog
import logging
import logging.config
import json
import sys
from datetime import datetime
from typing import Dict, Any, Optional
import os
from pathlib import Path


# Configure standard logging to work with structlog
def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None, 
                  json_format: bool = True) -> structlog.BoundLogger:
    """
    Setup centralized logging configuration with structlog
    """
    # Determine log level
    level = getattr(logging, log_level.upper())
    
    # Configure processors for structlog
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    
    if json_format:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )
    
    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter('%(message)s')
        file_handler.setFormatter(file_formatter)
        
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)
    
    return structlog.get_logger()


class Logger:
    """Centralized logger for ZeroGravity platform"""
    
    def __init__(self, name: str = "zerogravity", log_level: str = "INFO", 
                 log_file: Optional[str] = None):
        self.name = name
        self.logger = setup_logging(log_level, log_file)
        self.bound_logger = self.logger.bind(component=name)
    
    def bind(self, **kwargs) -> structlog.BoundLogger:
        """Bind additional context to the logger"""
        return self.bound_logger.bind(**kwargs)
    
    def info(self, message: str, **context) -> None:
        """Log info level message"""
        self.bound_logger.info(message, **context)
    
    def debug(self, message: str, **context) -> None:
        """Log debug level message"""
        self.bound_logger.debug(message, **context)
    
    def warning(self, message: str, **context) -> None:
        """Log warning level message"""
        self.bound_logger.warning(message, **context)
    
    def error(self, message: str, **context) -> None:
        """Log error level message"""
        self.bound_logger.error(message, **context)
    
    def critical(self, message: str, **context) -> None:
        """Log critical level message"""
        self.bound_logger.critical(message, **context)
    
    def log_execution(self, operation: str, status: str, duration: float = None, 
                     **context) -> None:
        """Log execution details"""
        self.bound_logger.info(
            "execution_log",
            operation=operation,
            status=status,
            duration=duration,
            **context
        )
    
    def log_security_event(self, event_type: str, user_id: str = None, 
                          ip_address: str = None, **context) -> None:
        """Log security-related events"""
        self.bound_logger.warning(
            "security_event",
            event_type=event_type,
            user_id=user_id,
            ip_address=ip_address,
            **context
        )
    
    def log_agent_activity(self, agent_role: str, session_id: str, 
                          action: str, **context) -> None:
        """Log agent-specific activities"""
        self.bound_logger.info(
            "agent_activity",
            agent_role=agent_role,
            session_id=session_id,
            action=action,
            **context
        )
    
    def log_api_request(self, endpoint: str, method: str, status_code: int,
                       user_id: str = None, **context) -> None:
        """Log API request details"""
        self.bound_logger.info(
            "api_request",
            endpoint=endpoint,
            method=method,
            status_code=status_code,
            user_id=user_id,
            **context
        )


class AuditLogger:
    """Specialized logger for audit trails and compliance"""
    
    def __init__(self, log_file: str = "audit.log"):
        self.log_file = Path(log_file)
        self.logger = Logger(name="audit")
    
    def log_decision(self, user_id: str, agent_role: str, decision: str, 
                    input_data: Any, output_data: Any, session_id: str = None) -> None:
        """Log agent decisions for audit and compliance"""
        audit_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": "agent_decision",
            "user_id": user_id,
            "agent_role": agent_role,
            "decision": decision,
            "input_data": str(input_data)[:1000],  # Limit data size
            "output_data": str(output_data)[:1000],  # Limit data size
            "session_id": session_id
        }
        
        self.logger.info("audit_decision", **audit_entry)
        
        # Also write to dedicated audit file
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(audit_entry) + "\n")
    
    def log_access(self, user_id: str, resource: str, action: str, 
                  success: bool, session_id: str = None) -> None:
        """Log access to resources"""
        audit_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": "access_log",
            "user_id": user_id,
            "resource": resource,
            "action": action,
            "success": success,
            "session_id": session_id
        }
        
        self.logger.info("audit_access", **audit_entry)
        
        # Also write to dedicated audit file
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(audit_entry) + "\n")


# Global logger instances
main_logger = Logger("zerogravity")
audit_logger = AuditLogger()


def get_logger(name: str = "zerogravity") -> Logger:
    """Get a logger instance by name"""
    return Logger(name)


def log_api_call(endpoint: str, method: str, status_code: int, 
                user_id: str = None, duration: float = None) -> None:
    """Helper function to log API calls"""
    main_logger.log_api_request(
        endpoint=endpoint,
        method=method,
        status_code=status_code,
        user_id=user_id,
        duration=duration
    )


def log_agent_execution(agent_role: str, session_id: str, 
                       input_data: Any, output_data: Any) -> None:
    """Helper function to log agent execution"""
    main_logger.log_agent_activity(
        agent_role=agent_role,
        session_id=session_id,
        action="execution",
        input=str(input_data)[:500],
        output=str(output_data)[:500]
    )


def log_security_violation(event_type: str, user_id: str = None, 
                          ip_address: str = None, details: str = None) -> None:
    """Helper function to log security violations"""
    main_logger.log_security_event(
        event_type=event_type,
        user_id=user_id,
        ip_address=ip_address,
        details=details
    )


# Initialize loggers
def init_loggers() -> None:
    """Initialize all loggers with configuration from environment"""
    log_level = os.getenv("LOG_LEVEL", "INFO")
    log_file = os.getenv("LOG_FILE")
    json_format = os.getenv("LOG_JSON_FORMAT", "true").lower() == "true"
    
    # Setup main logger
    global main_logger
    main_logger = Logger("zerogravity", log_level, log_file)
    
    # Setup audit logger
    audit_file = os.getenv("AUDIT_LOG_FILE", "audit.log")
    global audit_logger
    audit_logger = AuditLogger(audit_file)


# Initialize loggers when module is imported
init_loggers()
