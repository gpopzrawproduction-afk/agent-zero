"""
Error Handling and Retry Mechanisms for ZeroGravity

This module provides comprehensive error handling, retry mechanisms,
and fallback strategies for the ZeroGravity platform.
"""
import time
import functools
import random
import logging
from typing import Callable, Any, Optional, Dict, List, Type
from enum import Enum
from datetime import datetime
import traceback
import asyncio
from contextlib import contextmanager


class RetryStrategy(Enum):
    """Different retry strategies"""
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIXED_DELAY = "fixed_delay"
    FIBONACCI = "fibonacci"


class ErrorCategory(Enum):
    """Categories of errors for better handling"""
    NETWORK_ERROR = "network_error"
    API_ERROR = "api_error"
    VALIDATION_ERROR = "validation_error"
    RESOURCE_ERROR = "resource_error"
    BUSINESS_LOGIC_ERROR = "business_logic_error"
    UNKNOWN_ERROR = "unknown_error"


class ZeroGravityError(Exception):
    """Base exception for ZeroGravity platform"""
    def __init__(self, message: str, category: ErrorCategory = ErrorCategory.UNKNOWN_ERROR, 
                 original_exception: Exception = None, context: Dict[str, Any] = None):
        super().__init__(message)
        self.category = category
        self.original_exception = original_exception
        self.context = context or {}
        self.timestamp = datetime.utcnow()
        self.traceback = traceback.format_exc() if original_exception else None


class RetryConfig:
    """Configuration for retry mechanisms"""
    def __init__(self, max_retries: int = 3, delay: float = 1.0, 
                 strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF,
                 backoff_factor: float = 2.0, jitter: bool = True,
                 retry_on_exceptions: List[Type[Exception]] = None):
        self.max_retries = max_retries
        self.delay = delay
        self.strategy = strategy
        self.backoff_factor = backoff_factor
        self.jitter = jitter
        self.retry_on_exceptions = retry_on_exceptions or [Exception]


def calculate_delay(attempt: int, config: RetryConfig) -> float:
    """Calculate delay based on retry strategy and attempt number"""
    if config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
        delay = config.delay * (config.backoff_factor ** attempt)
    elif config.strategy == RetryStrategy.LINEAR_BACKOFF:
        delay = config.delay * (attempt + 1)
    elif config.strategy == RetryStrategy.FIXED_DELAY:
        delay = config.delay
    elif config.strategy == RetryStrategy.FIBONACCI:
        delay = config.delay * fibonacci(attempt + 1)
    else:
        delay = config.delay
    
    # Add jitter to prevent thundering herd
    if config.jitter:
        delay = delay * (0.5 + random.random() * 0.5)
    
    return delay


def fibonacci(n: int) -> int:
    """Calculate nth Fibonacci number"""
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


def retry_on_failure(config: RetryConfig = None):
    """Decorator for retrying failed operations"""
    if config is None:
        config = RetryConfig()
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(config.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except tuple(config.retry_on_exceptions) as e:
                    last_exception = e
                    if attempt < config.max_retries:
                        delay = calculate_delay(attempt, config)
                        logging.warning(f"Function {func.__name__} failed on attempt {attempt + 1}, "
                                      f"retrying in {delay:.2f}s: {str(e)}")
                        time.sleep(delay)
                    else:
                        logging.error(f"Function {func.__name__} failed after {config.max_retries} attempts: {str(e)}")
            
            # If all retries failed, raise the last exception
            raise last_exception
        return wrapper
    return decorator


async def async_retry_on_failure(config: RetryConfig = None):
    """Async version of retry decorator"""
    if config is None:
        config = RetryConfig()
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(config.max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except tuple(config.retry_on_exceptions) as e:
                    last_exception = e
                    if attempt < config.max_retries:
                        delay = calculate_delay(attempt, config)
                        logging.warning(f"Async function {func.__name__} failed on attempt {attempt + 1}, "
                                      f"retrying in {delay:.2f}s: {str(e)}")
                        await asyncio.sleep(delay)
                    else:
                        logging.error(f"Async function {func.__name__} failed after {config.max_retries} attempts: {str(e)}")
            
            raise last_exception
        return wrapper
    return decorator


class CircuitBreakerState(Enum):
    """States for circuit breaker pattern"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit broken, requests fail fast
    HALF_OPEN = "half_open" # Testing if circuit can be closed


class CircuitBreaker:
    """Circuit breaker implementation to prevent cascading failures"""
    def __init__(self, failure_threshold: int = 5, timeout: float = 60.0, 
                 success_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.success_threshold = success_threshold
        
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.success_count = 0
        self.lock = asyncio.Lock()
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        async with self.lock:
            if self.state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitBreakerState.HALF_OPEN
                else:
                    raise ZeroGravityError(
                        f"Circuit breaker is OPEN, request rejected: {func.__name__}",
                        ErrorCategory.RESOURCE_ERROR
                    )
        
        try:
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            
            await self._on_success()
            return result
            
        except Exception as e:
            await self._on_failure()
            raise e
    
    async def _on_success(self):
        """Handle successful execution"""
        async with self.lock:
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.success_threshold:
                    await self._close()
            else:
                self.failure_count = 0  # Reset on success
    
    async def _on_failure(self):
        """Handle failed execution"""
        async with self.lock:
            self.failure_count += 1
            self.last_failure_time = datetime.utcnow()
            
            if self.state != CircuitBreakerState.OPEN and self.failure_count >= self.failure_threshold:
                await self._open()
    
    async def _open(self):
        """Open the circuit"""
        self.state = CircuitBreakerState.OPEN
        logging.warning("Circuit breaker OPENED")
    
    async def _close(self):
        """Close the circuit"""
        async with self.lock:
            self.state = CircuitBreakerState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.last_failure_time = None
            logging.info("Circuit breaker CLOSED")
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit"""
        if self.last_failure_time is None:
            return False
        return (datetime.utcnow() - self.last_failure_time).total_seconds() > self.timeout


class FallbackManager:
    """Manages fallback strategies for different scenarios"""
    
    def __init__(self):
        self.fallback_strategies = {}
    
    def register_fallback(self, error_type: Type[Exception], fallback_func: Callable):
        """Register a fallback function for a specific error type"""
        self.fallback_strategies[error_type] = fallback_func
    
    def execute_with_fallback(self, primary_func: Callable, fallback_func: Callable = None, 
                            *args, **kwargs) -> Any:
        """Execute primary function with fallback"""
        try:
            return primary_func(*args, **kwargs)
        except Exception as e:
            # Check if we have a registered fallback for this error type
            for error_type, registered_fallback in self.fallback_strategies.items():
                if isinstance(e, error_type):
                    logging.warning(f"Using registered fallback for {error_type.__name__}: {str(e)}")
                    return registered_fallback(e, *args, **kwargs)
            
            # Use provided fallback if available
            if fallback_func:
                logging.warning(f"Using provided fallback: {str(e)}")
                return fallback_func(e, *args, **kwargs)
            
            # Use default fallback
            logging.warning(f"Using default fallback: {str(e)}")
            return self._default_fallback(e, *args, **kwargs)
    
    def _default_fallback(self, exception: Exception, *args, **kwargs) -> Any:
        """Default fallback behavior"""
        logging.error(f"Default fallback executed for exception: {str(exception)}")
        # Return a safe default value based on expected return type
        # This is a simplified version - in practice, you'd want more sophisticated type detection
        return {"error": str(exception), "fallback_used": True}


class ErrorHandler:
    """Centralized error handler for the ZeroGravity platform"""
    
    def __init__(self):
        self.fallback_manager = FallbackManager()
        self.circuit_breaker = CircuitBreaker()
        self.error_log = []
    
    def log_error(self, error: Exception, context: Dict[str, Any] = None, 
                  severity: str = "ERROR"):
        """Log an error with context"""
        error_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "severity": severity,
            "error_type": type(error).__name__,
            "message": str(error),
            "context": context or {},
            "traceback": traceback.format_exc()
        }
        self.error_log.append(error_entry)
        logging.log(getattr(logging, severity), f"{error_entry['error_type']}: {error_entry['message']}")
    
    @contextmanager
    def error_context(self, context: Dict[str, Any]):
        """Context manager for adding context to error handling"""
        try:
            yield
        except Exception as e:
            self.log_error(e, context)
            raise
    
    def safe_execute(self, func: Callable, fallback_func: Callable = None, 
                    context: Dict[str, Any] = None, *args, **kwargs) -> Any:
        """Safely execute a function with comprehensive error handling"""
        try:
            with self.error_context(context or {}):
                if asyncio.iscoroutinefunction(func):
                    return asyncio.run(self.circuit_breaker.call(func, *args, **kwargs))
                else:
                    return func(*args, **kwargs)
        except Exception as e:
            return self.fallback_manager.execute_with_fallback(func, fallback_func, *args, **kwargs)
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get a summary of errors"""
        if not self.error_log:
            return {"total_errors": 0, "recent_errors": []}
        
        recent_errors = self.error_log[-10:]  # Last 10 errors
        error_types = {}
        
        for entry in self.error_log:
            error_type = entry["error_type"]
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        return {
            "total_errors": len(self.error_log),
            "recent_errors": recent_errors,
            "error_types": error_types,
            "unique_error_types": len(error_types)
        }


# Global error handler instance
error_handler = ErrorHandler()
