"""
API Rate Limiting System for ZeroGravity

This module implements rate limiting for the ZeroGravity API
to protect from abuse and ensure fair usage.
"""
import time
import hashlib
import threading
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
import redis
import os


class RateLimitType(Enum):
    """Types of rate limits"""
    REQUESTS_PER_MINUTE = "requests_per_minute"
    REQUESTS_PER_HOUR = "requests_per_hour"
    REQUESTS_PER_DAY = "requests_per_day"
    CONCURRENT_REQUESTS = "concurrent_requests"


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting"""
    limit: int # Number of requests allowed
    window: int  # Time window in seconds
    limit_type: RateLimitType
    per_user: bool = True  # Whether limit is per user or global


class RateLimitStore:
    """Base class for rate limit storage"""
    
    def __init__(self):
        pass
    
    def get_count(self, key: str) -> int:
        """Get current count for a key"""
        raise NotImplementedError
    
    def increment(self, key: str, expiry: int) -> int:
        """Increment count for a key and set expiry"""
        raise NotImplementedError
    
    def reset(self, key: str):
        """Reset count for a key"""
        raise NotImplementedError


class InMemoryRateLimitStore(RateLimitStore):
    """In-memory rate limit store using a dictionary"""
    
    def __init__(self):
        super().__init__()
        self.store: Dict[str, Tuple[int, float]] = {}  # (count, expiry_timestamp)
        self.lock = threading.Lock()
    
    def get_count(self, key: str) -> int:
        with self.lock:
            if key in self.store:
                count, expiry = self.store[key]
                if time.time() > expiry:
                    # Entry has expired, remove it
                    del self.store[key]
                    return 0
                return count
            return 0
    
    def increment(self, key: str, expiry: int) -> int:
        with self.lock:
            current_time = time.time()
            expiry_time = current_time + expiry
            
            if key in self.store:
                count, stored_expiry = self.store[key]
                if current_time > stored_expiry:
                    # Entry has expired, reset
                    count = 1
                else:
                    count += 1
            else:
                count = 1
            
            self.store[key] = (count, expiry_time)
            return count
    
    def reset(self, key: str):
        with self.lock:
            if key in self.store:
                del self.store[key]


class RedisRateLimitStore(RateLimitStore):
    """Redis-based rate limit store"""
    
    def __init__(self, redis_url: str = None):
        super().__init__()
        redis_url = redis_url or os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        self.redis_client = redis.from_url(redis_url)
    
    def get_count(self, key: str) -> int:
        try:
            count = self.redis_client.get(key)
            return int(count) if count else 0
        except:
            return 0  # If Redis is unavailable, allow request
    
    def increment(self, key: str, expiry: int) -> int:
        try:
            # Use Redis INCR to atomically increment the counter
            count = self.redis_client.incr(key)
            # Set expiry if this is the first increment
            if count == 1:
                self.redis_client.expire(key, expiry)
            return count
        except:
            return 0  # If Redis is unavailable, allow request
    
    def reset(self, key: str):
        try:
            self.redis_client.delete(key)
        except:
            pass # If Redis is unavailable, ignore


class RateLimiter:
    """Main rate limiter class"""
    
    def __init__(self, store: RateLimitStore = None):
        self.store = store or InMemoryRateLimitStore()
        self.default_configs: Dict[str, RateLimitConfig] = {}
        self._setup_default_configs()
    
    def _setup_default_configs(self):
        """Setup default rate limit configurations"""
        self.default_configs = {
            "api_default": RateLimitConfig(
                limit=100,  # 100 requests
                window=60,  # per minute
                limit_type=RateLimitType.REQUESTS_PER_MINUTE
            ),
            "api_high_tier": RateLimitConfig(
                limit=100,  # 1000 requests
                window=60,  # per minute
                limit_type=RateLimitType.REQUESTS_PER_MINUTE
            ),
            "api_low_tier": RateLimitConfig(
                limit=10,  # 10 requests
                window=60,  # per minute
                limit_type=RateLimitType.REQUESTS_PER_MINUTE
            )
        }
    
    def _generate_key(self, identifier: str, config: RateLimitConfig, endpoint: str = None) -> str:
        """Generate a unique key for rate limiting"""
        if endpoint:
            key = f"rate_limit:{endpoint}:{identifier}:{config.limit_type.value}"
        else:
            key = f"rate_limit:{identifier}:{config.limit_type.value}"
        
        # Use hash to ensure key is not too long
        return hashlib.sha256(key.encode()).hexdigest()
    
    def is_allowed(self, identifier: str, config_name: str = "api_default", 
                   endpoint: str = None, custom_config: RateLimitConfig = None) -> Tuple[bool, Dict[str, int]]:
        """
        Check if a request is allowed based on rate limits
        
        Returns:
            Tuple of (is_allowed, rate_limit_info)
            rate_limit_info contains: {'limit': int, 'remaining': int, 'reset_time': int}
        """
        config = custom_config or self.default_configs.get(config_name)
        if not config:
            raise ValueError(f"Rate limit configuration '{config_name}' not found")
        
        key = self._generate_key(identifier, config, endpoint)
        
        # Get current count
        current_count = self.store.get_count(key)
        
        # Check if limit is exceeded
        is_allowed = current_count < config.limit
        
        # Calculate remaining requests
        remaining = max(0, config.limit - current_count)
        
        # Calculate reset time (in seconds since epoch)
        # For now, we'll use a simple calculation based on window
        reset_time = int(time.time()) + config.window
        
        rate_info = {
            'limit': config.limit,
            'remaining': remaining,
            'reset_time': reset_time
        }
        
        if is_allowed:
            # Increment the counter
            self.store.increment(key, config.window)
        
        return is_allowed, rate_info
    
    def get_rate_limit_headers(self, rate_info: Dict[str, int]) -> Dict[str, str]:
        """Get rate limit headers for API response"""
        return {
            'X-RateLimit-Limit': str(rate_info['limit']),
            'X-RateLimit-Remaining': str(rate_info['remaining']),
            'X-RateLimit-Reset': str(rate_info['reset_time'])
        }
    
    def apply_rate_limit(self, identifier: str, config_name: str = "api_default", 
                        endpoint: str = None, custom_config: RateLimitConfig = None) -> Tuple[bool, Dict[str, str]]:
        """
        Apply rate limiting and return headers for API response
        
        Returns:
            Tuple of (is_allowed, headers)
        """
        is_allowed, rate_info = self.is_allowed(identifier, config_name, endpoint, custom_config)
        headers = self.get_rate_limit_headers(rate_info)
        
        return is_allowed, headers


class UserRateLimiter(RateLimiter):
    """Rate limiter specifically for user-based rate limiting"""
    
    def check_user_limit(self, user_id: str, tier: str = "default", 
                        endpoint: str = None) -> Tuple[bool, Dict[str, str]]:
        """
        Check rate limit for a specific user
        
        Args:
            user_id: The ID of the user
            tier: User tier (default, premium, etc.)
            endpoint: Specific API endpoint (optional)
        
        Returns:
            Tuple of (is_allowed, headers)
        """
        config_name = f"api_{tier}_tier" if tier != "default" else "api_default"
        return self.apply_rate_limit(f"user:{user_id}", config_name, endpoint)
    
    def check_ip_limit(self, ip_address: str, endpoint: str = None) -> Tuple[bool, Dict[str, str]]:
        """
        Check rate limit for a specific IP address
        
        Args:
            ip_address: The IP address to check
            endpoint: Specific API endpoint (optional)
        
        Returns:
            Tuple of (is_allowed, headers)
        """
        return self.apply_rate_limit(f"ip:{ip_address}", "api_default", endpoint)
    
    def check_endpoint_limit(self, endpoint: str) -> Tuple[bool, Dict[str, str]]:
        """
        Check global rate limit for an endpoint
        
        Args:
            endpoint: The API endpoint to check
            
        Returns:
            Tuple of (is_allowed, headers)
        """
        # For global endpoint limits, we'll use a special identifier
        return self.apply_rate_limit(f"global:{endpoint}", "api_default", endpoint)


class RateLimitMiddleware:
    """Middleware class for integrating rate limiting into web frameworks"""
    
    def __init__(self, rate_limiter: RateLimiter = None):
        self.rate_limiter = rate_limiter or UserRateLimiter()
    
    def process_request(self, user_id: str = None, ip_address: str = None, 
                      endpoint: str = None, tier: str = "default") -> Tuple[bool, Dict[str, str], str]:
        """
        Process a request through rate limiting
        
        Args:
            user_id: User ID (if authenticated)
            ip_address: IP address of the request
            endpoint: API endpoint being accessed
            tier: User tier for rate limiting
            
        Returns:
            Tuple of (is_allowed, headers, error_message)
        """
        # First check user-specific limit if user is authenticated
        if user_id:
            is_allowed, headers = self.rate_limiter.check_user_limit(user_id, tier, endpoint)
            if not is_allowed:
                return False, headers, f"Rate limit exceeded for user {user_id}"
        
        # Then check IP-based limit
        if ip_address:
            is_allowed, headers = self.rate_limiter.check_ip_limit(ip_address, endpoint)
            if not is_allowed:
                return False, headers, f"Rate limit exceeded for IP {ip_address}"
        
        # Finally check endpoint-specific global limit
        is_allowed, headers = self.rate_limiter.check_endpoint_limit(endpoint)
        if not is_allowed:
            return False, headers, f"Rate limit exceeded for endpoint {endpoint}"
        
        # If we get here, the request is allowed
        # We need to get fresh headers after the final check
        _, headers = self.rate_limiter.check_endpoint_limit(endpoint)
        
        return True, headers, None


# Global rate limiter instances
default_rate_limiter = UserRateLimiter()
redis_rate_limiter = None

# Initialize Redis rate limiter if Redis is available
try:
    redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
    redis_client = redis.from_url(redis_url)
    redis_client.ping()  # Test connection
    redis_rate_limiter = UserRateLimiter(RedisRateLimitStore(redis_url))
except:
    # Redis not available, continue with in-memory store
    pass

# Use Redis if available, otherwise use in-memory
rate_limiter = redis_rate_limiter or default_rate_limiter


def check_rate_limit(user_id: str = None, ip_address: str = None, 
                    endpoint: str = None, tier: str = "default") -> Tuple[bool, Dict[str, str], str]:
    """
    Convenience function to check rate limits
    
    Args:
        user_id: User ID (if authenticated)
        ip_address: IP address of the request
        endpoint: API endpoint being accessed
        tier: User tier for rate limiting
        
    Returns:
        Tuple of (is_allowed, headers, error_message)
    """
    return rate_limiter.check_user_limit(user_id, tier, endpoint) if user_id else \
           rate_limiter.check_ip_limit(ip_address, endpoint) if ip_address else \
           rate_limiter.check_endpoint_limit(endpoint)


def get_rate_limit_headers(user_id: str = None, ip_address: str = None, 
                          endpoint: str = None, tier: str = "default") -> Dict[str, str]:
    """
    Convenience function to get rate limit headers for a request
    
    Args:
        user_id: User ID (if authenticated)
        ip_address: IP address of the request
        endpoint: API endpoint being accessed
        tier: User tier for rate limiting
        
    Returns:
        Dictionary of rate limit headers
    """
    is_allowed, headers, error_msg = check_rate_limit(user_id, ip_address, endpoint, tier)
    return headers
