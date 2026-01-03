"""
LLM Response Caching System for ZeroGravity

This module provides caching capabilities for LLM responses
to reduce API costs and improve response times for repeated prompts.
"""
import hashlib
import json
import time
from typing import Any, Dict, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import pickle
import sqlite3
import threading


class BaseCache:
    """Abstract base class for cache implementations"""
    
    def __init__(self):
        self.lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Tuple[Any, datetime]]:
        """Get a cached response and its creation time"""
        raise NotImplementedError
    
    def set(self, key: str, value: Any, ttl_seconds: int = 3600):
        """Set a cached response with TTL"""
        raise NotImplementedError
    
    def delete(self, key: str):
        """Delete a cached response"""
        raise NotImplementedError
    
    def clear(self):
        """Clear all cached responses"""
        raise NotImplementedError


class InMemoryCache(BaseCache):
    """Simple in-memory cache implementation"""
    
    def __init__(self):
        super().__init__()
        self.cache: Dict[str, Tuple[Any, datetime]] = {}
    
    def get(self, key: str) -> Optional[Tuple[Any, datetime]]:
        with self.lock:
            if key in self.cache:
                value, created_at = self.cache[key]
                return value, created_at
            return None
    
    def set(self, key: str, value: Any, ttl_seconds: int = 3600):
        with self.lock:
            created_at = datetime.utcnow()
            self.cache[key] = (value, created_at)
    
    def delete(self, key: str):
        with self.lock:
            if key in self.cache:
                del self.cache[key]
    
    def clear(self):
        with self.lock:
            self.cache.clear()


class FileCache(BaseCache):
    """File-based cache implementation"""
    
    def __init__(self, cache_dir: str = "cache"):
        super().__init__()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def _get_cache_file_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.cache"
    
    def _is_expired(self, created_at: datetime, ttl_seconds: int) -> bool:
        return (datetime.utcnow() - created_at).total_seconds() > ttl_seconds
    
    def get(self, key: str) -> Optional[Tuple[Any, datetime]]:
        cache_file = self._get_cache_file_path(key)
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
                value, created_at, ttl_seconds = data
                if self._is_expired(created_at, ttl_seconds):
                    cache_file.unlink()  # Remove expired cache
                    return None
                return value, created_at
        except (pickle.PickleError, EOFError, FileNotFoundError):
            # If there's an error reading the cache file, remove it
            cache_file.unlink(missing_ok=True)
            return None
    
    def set(self, key: str, value: Any, ttl_seconds: int = 3600):
        cache_file = self._get_cache_file_path(key)
        created_at = datetime.utcnow()
        data = (value, created_at, ttl_seconds)
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
        except (pickle.PickleError, OSError):
            # If there's an error writing the cache file, ignore
            pass
    
    def delete(self, key: str):
        cache_file = self._get_cache_file_path(key)
        cache_file.unlink(missing_ok=True)
    
    def clear(self):
        for cache_file in self.cache_dir.glob("*.cache"):
            cache_file.unlink()


class SQLiteCache(BaseCache):
    """SQLite-based cache implementation"""
    
    def __init__(self, db_path: str = "cache.db"):
        super().__init__()
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value BLOB,
                    created_at TIMESTAMP,
                    ttl_seconds INTEGER
                )
            ''')
            conn.commit()
    
    def get(self, key: str) -> Optional[Tuple[Any, datetime]]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                'SELECT value, created_at, ttl_seconds FROM cache WHERE key = ?',
                (key,)
            )
            row = cursor.fetchone()
            
            if row:
                value, created_at_str, ttl_seconds = row
                created_at = datetime.fromisoformat(created_at_str)
                
                if (datetime.utcnow() - created_at).total_seconds() > ttl_seconds:
                    # Remove expired entry
                    conn.execute('DELETE FROM cache WHERE key = ?', (key,))
                    conn.commit()
                    return None
                
                try:
                    value = pickle.loads(value)
                    return value, created_at
                except pickle.PickleError:
                    # Remove corrupted entry
                    conn.execute('DELETE FROM cache WHERE key = ?', (key,))
                    conn.commit()
                    return None
            return None
    
    def set(self, key: str, value: Any, ttl_seconds: int = 3600):
        with sqlite3.connect(self.db_path) as conn:
            created_at = datetime.utcnow()
            value_bytes = pickle.dumps(value)
            
            conn.execute('''
                INSERT OR REPLACE INTO cache (key, value, created_at, ttl_seconds)
                VALUES (?, ?, ?, ?)
            ''', (key, value_bytes, created_at.isoformat(), ttl_seconds))
            conn.commit()
    
    def delete(self, key: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('DELETE FROM cache WHERE key = ?', (key,))
            conn.commit()
    
    def clear(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('DELETE FROM cache')
            conn.commit()


class CacheManager:
    """Manages caching for LLM responses"""
    
    def __init__(self, cache_backend: str = "inmemory", **kwargs):
        self.backend = self._create_backend(cache_backend, **kwargs)
        self.enabled = True
    
    def _create_backend(self, backend_type: str, **kwargs):
        if backend_type.lower() == "inmemory":
            return InMemoryCache()
        elif backend_type.lower() == "file":
            cache_dir = kwargs.get("cache_dir", "cache")
            return FileCache(cache_dir)
        elif backend_type.lower() == "sqlite":
            db_path = kwargs.get("db_path", "cache.db")
            return SQLiteCache(db_path)
        else:
            raise ValueError(f"Unsupported cache backend: {backend_type}")
    
    def generate_key(self, messages: list, model: str, **kwargs) -> str:
        """Generate a unique cache key from the input parameters"""
        # Create a hash of the input parameters to use as cache key
        cache_input = {
            "messages": messages,
            "model": model,
            "kwargs": kwargs
        }
        cache_input_str = json.dumps(cache_input, sort_keys=True, default=str)
        return hashlib.sha256(cache_input_str.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Tuple[Any, datetime]]:
        """Get a cached response"""
        if not self.enabled:
            return None
        return self.backend.get(key)
    
    def set(self, key: str, value: Any, ttl_seconds: int = 3600):
        """Set a cached response"""
        if self.enabled:
            self.backend.set(key, value, ttl_seconds)
    
    def delete(self, key: str):
        """Delete a cached response"""
        if self.enabled:
            self.backend.delete(key)
    
    def clear(self):
        """Clear all cached responses"""
        if self.enabled:
            self.backend.clear()
    
    def get_cached_response(self, messages: list, model: str, ttl_seconds: int = 3600, **kwargs) -> Optional[Any]:
        """Get a cached response for the given parameters"""
        if not self.enabled:
            return None
        
        key = self.generate_key(messages, model, **kwargs)
        result = self.get(key)
        
        if result:
            cached_value, created_at = result
            return cached_value
        
        return None
    
    def cache_response(self, messages: list, model: str, response: Any, ttl_seconds: int = 3600, **kwargs):
        """Cache a response for the given parameters"""
        if not self.enabled:
            return
        
        key = self.generate_key(messages, model, **kwargs)
        self.set(key, response, ttl_seconds)


# Global cache manager instance
cache_manager = CacheManager()
