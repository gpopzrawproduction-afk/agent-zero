"""
Configuration Management System for ZeroGravity

This module implements a comprehensive configuration management system
for the ZeroGravity platform, handling environment-specific settings,
validation, and secure credential management.
"""
import os
import json
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union, List
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import secrets
import logging
from datetime import datetime
from pydantic import BaseModel, Field, validator, ValidationError
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64


class Environment(Enum):
    """Deployment environments"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


@dataclass
class DatabaseConfig:
    """Database configuration settings"""
    host: str = "localhost"
    port: int = 5432
    name: str = "zerogravity"
    user: str = "zerogravity_user"
    password: str = ""
    ssl_mode: str = "prefer"
    pool_size: int = 20
    max_overflow: int = 30
    echo_sql: bool = False
    ssl_cert_path: Optional[str] = None
    ssl_key_path: Optional[str] = None
    ssl_ca_path: Optional[str] = None
    
    def get_database_url(self) -> str:
        """Generate database URL from configuration"""
        ssl_param = f"?sslmode={self.ssl_mode}" if self.ssl_mode else ""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}{ssl_param}"


@dataclass
class RedisConfig:
    """Redis configuration settings"""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    ssl: bool = False
    ssl_cert_reqs: str = "required"
    ssl_ca_certs: Optional[str] = None
    connection_pool_size: int = 20
    socket_keepalive: bool = True
    socket_connect_timeout: int = 5
    socket_timeout: int = 10
    retry_on_timeout: bool = True
    health_check_interval: int = 30


@dataclass
class LLMProviderConfig:
    """LLM provider configuration settings"""
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    openai_base_url: str = "https://api.openai.com/v1"
    anthropic_base_url: str = "https://api.anthropic.com/v1"
    default_model: str = "gpt-4-turbo"
    timeout: int = 30
    max_retries: int = 3
    rate_limit_requests: int = 1000
    rate_limit_window: int = 60  # seconds
    max_tokens: int = 4096
    temperature: float = 0.7
    streaming_enabled: bool = True
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600  # 1 hour default


@dataclass
class SecurityConfig:
    """Security configuration settings"""
    jwt_secret: str = ""
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 1440  # 24 hours
    password_hash_rounds: int = 12
    ssl_enabled: bool = False
    ssl_cert_path: Optional[str] = None
    ssl_key_path: Optional[str] = None
    cors_allow_origins: List[str] = field(default_factory=list)
    cors_allow_credentials: bool = True
    cors_allow_methods: List[str] = field(default_factory=lambda: ["*"])
    cors_allow_headers: List[str] = field(default_factory=lambda: ["*"])
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # seconds
    input_validation_enabled: bool = True
    output_sanitization_enabled: bool = True
    encryption_key: Optional[str] = None  # Should be a base64-encoded 32-byte key


@dataclass
class MonitoringConfig:
    """Monitoring configuration settings"""
    prometheus_enabled: bool = True
    prometheus_port: int = 9090
    grafana_enabled: bool = True
    grafana_port: int = 3000
    jaeger_enabled: bool = False
    jaeger_port: int = 16686
    logging_level: str = "INFO"
    logging_format: str = "json"  # json or text
    metrics_retention_days: int = 30
    tracing_enabled: bool = False
    tracing_sample_rate: float = 1.0
    health_check_enabled: bool = True
    health_check_interval: int = 30  # seconds


@dataclass
class TaskQueueConfig:
    """Task queue configuration settings"""
    broker_url: str = "redis://localhost:6379/0"
    backend_url: str = "redis://localhost:6379/1"
    default_queue: str = "default"
    worker_concurrency: int = 4
    worker_prefetch_multiplier: int = 1
    task_ignore_result: bool = False
    result_expires: int = 3600  # 1 hour
    task_routes: Dict[str, str] = field(default_factory=dict)
    beat_scheduler: str = "database"  # or "redis", "database"
    task_serializer: str = "json"
    accept_content: List[str] = field(default_factory=lambda: ["json"])


@dataclass
class APIConfig:
    """API configuration settings"""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    reload: bool = False
    log_level: str = "info"
    timeout_keep_alive: int = 5
    limit_concurrency: Optional[int] = None
    limit_max_requests: Optional[int] = None
    backlog: int = 2048
    api_v1_prefix: str = "/api/v1"
    docs_url: str = "/docs"
    redoc_url: str = "/redoc"
    openapi_url: str = "/openapi.json"
    webhooks_enabled: bool = True
    webhooks_path: str = "/webhooks"
    streaming_enabled: bool = True
    max_upload_size: int = 10 * 1024 * 1024  # 10MB


@dataclass
class PluginConfig:
    """Plugin system configuration settings"""
    enabled: bool = True
    plugins_dir: str = "plugins"
    auto_discover: bool = True
    allow_remote_plugins: bool = False
    plugin_whitelist: List[str] = field(default_factory=list)
    plugin_blacklist: List[str] = field(default_factory=list)
    plugin_timeout: int = 30  # seconds
    plugin_memory_limit: int = 100 * 1024 * 1024  # 100MB


@dataclass
class DeploymentConfig:
    """Main deployment configuration"""
    environment: Environment = Environment.DEVELOPMENT
    project_name: str = "zerogravity"
    version: str = "1.0.0"
    debug: bool = False
    log_level: str = "INFO"
    
    # Component configurations
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    llm_providers: LLMProviderConfig = field(default_factory=LLMProviderConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    task_queue: TaskQueueConfig = field(default_factory=TaskQueueConfig)
    api: APIConfig = field(default_factory=APIConfig)
    plugins: PluginConfig = field(default_factory=PluginConfig)
    
    # Deployment-specific settings
    replica_count: int = 1
    resource_limits: Dict[str, str] = field(default_factory=lambda: {
        "cpu": "1000m",
        "memory": "2Gi"
    })
    resource_requests: Dict[str, str] = field(default_factory=lambda: {
        "cpu": "100m", 
        "memory": "256Mi"
    })
    
    # File paths
    base_dir: str = ""
    config_dir: str = "config"
    logs_dir: str = "logs"
    data_dir: str = "data"
    
    def __post_init__(self):
        """Post-initialization setup"""
        if not self.base_dir:
            self.base_dir = str(Path.cwd())
        
        # Set default CORS origins based on environment
        if not self.security.cors_allow_origins:
            if self.environment == Environment.PRODUCTION:
                self.security.cors_allow_origins = ["https://*.zerogravity.ai"]
            else:
                self.security.cors_allow_origins = ["*"]
        
        # Set default JWT secret if not provided (only for non-production)
        if not self.security.jwt_secret:
            if self.environment != Environment.PRODUCTION:
                self.security.jwt_secret = "dev-jwt-secret-change-in-production"
            else:
                raise ValueError("JWT secret must be set for production environment")
        
        # Set default encryption key if not provided
        if not self.security.encryption_key:
            self.security.encryption_key = Fernet.generate_key().decode()


class ConfigValidator(BaseModel):
    """Pydantic model for validating configuration"""
    environment: Environment
    project_name: str = Field(..., min_length=1, max_length=100)
    version: str = Field(..., regex=r"^\d+\.\d+\.\d+(-[a-zA-Z0-9]+)?$")
    debug: bool
    log_level: str = Field(..., regex=r"^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    
    # Database validation
    database_host: str
    database_port: int = Field(ge=1, le=65535)
    database_name: str
    database_user: str
    
    # Security validation
    jwt_secret: str
    cors_allow_origins: List[str]
    
    # LLM provider validation
    llm_default_model: str
    llm_timeout: int = Field(ge=1, le=300)
    llm_max_retries: int = Field(ge=0, le=10)
    
    class Config:
        """Pydantic configuration"""
        arbitrary_types_allowed = True


class ConfigManager:
    """Main configuration manager for ZeroGravity"""
    
    def __init__(self, config_path: Optional[str] = None, environment: Optional[Environment] = None):
        self.config_path = config_path or "config"
        self.environment = environment or self._detect_environment()
        self.config: Optional[DeploymentConfig] = None
        self.logger = logging.getLogger("ConfigManager")
        self._encrypted_values = {}
        
        # Initialize configuration
        self.load_config()
    
    def _detect_environment(self) -> Environment:
        """Detect the current environment from environment variables"""
        env_str = os.getenv("ENVIRONMENT", os.getenv("ZEROGRAVITY_ENV", "development")).lower()
        try:
            return Environment(env_str)
        except ValueError:
            self.logger.warning(f"Unknown environment '{env_str}', defaulting to development")
            return Environment.DEVELOPMENT
    
    def load_config(self, config_file: Optional[str] = None) -> DeploymentConfig:
        """Load configuration from file or environment variables"""
        if config_file is None:
            config_file = self._get_config_file_path()
        
        # Try to load from YAML file first
        if Path(config_file).exists():
            config_data = self._load_yaml_config(config_file)
        else:
            # Fall back to environment variables
            config_data = self._load_from_environment()
        
        # Merge with environment variables
        config_data = self._merge_with_environment(config_data)
        
        # Validate configuration
        self._validate_config(config_data)
        
        # Create configuration object
        self.config = self._create_config_from_data(config_data)
        
        # Log configuration loading
        self.logger.info(f"Configuration loaded for {self.environment.value} environment")
        
        return self.config
    
    def _get_config_file_path(self) -> str:
        """Get the appropriate config file path based on environment"""
        config_dir = Path(self.config_path)
        
        # Try environment-specific config first
        env_config = config_dir / f"config.{self.environment.value}.yaml"
        if env_config.exists():
            return str(env_config)
        
        # Fall back to default config
        default_config = config_dir / "config.yaml"
        if default_config.exists():
            return str(default_config)
        
        # Try JSON format
        env_config_json = config_dir / f"config.{self.environment.value}.json"
        if env_config_json.exists():
            return str(env_config_json)
        
        default_config_json = config_dir / "config.json"
        if default_config_json.exists():
            return str(default_config_json)
        
        # If no config file exists, return None to use environment variables only
        return ""
    
    def _load_yaml_config(self, config_file: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        with open(config_file, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        if config_data is None:
            config_data = {}
        
        return config_data
    
    def _load_from_environment(self) -> Dict[str, Any]:
        """Load configuration from environment variables"""
        config_data = {}
        
        # Load database config
        config_data["database"] = {
            "host": os.getenv("DATABASE_HOST", "localhost"),
            "port": int(os.getenv("DATABASE_PORT", "5432")),
            "name": os.getenv("DATABASE_NAME", "zerogravity"),
            "user": os.getenv("DATABASE_USER", "zerogravity_user"),
            "password": os.getenv("DATABASE_PASSWORD", ""),
            "ssl_mode": os.getenv("DATABASE_SSL_MODE", "prefer"),
            "pool_size": int(os.getenv("DATABASE_POOL_SIZE", "20")),
            "max_overflow": int(os.getenv("DATABASE_MAX_OVERFLOW", "30")),
            "echo_sql": os.getenv("DATABASE_ECHO_SQL", "false").lower() == "true"
        }
        
        # Load Redis config
        config_data["redis"] = {
            "host": os.getenv("REDIS_HOST", "localhost"),
            "port": int(os.getenv("REDIS_PORT", "6379")),
            "db": int(os.getenv("REDIS_DB", "0")),
            "password": os.getenv("REDIS_PASSWORD"),
            "ssl": os.getenv("REDIS_SSL", "false").lower() == "true",
            "connection_pool_size": int(os.getenv("REDIS_POOL_SIZE", "20"))
        }
        
        # Load LLM provider config
        config_data["llm_providers"] = {
            "openai_api_key": os.getenv("OPENAI_API_KEY"),
            "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY"),
            "openai_base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            "anthropic_base_url": os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1"),
            "default_model": os.getenv("LLM_DEFAULT_MODEL", "gpt-4-turbo"),
            "timeout": int(os.getenv("LLM_TIMEOUT", "30")),
            "max_retries": int(os.getenv("LLM_MAX_RETRIES", "3")),
            "rate_limit_requests": int(os.getenv("LLM_RATE_LIMIT_REQUESTS", "1000")),
            "rate_limit_window": int(os.getenv("LLM_RATE_LIMIT_WINDOW", "60")),
            "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "4096")),
            "temperature": float(os.getenv("LLM_TEMPERATURE", "0.7")),
            "streaming_enabled": os.getenv("LLM_STREAMING_ENABLED", "true").lower() == "true",
            "cache_enabled": os.getenv("LLM_CACHE_ENABLED", "true").lower() == "true",
            "cache_ttl_seconds": int(os.getenv("LLM_CACHE_TTL_SECONDS", "3600"))
        }
        
        # Load security config
        config_data["security"] = {
            "jwt_secret": os.getenv("JWT_SECRET", ""),
            "jwt_algorithm": os.getenv("JWT_ALGORITHM", "HS256"),
            "jwt_expire_minutes": int(os.getenv("JWT_EXPIRE_MINUTES", "1440")),
            "password_hash_rounds": int(os.getenv("PASSWORD_HASH_ROUNDS", "12")),
            "ssl_enabled": os.getenv("SSL_ENABLED", "false").lower() == "true",
            "ssl_cert_path": os.getenv("SSL_CERT_PATH"),
            "ssl_key_path": os.getenv("SSL_KEY_PATH"),
            "cors_allow_origins": os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
            "rate_limit_enabled": os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true",
            "rate_limit_requests": int(os.getenv("RATE_LIMIT_REQUESTS", "100")),
            "rate_limit_window": int(os.getenv("RATE_LIMIT_WINDOW", "60")),
            "input_validation_enabled": os.getenv("INPUT_VALIDATION_ENABLED", "true").lower() == "true",
            "output_sanitization_enabled": os.getenv("OUTPUT_SANITIZATION_ENABLED", "true").lower() == "true",
            "encryption_key": os.getenv("ENCRYPTION_KEY")
        }
        
        # Load monitoring config
        config_data["monitoring"] = {
            "prometheus_enabled": os.getenv("PROMETHEUS_ENABLED", "true").lower() == "true",
            "prometheus_port": int(os.getenv("PROMETHEUS_PORT", "9090")),
            "grafana_enabled": os.getenv("GRAFANA_ENABLED", "true").lower() == "true",
            "grafana_port": int(os.getenv("GRAFANA_PORT", "3000")),
            "jaeger_enabled": os.getenv("JAEGER_ENABLED", "false").lower() == "true",
            "jaeger_port": int(os.getenv("JAEGER_PORT", "16686")),
            "logging_level": os.getenv("LOGGING_LEVEL", "INFO"),
            "logging_format": os.getenv("LOGGING_FORMAT", "json"),
            "metrics_retention_days": int(os.getenv("METRICS_RETENTION_DAYS", "30")),
            "tracing_enabled": os.getenv("TRACING_ENABLED", "false").lower() == "true",
            "tracing_sample_rate": float(os.getenv("TRACING_SAMPLE_RATE", "1.0")),
            "health_check_enabled": os.getenv("HEALTH_CHECK_ENABLED", "true").lower() == "true",
            "health_check_interval": int(os.getenv("HEALTH_CHECK_INTERVAL", "30"))
        }
        
        # Load API config
        config_data["api"] = {
            "host": os.getenv("API_HOST", "0.0.0.0"),
            "port": int(os.getenv("API_PORT", "8000")),
            "workers": int(os.getenv("API_WORKERS", "1")),
            "reload": os.getenv("API_RELOAD", "false").lower() == "true",
            "log_level": os.getenv("API_LOG_LEVEL", "info"),
            "timeout_keep_alive": int(os.getenv("API_TIMEOUT_KEEP_ALIVE", "5")),
            "api_v1_prefix": os.getenv("API_V1_PREFIX", "/api/v1"),
            "docs_url": os.getenv("API_DOCS_URL", "/docs"),
            "redoc_url": os.getenv("API_REDOC_URL", "/redoc"),
            "openapi_url": os.getenv("API_OPENAPI_URL", "/openapi.json"),
            "webhooks_enabled": os.getenv("WEBHOOKS_ENABLED", "true").lower() == "true",
            "webhooks_path": os.getenv("WEBHOOKS_PATH", "/webhooks"),
            "streaming_enabled": os.getenv("STREAMING_ENABLED", "true").lower() == "true",
            "max_upload_size": int(os.getenv("MAX_UPLOAD_SIZE", "10485760"))  # 10MB
        }
        
        return config_data
    
    def _merge_with_environment(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Merge config data with environment variables (env vars take precedence)"""
        # This would merge environment variables with config file data
        # For brevity, this is a simplified version - in a real implementation,
        # you would have more sophisticated merging logic
        return config_data
    
    def _validate_config(self, config_data: Dict[str, Any]) -> None:
        """Validate configuration data"""
        try:
            # Flatten config for validation
            flat_data = {
                "environment": self.environment,
                "project_name": config_data.get("project_name", "zerogravity"),
                "version": config_data.get("version", "1.0.0"),
                "debug": config_data.get("debug", False),
                "log_level": config_data.get("log_level", "INFO"),
                "database_host": config_data.get("database", {}).get("host", "localhost"),
                "database_port": config_data.get("database", {}).get("port", 5432),
                "database_name": config_data.get("database", {}).get("name", "zerogravity"),
                "database_user": config_data.get("database", {}).get("user", "zerogravity_user"),
                "jwt_secret": config_data.get("security", {}).get("jwt_secret", ""),
                "cors_allow_origins": config_data.get("security", {}).get("cors_allow_origins", ["*"]),
                "llm_default_model": config_data.get("llm_providers", {}).get("default_model", "gpt-4-turbo"),
                "llm_timeout": config_data.get("llm_providers", {}).get("timeout", 30),
                "llm_max_retries": config_data.get("llm_providers", {}).get("max_retries", 3)
            }
            
            # Validate with Pydantic
            ConfigValidator(**flat_data)
            
        except ValidationError as e:
            error_messages = []
            for error in e.errors():
                field = ".".join(str(loc) for loc in error['loc'])
                message = f"{field}: {error['msg']}"
                error_messages.append(message)
            
            raise ValueError(f"Configuration validation failed: {'; '.join(error_messages)}")
    
    def _create_config_from_data(self, config_data: Dict[str, Any]) -> DeploymentConfig:
        """Create DeploymentConfig from data dictionary"""
        # Create component configs
        database_config = DatabaseConfig(**config_data.get("database", {}))
        redis_config = RedisConfig(**config_data.get("redis", {}))
        llm_config = LLMProviderConfig(**config_data.get("llm_providers", {}))
        security_config = SecurityConfig(**config_data.get("security", {}))
        monitoring_config = MonitoringConfig(**config_data.get("monitoring", {}))
        api_config = APIConfig(**config_data.get("api", {}))
        
        # Create and return deployment config
        return DeploymentConfig(
            environment=self.environment,
            project_name=config_data.get("project_name", "zerogravity"),
            version=config_data.get("version", "1.0.0"),
            debug=config_data.get("debug", False),
            log_level=config_data.get("log_level", "INFO"),
            database=database_config,
            redis=redis_config,
            llm_providers=llm_config,
            security=security_config,
            monitoring=monitoring_config,
            api=api_config
        )
    
    def get_config(self) -> DeploymentConfig:
        """Get the current configuration"""
        if self.config is None:
            self.load_config()
        return self.config
    
    def reload_config(self) -> DeploymentConfig:
        """Reload the configuration"""
        return self.load_config()
    
    def validate_config(self) -> List[str]:
        """Validate the current configuration and return any errors"""
        errors = []
        
        if self.config is None:
            return ["Configuration not loaded"]
        
        # Validate environment-specific requirements
        if self.config.environment == Environment.PRODUCTION:
            if not self.config.security.jwt_secret or self.config.security.jwt_secret == "dev-jwt-secret-change-in-production":
                errors.append("JWT secret must be set for production environment")
            
            if not self.config.database.password:
                errors.append("Database password must be set for production environment")
            
            if not self.config.llm_providers.openai_api_key and not self.config.llm_providers.anthropic_api_key:
                errors.append("At least one LLM provider API key must be set for production environment")
        
        # Validate database configuration
        if not self.config.database.host:
            errors.append("Database host must be specified")
        
        if self.config.database.port < 1 or self.config.database.port > 65535:
            errors.append("Database port must be between 1 and 65535")
        
        # Validate Redis configuration
        if not self.config.redis.host:
            errors.append("Redis host must be specified")
        
        if self.config.redis.port < 1 or self.config.redis.port > 65535:
            errors.append("Redis port must be between 1 and 65535")
        
        # Validate API configuration
        if self.config.api.port < 1 or self.config.api.port > 65535:
            errors.append("API port must be between 1 and 65535")
        
        if self.config.api.workers < 1:
            errors.append("API workers must be at least 1")
        
        return errors
    
    def encrypt_value(self, value: str) -> str:
        """Encrypt a sensitive value"""
        if not self.config or not self.config.security.encryption_key:
            raise ValueError("Encryption key not configured")
        
        fernet = Fernet(self.config.security.encryption_key.encode())
        encrypted_value = fernet.encrypt(value.encode())
        return encrypted_value.decode()
    
    def decrypt_value(self, encrypted_value: str) -> str:
        """Decrypt an encrypted value"""
        if not self.config or not self.config.security.encryption_key:
            raise ValueError("Encryption key not configured")
        
        fernet = Fernet(self.config.security.encryption_key.encode())
        decrypted_value = fernet.decrypt(encrypted_value.encode())
        return decrypted_value.decode()
    
    def mask_sensitive_data(self, data: Union[Dict[str, Any], str]) -> Union[Dict[str, Any], str]:
        """Mask sensitive data in logs or output"""
        if isinstance(data, str):
            # If it's a string, check if it contains sensitive info
            sensitive_keywords = ["password", "secret", "key", "token", "api"]
            if any(keyword in data.lower() for keyword in sensitive_keywords):
                return "*" * len(data)
            return data
        
        elif isinstance(data, dict):
            masked_data = {}
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    masked_data[key] = self.mask_sensitive_data(value)
                elif isinstance(key, str) and any(keyword in key.lower() for keyword in ["password", "secret", "key", "token", "api"]):
                    masked_data[key] = "*" * len(str(value))
                else:
                    masked_data[key] = value
            return masked_data
        
        return data
    
    def get_database_url(self) -> str:
        """Get the database URL for SQLAlchemy"""
        return self.config.database.get_database_url() if self.config else ""
    
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.environment == Environment.PRODUCTION
    
    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.environment == Environment.DEVELOPMENT
    
    def is_testing(self) -> bool:
        """Check if running in testing environment"""
        return self.environment == Environment.TESTING


# Global configuration manager instance
config_manager: Optional[ConfigManager] = None


def init_config(environment: Optional[str] = None, config_path: Optional[str] = None) -> ConfigManager:
    """Initialize the configuration manager"""
    global config_manager
    env = Environment(environment.lower()) if environment else None
    config_manager = ConfigManager(config_path, env)
    
    # Setup logging based on config
    logging.basicConfig(
        level=getattr(logging, config_manager.get_config().log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    return config_manager


def get_config() -> Optional[DeploymentConfig]:
    """Get the global configuration"""
    global config_manager
    if config_manager is None:
        config_manager = ConfigManager()
    return config_manager.get_config()


def get_config_manager() -> Optional[ConfigManager]:
    """Get the global configuration manager instance"""
    return config_manager


def reload_config() -> Optional[DeploymentConfig]:
    """Reload the configuration"""
    global config_manager
    if config_manager is None:
        config_manager = ConfigManager()
    return config_manager.reload_config()


def validate_config() -> List[str]:
    """Validate the current configuration"""
    global config_manager
    if config_manager is None:
        config_manager = ConfigManager()
    return config_manager.validate_config()


def get_database_url() -> str:
    """Get the database URL from config"""
    config = get_config()
    return config.database.get_database_url() if config else ""


def is_production() -> bool:
    """Check if running in production"""
    config_manager_instance = get_config_manager()
    return config_manager_instance.is_production() if config_manager_instance else False


def is_development() -> bool:
    """Check if running in development"""
    config_manager_instance = get_config_manager()
    return config_manager_instance.is_development() if config_manager_instance else False


def is_testing() -> bool:
    """Check if running in testing mode"""
    config_manager_instance = get_config_manager()
    return config_manager_instance.is_testing() if config_manager_instance else False


# Initialize the config manager when this module is imported
def _initialize_global_config():
    """Initialize the global configuration manager"""
    global config_manager
    if config_manager is None:
        try:
            config_manager = ConfigManager()
        except Exception as e:
            print(f"Warning: Could not initialize config manager: {e}")
            # Create a minimal config for basic functionality
            config_manager = ConfigManager()
            # Set a basic config to avoid further errors
            config_manager.config = DeploymentConfig(environment=_detect_environment())


# Detect environment helper function
def _detect_environment() -> Environment:
    """Helper to detect environment"""
    env_str = os.getenv("ENVIRONMENT", os.getenv("ZEROGRAVITY_ENV", "development")).lower()
    try:
        return Environment(env_str)
    except ValueError:
        return Environment.DEVELOPMENT


# Initialize the global config when module is imported
_initialize_global_config()


if __name__ == "__main__":
    # For testing purposes
    print("Testing ZeroGravity Configuration Manager...")
    
    # Initialize config
    config = init_config()
    
    print(f"Configuration loaded for: {config.environment.value}")
    print(f"Project: {config.project_name}")
    print(f"Version: {config.version}")
    print(f"Debug mode: {config.debug}")
    print(f"Database: {config.database.host}:{config.database.port}/{config.database.name}")
    print(f"LLM Provider: {config.llm_providers.default_model}")
    
    # Validate config
    errors = validate_config()
    if errors:
        print(f"Configuration errors: {errors}")
    else:
        print("Configuration is valid")
    
    # Test encryption/decryption
    if config.security.encryption_key:
        test_value = "sensitive_data"
        encrypted = config_manager.encrypt_value(test_value)
        decrypted = config_manager.decrypt_value(encrypted)
        print(f"Encryption test: {test_value} -> {encrypted} -> {decrypted}")
    
    # Test sensitive data masking
    test_data = {
        "password": "secret123",
        "api_key": "sk-1234567890",
        "normal_field": "normal_value"
    }
    masked = config_manager.mask_sensitive_data(test_data)
    print(f"Sensitive data masking: {test_data} -> {masked}")
